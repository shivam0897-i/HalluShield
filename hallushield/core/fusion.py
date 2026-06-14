"""FusionScorer: combine the enabled signals into one groundedness score.

Weighted mean over the enabled signals, minus an additive contradiction
penalty, clipped to [0, 1] (HALLUSHIELD.md §7.6). With a single enabled
signal this is just that signal's score, so the same code path serves every
ablation configuration.
"""

from __future__ import annotations

from .types import Chunk, Signal, SignalResult


class FusionScorer:
    def __init__(
        self,
        signals: list[Signal],
        weights: dict[str, float] | None = None,
        contradiction_penalty: float = 0.25,
    ) -> None:
        self.signals = list(signals)
        self.weights = weights or {}
        self.contradiction_penalty = contradiction_penalty

    def score_claim(
        self, claim: str, chunks: list[Chunk]
    ) -> tuple[float, dict[str, SignalResult], str | None]:
        """Return (fused_score, per-signal results, best supporting chunk id)."""
        results: dict[str, SignalResult] = {s.name: s.score(claim, chunks) for s in self.signals}
        if not results:
            return 0.0, results, None

        total_w = sum(self.weights.get(name, 1.0) for name in results)
        if total_w > 0:
            weighted = sum(self.weights.get(name, 1.0) * r.score for name, r in results.items())
            fused = weighted / total_w
        else:
            # All enabled signals had weight 0 (or negative) — fall back to an
            # unweighted mean rather than silently collapsing the score to 0.
            fused = sum(r.score for r in results.values()) / len(results)

        contradiction = max((r.contradiction for r in results.values()), default=0.0)
        fused -= self.contradiction_penalty * contradiction
        fused = max(0.0, min(1.0, fused))

        # Attribution: the supporting chunk from the highest-scoring signal
        # that actually identified one.
        attributing = [r for r in results.values() if r.best_chunk_id is not None]
        best_chunk = max(attributing, key=lambda r: r.score).best_chunk_id if attributing else None

        return fused, results, best_chunk
