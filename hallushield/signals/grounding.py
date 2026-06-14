"""Core grounding signal — wraps LettuceDetect (Phase 1).

LettuceDetect (ModernBERT, 8k context, token-level span detection, ~79% F1 on
RAGTruth) is the adopted core grounding scorer. It predicts which spans of an
answer are unsupported by the context; we turn that into a per-claim
groundedness score = 1 - (fraction of the claim flagged as unsupported).

Lazy-imports lettucedetect so the package installs without it. To enable:
    pip install -e ".[ml]"
then set  config.ENABLED_SIGNALS = ["grounding"].
"""

from __future__ import annotations

from ..config import MODELS
from ..core.types import Chunk, SignalResult


class GroundingSignal:
    name = "grounding"

    def __init__(self, model_path: str | None = None, method: str = "transformer") -> None:
        try:
            from lettucedetect.models.inference import HallucinationDetector  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise ImportError(
                "GroundingSignal needs the ML extras. Install: pip install -e \".[ml]\""
            ) from exc
        self._detector = HallucinationDetector(
            method=method, model_path=model_path or MODELS["grounding"]
        )

    def score(self, claim: str, chunks: list[Chunk], query: str = "") -> SignalResult:
        context = [c.text for c in chunks]
        # LettuceDetect returns spans of `claim` that are unsupported by context.
        # Threading the real query materially improves detection (the model was
        # trained on QA/summarization with a question); empty query under-detects.
        spans = self._detector.predict(
            context=context, question=query, answer=claim, output_format="spans"
        )
        flagged = sum(max(0, s["end"] - s["start"]) for s in spans)
        # Spans can overlap, so flagged length may exceed len(claim); clamp the
        # unsupported fraction to [0, 1] before inverting it.
        coverage = min(1.0, flagged / len(claim)) if claim else 0.0
        score = 1.0 - coverage

        # Attribution: lexical-overlap fallback to pick the most relevant chunk
        # for the UI (LettuceDetect scores support, it doesn't rank chunks).
        from ..core.text import content_tokens  # noqa: PLC0415

        ct = content_tokens(claim)
        best_id = None
        if ct and chunks:
            best_id = max(
                chunks, key=lambda c: len(ct & content_tokens(c.text))
            ).id
        evidence = "; ".join(s.get("text", "") for s in spans)[:160]
        return SignalResult(self.name, score=score, evidence=evidence, best_chunk_id=best_id)
