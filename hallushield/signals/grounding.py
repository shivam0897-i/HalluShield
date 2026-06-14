"""Core grounding signal — wraps LettuceDetect (Phase 1).

LettuceDetect (ModernBERT, 8k context, token-level span detection, ~79% F1 on
RAGTruth) is the adopted core grounding scorer. It predicts which spans of an
answer are unsupported by the context; we turn that into a per-claim
groundedness score using span-sensitive aggregation (see config.GROUNDING_SPAN_PENALTY).

Lazy-imports lettucedetect so the package installs without it. To enable:
    pip install -e ".[ml]"
then set  config.ENABLED_SIGNALS = ["grounding"].

The detector is injectable: pass `detector=<obj>` to bypass the real import
entirely (mirrors the LogicJudgeSignal pattern for unit testing).
"""

from __future__ import annotations

from typing import Any

from ..config import GROUNDING_SPAN_PENALTY, MODELS
from ..core.types import Chunk, SignalResult


def _build_detector(model_path: str | None, method: str) -> Any:
    """Lazily construct the real HallucinationDetector; raises ImportError without .[ml]."""
    try:
        from lettucedetect.models.inference import HallucinationDetector  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise ImportError(
            "GroundingSignal needs the ML extras. Install: pip install -e \".[ml]\""
        ) from exc
    return HallucinationDetector(
        method=method, model_path=model_path or MODELS["grounding"]
    )


class GroundingSignal:
    name = "grounding"

    def __init__(
        self,
        model_path: str | None = None,
        method: str = "transformer",
        detector: Any = None,
    ) -> None:
        # Use the injected detector as-is; only build the real one when needed.
        if detector is not None:
            self._detector = detector
        else:
            self._detector = _build_detector(model_path, method)

    def score(self, claim: str, chunks: list[Chunk], query: str = "") -> SignalResult:
        if not claim.strip():
            return SignalResult(self.name, 0.0)

        context = [c.text for c in chunks]
        # LettuceDetect returns spans of `claim` that are unsupported by context.
        # Threading the real query materially improves detection (the model was
        # trained on QA/summarization with a question); empty query under-detects.
        spans = self._detector.predict(
            context=context, question=query, answer=claim, output_format="spans"
        )
        flagged = sum(max(0, s["end"] - s["start"]) for s in spans)
        # Spans can overlap; clamp unsupported fraction to [0, 1].
        coverage = min(1.0, flagged / len(claim))
        # Apply a flat penalty whenever any span is flagged, so a short but
        # critical hallucination (e.g. a wrong dosage of only ~4 chars in a
        # 50-char claim) cannot sneak past the strict medical threshold on
        # coverage alone.  No spans → penalty is 0 → score is 1.0.
        penalty = GROUNDING_SPAN_PENALTY if spans else 0.0
        score = max(0.0, 1.0 - coverage - penalty)

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
