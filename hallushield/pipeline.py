"""The validate pipeline — the core of `POST /validate`.

`validate(answer, chunks, domain)` splits the answer into sentences, scores
each via the fused enabled signals, assigns a per-sentence verdict, and rolls
up to an answer-level score/verdict. Retrieval and generation are added in
Phase 1; this function is the part the firewall is actually built around and
is reused by both the API route and the eval harness.
"""

from __future__ import annotations

from . import config
from .core.fusion import FusionScorer
from .core.sentences import split_sentences
from .core.types import Chunk, ClaimScore, ValidationResult, Verdict, worst
from .decision.engine import decide
from .signals import build_signal


def build_default_fusion() -> FusionScorer:
    """Construct the fusion scorer from config (the ablation control point)."""
    signals = [build_signal(name) for name in config.ENABLED_SIGNALS]
    return FusionScorer(
        signals,
        weights=config.DEFAULT_WEIGHTS,
        contradiction_penalty=config.CONTRADICTION_PENALTY,
    )


def validate(
    answer: str,
    chunks: list[Chunk],
    domain: str = "general",
    fusion: FusionScorer | None = None,
) -> ValidationResult:
    fusion = fusion or build_default_fusion()

    claims: list[ClaimScore] = []
    for i, sentence in enumerate(split_sentences(answer)):
        fused, results, best_chunk = fusion.score_claim(sentence, chunks)
        claims.append(
            ClaimScore(
                sentence_id=i,
                claim=sentence,
                fused_score=fused,
                verdict=decide(fused, domain),
                supporting_chunk=best_chunk,
                signals=results,
            )
        )

    if claims:
        scores = [c.fused_score for c in claims]
        answer_score = min(scores) if config.ANSWER_AGGREGATE == "min" else sum(scores) / len(scores)
        # Firewall semantics: the answer is only as good as its worst claim.
        answer_verdict = worst([c.verdict for c in claims])
    else:
        # Fail CLOSED: no checkable content (empty/whitespace/non-sentence output,
        # e.g. a stripped refusal). Emitting PASS@1.0 here would let unvalidated
        # output clear even the strictest threshold — a firewall must not pass
        # what it never checked.
        answer_score, answer_verdict = 0.0, Verdict.HEAL

    return ValidationResult(
        verdict=answer_verdict,
        answer_score=answer_score,
        domain=domain,
        claims=claims,
    )
