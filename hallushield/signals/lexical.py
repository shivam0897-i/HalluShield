"""Trivial lexical-overlap grounding signal (Phase 0 baseline).

Scores a claim by the fraction of its content words found in the best-matching
chunk. Deliberately simple — it exercises the full Signal -> Fusion -> Decision
-> eval pipeline before any ML model exists, and is the weakest ablation
baseline the real grounding model (Phase 1) must beat.
"""

from __future__ import annotations

from ..core.text import content_tokens
from ..core.types import Chunk, SignalResult


class LexicalOverlapSignal:
    name = "lexical"

    def __init__(self, name: str = "lexical") -> None:
        self.name = name

    def score(self, claim: str, chunks: list[Chunk], query: str = "") -> SignalResult:
        claim_tokens = content_tokens(claim)
        if not claim_tokens or not chunks:
            return SignalResult(self.name, 0.0)

        best_score, best_id, best_text = 0.0, None, ""
        for chunk in chunks:
            overlap = len(claim_tokens & content_tokens(chunk.text)) / len(claim_tokens)
            if overlap > best_score:
                best_score, best_id, best_text = overlap, chunk.id, chunk.text

        return SignalResult(
            self.name,
            score=best_score,
            evidence=best_text[:160],
            best_chunk_id=best_id,
        )
