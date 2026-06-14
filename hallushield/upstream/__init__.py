"""Upstream Shield — pre-generation validation of retrieved chunks.

The genuinely novel layer: before the LLM generates, score each chunk's source
credibility and detect contradictions *between* retrieved chunks, then
down-weight (via `Chunk.weight`) or drop low-credibility chunks and surface
conflicting evidence to the decision layer. Grounded in ChunkRAG, RA-RAG,
RAGRank, EcoSafeRAG; threat model from PoisonedRAG.
"""

from __future__ import annotations

from ..core.types import Chunk
from .inter_chunk import InterChunkContradiction, heuristic_contradiction
from .shield import ShieldReport, UpstreamShield
from .source_credibility import SourceCredibilityScorer


def build_shield(drop_below: float | None = None, nli=None) -> UpstreamShield:
    return UpstreamShield(
        contradiction=InterChunkContradiction(nli=nli),
        drop_below=drop_below,
    )


__all__ = [
    "Chunk",
    "InterChunkContradiction",
    "ShieldReport",
    "SourceCredibilityScorer",
    "UpstreamShield",
    "build_shield",
    "heuristic_contradiction",
]
