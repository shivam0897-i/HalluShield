"""UpstreamShield orchestrator: credibility-score + down-weight/drop chunks and
detect inter-chunk contradictions, before generation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..core.types import Chunk
from .inter_chunk import InterChunkContradiction
from .source_credibility import SourceCredibilityScorer


@dataclass
class ShieldReport:
    kept: list[Chunk]                                   # chunks (with weight set) to use
    dropped: list[Chunk] = field(default_factory=list)  # removed (below drop_below)
    contradictions: list[tuple[str, str, float]] = field(default_factory=list)
    credibility: dict[str, float] = field(default_factory=dict)  # chunk_id -> score


class UpstreamShield:
    def __init__(
        self,
        credibility: SourceCredibilityScorer | None = None,
        contradiction: InterChunkContradiction | None = None,
        drop_below: float | None = None,
    ) -> None:
        self.credibility = credibility or SourceCredibilityScorer()
        self.contradiction = contradiction or InterChunkContradiction()
        # None -> never drop, only down-weight (the LLM still gets cleaner-weighted
        # context); set a float to filter clearly low-credibility chunks.
        self.drop_below = drop_below

    def validate(self, chunks: list[Chunk]) -> ShieldReport:
        kept: list[Chunk] = []
        dropped: list[Chunk] = []
        cred: dict[str, float] = {}
        for c in chunks:
            s = self.credibility.score(c.source)
            cred[c.id] = s
            weighted = replace(c, weight=s)  # frozen Chunk -> copy with new weight
            if self.drop_below is not None and s < self.drop_below:
                dropped.append(weighted)
            else:
                kept.append(weighted)
        contradictions = self.contradiction.detect(kept)
        return ShieldReport(kept=kept, dropped=dropped, contradictions=contradictions, credibility=cred)
