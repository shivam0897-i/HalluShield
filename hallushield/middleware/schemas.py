"""Request/response schemas for POST /validate (mirrors HALLUSHIELD.md §18)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChunkIn(BaseModel):
    id: str
    text: str
    source: str | None = None


class ValidateOptions(BaseModel):
    return_claim_scores: bool = True
    max_heal_attempts: int = 3  # reserved for Phase 4


class ValidateRequest(BaseModel):
    query: str
    answer: str = Field(..., max_length=50_000)  # bound work + payload size
    chunks: list[ChunkIn]
    domain: str = "general"
    options: ValidateOptions = Field(default_factory=ValidateOptions)

    @field_validator("chunks")
    @classmethod
    def _unique_nonempty_ids(cls, v: list[ChunkIn]) -> list[ChunkIn]:
        ids = [c.id for c in v]
        if any(not i.strip() for i in ids):
            raise ValueError("chunk id must be non-empty")
        if len(set(ids)) != len(ids):
            raise ValueError("chunk ids must be unique (supporting_chunk must join back to one chunk)")
        return v


class ClaimOut(BaseModel):
    id: str
    text: str
    score: float
    verdict: str
    supporting_chunk: str | None = None


class ValidateResponse(BaseModel):
    verdict: str
    answer_score: float
    domain: str
    claims: list[ClaimOut] = Field(default_factory=list)
    latency_ms: int
    healed: bool = False
