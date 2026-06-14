"""FastAPI app.  Run:  uvicorn hallushield.middleware.app:app --reload

POST /validate takes a query, an already-generated answer, and the retrieved
chunks, and returns the firewall verdict with per-sentence scores and source
attribution (HALLUSHIELD.md §18). Generation/retrieval are the caller's job
here — this is the middleware contract a developer drops into their RAG app.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException

from ..core.types import Chunk
from ..pipeline import build_default_fusion, validate
from .schemas import ClaimOut, ValidateRequest, ValidateResponse

app = FastAPI(title="HalluShield", version="0.1.0")

# Built lazily on first use and cached. Constructing a heavy signal (e.g. the
# grounding model) must not crash the whole ASGI app at import time.
_fusion = None


def _get_fusion():
    global _fusion
    if _fusion is None:
        _fusion = build_default_fusion()
    return _fusion


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/validate", response_model=ValidateResponse)
def validate_endpoint(req: ValidateRequest) -> ValidateResponse:
    started = time.perf_counter()
    chunks = [Chunk(id=c.id, text=c.text, source=c.source) for c in req.chunks]
    try:
        result = validate(req.answer, chunks, req.domain, fusion=_get_fusion(), query=req.query)
    except Exception as exc:  # noqa: BLE001 — fail CLOSED; never return PASS on error
        raise HTTPException(
            status_code=500, detail={"error": "firewall_error", "message": str(exc)}
        ) from exc

    claims = []
    if req.options.return_claim_scores:
        claims = [
            ClaimOut(
                id=f"c{c.sentence_id}",
                text=c.claim,
                score=round(c.fused_score, 4),
                verdict=c.verdict.value,
                supporting_chunk=c.supporting_chunk,
            )
            for c in result.claims
        ]

    return ValidateResponse(
        verdict=result.verdict.value,
        answer_score=round(result.answer_score, 4),
        domain=result.domain,
        claims=claims,
        latency_ms=max(1, round((time.perf_counter() - started) * 1000)),
        healed=False,
    )
