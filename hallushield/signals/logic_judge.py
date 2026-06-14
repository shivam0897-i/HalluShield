"""Logic / reasoning signal — LLM-as-judge groundedness (Phase 2).

A single LLM call scores how well the *reasoning* of a claim is supported by
the retrieved context, returning a continuous [0, 1] groundedness score (not
binary — see HALLUSHIELD.md §7.5). This is the second fusion signal; with the
core grounding signal it forms the `["grounding", "logic"]` ablation row.

The judge is an injected callable `(prompt: str) -> str`, so the signal is
fully unit-testable with a stub. When no judge is given it lazily builds a
LiteLLM-backed one (provider-agnostic, API-only) — which needs the api extras:
    pip install -e ".[api]"
Constructing without those extras raises ImportError, so the ablation harness
skips the row cleanly (same contract as the grounding signal).
"""

from __future__ import annotations

import re
from typing import Callable

from ..config import MODELS
from ..core.text import content_tokens
from ..core.types import Chunk, SignalResult

_PROMPT = """You are a strict fact-checker. Decide how well the CLAIM is supported by the CONTEXT.

QUESTION: {query}

CONTEXT:
{context}

CLAIM: {claim}

Reply with ONLY a number between 0 and 1 (no words): 1.0 = the claim is fully
supported by the context; 0.0 = it is unsupported by or contradicts the context."""

_NUMBER = re.compile(r"[01](?:\.\d+)?|\.\d+")


def _parse_score(text: str) -> float:
    """Extract the first number in [0,1] from the judge's reply; 0.5 if none."""
    match = _NUMBER.search(text or "")
    if not match:
        return 0.5  # uncertain — the judge did not return a parseable score
    return max(0.0, min(1.0, float(match.group())))


def _build_litellm_judge(model: str) -> Callable[[str], str]:
    try:
        import litellm  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise ImportError(
            "LogicJudgeSignal needs the api extras. Install: pip install -e \".[api]\""
        ) from exc

    def judge(prompt: str) -> str:
        resp = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0
        )
        return resp["choices"][0]["message"]["content"]

    return judge


class LogicJudgeSignal:
    name = "logic"

    def __init__(self, judge: Callable[[str], str] | None = None, model: str | None = None) -> None:
        self._judge = judge or _build_litellm_judge(model or MODELS["judge"])

    def score(self, claim: str, chunks: list[Chunk], query: str = "") -> SignalResult:
        if not claim.strip():
            return SignalResult(self.name, 0.0)
        context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks) or "(no context)"
        reply = self._judge(_PROMPT.format(query=query or "(not provided)", context=context, claim=claim))
        score = _parse_score(reply)

        ct = content_tokens(claim)
        best_id = None
        if ct and chunks:
            best_id = max(chunks, key=lambda c: len(ct & content_tokens(c.text))).id
        return SignalResult(self.name, score=score, evidence=str(reply)[:160], best_chunk_id=best_id)
