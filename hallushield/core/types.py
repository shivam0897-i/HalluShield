"""Core data types and the Signal protocol every scorer implements.

The `Signal` protocol is the architectural spine: each detection method
(lexical, grounding, logic, ...) returns a uniform `SignalResult`, and the
`FusionScorer` combines whichever signals are enabled. Adding/removing a
signal is a config change, which makes the ablation study trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Chunk:
    """A retrieved context chunk."""

    id: str
    text: str
    source: str | None = None
    # Credibility/trust weight in [0, 1]; default 1.0. The Upstream Shield
    # (Phase 3) lowers this for low-credibility chunks so fusion/attribution can
    # honour it without changing the Signal protocol.
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SignalResult:
    """One signal's verdict on one claim."""

    name: str
    score: float  # groundedness in [0, 1] — higher = better supported
    evidence: str = ""  # short supporting snippet (for the UI/attribution)
    best_chunk_id: str | None = None  # chunk that best supports the claim
    contradiction: float = 0.0  # [0, 1] — how strongly some chunk contradicts


@runtime_checkable
class Signal(Protocol):
    """Anything that scores a claim against retrieved chunks.

    `query` (the original user question) is optional context — some signals
    (grounding, logic-judge) score better with it; others (lexical) ignore it.
    """

    name: str

    def score(self, claim: str, chunks: list[Chunk], query: str = "") -> SignalResult: ...


class Verdict(str, Enum):
    """Decision-engine output. Ordered PASS < WARN < HEAL by severity."""

    PASS = "PASS"
    WARN = "WARN"
    HEAL = "HEAL"  # blocked -> trigger self-healing


_SEVERITY = {Verdict.PASS: 0, Verdict.WARN: 1, Verdict.HEAL: 2}


def worst(verdicts: list[Verdict]) -> Verdict:
    """Most severe verdict in a list (PASS if empty)."""
    return max(verdicts, key=lambda v: _SEVERITY[v], default=Verdict.PASS)


@dataclass
class ClaimScore:
    """Fused score + per-signal breakdown for a single sentence/claim."""

    sentence_id: int
    claim: str
    fused_score: float
    verdict: Verdict
    supporting_chunk: str | None
    signals: dict[str, SignalResult] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """The firewall's verdict on a full answer."""

    verdict: Verdict
    answer_score: float
    domain: str
    claims: list[ClaimScore] = field(default_factory=list)
