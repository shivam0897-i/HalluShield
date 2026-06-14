"""Prompt-only self-healing loop (Phase 4).

When a claim is blocked (verdict HEAL), instead of dead-ending we re-query →
re-retrieve → re-generate → re-score, up to `max_attempts`, returning an honest
fallback if healing fails. This is the defensible, finetuning-free adaptation of
Corrective-RAG / Self-RAG (HALLUSHIELD.md §7.8): the "reflection" is the
firewall's own verdict, and re-querying is plain prompting.

Retriever and Generator are injected (same protocols as the firewall), so the
loop is fully unit-testable with the dependency-free stubs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.fusion import FusionScorer
from ..core.types import Chunk, Verdict
from ..generation import Generator
from ..pipeline import build_default_fusion, validate
from ..retrieval import Retriever

FALLBACK_MESSAGE = "I don't have enough information to answer this accurately."


def rephrase_for_claim(query: str, failed_claim: str) -> str:
    """Re-ask the original question, flagging the prior unsupported claim to correct.

    Framing matters: asking the model to *verify* the false claim makes an honest,
    context-grounded model refuse ("I don't have enough information"). Asking it to
    *re-answer the question correctly from context* lets it recover the right answer.

    `failed_claim` comes from the firewall's own sentence splitter (not untrusted
    user input); treat it as adversarial if that ever changes.
    """
    return (
        f"{query.strip()} Answer accurately using only the provided context. "
        f"A previous attempt made an unsupported claim that must be corrected: "
        f"{failed_claim.strip()}"
    )


@dataclass
class HealResult:
    healed: bool
    answer: str          # healed segment, or the honest fallback message
    attempts: int
    score: float = 0.0
    chunks: list[Chunk] = field(default_factory=list)


class SelfHealer:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        fusion: FusionScorer | None = None,
        domain: str = "general",
        k: int = 3,
        max_attempts: int = 3,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.fusion = fusion or build_default_fusion()
        self.domain = domain
        self.k = k
        self.max_attempts = max_attempts

    def heal(self, query: str, failed_claim: str, domain: str | None = None) -> HealResult:
        domain = domain or self.domain
        for attempt in range(1, self.max_attempts + 1):
            sub_query = rephrase_for_claim(query, failed_claim)
            chunks = self.retriever.search(sub_query, k=self.k)
            segment = self.generator.generate(sub_query, chunks)
            result = validate(segment, chunks, domain, fusion=self.fusion, query=query)
            if result.verdict is Verdict.PASS:
                return HealResult(True, segment, attempt, result.answer_score, chunks)
        # Exhausted attempts — fail honest, not silent.
        return HealResult(False, FALLBACK_MESSAGE, self.max_attempts)
