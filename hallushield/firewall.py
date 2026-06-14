"""The HalluShield orchestrator: retrieve -> generate -> validate.

Ties a Retriever and Generator to the validation pipeline. Both dependencies
are injected and satisfy simple protocols, so the same firewall runs with the
dependency-free stack (InMemoryRetriever + StubGenerator) in tests/eval and the
real stack (FaissRetriever + LiteLLMGenerator) in production — no code change.

The self-healing loop (Phase 4) will hook in here, after `validate`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core.fusion import FusionScorer
from .core.types import Chunk, ValidationResult, Verdict
from .decision.self_healing import HealResult, SelfHealer
from .generation import Generator
from .pipeline import build_default_fusion, validate
from .retrieval import Retriever
from .upstream.shield import ShieldReport, UpstreamShield


@dataclass
class AnswerResult:
    query: str
    answer: str
    chunks: list[Chunk]
    validation: ValidationResult
    shield: ShieldReport | None = None
    healed: bool = False
    heal: HealResult | None = None


class HalluShield:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        fusion: FusionScorer | None = None,
        domain: str = "general",
        k: int = 5,
        shield: UpstreamShield | None = None,
        healer: SelfHealer | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.fusion = fusion or build_default_fusion()
        self.domain = domain
        self.k = k
        self.shield = shield
        self.healer = healer

    def answer(self, query: str, domain: str | None = None) -> AnswerResult:
        domain = domain or self.domain
        chunks = self.retriever.search(query, k=self.k)

        report: ShieldReport | None = None
        if self.shield is not None:
            # Upstream Shield runs BEFORE generation: down-weight/drop low-
            # credibility chunks so the LLM sees cleaner context.
            report = self.shield.validate(chunks)
            chunks = report.kept

        raw = self.generator.generate(query, chunks)
        result = validate(raw, chunks, domain, fusion=self.fusion, query=query)

        healed = False
        heal: HealResult | None = None
        if self.healer is not None and result.verdict is Verdict.HEAL:
            # Heal the worst (lowest-scoring) claim; re-validate the new answer.
            failed = min(result.claims, key=lambda c: c.fused_score).claim if result.claims else query
            heal = self.healer.heal(query, failed, domain)
            if heal.healed:
                raw = heal.answer
                chunks = heal.chunks or chunks
                result = validate(raw, chunks, domain, fusion=self.fusion)
                healed = True

        return AnswerResult(
            query=query, answer=raw, chunks=chunks, validation=result,
            shield=report, healed=healed, heal=heal,
        )
