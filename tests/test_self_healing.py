from hallushield.core.types import Chunk, Verdict
from hallushield.decision.self_healing import (
    FALLBACK_MESSAGE,
    SelfHealer,
    rephrase_for_claim,
)
from hallushield.firewall import HalluShield
from hallushield.generation import StubGenerator
from hallushield.retrieval import InMemoryRetriever

CORPUS = [Chunk("c1", "Paris is the capital of France.")]


def test_rephrase_targets_the_failed_claim():
    q = rephrase_for_claim("What is the capital?", "Berlin is the capital of France")
    assert "What is the capital?" in q
    assert "Berlin is the capital of France" in q


def test_heal_succeeds_first_attempt():
    healer = SelfHealer(
        InMemoryRetriever(CORPUS),
        StubGenerator(fixed="Paris is the capital of France."),
    )
    r = healer.heal("capital of France", "Berlin is the capital of France")
    assert r.healed is True
    assert r.attempts == 1
    assert "Paris is the capital of France." in r.answer


def test_heal_fails_returns_honest_fallback():
    healer = SelfHealer(
        InMemoryRetriever(CORPUS),
        StubGenerator(fixed="Aliens built the pyramids on Mars."),
        max_attempts=3,
    )
    r = healer.heal("capital of France", "Berlin is the capital")
    assert r.healed is False
    assert r.attempts == 3
    assert r.answer == FALLBACK_MESSAGE


def test_heal_succeeds_on_retry():
    class FlakyGen:
        def __init__(self):
            self.calls = 0

        def generate(self, query, chunks):
            self.calls += 1
            return "Paris is the capital of France." if self.calls >= 2 else "Unrelated nonsense here."

    healer = SelfHealer(InMemoryRetriever(CORPUS), FlakyGen(), max_attempts=3)
    r = healer.heal("capital of France", "bad claim")
    assert r.healed is True
    assert r.attempts == 2


def test_firewall_heals_a_blocked_answer():
    healer = SelfHealer(InMemoryRetriever(CORPUS), StubGenerator(fixed="Paris is the capital of France."))
    fw = HalluShield(
        InMemoryRetriever(CORPUS),
        StubGenerator(fixed="Paris is a small village on Mars."),  # blocked
        healer=healer,
    )
    res = fw.answer("capital of France")
    assert res.healed is True
    assert res.validation.verdict is Verdict.PASS
    assert "Paris is the capital of France." in res.answer


def test_firewall_does_not_heal_when_answer_passes():
    healer = SelfHealer(InMemoryRetriever(CORPUS), StubGenerator(fixed="x"))
    fw = HalluShield(
        InMemoryRetriever(CORPUS),
        StubGenerator(fixed="Paris is the capital of France."),  # passes
        healer=healer,
    )
    res = fw.answer("capital of France")
    assert res.healed is False
    assert res.heal is None
