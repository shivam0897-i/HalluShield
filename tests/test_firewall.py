from hallushield.core.types import Chunk, Verdict
from hallushield.firewall import HalluShield
from hallushield.generation import StubGenerator
from hallushield.retrieval import InMemoryRetriever

CORPUS = [
    Chunk("c1", "Metformin initial dose is 500mg twice daily with meals.", source="ADA 2024"),
    Chunk("c2", "Paris is the capital of France.", source="Wikipedia"),
]


def test_grounded_answer_passes_end_to_end():
    # StubGenerator echoes the top chunk -> trivially grounded -> PASS.
    fw = HalluShield(InMemoryRetriever(CORPUS), StubGenerator(), domain="general")
    res = fw.answer("What is the metformin dose?")
    assert res.chunks[0].id == "c1"
    assert res.validation.verdict is Verdict.PASS
    assert res.validation.claims[0].supporting_chunk == "c1"


def test_fabricated_answer_is_blocked():
    fw = HalluShield(
        InMemoryRetriever(CORPUS),
        StubGenerator(fixed="Metformin should be injected at 5000mg every hour."),
        domain="medical",
    )
    res = fw.answer("metformin dose")
    assert res.validation.verdict is Verdict.HEAL


def test_answer_carries_query_and_chunks():
    fw = HalluShield(InMemoryRetriever(CORPUS), StubGenerator())
    res = fw.answer("capital of France")
    assert res.query == "capital of France"
    assert any(c.id == "c2" for c in res.chunks)
