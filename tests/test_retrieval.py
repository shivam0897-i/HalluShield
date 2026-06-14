from hallushield.core.types import Chunk
from hallushield.retrieval import InMemoryRetriever, build_retriever

CORPUS = [
    Chunk("c1", "Metformin initial dose is 500mg twice daily with meals."),
    Chunk("c2", "Aspirin reduces fever and inflammation."),
    Chunk("c3", "Paris is the capital of France."),
]


def test_ranks_most_relevant_first():
    r = InMemoryRetriever(CORPUS)
    hits = r.search("What is the metformin dose?", k=2)
    assert hits[0].id == "c1"
    assert len(hits) <= 2


def test_respects_k():
    r = InMemoryRetriever(CORPUS)
    assert len(r.search("aspirin fever", k=1)) == 1


def test_empty_query_returns_corpus_head():
    r = InMemoryRetriever(CORPUS)
    assert len(r.search("", k=2)) == 2


def test_build_retriever_factory():
    r = build_retriever(CORPUS, backend="memory")
    assert isinstance(r, InMemoryRetriever)
