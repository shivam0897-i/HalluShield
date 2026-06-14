from hallushield.core.types import Chunk
from hallushield.firewall import HalluShield
from hallushield.generation import StubGenerator
from hallushield.retrieval import InMemoryRetriever
from hallushield.upstream import (
    InterChunkContradiction,
    SourceCredibilityScorer,
    UpstreamShield,
    build_shield,
    heuristic_contradiction,
)


# --- source credibility -------------------------------------------------- #
def test_credibility_tiers():
    sc = SourceCredibilityScorer()
    assert sc.score("PubMed Central") >= 0.9
    assert sc.score("ADA Clinical Guidelines 2024") >= 0.9
    assert sc.score("some random blog") <= 0.35
    assert sc.score("reddit.com/r/health") <= 0.25


def test_credibility_unknown_is_default():
    sc = SourceCredibilityScorer(default=0.5)
    assert sc.score("Mystery Source") == 0.5
    assert sc.score(None) == 0.5


def test_credibility_conservative_on_multiple_matches():
    # both ".edu" (0.85) and "blog" (0.3) present -> lowest wins
    assert SourceCredibilityScorer().score("prof.blog.harvard.edu") == 0.3


# --- inter-chunk contradiction ------------------------------------------- #
def test_heuristic_flags_conflicting_dosages():
    a = "Metformin dose is 500mg twice daily."
    b = "Metformin dose is 100mg once daily."
    assert heuristic_contradiction(a, b) >= 0.5


def test_heuristic_ignores_unrelated_chunks():
    assert heuristic_contradiction("Paris is in France.", "Metformin is a drug.") == 0.0


def test_inter_chunk_detect_reports_pairs():
    chunks = [
        Chunk("c1", "Metformin dose is 500mg twice daily."),
        Chunk("c2", "Metformin dose is 100mg once daily."),
        Chunk("c3", "Aspirin reduces fever."),
    ]
    pairs = InterChunkContradiction().detect(chunks)
    ids = {(a, b) for a, b, _ in pairs}
    assert ("c1", "c2") in ids
    assert all("c3" not in (a, b) for a, b, _ in pairs)


def test_injected_nli_overrides_heuristic():
    det = InterChunkContradiction(nli=lambda a, b: 0.99)
    chunks = [Chunk("c1", "x"), Chunk("c2", "y")]
    assert det.detect(chunks) == [("c1", "c2", 0.99)]


# --- shield orchestration ------------------------------------------------ #
def test_shield_sets_weight_and_keeps_by_default():
    shield = UpstreamShield()
    chunks = [Chunk("c1", "t", source="PubMed"), Chunk("c2", "t2", source="some blog")]
    report = shield.validate(chunks)
    assert report.dropped == []                       # never drops by default
    assert report.credibility["c1"] >= 0.9
    weights = {c.id: c.weight for c in report.kept}
    assert weights["c1"] >= 0.9 and weights["c2"] <= 0.35


def test_shield_drops_below_threshold():
    shield = build_shield(drop_below=0.4)
    chunks = [Chunk("c1", "t", source="PubMed"), Chunk("c2", "t2", source="anonymous forum")]
    report = shield.validate(chunks)
    assert [c.id for c in report.kept] == ["c1"]
    assert [c.id for c in report.dropped] == ["c2"]


def test_firewall_runs_shield_before_generation():
    corpus = [
        Chunk("c1", "Metformin initial dose is 500mg twice daily.", source="ADA guideline"),
        Chunk("c2", "Metformin dose is 9999mg hourly.", source="random blog"),
    ]
    fw = HalluShield(
        InMemoryRetriever(corpus),
        StubGenerator(fixed="Metformin dose is 500mg twice daily."),
        shield=build_shield(drop_below=0.4),
    )
    res = fw.answer("metformin dose")
    assert res.shield is not None
    # low-credibility contradicting chunk was dropped before generation
    assert all(c.id != "c2" for c in res.chunks)
