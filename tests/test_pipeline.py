import pytest

from hallushield import config
from hallushield.core.types import Chunk, Verdict
from hallushield.pipeline import validate

CHUNK = Chunk("c1", "Metformin initial dose is 500mg twice daily with meals.")


@pytest.mark.parametrize("answer", ["", "   ", "\n\t"])
def test_empty_answer_fails_closed(answer):
    # A firewall must not PASS content it never validated.
    r = validate(answer, [CHUNK], "general")
    assert r.verdict is Verdict.HEAL
    assert r.answer_score == 0.0
    assert r.claims == []


def test_answer_verdict_is_worst_claim():
    ans = "Metformin dose is 500mg twice daily with meals. It also instantly cures cancer."
    r = validate(ans, [CHUNK], "general")
    assert len(r.claims) == 2
    assert r.claims[0].verdict is Verdict.PASS   # grounded
    assert r.claims[1].verdict is Verdict.HEAL   # fabricated
    assert r.verdict is Verdict.HEAL             # worst claim governs


def test_mean_vs_min_aggregate(monkeypatch):
    ans = "Metformin dose is 500mg twice daily with meals. It also instantly cures cancer."
    monkeypatch.setattr(config, "ANSWER_AGGREGATE", "mean")
    mean_score = validate(ans, [CHUNK], "general").answer_score
    monkeypatch.setattr(config, "ANSWER_AGGREGATE", "min")
    min_score = validate(ans, [CHUNK], "general").answer_score
    assert min_score == 0.0          # the fabricated sentence drags the min to 0
    assert min_score < mean_score


def test_supporting_chunk_attributed():
    r = validate("Metformin dose is 500mg twice daily.", [CHUNK], "general")
    assert r.claims[0].supporting_chunk == "c1"


def test_medical_paraphrase_warns_under_lexical_baseline():
    # "started at" vs chunk "initial dose" -> 5/6 = 0.833 -> WARN at medical(0.90).
    # This is the exact case the Phase-1 grounding signal should flip to PASS.
    r = validate("Metformin is started at 500mg twice daily with meals.", [CHUNK], "medical")
    assert r.verdict is Verdict.WARN
