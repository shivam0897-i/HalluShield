import pytest

from hallushield.core.types import Verdict, worst
from hallushield.decision.engine import decide


@pytest.mark.parametrize(
    "score,domain,expected",
    [
        (0.95, "medical", Verdict.PASS),   # >= 0.90
        (0.80, "medical", Verdict.WARN),   # 0.70..0.90
        (0.50, "medical", Verdict.HEAL),   # < 0.70
        (0.80, "general", Verdict.PASS),   # >= 0.75
        (0.60, "general", Verdict.WARN),   # 0.50..0.75
        (0.30, "general", Verdict.HEAL),   # < 0.50
    ],
)
def test_domain_thresholds(score, domain, expected):
    assert decide(score, domain) is expected


def test_same_score_stricter_in_medical_than_general():
    # 0.80 passes in general but only warns in medical.
    assert decide(0.80, "general") is Verdict.PASS
    assert decide(0.80, "medical") is Verdict.WARN


def test_unknown_domain_falls_back_to_general():
    assert decide(0.80, "astrology") is Verdict.PASS


def test_worst_verdict_ordering():
    assert worst([Verdict.PASS, Verdict.WARN, Verdict.HEAL]) is Verdict.HEAL
    assert worst([Verdict.PASS, Verdict.WARN]) is Verdict.WARN
    assert worst([]) is Verdict.PASS
