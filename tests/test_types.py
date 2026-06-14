import pytest

from hallushield.core.types import Verdict, _SEVERITY, worst


@pytest.mark.parametrize(
    "verdicts,expected",
    [
        ([Verdict.PASS, Verdict.WARN, Verdict.HEAL], Verdict.HEAL),
        ([Verdict.HEAL, Verdict.PASS], Verdict.HEAL),   # max NOT last
        ([Verdict.WARN, Verdict.PASS], Verdict.WARN),   # max NOT last
        ([Verdict.PASS], Verdict.PASS),
        ([Verdict.HEAL], Verdict.HEAL),
        ([Verdict.WARN, Verdict.WARN], Verdict.WARN),
        ([], Verdict.PASS),                             # empty default
    ],
)
def test_worst_is_order_independent(verdicts, expected):
    assert worst(verdicts) is expected


def test_severity_map_covers_all_verdicts():
    assert set(_SEVERITY) == set(Verdict)
