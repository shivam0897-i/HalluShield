import pytest

from benchmarks.ragtruth_eval import Example, predict_hallucinated, prf1
from hallushield.core.types import Chunk
from hallushield.pipeline import build_default_fusion


def test_prf1_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        prf1([True, False], [True])


def test_prf1_known_values():
    m = prf1([True, False, False, False], [True, True, False, False])
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 0, 1, 2)
    assert m.precision == 1.0
    assert m.recall == 0.5


def test_empty_answer_is_predicted_hallucinated():
    # Fail-closed empty answer (verdict HEAL) must be flagged, not a false negative.
    fusion = build_default_fusion()
    ex = Example("e", "   ", [Chunk("c1", "anything")], hallucinated=True)
    assert predict_hallucinated(ex, fusion) is True


def test_flag_warn_false_only_flags_heal():
    fusion = build_default_fusion()
    # Medical paraphrase -> WARN. flag_warn=False (precision-leaning) should NOT flag it.
    chunk = Chunk("c1", "Metformin initial dose is 500mg twice daily with meals.")
    ex = Example("w", "Metformin is started at 500mg twice daily with meals.", [chunk],
                 hallucinated=False, domain="medical")
    assert predict_hallucinated(ex, fusion, flag_warn=True) is True    # WARN counts
    assert predict_hallucinated(ex, fusion, flag_warn=False) is False  # only HEAL counts
