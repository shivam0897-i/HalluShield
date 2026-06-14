from benchmarks.ragtruth_eval import demo_examples, evaluate, predict_hallucinated, prf1
from hallushield.pipeline import build_default_fusion


def test_prf1_known_values():
    # gold:  T T F F ;  preds: T F F F  -> tp=1 fp=0 fn=1 tn=2
    m = prf1([True, False, False, False], [True, True, False, False])
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 0, 1, 2)
    assert m.precision == 1.0
    assert m.recall == 0.5
    assert abs(m.f1 - (2 * 1.0 * 0.5) / 1.5) < 1e-9


def test_prf1_handles_empty_positive_predictions():
    m = prf1([False, False], [True, False])
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0


def test_demo_predictions_separate_grounded_from_fabricated():
    fusion = build_default_fusion()
    by_id = {ex.id: predict_hallucinated(ex, fusion) for ex in demo_examples()}
    # grounded answers not flagged; fabricated ones flagged
    assert by_id["g1"] is False
    assert by_id["g2"] is False
    assert by_id["h1"] is True
    assert by_id["h2"] is True


def test_demo_harness_runs_green_with_perfect_separation():
    m = evaluate(demo_examples())
    assert m.n == 4
    assert m.f1 == 1.0  # the trivial baseline cleanly separates this fixture
