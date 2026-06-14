import importlib.util

from benchmarks.ablation import build_fusion_for, run_ablation
from benchmarks.ragtruth_eval import demo_examples

_HAS_GROUNDING = importlib.util.find_spec("lettucedetect") is not None
_HAS_LOGIC = importlib.util.find_spec("litellm") is not None


def test_lexical_row_runs_and_others_skip_gracefully():
    rows = run_ablation(demo_examples())
    by_signals = {"+".join(r["signals"]): r for r in rows}

    # The trivial baseline runs and separates the demo fixture.
    lexical = by_signals["lexical"]
    assert lexical["status"] == "ok"
    assert lexical["f1"] == 1.0

    # Grounding runs iff the ML extra is installed; otherwise it is skipped
    # (reported, not silently dropped).
    assert by_signals["grounding"]["status"] == ("ok" if _HAS_GROUNDING else "skipped")

    # grounding+logic also needs the api extra (litellm); without it the row skips.
    if not (_HAS_GROUNDING and _HAS_LOGIC):
        assert by_signals["grounding+logic"]["status"] == "skipped"


def test_every_config_is_reported():
    # No silent truncation: one row per requested config.
    rows = run_ablation(demo_examples())
    assert len(rows) == 3


def test_build_fusion_for_lexical_is_usable():
    fusion = build_fusion_for(["lexical"])
    score, results, _ = fusion.score_claim("Paris is the capital", [])
    assert score == 0.0  # no chunks -> ungrounded
    assert "lexical" in results
