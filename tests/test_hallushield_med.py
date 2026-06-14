import random

from benchmarks.hallushield_med import (
    SEED_PAIRS,
    build_med_examples,
    inject_entity,
    inject_hallucination,
    inject_negation,
    inject_numeric,
    write_jsonl,
)
from benchmarks.ragtruth_eval import load_jsonl
from hallushield.core.types import Verdict
from hallushield.pipeline import validate


def test_inject_numeric_changes_a_number():
    s, changed = inject_numeric("Take 500mg twice daily", random.Random(0))
    assert changed
    assert "500mg" not in s and "5000" in s


def test_inject_entity_swaps_clinical_term():
    s, changed = inject_entity("This will increase blood pressure", random.Random(0))
    assert changed and "decrease" in s


def test_inject_negation_negates_assertion():
    s, changed = inject_negation("Aspirin is safe", random.Random(0))
    assert changed and "is not" in s


def test_inject_negation_fallback_always_introduces_falsehood():
    s, changed = inject_negation("Take it now", random.Random(0))
    assert changed and s.startswith("It is not true that")


def test_inject_hallucination_is_deterministic_and_spanned():
    answer = "Metformin is started at 500mg twice daily with meals."
    n1, c1, k1, sp1 = inject_hallucination(answer, seed=3)
    n2, c2, k2, sp2 = inject_hallucination(answer, seed=3)
    assert c1 and (n1, sp1, k1) == (n2, sp2, k2)  # same seed -> same output
    assert n1 != answer
    (start, end), = sp1
    assert 0 <= start < end <= len(n1)


def test_build_med_examples_makes_clean_and_hallucinated():
    examples = build_med_examples(SEED_PAIRS, seed=0)
    assert any(not e.hallucinated for e in examples)
    halu = [e for e in examples if e.hallucinated]
    assert halu
    assert all(e.gold_spans for e in halu)  # every injected example is span-labelled


def test_jsonl_roundtrip_preserves_labels_and_spans(tmp_path):
    examples = build_med_examples(SEED_PAIRS, seed=1)
    path = tmp_path / "med.jsonl"
    write_jsonl(examples, str(path))
    loaded = load_jsonl(str(path))
    assert len(loaded) == len(examples)
    halu = [e for e in loaded if e.hallucinated]
    assert halu and all(e.gold_spans for e in halu)


def test_injected_examples_are_caught_by_the_firewall():
    halu = [e for e in build_med_examples(SEED_PAIRS, seed=0) if e.hallucinated]
    flagged = sum(
        validate(e.answer, e.chunks, e.domain).verdict is not Verdict.PASS for e in halu
    )
    assert flagged >= len(halu) // 2  # even the lexical baseline catches most injections
