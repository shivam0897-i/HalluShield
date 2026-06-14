from benchmarks.ragtruth_eval import Example
from benchmarks.span_eval import char_span_prf1, evaluate_spans, predicted_spans
from hallushield.core.sentences import split_sentences_with_offsets
from hallushield.core.types import Chunk
from hallushield.pipeline import validate

ANSWER = "Paris is the capital of France. Berlin invented gravity in 1822."
CHUNKS = [Chunk("c1", "Paris is the capital of France.")]


def test_offsets_are_exact_for_unmerged_sentences():
    offs = split_sentences_with_offsets("First one. Second two.")
    assert [(t, s, e) for t, s, e in offs] == [("First one.", 0, 10), ("Second two.", 11, 22)]


def test_offsets_index_back_into_original_text():
    offs = split_sentences_with_offsets(ANSWER)
    for text, start, end in offs:
        assert ANSWER[start:end] == text


def test_char_span_prf1_perfect_partial_disjoint():
    assert char_span_prf1([(0, 10)], [(0, 10)], 20).f1 == 1.0
    assert char_span_prf1([(0, 5)], [(10, 15)], 20).f1 == 0.0
    m = char_span_prf1([(0, 10)], [(5, 15)], 20)  # tp=5 fp=5 fn=5
    assert (m.tp, m.fp, m.fn) == (5, 5, 5)
    assert abs(m.f1 - 0.5) < 1e-9


def test_overlapping_pred_spans_do_not_double_count():
    # two overlapping predicted spans cover chars 0..10 once
    m = char_span_prf1([(0, 8), (4, 10)], [(0, 10)], 10)
    assert m.tp == 10 and m.fp == 0


def test_predicted_spans_flag_only_the_fabricated_sentence():
    res = validate(ANSWER, CHUNKS, "general")
    pred = predicted_spans(ANSWER, res)
    assert len(pred) == 1
    start, end = pred[0]
    assert ANSWER[start:end].startswith("Berlin")


def test_evaluate_spans_rewards_overlap_with_gold():
    gold = [(ANSWER.index("Berlin"), len(ANSWER))]
    ex = Example("x", ANSWER, CHUNKS, hallucinated=True, gold_spans=gold)
    m = evaluate_spans([ex])
    assert m.f1 > 0.9


def test_evaluate_spans_handles_no_gold_spans():
    ex = Example("x", ANSWER, CHUNKS, hallucinated=True)  # gold_spans=[]
    m = evaluate_spans([ex])
    assert m.f1 == 0.0  # no gold -> no true positives, but must not crash
