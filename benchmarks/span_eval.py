"""Span-level (character) hallucination scoring.

RAGTruth annotates hallucinated *spans* (char offsets into the response) and
LettuceDetect reports a span-level F1, so example-level F1 alone isn't a fair
comparison. Here we score character overlap between the firewall's predicted
spans and the gold spans:

* gold spans  = RAGTruth `labels` char ranges (carried on `Example.gold_spans`)
* pred spans  = char ranges of the sentences the firewall did NOT pass

A predicted char that lies inside any gold span is a TP, a predicted char
outside all gold spans is an FP, a gold char not predicted is an FN. Metrics are
micro-averaged over all characters in the dataset (the standard span metric).

    python -m benchmarks.span_eval --responses R.jsonl --source-info S.jsonl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from hallushield.core.sentences import split_sentences_with_offsets
from hallushield.core.types import ValidationResult, Verdict
from hallushield.pipeline import build_default_fusion, validate

from .ragtruth_eval import Example, load_examples

Span = tuple[int, int]


@dataclass
class SpanMetrics:
    precision: float
    recall: float
    f1: float
    tp: int = 0
    fp: int = 0
    fn: int = 0


def _char_mask(spans: list[Span], length: int) -> bytearray:
    mask = bytearray(length)
    for start, end in spans:
        for i in range(max(0, start), min(length, end)):
            mask[i] = 1
    return mask


def char_span_prf1(pred_spans: list[Span], gold_spans: list[Span], length: int) -> SpanMetrics:
    """Character-level precision/recall/F1 for one answer of `length` chars."""
    p = _char_mask(pred_spans, length)
    g = _char_mask(gold_spans, length)
    tp = sum(1 for i in range(length) if p[i] and g[i])
    fp = sum(1 for i in range(length) if p[i] and not g[i])
    fn = sum(1 for i in range(length) if not p[i] and g[i])
    return _metrics(tp, fp, fn)


def _metrics(tp: int, fp: int, fn: int) -> SpanMetrics:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return SpanMetrics(precision, recall, f1, tp, fp, fn)


def predicted_spans(answer: str, validation: ValidationResult, flag_warn: bool = True) -> list[Span]:
    """Char ranges of the sentences the firewall flagged (verdict != PASS)."""
    offsets = split_sentences_with_offsets(answer)
    spans: list[Span] = []
    for (_, start, end), claim in zip(offsets, validation.claims):
        flagged = claim.verdict is not Verdict.PASS if flag_warn else claim.verdict is Verdict.HEAL
        if flagged:
            spans.append((start, end))
    return spans


def evaluate_spans(examples: list[Example], fusion=None, flag_warn: bool = True) -> SpanMetrics:
    """Micro-averaged char-level span F1 over the dataset."""
    fusion = fusion or build_default_fusion()
    tp = fp = fn = 0
    for ex in examples:
        result = validate(ex.answer, ex.chunks, ex.domain, fusion=fusion, query=ex.query)
        pred = predicted_spans(ex.answer, result, flag_warn=flag_warn)
        m = char_span_prf1(pred, ex.gold_spans, len(ex.answer))
        tp += m.tp
        fp += m.fp
        fn += m.fn
    return _metrics(tp, fp, fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGTruth span-level eval")
    parser.add_argument("--responses", help="real RAGTruth response.jsonl")
    parser.add_argument("--source-info", help="real RAGTruth source_info.jsonl")
    parser.add_argument("--jsonl", help="self-contained JSONL dump")
    parser.add_argument("--split", help="RAGTruth split filter, e.g. train|test")
    args = parser.parse_args()

    examples, source = load_examples(
        responses=args.responses, source_info=args.source_info, jsonl=args.jsonl, split=args.split
    )
    if not any(ex.gold_spans for ex in examples):
        print("NOTE: no gold spans present (demo/jsonl have none) — span F1 needs real RAGTruth labels.")
    m = evaluate_spans(examples)
    from hallushield import config

    print(f"dataset:  {source}  (n={len(examples)})")
    print(f"signals:  {config.ENABLED_SIGNALS}")
    print(f"[span] precision={m.precision:.3f}  recall={m.recall:.3f}  f1={m.f1:.3f}")
    print(f"tp_chars={m.tp} fp_chars={m.fp} fn_chars={m.fn}")


if __name__ == "__main__":
    main()
