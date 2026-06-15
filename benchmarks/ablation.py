"""Ablation harness — run the eval across signal configurations.

The `Signal` interface is the ablation knob: each row is just a different
`FusionScorer` over a different enabled-signal set, so the project's results
section is generated here rather than hand-assembled. Configurations whose
signals need optional extras that aren't installed (or aren't built yet) are
**skipped and reported** — never silently dropped, so the table can't
overstate coverage.

    python -m benchmarks.ablation --demo
    python -m benchmarks.ablation --data path/to/ragtruth.jsonl --out results.json
"""

from __future__ import annotations

import argparse
import json

from hallushield import config
from hallushield.core.fusion import FusionScorer
from hallushield.signals import build_signal

from .ragtruth_eval import Example, evaluate, load_examples, sample_examples

# Ordered ablation (plan §Evaluation): trivial baseline, then the core grounding
# model, then each added signal. Rows for signals not yet installed/built are
# skipped at runtime.
DEFAULT_CONFIGS: list[list[str]] = [
    ["lexical"],
    ["grounding"],
    ["grounding", "logic"],
]


def build_fusion_for(signal_names: list[str]) -> FusionScorer:
    """May raise ImportError if a signal's optional deps aren't installed."""
    signals = [build_signal(name) for name in signal_names]
    return FusionScorer(
        signals,
        weights=config.DEFAULT_WEIGHTS,
        contradiction_penalty=config.CONTRADICTION_PENALTY,
    )


def run_ablation(examples: list[Example], configs: list[list[str]] | None = None) -> list[dict]:
    rows: list[dict] = []
    for names in configs or DEFAULT_CONFIGS:
        try:
            fusion = build_fusion_for(names)
        except ImportError as exc:
            rows.append({"signals": names, "status": "skipped", "reason": str(exc).splitlines()[0]})
            continue
        m = evaluate(examples, fusion=fusion)
        rows.append(
            {
                "signals": names,
                "status": "ok",
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "f1": round(m.f1, 4),
                "n": m.n,
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
                "tn": m.tn,
            }
        )
    return rows


def _print_table(source: str, n: int, rows: list[dict]) -> None:
    print(f"dataset: {source}  (n={n})\n")
    print(f"{'signals':<26}{'status':<8}{'P':>7}{'R':>7}{'F1':>7}")
    print("-" * 55)
    for r in rows:
        label = "+".join(r["signals"])
        if r["status"] == "ok":
            print(f"{label:<26}{'ok':<8}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}")
        else:
            print(f"{label:<26}{'SKIP':<8}   ({r['reason'][:38]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="HalluShield ablation over signal sets")
    parser.add_argument("--responses", help="real RAGTruth response.jsonl")
    parser.add_argument("--source-info", help="real RAGTruth source_info.jsonl")
    parser.add_argument("--jsonl", help="self-contained JSONL dump (custom/injected)")
    parser.add_argument("--split", help="RAGTruth split filter, e.g. train|test")
    parser.add_argument("--demo", action="store_true", help="use the built-in synthetic fixture")
    parser.add_argument("--limit", type=int, default=0, help="evaluate a random sample of this many")
    parser.add_argument("--out", default="results.json", help="where to write the results JSON")
    args = parser.parse_args()

    examples, source = load_examples(
        responses=args.responses, source_info=args.source_info, jsonl=args.jsonl, split=args.split
    )
    examples = sample_examples(examples, args.limit)
    rows = run_ablation(examples)

    _print_table(source, len(examples), rows)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"dataset": source, "n": len(examples), "rows": rows}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
