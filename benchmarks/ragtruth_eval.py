"""RAGTruth evaluation harness — example-level hallucination detection.

Built before any model so the metric is trustworthy from day one. Three input
modes:

* ``--demo``  — tiny built-in synthetic fixture (smoke test; NOT a benchmark).
* ``--jsonl PATH`` — a self-contained JSONL dump (one object per line with
  ``response``/``hallucinated``/``chunks``). Use this for custom/injected sets
  such as HalluShield-Med.
* ``--responses R.jsonl --source-info S.jsonl`` — the **real RAGTruth release**
  (arXiv:2402.07067, ParticleMedia/RAGTruth), which ships two files joined by
  ``source_id``. Gold label = the response's ``labels`` list is non-empty; the
  context comes from ``source_info``; ``quality`` != "good" rows (truncated /
  incorrect_refusal) are excluded by default.

We evaluate the *example-level* task: predict whether a response contains any
hallucination, and score precision/recall/F1 for the positive (hallucinated)
class. Until you run one of the real-data modes, only ``--demo`` numbers exist
and they are not a benchmark result.
"""

from __future__ import annotations

import argparse
import json

from hallushield.core.types import Chunk, Verdict
from hallushield.pipeline import build_default_fusion, validate

# RAGTruth task_type -> firewall domain. RAGTruth is general-domain (QA /
# summarization / data-to-text), so it uses the general thresholds; we keep the
# task_type for optional stratified reporting.
_TASK_DOMAIN = {"QA": "general", "Summary": "general", "Data2txt": "general"}


class Example:
    def __init__(self, id, answer, chunks, hallucinated, domain="general", task_type=None):
        self.id = id
        self.answer = answer
        self.chunks = chunks
        self.hallucinated = hallucinated
        self.domain = domain
        self.task_type = task_type


class Metrics:
    def __init__(self, precision, recall, f1, tp=0, fp=0, fn=0, tn=0, n=0):
        self.precision, self.recall, self.f1 = precision, recall, f1
        self.tp, self.fp, self.fn, self.tn, self.n = tp, fp, fn, tn, n


def prf1(preds: list[bool], gold: list[bool]) -> Metrics:
    """Precision/recall/F1 for the positive (hallucinated) class.

    Refuses mismatched lengths — a silent zip-truncation would report a metric
    over fewer examples than `n` claims.
    """
    if len(preds) != len(gold):
        raise ValueError(f"preds/gold length mismatch: {len(preds)} != {len(gold)}")
    tp = sum(p and g for p, g in zip(preds, gold))
    fp = sum(p and not g for p, g in zip(preds, gold))
    fn = sum(not p and g for p, g in zip(preds, gold))
    tn = sum(not p and not g for p, g in zip(preds, gold))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Metrics(precision, recall, f1, tp, fp, fn, tn, len(preds))


def predict_hallucinated(example: Example, fusion, flag_warn: bool = True) -> bool:
    """Predict whether the response is hallucinated.

    Uses the answer-level verdict (so empty/degenerate answers, which fail
    CLOSED to HEAL, are correctly flagged). `flag_warn=True` counts WARN as
    hallucinated (recall-leaning, matches the firewall's "don't pass it
    unreviewed" stance); `flag_warn=False` flags only HEAL (precision-leaning).
    Report which setting a number was produced under.
    """
    result = validate(example.answer, example.chunks, example.domain, fusion=fusion)
    if flag_warn:
        return result.verdict is not Verdict.PASS
    return result.verdict is Verdict.HEAL


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_jsonl(path: str) -> list[Example]:
    """Self-contained JSONL: {id?, response, hallucinated, chunks:[{id,text,source?}], domain?}."""
    examples: list[Example] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunks = [
                Chunk(id=c.get("id", str(i)), text=c["text"], source=c.get("source"))
                for i, c in enumerate(row.get("chunks", []))
            ]
            examples.append(
                Example(
                    id=str(row.get("id", len(examples))),
                    answer=row["response"],
                    chunks=chunks,
                    hallucinated=bool(row["hallucinated"]),
                    domain=row.get("domain", "general"),
                )
            )
    return examples


def _chunks_from_source(source_id: str, task_type: str, source_info) -> list[Chunk]:
    """Build context Chunks from a RAGTruth source_info record."""
    if isinstance(source_info, dict):
        passages = source_info.get("passages") or source_info.get("source_info") or ""
        if not passages:
            # Data2txt and similar: serialise the structured record as one chunk.
            passages = json.dumps(source_info, ensure_ascii=False)
    else:
        passages = str(source_info)
    parts = [p.strip() for p in passages.split("\n\n") if p.strip()] or [passages]
    return [Chunk(id=f"{source_id}:{i}", text=p, source=task_type) for i, p in enumerate(parts)]


def load_ragtruth(
    responses_path: str,
    source_info_path: str,
    *,
    split: str | None = None,
    include_non_good: bool = False,
) -> list[Example]:
    """Load the real two-file RAGTruth release (joined by source_id).

    response.jsonl rows: {id, source_id, model, labels, split, quality, response}
    source_info.jsonl rows: {source_id, task_type, source_info, prompt}
    Gold hallucinated = labels is non-empty.
    """
    sources: dict[str, dict] = {}
    with open(source_info_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                sources[row["source_id"]] = row

    examples: list[Example] = []
    skipped_quality = skipped_split = 0
    with open(responses_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split is not None and row.get("split") not in (None, split):
                skipped_split += 1
                continue
            if not include_non_good and row.get("quality", "good") != "good":
                skipped_quality += 1
                continue
            src = sources.get(row["source_id"], {})
            task_type = src.get("task_type", "QA")
            examples.append(
                Example(
                    id=str(row.get("id")),
                    answer=row["response"],
                    chunks=_chunks_from_source(row["source_id"], task_type, src.get("source_info", "")),
                    hallucinated=len(row.get("labels", [])) > 0,
                    domain=_TASK_DOMAIN.get(task_type, "general"),
                    task_type=task_type,
                )
            )
    if skipped_quality or skipped_split:
        print(f"[loader] excluded {skipped_quality} non-good rows, {skipped_split} off-split rows")
    return examples


def load_examples(
    *,
    responses: str | None = None,
    source_info: str | None = None,
    jsonl: str | None = None,
    split: str | None = None,
) -> tuple[list[Example], str]:
    """Pick a loader from the given paths; returns (examples, source_label)."""
    if responses and source_info:
        return load_ragtruth(responses, source_info, split=split), f"RAGTruth({responses})"
    if jsonl:
        return load_jsonl(jsonl), jsonl
    return demo_examples(), "built-in demo fixture (SMOKE TEST - not a benchmark)"


def demo_examples() -> list[Example]:
    """Tiny synthetic set: grounded answers vs. fabricated ones."""
    metformin = Chunk("c1", "Metformin initial dose is 500mg twice daily with meals.", "ADA 2024")
    paris = Chunk("c2", "Paris is the capital of France and sits on the river Seine.", "Wikipedia")
    return [
        Example("g1", "Metformin is started at 500mg twice daily with meals.", [metformin], False),
        Example("g2", "Paris is the capital of France.", [paris], False),
        Example("h1", "Metformin should be injected at 5000mg every hour.", [metformin], True),
        Example("h2", "Berlin is the capital of France and has no rivers.", [paris], True),
    ]


def evaluate(examples: list[Example], fusion=None, flag_warn: bool = True) -> Metrics:
    fusion = fusion or build_default_fusion()
    preds = [predict_hallucinated(ex, fusion, flag_warn=flag_warn) for ex in examples]
    gold = [ex.hallucinated for ex in examples]
    return prf1(preds, gold)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGTruth example-level eval")
    parser.add_argument("--responses", help="real RAGTruth response.jsonl")
    parser.add_argument("--source-info", help="real RAGTruth source_info.jsonl")
    parser.add_argument("--jsonl", help="self-contained JSONL dump (custom/injected)")
    parser.add_argument("--split", help="RAGTruth split filter, e.g. train|test")
    parser.add_argument("--demo", action="store_true", help="built-in synthetic fixture")
    args = parser.parse_args()

    examples, source = load_examples(
        responses=args.responses, source_info=args.source_info, jsonl=args.jsonl, split=args.split
    )
    is_demo = not (args.responses and args.source_info) and not args.jsonl
    m = evaluate(examples)

    from hallushield import config

    if is_demo:
        print("SMOKE TEST - synthetic fixture, NOT a RAGTruth benchmark result.\n")
    print(f"dataset:  {source}  (n={m.n})")
    print(f"signals:  {config.ENABLED_SIGNALS}")
    label = "smoke" if is_demo else "metric"
    print(f"[{label}] precision={m.precision:.3f}  recall={m.recall:.3f}  f1={m.f1:.3f}")
    print(f"tp={m.tp} fp={m.fp} fn={m.fn} tn={m.tn}")


if __name__ == "__main__":
    main()
