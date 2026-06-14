"""HalluShield-Med — controlled hallucination injection for a medical eval set.

Turns grounded medical Q&A (answer supported by a context passage) into labelled
hallucination samples by injecting one of three controlled error types into a
single sentence, recording the corrupted sentence's char span as the gold label.
Each source pair yields a clean (grounded, not hallucinated) example and an
injected (hallucinated) example, so detectors are scored on both precision and
recall. Output is the self-contained JSONL that `ragtruth_eval.load_jsonl` reads,
with example-level labels AND span-level gold spans.

Injection types (HALLUSHIELD.md taxonomy minus omission, which v1 does not do):
* numeric  — change a dosage/number (500mg -> 5000mg)
* entity   — swap a clinical term for its opposite (increase->decrease, twice->once)
* negation — negate an assertion (is -> is not)

    python -m benchmarks.hallushield_med --demo --out datasets/hallushield_med.jsonl
    python -m benchmarks.hallushield_med --pairs my_pairs.jsonl --out med.jsonl --seed 7

`--pairs` expects JSONL of {id?, question?, answer, context} where `context` is a
string or a list of strings (convert MedQuAD into this shape first).
"""

from __future__ import annotations

import argparse
import json
import random
import re

from hallushield.core.sentences import split_sentences_with_offsets
from hallushield.core.types import Chunk

from .ragtruth_eval import Example

_NUM = re.compile(r"\d+(?:\.\d+)?")

# Clinical / factual term swaps (first match wins, word-boundary, case-insensitive).
_ENTITY_SWAPS: dict[str, str] = {
    "increase": "decrease", "increases": "decreases", "decrease": "increase",
    "high": "low", "low": "high", "twice": "once", "once": "three times",
    "daily": "weekly", "before": "after", "after": "before", "raise": "lower",
    "lower": "raise", "benign": "malignant", "malignant": "benign",
    "effective": "ineffective", "safe": "unsafe", "recommended": "contraindicated",
}


def inject_numeric(sentence: str, rng: random.Random) -> tuple[str, bool]:
    matches = list(_NUM.finditer(sentence))
    if not matches:
        return sentence, False
    m = rng.choice(matches)
    value = m.group()
    new = str(round(float(value) * 10, 2)) if "." in value else str(int(value) * 10)
    return sentence[: m.start()] + new + sentence[m.end():], True


def inject_entity(sentence: str, rng: random.Random) -> tuple[str, bool]:
    for key, repl in _ENTITY_SWAPS.items():
        m = re.search(rf"\b{re.escape(key)}\b", sentence, re.IGNORECASE)
        if m:
            return sentence[: m.start()] + repl + sentence[m.end():], True
    return sentence, False


def inject_negation(sentence: str, rng: random.Random) -> tuple[str, bool]:
    for pat, repl in ((" is ", " is not "), (" are ", " are not "),
                      (" can ", " cannot "), (" should ", " should not ")):
        if pat in sentence:
            return sentence.replace(pat, repl, 1), True
    # Fallback that always introduces a falsehood.
    return f"It is not true that {sentence[:1].lower()}{sentence[1:]}", True


_INJECTORS = {"numeric": inject_numeric, "entity": inject_entity, "negation": inject_negation}
# Try meaning-preserving-looking corruptions first; negation always succeeds, last.
_ORDER = ("numeric", "entity", "negation")


def inject_hallucination(answer: str, seed: int = 0) -> tuple[str, bool, str | None, list[tuple[int, int]]]:
    """Corrupt one sentence of `answer`. Returns (new_answer, changed, type, gold_spans)."""
    rng = random.Random(seed)
    sentences = split_sentences_with_offsets(answer)
    if not sentences:
        return answer, False, None, []
    text, start, end = sentences[rng.randrange(len(sentences))]
    for kind in _ORDER:
        mutated, changed = _INJECTORS[kind](text, rng)
        if changed and mutated != text:
            new_answer = answer[:start] + mutated + answer[end:]
            return new_answer, True, kind, [(start, start + len(mutated))]
    return answer, False, None, []


def _chunks(context, pair_id: str) -> list[Chunk]:
    texts = context if isinstance(context, list) else [context]
    return [Chunk(id=f"{pair_id}:{i}", text=t, source="MedQuAD") for i, t in enumerate(texts)]


def build_med_examples(pairs: list[dict], seed: int = 0) -> list[Example]:
    """Each pair -> a clean example + an injected hallucinated example."""
    examples: list[Example] = []
    for i, pair in enumerate(pairs):
        pid = str(pair.get("id", i))
        answer = pair["answer"]
        question = pair.get("question", "")
        chunks = _chunks(pair["context"], pid)
        examples.append(Example(f"{pid}-clean", answer, chunks, hallucinated=False,
                                domain="medical", query=question))
        new_answer, changed, kind, spans = inject_hallucination(answer, seed=seed + i)
        if changed:
            ex = Example(f"{pid}-halu", new_answer, chunks, hallucinated=True,
                         domain="medical", gold_spans=spans, query=question)
            ex.injection_type = kind  # type: ignore[attr-defined]
            examples.append(ex)
    return examples


def write_jsonl(examples: list[Example], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps({
                "id": ex.id,
                "question": ex.query,
                "response": ex.answer,
                "hallucinated": ex.hallucinated,
                "domain": ex.domain,
                "chunks": [{"id": c.id, "text": c.text, "source": c.source} for c in ex.chunks],
                "gold_spans": [list(s) for s in ex.gold_spans],
                "injection": getattr(ex, "injection_type", None),
            }, ensure_ascii=False) + "\n")


SEED_PAIRS: list[dict] = [
    {"id": "metformin", "question": "What is the recommended starting dose of metformin?",
     "answer": "Metformin is started at 500mg twice daily with meals.",
     "context": "Metformin initial dose is 500mg twice daily with meals to reduce GI upset."},
    {"id": "aspirin", "question": "Can low-dose aspirin help prevent heart attacks?",
     "answer": "Low-dose aspirin can reduce the risk of heart attack.",
     "context": "Low-dose aspirin reduces the risk of myocardial infarction in high-risk adults."},
    {"id": "insulin", "question": "How should unopened insulin be stored?",
     "answer": "Insulin should be stored in a refrigerator before first use.",
     "context": "Unopened insulin vials should be stored refrigerated at 2 to 8 degrees Celsius."},
    {"id": "bp", "question": "What is a normal resting blood pressure?",
     "answer": "A normal resting blood pressure is below 120 over 80 mmHg.",
     "context": "Normal resting blood pressure is defined as below 120/80 mmHg."},
    {"id": "amox", "question": "How often is amoxicillin taken?",
     "answer": "Amoxicillin is taken three times daily for 7 days.",
     "context": "Amoxicillin is typically taken three times daily for a 7 day course."},
]


def load_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the HalluShield-Med injected eval set")
    parser.add_argument("--pairs", help="JSONL of {answer, context, id?} medical pairs")
    parser.add_argument("--demo", action="store_true", help="use the built-in seed pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="datasets/hallushield_med.jsonl")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs) if args.pairs else SEED_PAIRS
    examples = build_med_examples(pairs, seed=args.seed)
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(examples, args.out)
    halu = sum(1 for e in examples if e.hallucinated)
    print(f"wrote {len(examples)} examples ({halu} hallucinated, {len(examples) - halu} clean) -> {args.out}")


if __name__ == "__main__":
    main()
