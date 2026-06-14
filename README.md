# 🔥 HalluShield

Real-time hallucination firewall for RAG systems. Final-year B.Tech project.

Validates an LLM answer **before it reaches the user**: scores every sentence
for groundedness against the retrieved chunks, attaches per-sentence source
citations, and (later phases) validates the chunks upstream and self-heals
blocked claims.

- **Design doc / single source of truth:** [HALLUSHIELD.md](HALLUSHIELD.md)
- **Build plan (descoped, current):** `.claude/plans/` — see the approved plan.

## Status

**Phase 0 complete** — skeleton, the pluggable `Signal` interface, fusion,
domain-adaptive decision engine, the `validate` pipeline, and the RAGTruth eval
harness all run with a trivial lexical baseline and **zero ML dependencies**.
Phase 1 swaps in the real grounding model (LettuceDetect / MiniCheck).

## Quick start

```bash
python -m pip install -e ".[dev]"     # Phase 0 needs nothing heavier
python -m pytest -q                    # 22 tests, all green
python -m benchmarks.ragtruth_eval --demo   # eval harness on a synthetic fixture
```

Use the firewall directly:

```python
from hallushield import Chunk, validate

result = validate(
    answer="Metformin is started at 500mg twice daily with meals.",
    chunks=[Chunk("c1", "Metformin initial dose is 500mg twice daily.", source="ADA 2024")],
    domain="medical",
)
print(result.verdict, round(result.answer_score, 2))
for c in result.claims:
    print(c.sentence_id, c.verdict, c.supporting_chunk, c.claim)
```

## Architecture (the one idea that matters)

Every detection method implements one `Signal` protocol
(`score(claim, chunks) -> SignalResult`). `FusionScorer` combines whichever
signals are **enabled in `config.ENABLED_SIGNALS`** — so the ablation study
(core → +logic → +upstream → +self-heal) is a config change, not a rewrite.

```
hallushield/
  config.py            # thresholds, weights, ENABLED_SIGNALS (ablation knob)
  core/                # types, sentence splitter, FusionScorer
  signals/             # lexical (now); grounding, logic (next)
  decision/            # score -> PASS/WARN/HEAL; self-healing (later)
  upstream/            # chunk poisoning / credibility shield (later)
  pipeline.py          # validate(answer, chunks, domain) — core of POST /validate
benchmarks/            # RAGTruth loader + P/R/F1 + ablation
tests/
```

## Real benchmark

`python -m benchmarks.ragtruth_eval --data path/to/ragtruth.jsonl`
(see the loader docstring for the expected JSONL schema).
