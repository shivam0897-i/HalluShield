# Roadmap

The library, FastAPI middleware, evaluation harness, and demo are complete and tested. The remaining work is primarily **data acquisition** and **tuning on real data** rather than new code — every harness below is already implemented and unit-tested.

## Results — RAGTruth test split

Run on the real RAGTruth release (ParticleMedia/RAGTruth); precision/recall/F1 for the positive (hallucinated) class:

| Signals | example-F1 | span-F1 | n |
| --- | --- | --- | --- |
| lexical (baseline) | 0.544 | 0.102 | 2,675 (full test split) |
| grounding | 0.734 | 0.375 | 150 (random sample) |
| grounding + logic | 0.765 | 0.365 | 30 (random sample) |

Each added signal raises example-level F1; grounding (~0.73) is in the range LettuceDetect reports. **Caveats:** the grounding and fusion rows use random sub-samples — the grounding model runs per sentence on CPU and the LLM-judge makes one API call per sentence, so the full-split fusion run is cost/time-bound. The rows therefore use different `n` and are indicative, not a like-for-like comparison.

Reproduce (`--limit` bounds the API-backed logic row):

```bash
mkdir -p datasets
curl -sSL -o datasets/response.jsonl    https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl
curl -sSL -o datasets/source_info.jsonl https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl
python -m benchmarks.ablation  --responses datasets/response.jsonl --source-info datasets/source_info.jsonl --split test --limit 150
python -m benchmarks.span_eval --responses datasets/response.jsonl --source-info datasets/source_info.jsonl --split test --limit 150
```

**For a definitive headline:** run grounding-only over the full test split (free, local — no API), and the fusion row on a larger sample.

## More benchmarking

- **HalluShield-Med at scale.** The injection generator ships with a small built-in seed set. To reach a larger sample, convert medical Q&A (e.g. MedQuAD) into the `--pairs` JSONL shape (`{answer, context, question}`) and run `benchmarks/hallushield_med`.

## Tuning (requires a labelled dev split)

- **Fusion weight calibration.** `config.DEFAULT_WEIGHTS` (α/β/γ) currently use sensible defaults; calibrate on a dev split.
- **Grounding penalty / thresholds.** `GROUNDING_SPAN_PENALTY` and the per-domain `THRESHOLDS` can be tuned once real data is available.
- **Grounding model.** The base LettuceDetect model misses some subtle entity/numeric edits (it returns no span); the larger model or a fine-tuned variant improves recall. The LLM-judge signal complements it — fusing both recovers the misses.

## Performance

- **Latency profiling.** Measure end-to-end latency with the real grounding model and LLM-judge under load.

## Future scope

- Omission detection (facts present in the context but absent from the answer).
- Wire the Upstream Shield's injectable NLI seam to a cross-encoder NLI model.
- Optional consistency-sampling uncertainty signal as an ablation.
