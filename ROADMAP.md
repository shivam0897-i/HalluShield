# Roadmap

The library, FastAPI middleware, evaluation harness, and demo are complete and tested. The remaining work is primarily **data acquisition** and **tuning on real data** rather than new code — every harness below is already implemented and unit-tested.

## Benchmarking on real data

- **RAGTruth headline F1.** The harness (`benchmarks/ragtruth_eval`, `benchmarks/span_eval`, `benchmarks/ablation`) is ready; it needs the RAGTruth release. Download `response.jsonl` and `source_info.jsonl` (ParticleMedia/RAGTruth) into `datasets/`, then:

  ```bash
  python -m benchmarks.ablation \
      --responses datasets/response.jsonl \
      --source-info datasets/source_info.jsonl \
      --split test
  ```

  Results to date are on the synthetic `HalluShield-Med` set (`n = 10`) and are illustrative only.

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
