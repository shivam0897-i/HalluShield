# 🔥 HalluShield

**A real-time hallucination firewall for Retrieval-Augmented Generation (RAG) systems.**

HalluShield is a middleware layer that validates an LLM's answer **before it reaches the user**. It scores every sentence for groundedness against the retrieved context, attaches per-sentence source citations, validates the retrieved chunks *before* generation, and can automatically re-generate blocked answers — so unsupported or fabricated claims are caught at request time rather than discovered later.

It is **model-agnostic** (works with any LLM via [LiteLLM](https://github.com/BerriAI/litellm)), **vector-store-agnostic**, and ships with a dependency-free baseline so the full pipeline runs and is testable without a GPU or API key.

---

## Why

RAG reduces hallucinations but does not eliminate them — a model can still ignore the retrieved context, misread it, or confabulate details. Existing tooling tends to be either **offline** (batch evaluation after deployment) or **coarse** (a single answer-level pass/fail score). HalluShield targets the gap: **real-time, per-sentence interception with source attribution and self-healing**, with stricter thresholds for high-stakes domains such as medicine and law.

> ⚠️ **A `PASS` verdict means a claim is *grounded in the retrieved chunks* — not that it is factually or clinically correct.** The chunks themselves may be wrong. HalluShield is a safety layer, not a source of truth; always keep a human in the loop for high-stakes use.

---

## Features

- **Per-sentence groundedness scoring** with inline source-chunk citations.
- **Pluggable detection signals** behind a single `Signal` interface:
  - *grounding* — span-level NLI via [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) (ModernBERT, 8k context).
  - *logic* — an LLM-as-judge groundedness score (any provider via LiteLLM).
  - *lexical* — a dependency-free token-overlap baseline (default).
- **Upstream Shield** — validates retrieved chunks *before* generation: heuristic source-credibility scoring and inter-chunk contradiction detection, down-weighting or dropping low-credibility context.
- **Self-healing loop** — on a blocked answer, re-query → re-retrieve → re-generate → re-score, with an honest "not enough information" fallback.
- **Domain-adaptive thresholds** — stricter for `medical` / `legal` / `finance`, relaxed for `general`.
- **Drop-in middleware** — a FastAPI `POST /validate` endpoint, plus a Python API.
- **Evaluation harness** — RAGTruth example- and span-level F1, an ablation runner, and a synthetic medical injection set.

---

## How it works

A standard RAG call is wrapped in four layers:

```
Query → Retrieve → [Upstream Shield] → Generate
      → Validate (per-sentence scoring) → Decide (PASS / WARN / HEAL) → [Self-Heal] → User
```

Every detection method implements one protocol:

```python
class Signal(Protocol):
    name: str
    def score(self, claim: str, chunks: list[Chunk], query: str = "") -> SignalResult: ...
```

`FusionScorer` combines whichever signals are listed in `config.ENABLED_SIGNALS`, so adding or removing a detector — and reproducing the ablation study — is a one-line configuration change rather than a code change.

---

## Installation

Requires Python 3.11+.

```bash
pip install -e ".[dev]"          # core library + tests (no ML/API stack)
pip install -e ".[ml]"           # grounding model + embeddings + vector store
pip install -e ".[api]"          # FastAPI middleware + LiteLLM generation/judge
pip install -e ".[demo]"         # Streamlit dashboard
```

The library is fully functional with `[dev]` only (using the lexical baseline). The heavier extras enable the real grounding model, the LLM-judge, retrieval, and the API server.

---

## Quick start

### Validate an answer

```python
from hallushield import Chunk, validate

result = validate(
    query="What is the recommended dose of metformin?",
    answer="Metformin is started at 500mg twice daily with meals.",
    chunks=[Chunk("c1", "Metformin initial dose is 500mg twice daily.", source="ADA 2024")],
    domain="medical",
)

print(result.verdict, round(result.answer_score, 2))      # e.g. PASS 0.95
for c in result.claims:
    print(c.verdict, c.supporting_chunk, c.claim)          # per-sentence verdict + citation
```

### Run the full firewall (retrieve → shield → generate → validate → heal)

```python
from hallushield.firewall import HalluShield
from hallushield.retrieval import InMemoryRetriever
from hallushield.generation import StubGenerator        # or LiteLLMGenerator for a real model
from hallushield.upstream import build_shield
from hallushield.decision.self_healing import SelfHealer

corpus = [Chunk("c1", "Metformin initial dose is 500mg twice daily.", source="ADA guideline")]
retriever = InMemoryRetriever(corpus)

firewall = HalluShield(
    retriever, StubGenerator(),
    domain="medical",
    shield=build_shield(drop_below=0.4),                  # drop low-credibility chunks
    healer=SelfHealer(retriever, StubGenerator()),        # auto-fix blocked answers
)

answer = firewall.answer("What is the metformin dose?")
print(answer.validation.verdict, answer.healed)
```

### Serve as middleware

```bash
uvicorn hallushield.middleware.app:app --reload
```

```http
POST /validate
{ "query": "...", "answer": "...", "chunks": [{"id": "c1", "text": "...", "source": "..."}], "domain": "medical" }
```

---

## Configuration

| Setting | Where | Purpose |
| --- | --- | --- |
| `ENABLED_SIGNALS` | `hallushield/config.py` | Which detectors run, e.g. `["grounding", "logic"]`. The ablation control. |
| `THRESHOLDS` | `hallushield/config.py` | Per-domain PASS/WARN cut-offs (medical `0.90` … general `0.75`). |
| `MODELS` | `hallushield/config.py` | Grounding / embedding / judge / generator model ids. |
| API keys & model overrides | `.env` (gitignored) | `OPENROUTER_API_KEY`, `HALLUSHIELD_JUDGE_MODEL`, etc. Loaded automatically. |

To switch from the baseline to the real grounding model:

```python
# hallushield/config.py
ENABLED_SIGNALS = ["grounding"]          # requires: pip install -e ".[ml]"
```

---

## Evaluation

The headline benchmark is **[RAGTruth](https://arxiv.org/abs/2402.07067)** (example- and span-level F1).

```bash
# example-level F1
python -m benchmarks.ragtruth_eval --responses response.jsonl --source-info source_info.jsonl
# span-level (character) F1 — comparable to LettuceDetect
python -m benchmarks.span_eval     --responses response.jsonl --source-info source_info.jsonl
# ablation across signal configurations -> results.json
python -m benchmarks.ablation      --responses response.jsonl --source-info source_info.jsonl
```

### Synthetic medical injection set

`HalluShield-Med` builds a labelled evaluation set by injecting controlled errors (numeric, entity, negation) into grounded medical Q&A, with gold spans for span-level scoring:

```bash
python -m benchmarks.hallushield_med --demo --out datasets/hallushield_med.jsonl
python -m benchmarks.ablation --jsonl datasets/hallushield_med.jsonl
```

**Illustrative ablation** (synthetic `HalluShield-Med`, `n = 10` — demonstrates the methodology, not a statistically significant result):

| Signals | Precision | Recall | F1 |
| --- | --- | --- | --- |
| lexical (baseline) | 0.50 | 1.00 | 0.67 |
| grounding | 1.00 | 0.40 | 0.57 |
| **grounding + logic** | **1.00** | **1.00** | **1.00** |

The baseline over-flags paraphrases (low precision); grounding alone is precise but misses subtle edits (low recall); fusing the two recovers both — the case for multi-signal fusion.

---

## Demo

```bash
streamlit run dashboard/app.py
```

An interactive dashboard showing the live verdict, colour-coded per-sentence groundedness, source citations, the Upstream Shield report, and the self-healing trace.

---

## Project structure

```
hallushield/
  config.py            # thresholds, fusion weights, ENABLED_SIGNALS, model ids
  core/                # types + Signal protocol, sentence splitter, FusionScorer
  signals/             # lexical (baseline), grounding (LettuceDetect), logic (LLM-judge)
  retrieval/           # Retriever protocol — in-memory + FAISS
  generation/          # Generator protocol — deterministic stub + LiteLLM
  upstream/            # Upstream Shield: source credibility + inter-chunk contradiction
  decision/            # score -> PASS/WARN/HEAL; prompt-only self-healing loop
  pipeline.py          # validate(answer, chunks, domain, query) — core of /validate
  firewall.py          # HalluShield: retrieve -> shield -> generate -> validate -> heal
  middleware/          # FastAPI app + schemas
dashboard/             # Streamlit demo
benchmarks/            # RAGTruth eval, span-level F1, ablation, HalluShield-Med
tests/
```

Each external dependency has a dependency-free implementation (testable now) and a lazy real implementation behind the same protocol, so the suite stays green without the ML/API stack.

```bash
python -m pytest -q
```

---

## Limitations

- **Grounded ≠ correct.** A `PASS` only means the claim is supported by the retrieved chunks.
- **No omission detection.** HalluShield flags what the model *wrongly states*, not critical facts it *omits*.
- **English-only** detection models; **short-context** truncation for very long chunks.
- Token-level model confidence (logprobs) is **not** used — most hosted LLM APIs do not expose it.

These are deliberate scope choices; contributions extending them are welcome.

---

## Acknowledgements

Builds on ideas and tools from LettuceDetect, MiniCheck, RAGTruth, RAGAS, and Corrective/Self-RAG.

## License

Released under the MIT License.
