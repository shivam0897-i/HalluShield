# 🔥 HalluShield

**A real-time hallucination firewall for Retrieval-Augmented Generation (RAG) systems.**

HalluShield is a middleware layer you place between your RAG pipeline and your users. It takes a `query`, your LLM's `answer`, and the `chunks` you retrieved, then scores **every sentence** of the answer for groundedness against those chunks, attaches **per-sentence source citations**, and returns a verdict — `PASS`, `WARN`, or `HEAL` — so unsupported or fabricated claims are caught *before* the user sees them.

It is **model-agnostic** (any LLM via [LiteLLM](https://github.com/BerriAI/litellm)), **vector-store-agnostic** (bring your own retriever), and ships with a dependency-free baseline so it runs and is testable with no GPU or API key.

> ⚠️ **`PASS` means a claim is *grounded in the retrieved chunks* — not that it is factually or clinically correct.** The chunks themselves may be wrong. HalluShield is a safety layer, not a source of truth; keep a human in the loop for high-stakes use.

---

## What you get

- **Per-sentence groundedness scoring** with inline source-chunk citations.
- **A single verdict** (`PASS` / `WARN` / `HEAL`) plus a numeric score, with **domain-adaptive thresholds** (stricter for `medical` / `legal` / `finance`).
- **Pluggable detection signals** behind one interface: a *lexical* baseline (default, no deps), a *grounding* model ([LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)), and an *LLM-as-judge* (`logic`) — combined by a fusion scorer.
- **Upstream Shield** — validates retrieved chunks *before* generation (source-credibility + inter-chunk contradiction), down-weighting or dropping bad context.
- **Self-healing loop** — on a blocked answer, re-query → re-retrieve → re-generate → re-score, with an honest "not enough information" fallback.
- **Two integration modes** — call it in-process (Python) or run it as an HTTP service.

---

## Installation

Requires Python 3.11+.

```bash
pip install -e ".[dev]"          # core library + tests (no ML/API stack)
pip install -e ".[ml]"           # real grounding model + embeddings + FAISS
pip install -e ".[api]"          # FastAPI middleware + LiteLLM generation/judge
pip install -e ".[demo]"         # Streamlit dashboard
```

Everything works with `[dev]` alone (lexical baseline). The extras enable the real models, retrieval, and the API server.

---

## Integrate into your RAG pipeline

HalluShield does **not** replace your retriever or LLM — it validates their output. You already produce a `query`, an `answer`, and the retrieved `chunks`; hand them over and branch on the verdict.

### Option A — in-process (Python)

```python
from hallushield import Chunk, validate
from hallushield.core.types import Verdict

# ↓ these come from YOUR existing RAG pipeline
query   = "What is the recommended dose of metformin?"
answer  = your_llm.generate(query, retrieved)          # your model
chunks  = [Chunk(id=d.id, text=d.text, source=d.source) for d in retrieved]

result = validate(query=query, answer=answer, chunks=chunks, domain="medical")

if result.verdict is Verdict.PASS:
    return answer                                       # safe to deliver
elif result.verdict is Verdict.WARN:
    return render_with_warnings(answer, result)         # highlight weak sentences
else:  # HEAL
    return "I don't have enough information to answer that accurately."
```

**What `result` gives you:**

```python
result.verdict        # Verdict.PASS | WARN | HEAL  (worst of all sentences)
result.answer_score   # mean groundedness in [0, 1]
result.domain         # the domain you passed
for claim in result.claims:
    claim.claim             # the sentence text
    claim.verdict           # PASS / WARN / HEAL for this sentence
    claim.fused_score       # [0, 1]
    claim.supporting_chunk  # id of the chunk that best supports it (your citation)
    claim.signals           # per-signal breakdown
```

### Option B — as an HTTP service (drop-in middleware)

```bash
pip install -e ".[api]"
uvicorn hallushield.middleware.app:app --reload
```

**Request** — `POST /validate`:

```json
{
  "query": "What is the recommended dose of Metformin?",
  "answer": "Metformin is started at 500mg twice daily with meals.",
  "chunks": [
    {"id": "chunk_1", "text": "Metformin initial dose is 500mg twice daily.", "source": "ADA 2024"}
  ],
  "domain": "medical",
  "options": {"return_claim_scores": true}
}
```

**Response:**

```json
{
  "verdict": "PASS",
  "answer_score": 0.95,
  "domain": "medical",
  "claims": [
    {"id": "c0", "text": "Metformin is started at 500mg twice daily with meals.",
     "score": 0.95, "verdict": "PASS", "supporting_chunk": "chunk_1"}
  ],
  "latency_ms": 8,
  "healed": false
}
```

Call this from any language. Chunk ids must be unique and non-empty so `supporting_chunk` is a reliable citation key. On an internal error the endpoint **fails closed** (HTTP 500, never a silent `PASS`).

### Option C — let HalluShield drive the whole loop

If you want HalluShield to retrieve, generate, shield, and self-heal for you:

```python
from hallushield.firewall import HalluShield
from hallushield.retrieval import build_retriever        # InMemory or FAISS
from hallushield.generation import build_generator       # stub or LiteLLM
from hallushield.upstream import build_shield
from hallushield.decision.self_healing import SelfHealer

retriever = build_retriever(corpus, backend="memory")    # or "faiss"
generator = build_generator(backend="litellm")           # any provider via LiteLLM
shield    = build_shield(drop_below=0.4)                  # pre-generation chunk filter
healer    = SelfHealer(retriever, generator, domain="medical")

firewall = HalluShield(retriever, generator, domain="medical", shield=shield, healer=healer)
result   = firewall.answer("What is the metformin dose?")
print(result.validation.verdict, result.healed, result.answer)
```

---

## How it works

A standard RAG call is wrapped in four layers:

```text
Query → Retrieve → [Upstream Shield] → Generate
      → Validate (per-sentence scoring) → Decide (PASS / WARN / HEAL) → [Self-Heal] → User
```

Every detection method implements one protocol, so signals are interchangeable and the ablation study is a config change, not a rewrite:

```python
class Signal(Protocol):
    name: str
    def score(self, claim: str, chunks: list[Chunk], query: str = "") -> SignalResult: ...
```

`FusionScorer` combines whichever signals are listed in `config.ENABLED_SIGNALS`.

---

## Going beyond the baseline

The default `ENABLED_SIGNALS = ["lexical"]` needs no extra dependencies and is good for trying the flow. For production accuracy, enable the real signals:

| To enable | Install | Then set |
| --- | --- | --- |
| Real grounding (LettuceDetect) | `pip install -e ".[ml]"` | `ENABLED_SIGNALS = ["grounding"]` |
| LLM-as-judge fusion | `pip install -e ".[api]"` + an API key in `.env` | `ENABLED_SIGNALS = ["grounding", "logic"]` |
| Dense retrieval (FAISS) | `pip install -e ".[ml]"` | `build_retriever(corpus, backend="faiss")` |
| Real generation | `pip install -e ".[api]"` + API key | `build_generator(backend="litellm")` |

API keys and model overrides live in a gitignored `.env` (auto-loaded):

```bash
OPENROUTER_API_KEY=...                       # or OPENAI_API_KEY / ANTHROPIC_API_KEY
HALLUSHIELD_JUDGE_MODEL=openrouter/openai/gpt-4o-mini
```

---

## Configuration

| Setting | Where | Purpose |
| --- | --- | --- |
| `ENABLED_SIGNALS` | `hallushield/config.py` | Which detectors run, e.g. `["grounding", "logic"]`. The ablation control. |
| `THRESHOLDS` | `hallushield/config.py` | Per-domain PASS/WARN cut-offs (medical `0.90` … general `0.75`). |
| `MODELS` | `hallushield/config.py` | Grounding / embedding / judge / generator model ids. |
| `GROUNDING_SPAN_PENALTY` | `hallushield/config.py` | Ensures a short but critical unsupported span (e.g. a wrong dosage) can't slip past a strict threshold on low character coverage. |
| API keys & model overrides | `.env` (gitignored) | `OPENROUTER_API_KEY`, `HALLUSHIELD_JUDGE_MODEL`, etc. |

---

## Evaluation

The headline benchmark is **[RAGTruth](https://arxiv.org/abs/2402.07067)** (example- and span-level F1).

```bash
python -m benchmarks.ragtruth_eval --responses response.jsonl --source-info source_info.jsonl
python -m benchmarks.span_eval     --responses response.jsonl --source-info source_info.jsonl
python -m benchmarks.ablation      --responses response.jsonl --source-info source_info.jsonl
```

**HalluShield-Med** generates a labelled medical set by injecting controlled errors (numeric / entity / negation) into grounded Q&A, with gold spans:

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

The baseline over-flags paraphrases; grounding alone is precise but misses subtle edits; fusing the two recovers both — the case for multi-signal fusion. See [ROADMAP.md](ROADMAP.md) to run the real RAGTruth benchmark.

---

## Demo

```bash
streamlit run dashboard/app.py
```

Live verdict, colour-coded per-sentence groundedness, source citations, the Upstream Shield report, and the self-healing trace.

---

## Project structure

```text
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

Each external dependency has a dependency-free implementation (testable now) and a lazy real one behind the same protocol, so the suite stays green without the ML/API stack.

```bash
python -m pytest -q
```

---

## Limitations

- **Grounded ≠ correct.** A `PASS` only means the claim is supported by the retrieved chunks.
- **No omission detection.** HalluShield flags what the model *wrongly states*, not facts it *omits*.
- **English-only** detection models; **short-context** truncation for very long chunks.
- Token-level model confidence (logprobs) is **not** used — most hosted LLM APIs don't expose it.

These are deliberate scope choices. See [ROADMAP.md](ROADMAP.md) for remaining work and how to run the real RAGTruth benchmark; contributions are welcome.

---

## Acknowledgements

Builds on ideas and tools from LettuceDetect, MiniCheck, RAGTruth, RAGAS, and Corrective/Self-RAG.

## License

Released under the [MIT License](LICENSE).
