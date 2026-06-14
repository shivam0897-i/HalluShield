"""Central configuration: domain thresholds, fusion weights, and signal toggles.

The signal toggles are the ablation knob — `ENABLED_SIGNALS` drives which
scorers the default pipeline runs, so an ablation row is one config change.
"""

from __future__ import annotations

import os

# Auto-load a local .env (gitignored) when python-dotenv is available, so API
# keys and model overrides are picked up without manual exports. No-op otherwise.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Domain-adaptive decision thresholds (see HALLUSHIELD.md §7.7).
# Stricter for high-stakes domains; relaxed for general queries.
THRESHOLDS: dict[str, dict[str, float]] = {
    "medical": {"pass": 0.90, "warn": 0.70},
    "legal": {"pass": 0.88, "warn": 0.65},
    "finance": {"pass": 0.85, "warn": 0.60},
    "general": {"pass": 0.75, "warn": 0.50},
}

# Which signals the default fusion runs. This is the ablation control:
#   ["lexical"]                      -> Phase 0 trivial baseline
#   ["grounding"]                    -> Phase 1 core (LettuceDetect/MiniCheck)
#   ["grounding", "logic"]           -> Phase 2
# Names must resolve in hallushield.signals.build_signal().
ENABLED_SIGNALS: list[str] = ["lexical"]

# Per-signal fusion weights (normalised over the enabled set at runtime).
DEFAULT_WEIGHTS: dict[str, float] = {
    "lexical": 1.0,
    "grounding": 0.6,   # core NLI/grounding signal (Phase 1)
    "logic": 0.4,       # LLM-as-judge reasoning signal (Phase 2)
}

# Additive penalty applied to the fused score when a signal reports a
# contradiction (see HALLUSHIELD.md §7.6 — additive, then clipped to [0,1]).
CONTRADICTION_PENALTY: float = 0.25

# How the per-sentence scores roll up to an answer-level score.
# "mean" reports average groundedness; the answer *verdict* is always the
# worst per-claim verdict (firewall semantics: one bad claim blocks).
ANSWER_AGGREGATE: str = "mean"

# Model identifiers, resolved in later phases (placeholders for now).
MODELS: dict[str, str] = {
    "grounding": "KRLabsOrg/lettucedect-base-modernbert-en-v1",
    "nli": "cross-encoder/nli-deberta-v3-base",
    "embeddings": "BAAI/bge-small-en-v1.5",
    "judge": os.environ.get("HALLUSHIELD_JUDGE_MODEL", "claude-opus-4-8"),        # LLM-as-judge (Phase 2)
    "generator": os.environ.get("HALLUSHIELD_GENERATOR_MODEL", "claude-opus-4-8"),  # answer generation (Phase 1)
}
