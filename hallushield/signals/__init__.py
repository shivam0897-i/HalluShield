"""Signal registry.

`build_signal(name)` maps a config name to a concrete `Signal`. Later phases
register "grounding" (LettuceDetect/MiniCheck) and "logic" (LLM-as-judge) here
behind lazy imports so Phase 0 needs no ML dependencies.
"""

from __future__ import annotations

from ..core.types import Signal
from .lexical import LexicalOverlapSignal


def build_signal(name: str) -> Signal:
    if name == "lexical":
        return LexicalOverlapSignal()
    if name == "grounding":
        from .grounding import GroundingSignal  # noqa: PLC0415 — lazy, Phase 1 (heavy deps)

        return GroundingSignal()
    if name == "logic":
        from .logic_judge import LogicJudgeSignal  # noqa: PLC0415 — lazy, Phase 2

        return LogicJudgeSignal()
    raise ValueError(f"Unknown signal: {name!r}")


__all__ = ["build_signal", "LexicalOverlapSignal"]
