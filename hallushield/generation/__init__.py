"""Generation layer: a Generator protocol with a deterministic stub (testable
now) and a LiteLLM-backed one (provider-agnostic API generation, Phase 1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.types import Chunk
from .stub import StubGenerator


@runtime_checkable
class Generator(Protocol):
    def generate(self, query: str, chunks: list[Chunk]) -> str: ...


def build_generator(backend: str = "stub", **kwargs) -> Generator:
    if backend == "stub":
        return StubGenerator(**kwargs)
    if backend == "litellm":
        from .litellm_gen import LiteLLMGenerator  # noqa: PLC0415 — lazy, needs api extras

        return LiteLLMGenerator(**kwargs)
    raise ValueError(f"Unknown generator backend: {backend!r}")


__all__ = ["Generator", "StubGenerator", "build_generator"]
