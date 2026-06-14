"""Retrieval layer: a Retriever protocol with a dependency-free in-memory
implementation (testable now) and a FAISS-backed one (real use, Phase 1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.types import Chunk
from .memory import InMemoryRetriever


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Chunk]: ...


def build_retriever(chunks: list[Chunk], backend: str = "memory", **kwargs) -> Retriever:
    if backend == "memory":
        return InMemoryRetriever(chunks)
    if backend == "faiss":
        from .faiss_store import FaissRetriever  # noqa: PLC0415 — lazy, needs ML extras

        return FaissRetriever(chunks, **kwargs)
    raise ValueError(f"Unknown retriever backend: {backend!r}")


__all__ = ["Retriever", "InMemoryRetriever", "build_retriever"]
