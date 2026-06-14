"""Dependency-free retriever: token-overlap ranking over an in-memory corpus.

Good enough for tests, demos, and the eval harness when no GPU/embeddings are
available. The FAISS retriever (faiss_store.py) is the production path; both
satisfy the same `Retriever` protocol, so the firewall code never changes.
"""

from __future__ import annotations

from ..core.text import content_tokens
from ..core.types import Chunk


class InMemoryRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tokens = [content_tokens(c.text) for c in self._chunks]

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        q = content_tokens(query)
        if not q:
            return self._chunks[:k]
        scored = [
            (len(q & toks) / len(q), chunk)
            for toks, chunk in zip(self._tokens, self._chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored[:k] if score > 0] or self._chunks[:k]
