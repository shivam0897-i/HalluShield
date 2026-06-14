"""FAISS-backed dense retriever (Phase 1 production path).

Lazy-imports sentence-transformers + faiss so the package installs and tests
run without the ML stack. Install with:  pip install -e ".[ml]"
"""

from __future__ import annotations

from ..config import MODELS
from ..core.types import Chunk


class FaissRetriever:
    def __init__(self, chunks: list[Chunk], model_name: str | None = None) -> None:
        try:
            import faiss  # noqa: PLC0415
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise ImportError(
                "FaissRetriever needs the ML extras. Install: pip install -e \".[ml]\""
            ) from exc

        self._chunks = list(chunks)
        self._model = SentenceTransformer(model_name or MODELS["embeddings"])
        embeddings = self._model.encode(
            [c.text for c in self._chunks], normalize_embeddings=True
        )
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        q = self._model.encode([query], normalize_embeddings=True)
        k = min(k, len(self._chunks))
        _, idx = self._index.search(q, k)
        return [self._chunks[i] for i in idx[0] if i >= 0]
