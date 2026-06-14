"""LiteLLM-backed generator — provider-agnostic, API-only (Phase 1).

Lazy-imports litellm so the package installs without it. The model id comes
from config.MODELS["generator"]; the API key is read from the environment by
litellm (e.g. ANTHROPIC_API_KEY / OPENAI_API_KEY). Install:  pip install -e ".[api]"

Prompt is grounded-RAG style: answer ONLY from context, else say "I don't know"
— this keeps the generator honest so the firewall measures grounding, not the
model's parametric knowledge.
"""

from __future__ import annotations

from ..config import MODELS
from ..core.types import Chunk

_SYSTEM = (
    "You are a careful assistant. Answer the question using ONLY the provided "
    "context passages. If the context does not contain the answer, say exactly: "
    "\"I don't have enough information to answer.\" Do not use outside knowledge."
)


def _format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)


class LiteLLMGenerator:
    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        try:
            import litellm  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise ImportError(
                "LiteLLMGenerator needs the api extras. Install: pip install -e \".[api]\""
            ) from exc
        self._litellm = litellm
        self._model = model or MODELS["generator"]
        self._temperature = temperature

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Context:\n{_format_context(chunks)}\n\nQuestion: {query}"},
        ]
        resp = self._litellm.completion(
            model=self._model, messages=messages, temperature=self._temperature
        )
        return resp["choices"][0]["message"]["content"]
