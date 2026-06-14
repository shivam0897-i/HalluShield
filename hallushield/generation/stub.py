"""Deterministic generator for tests, demos, and offline eval.

By default it returns the top retrieved chunk's text (a trivially grounded
answer), so the happy-path slice produces PASS without any API. Pass a fixed
string to simulate a specific (possibly fabricated) answer for BLOCK demos.
"""

from __future__ import annotations

from ..core.types import Chunk


class StubGenerator:
    def __init__(self, fixed: str | None = None) -> None:
        self._fixed = fixed

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        if self._fixed is not None:
            return self._fixed
        return chunks[0].text if chunks else "I don't have enough information to answer."
