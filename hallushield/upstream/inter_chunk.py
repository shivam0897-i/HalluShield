"""Inter-chunk contradiction detection.

Surfaces conflicting evidence *among* the retrieved chunks (e.g. one says 10mg,
another 100mg). The real path is an NLI cross-encoder injected as
`nli(a, b) -> contradiction_prob`; the dependency-free fallback is a heuristic
that flags same-topic chunks with conflicting numbers or mismatched negation.
Detected conflicts are reported (not silently removed), per HALLUSHIELD.md §7.1.
"""

from __future__ import annotations

import re
from typing import Callable

from ..core.text import content_tokens
from ..core.types import Chunk

_NEG = re.compile(r"\b(?:not|no|never|cannot|can't|without|denies|absence|none)\b", re.IGNORECASE)
_NUM = re.compile(r"\d+(?:\.\d+)?")


def heuristic_contradiction(a: str, b: str) -> float:
    """Cheap contradiction proxy in [0, 1] for two chunk texts."""
    ta, tb = content_tokens(a), content_tokens(b)
    shared = ta & tb
    if len(shared) < 2:  # unrelated topics can't really contradict
        return 0.0
    nums_a, nums_b = set(_NUM.findall(a)), set(_NUM.findall(b))
    if nums_a and nums_b and not (nums_a & nums_b):
        return 0.8  # same topic, entirely different numbers (e.g. dosages)
    if bool(_NEG.search(a)) != bool(_NEG.search(b)) and len(shared) >= 3:
        return 0.6  # one asserts, the other negates, on a shared topic
    return 0.0


class InterChunkContradiction:
    def __init__(self, nli: Callable[[str, str], float] | None = None, threshold: float = 0.5) -> None:
        self._score = nli or heuristic_contradiction
        self.threshold = threshold

    def detect(self, chunks: list[Chunk]) -> list[tuple[str, str, float]]:
        """Return [(chunk_id_a, chunk_id_b, contradiction_score), ...] above threshold."""
        conflicts: list[tuple[str, str, float]] = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                c = float(self._score(chunks[i].text, chunks[j].text))
                if c >= self.threshold:
                    conflicts.append((chunks[i].id, chunks[j].id, round(c, 3)))
        return conflicts
