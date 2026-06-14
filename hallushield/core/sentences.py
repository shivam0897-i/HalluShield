"""Lightweight sentence segmentation.

Phase 0/1 uses a regex splitter plus an abbreviation-merge pass so common
abbreviations (``e.g.``, ``i.e.``, ``Dr.``, ``vs.``, ``approx.``) and initials
(``A. Smith``) do NOT create false sentence boundaries — a false split would
fragment a claim and corrupt per-claim scoring. It is still a heuristic; Phase 1
can swap in spaCy behind this same `split_sentences` signature for broader
coverage (decimals, more abbreviations, clinical notation).
"""

from __future__ import annotations

import re

# Candidate split: sentence-ending punctuation, whitespace, then an opener
# (capital letter, bracket, or quote). The abbreviation merge below repairs
# over-eager splits.
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")

# Tokens that end with "." but are not sentence ends (lowercased, period stripped).
_ABBREVIATIONS = frozenset(
    """e.g i.e etc vs approx cf al et seq dr mr mrs ms prof sr jr st rev
    inc ltd co corp dept fig eq vol no pp ch sec art mt ave blvd""".split()
)


def _ends_with_abbreviation(segment: str) -> bool:
    """True if `segment` ends in an abbreviation/initial, so the split was false."""
    segment = segment.rstrip()
    if not segment.endswith("."):
        return False
    last = re.split(r"\s+", segment)[-1]
    core = last[:-1]  # drop trailing period
    if not core:
        return False
    if core.lower() in _ABBREVIATIONS:
        return True
    # Single-letter initial, e.g. "A." in "A. Smith".
    if len(core) == 1 and core.isalpha():
        return True
    # Multi-dot abbreviation, e.g. "e.g" / "i.e" (period already stripped once).
    if "." in core and core.replace(".", "").isalpha():
        return True
    return False


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    merged: list[str] = []
    for piece in _SPLIT.split(text):
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return [s.strip() for s in merged if s.strip()]
