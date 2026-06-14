"""Shared text utilities (tokenisation) used by retrieval and lexical scoring."""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9]+")

# Minimal stopword set so overlap reflects content words, not glue words.
_STOP = frozenset(
    """a an the of to in on at for and or but is are was were be been being
    this that these those it its as by with from into about than then so such
    can could should would may might will shall do does did has have had not""".split()
)


def content_tokens(text: str) -> set[str]:
    """Lowercased content-word tokens (stopwords and 1-char tokens removed)."""
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 1}
