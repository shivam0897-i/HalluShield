"""Source-credibility scoring (heuristic).

Maps a chunk's `source` string to a trust score in [0, 1] via substring tiers
(authoritative medical/gov/edu high; reference/news medium; forums/anon low).
Deliberately simple and transparent; a learned credibility model can replace
this behind the same `score(source) -> float` signature. The most conservative
matching tier wins, so a coincidental authoritative substring can't launder a
low-trust source.
"""

from __future__ import annotations

# Substring (lowercased) -> credibility score. Longest/most-specific intent,
# but matching is by containment; on multiple matches the lowest score wins.
DEFAULT_TIERS: dict[str, float] = {
    # high authority
    "pubmed": 0.95, "nih.gov": 0.95, "who.int": 0.95, "cochrane": 0.95,
    "nejm": 0.95, "lancet": 0.95, "ada ": 0.9, "guideline": 0.9,
    "peer-reviewed": 0.9, ".gov": 0.9, ".edu": 0.85, "journal": 0.85,
    # medium
    "wikipedia": 0.6, "reuters": 0.6, "news": 0.55, "textbook": 0.8,
    # low
    "blog": 0.3, "forum": 0.25, "reddit": 0.2, "quora": 0.2, "anonymous": 0.15,
}


class SourceCredibilityScorer:
    def __init__(self, tiers: dict[str, float] | None = None, default: float = 0.5) -> None:
        self.tiers = {k.lower(): v for k, v in (tiers or DEFAULT_TIERS).items()}
        self.default = default

    def score(self, source: str | None) -> float:
        if not source:
            return self.default
        s = source.lower()
        matches = [v for k, v in self.tiers.items() if k in s]
        return min(matches) if matches else self.default
