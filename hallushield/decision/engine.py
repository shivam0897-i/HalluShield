"""Decision engine: fused score + domain -> verdict.

Domain selects the threshold set (HALLUSHIELD.md §7.7); unknown domains fall
back to "general".
"""

from __future__ import annotations

from ..config import THRESHOLDS
from ..core.types import Verdict


def decide(score: float, domain: str = "general") -> Verdict:
    t = THRESHOLDS.get(domain, THRESHOLDS["general"])
    if score >= t["pass"]:
        return Verdict.PASS
    if score >= t["warn"]:
        return Verdict.WARN
    return Verdict.HEAL
