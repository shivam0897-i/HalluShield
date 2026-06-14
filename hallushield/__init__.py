"""HalluShield — real-time hallucination firewall for RAG systems.

Public surface kept small on purpose; import from submodules for internals.
"""

from .core.types import Chunk, ClaimScore, Signal, SignalResult, ValidationResult, Verdict
from .pipeline import build_default_fusion, validate

__all__ = [
    "Chunk",
    "ClaimScore",
    "Signal",
    "SignalResult",
    "ValidationResult",
    "Verdict",
    "validate",
    "build_default_fusion",
]
