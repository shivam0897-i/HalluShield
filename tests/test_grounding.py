"""Unit tests for hallushield.signals.grounding (injectable detector, no ML deps)."""

from __future__ import annotations

from hallushield.config import GROUNDING_SPAN_PENALTY
from hallushield.core.types import Chunk, Verdict
from hallushield.decision.engine import decide
from hallushield.signals.grounding import GroundingSignal

# ---------------------------------------------------------------------------
# Fake detector — records the kwargs it was last called with so tests can
# assert on forwarded arguments.  Returns whatever spans list it was given at
# construction time.
# ---------------------------------------------------------------------------

CHUNKS = [
    Chunk("c1", "Metformin initial dose is 500mg twice daily with meals."),
    Chunk("c2", "Ibuprofen 200mg is used for mild pain relief."),
]

# A 50-character claim containing a wrong dosage ("10mg" — 4 chars, ≈ 8 % of
# the claim).  Old formula: score = 1 - 0.08 = 0.92 → passes the 0.90 medical
# bar even though the dosage is hallucinated.  New formula adds a flat penalty.
LONG_CLAIM = "The patient was prescribed 10mg of ibuprofen daily"


class FakeDetector:
    """Minimal stand-in for HallucinationDetector.predict."""

    def __init__(self, spans: list[dict]) -> None:
        self._spans = spans
        self.last_kwargs: dict = {}

    def predict(self, **kwargs) -> list[dict]:
        self.last_kwargs = kwargs
        return self._spans


# ---------------------------------------------------------------------------
# No spans → score == 1.0 → PASS even at medical threshold
# ---------------------------------------------------------------------------


def test_no_spans_score_is_1():
    detector = FakeDetector(spans=[])
    sig = GroundingSignal(detector=detector)
    r = sig.score("Metformin 500mg twice daily", CHUNKS)
    assert r.score == 1.0
    assert r.name == "grounding"


def test_no_spans_passes_medical():
    detector = FakeDetector(spans=[])
    sig = GroundingSignal(detector=detector)
    r = sig.score("Metformin 500mg twice daily", CHUNKS)
    assert decide(r.score, "medical") == Verdict.PASS


# ---------------------------------------------------------------------------
# Short span in a long claim — regression test.
# A 4-char unsupported span ("10mg") in a ~50-char claim gives coverage ≈ 0.08.
# Old formula: score = 1 - 0.08 = 0.92 → PASS (wrong — hallucination missed).
# New formula: score = 1 - 0.08 - 0.4 = 0.52 → not PASS (correct).
# ---------------------------------------------------------------------------


def _span_for_wrong_dosage() -> list[dict]:
    """Flag the substring '10mg' (the hallucinated dosage) inside LONG_CLAIM."""
    start = LONG_CLAIM.index("10mg")
    end = start + len("10mg")
    return [{"start": start, "end": end, "text": "10mg"}]


def test_short_span_fails_medical_with_new_formula():
    spans = _span_for_wrong_dosage()
    detector = FakeDetector(spans=spans)
    sig = GroundingSignal(detector=detector)
    r = sig.score(LONG_CLAIM, CHUNKS)
    assert r.score < 0.90, f"Expected score < 0.90 (medical pass bar), got {r.score}"
    assert decide(r.score, "medical") != Verdict.PASS


def test_old_formula_would_have_passed_short_span():
    """Document that the old 1-coverage formula was insufficient."""
    spans = _span_for_wrong_dosage()
    flagged = sum(max(0, s["end"] - s["start"]) for s in spans)
    old_score = 1.0 - min(1.0, flagged / len(LONG_CLAIM))
    # Old score passes the 0.90 medical bar — this is the bug we fixed.
    assert old_score >= 0.90, (
        f"Old formula score was {old_score}; expected it to (wrongly) pass ≥ 0.90"
    )


# ---------------------------------------------------------------------------
# Span covering the whole claim → score == 0.0
# ---------------------------------------------------------------------------


def test_full_span_score_is_zero():
    claim = "The patient should take 5000mg of metformin."
    spans = [{"start": 0, "end": len(claim), "text": claim}]
    detector = FakeDetector(spans=spans)
    sig = GroundingSignal(detector=detector)
    r = sig.score(claim, CHUNKS)
    assert r.score == 0.0


# ---------------------------------------------------------------------------
# Empty / whitespace claim → score == 0.0
# ---------------------------------------------------------------------------


def test_empty_claim_returns_zero():
    detector = FakeDetector(spans=[])
    sig = GroundingSignal(detector=detector)
    assert sig.score("", CHUNKS).score == 0.0


def test_whitespace_claim_returns_zero():
    detector = FakeDetector(spans=[])
    sig = GroundingSignal(detector=detector)
    assert sig.score("   ", CHUNKS).score == 0.0


# ---------------------------------------------------------------------------
# query is forwarded to detector.predict as `question`
# ---------------------------------------------------------------------------


def test_query_forwarded_as_question():
    detector = FakeDetector(spans=[])
    sig = GroundingSignal(detector=detector)
    sig.score("Metformin 500mg", CHUNKS, query="What is the starting dose?")
    assert detector.last_kwargs.get("question") == "What is the starting dose?"


def test_empty_query_forwarded():
    detector = FakeDetector(spans=[])
    sig = GroundingSignal(detector=detector)
    sig.score("Metformin 500mg", CHUNKS, query="")
    assert detector.last_kwargs.get("question") == ""


# ---------------------------------------------------------------------------
# best_chunk_id attribution — lexical overlap picks the most relevant chunk
# ---------------------------------------------------------------------------


def test_best_chunk_id_picks_overlapping_chunk():
    detector = FakeDetector(spans=[])
    sig = GroundingSignal(detector=detector)
    r = sig.score("Metformin dose is 500mg", CHUNKS)
    # c1 shares more content tokens with the claim than c2 does.
    assert r.best_chunk_id == "c1"


def test_best_chunk_id_picks_other_chunk():
    detector = FakeDetector(spans=[])
    sig = GroundingSignal(detector=detector)
    r = sig.score("Ibuprofen 200mg pain relief", CHUNKS)
    assert r.best_chunk_id == "c2"


def test_best_chunk_id_none_when_no_chunks():
    detector = FakeDetector(spans=[])
    sig = GroundingSignal(detector=detector)
    r = sig.score("Metformin 500mg", [])
    assert r.best_chunk_id is None


# ---------------------------------------------------------------------------
# Penalty constant is in expected range (sanity guard against accidental edits)
# ---------------------------------------------------------------------------


def test_penalty_constant_sensible():
    assert 0.0 < GROUNDING_SPAN_PENALTY < 1.0


# ---------------------------------------------------------------------------
# Degenerate (zero-length) spans are ignored — no false penalty
# ---------------------------------------------------------------------------


def test_zero_length_span_is_ignored():
    # A span with start == end flags no characters and must not trigger the penalty.
    detector = FakeDetector(spans=[{"start": 5, "end": 5, "text": ""}])
    sig = GroundingSignal(detector=detector)
    r = sig.score("Metformin 500mg twice daily", CHUNKS)
    assert r.score == 1.0
