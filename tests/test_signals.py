from hallushield.core.types import Chunk
from hallushield.signals.lexical import LexicalOverlapSignal

CHUNKS = [
    Chunk("c1", "Metformin initial dose is 500mg twice daily with meals."),
    Chunk("c2", "Aspirin is used to reduce fever and inflammation."),
]


def test_overlapping_claim_scores_high_and_attributes():
    sig = LexicalOverlapSignal()
    r = sig.score("Metformin dose is 500mg twice daily", CHUNKS)
    assert r.score > 0.7
    assert r.best_chunk_id == "c1"


def test_unsupported_claim_scores_low():
    sig = LexicalOverlapSignal()
    r = sig.score("The patient should undergo immediate cardiac surgery", CHUNKS)
    assert r.score < 0.3


def test_overlapping_beats_unsupported():
    sig = LexicalOverlapSignal()
    grounded = sig.score("Metformin 500mg twice daily", CHUNKS).score
    invented = sig.score("Quantum teleportation cures diabetes", CHUNKS).score
    assert grounded > invented


def test_empty_inputs_are_safe():
    sig = LexicalOverlapSignal()
    assert sig.score("", CHUNKS).score == 0.0
    assert sig.score("anything", []).score == 0.0
