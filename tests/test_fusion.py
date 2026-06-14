from hallushield.core.fusion import FusionScorer
from hallushield.core.types import Chunk, SignalResult

CHUNKS = [Chunk("c1", "irrelevant")]


class _Stub:
    """A signal that returns whatever it was constructed with."""

    def __init__(self, name, score, contradiction=0.0, best_chunk_id="c1"):
        self.name = name
        self._score = score
        self._contradiction = contradiction
        self._best = best_chunk_id

    def score(self, claim, chunks, query=""):
        return SignalResult(self.name, self._score, best_chunk_id=self._best,
                            contradiction=self._contradiction)


def test_single_signal_passthrough():
    f = FusionScorer([_Stub("a", 0.8)])
    fused, results, best = f.score_claim("x", CHUNKS)
    assert abs(fused - 0.8) < 1e-9
    assert set(results) == {"a"}
    assert best == "c1"


def test_weighted_mean():
    f = FusionScorer([_Stub("a", 1.0), _Stub("b", 0.0)], weights={"a": 3.0, "b": 1.0})
    fused, _, _ = f.score_claim("x", CHUNKS)
    assert abs(fused - 0.75) < 1e-9  # (3*1 + 1*0) / 4


def test_contradiction_penalty_is_additive_and_clipped():
    f = FusionScorer([_Stub("a", 0.6, contradiction=1.0)], contradiction_penalty=0.25)
    fused, _, _ = f.score_claim("x", CHUNKS)
    assert abs(fused - 0.35) < 1e-9  # 0.6 - 0.25*1.0

    f2 = FusionScorer([_Stub("a", 0.1, contradiction=1.0)], contradiction_penalty=0.25)
    fused2, _, _ = f2.score_claim("x", CHUNKS)
    assert fused2 == 0.0  # clipped, not negative


def test_best_chunk_from_highest_scoring_signal():
    f = FusionScorer([_Stub("a", 0.2, best_chunk_id="cA"), _Stub("b", 0.9, best_chunk_id="cB")])
    _, _, best = f.score_claim("x", CHUNKS)
    assert best == "cB"


def test_no_signals_returns_zero():
    f = FusionScorer([])
    fused, results, best = f.score_claim("x", CHUNKS)
    assert fused == 0.0 and results == {} and best is None
