from hallushield.core.fusion import FusionScorer
from hallushield.core.types import Chunk
from hallushield.signals.lexical import LexicalOverlapSignal
from hallushield.signals.logic_judge import LogicJudgeSignal, _parse_score

CHUNKS = [Chunk("c1", "Metformin initial dose is 500mg twice daily.")]


def test_parse_score_variants():
    assert _parse_score("0.8") == 0.8
    assert _parse_score("The score is 0.95 overall") == 0.95
    assert _parse_score("1") == 1.0
    assert _parse_score("0") == 0.0
    assert _parse_score(".7") == 0.7


def test_parse_score_clamps_and_defaults():
    assert _parse_score("1.5") == 1.0          # clamped
    assert _parse_score("no number here") == 0.5  # uncertain default
    assert _parse_score("") == 0.5


def test_signal_uses_injected_judge():
    sig = LogicJudgeSignal(judge=lambda prompt: "0.9")
    r = sig.score("Metformin dose is 500mg", CHUNKS)
    assert r.score == 0.9
    assert r.name == "logic"
    assert r.best_chunk_id == "c1"


def test_judge_prompt_contains_claim_and_context():
    seen = {}

    def judge(prompt):
        seen["prompt"] = prompt
        return "0.5"

    LogicJudgeSignal(judge=judge).score("the dose question", CHUNKS)
    assert "the dose question" in seen["prompt"]
    assert "Metformin initial dose" in seen["prompt"]


def test_empty_claim_scores_zero():
    sig = LogicJudgeSignal(judge=lambda p: "1.0")
    assert sig.score("   ", CHUNKS).score == 0.0


def test_fusion_blends_lexical_and_logic():
    fusion = FusionScorer(
        [LexicalOverlapSignal(), LogicJudgeSignal(judge=lambda p: "1.0")],
        weights={"lexical": 1.0, "logic": 1.0},
    )
    score, results, best = fusion.score_claim("Metformin is begun at 500mg", CHUNKS)
    assert set(results) == {"lexical", "logic"}
    # lexical ~0.667 (paraphrase) blended with logic 1.0 -> between the two.
    assert 0.66 < score < 1.0
    assert best == "c1"
