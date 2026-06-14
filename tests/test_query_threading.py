from hallushield.core.fusion import FusionScorer
from hallushield.core.types import Chunk, SignalResult
from hallushield.pipeline import validate
from hallushield.signals.logic_judge import LogicJudgeSignal


class _QueryCapture:
    name = "cap"

    def __init__(self):
        self.seen = []

    def score(self, claim, chunks, query=""):
        self.seen.append(query)
        return SignalResult("cap", 1.0)


def test_fusion_threads_query_to_signals():
    sig = _QueryCapture()
    FusionScorer([sig]).score_claim("a claim", [], query="my question")
    assert sig.seen == ["my question"]


def test_validate_threads_query_to_signals():
    sig = _QueryCapture()
    validate("One sentence here.", [Chunk("c1", "x")], "general",
             fusion=FusionScorer([sig]), query="Q?")
    assert "Q?" in sig.seen


def test_validate_defaults_query_to_empty():
    sig = _QueryCapture()
    validate("One sentence.", [Chunk("c1", "x")], "general", fusion=FusionScorer([sig]))
    assert sig.seen == [""]


def test_logic_judge_prompt_includes_query():
    seen = {}

    def judge(prompt):
        seen["prompt"] = prompt
        return "0.5"

    LogicJudgeSignal(judge=judge).score("claim text", [Chunk("c1", "ctx")], query="my question")
    assert "my question" in seen["prompt"]
