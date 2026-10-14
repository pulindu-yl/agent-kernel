"""Dedicated behavior tests for the built-in OpikAKEvaluator adapter.

evaluate_by_score is exercised against the real (offline, no-network) `LevenshteinRatio` metric.
evaluate_by_llm mocks `GEval` in place (no real LLM/network call) since that's the one path that
would otherwise hit a live provider.
"""

from typing import ClassVar

import pytest
from opik.evaluation.metrics.score_result import ScoreResult

from agentkernel.test import (
    AKEvaluationCase,
    AKEvaluationError,
    AKMissingInput,
)
from agentkernel.test.config import AKTestConfig
from agentkernel.test.core.evaluator.opik import (
    _DEFAULT_EVALUATION_CRITERIA,
    OpikAKEvaluator,
)


@pytest.fixture
def evaluator():
    return OpikAKEvaluator(AKTestConfig())


# --- evaluate_by_score --------------------------------------------------------------------#


def test_evaluate_by_score_exact_match(evaluator):
    case = AKEvaluationCase(user_input="capital of France?", actual="Paris", expected="Paris")
    result = evaluator.evaluate_by_score(case)
    assert result.score == 1.0
    assert result.metric == "levenshtein_ratio"
    assert result.evaluator == "opik"
    assert result.passed is True


def test_evaluate_by_score_mismatch(evaluator):
    case = AKEvaluationCase(user_input="capital of France?", actual="London", expected="Paris")
    result = evaluator.evaluate_by_score(case)
    assert result.score == 0.0
    assert result.passed is False


def test_evaluate_by_score_passed_uses_case_threshold(evaluator):
    case = AKEvaluationCase(user_input="capital of France?", actual="London", expected="Paris", threshold=0.0)
    result = evaluator.evaluate_by_score(case)
    assert result.score == 0.0
    assert result.passed is True  # 0.0 >= the case's own threshold of 0.0


def test_evaluate_by_score_is_case_insensitive(evaluator):
    case = AKEvaluationCase(user_input="q", actual="PARIS", expected="paris")
    result = evaluator.evaluate_by_score(case)
    assert result.score == 1.0


def test_evaluate_by_score_missing_expected_raises(evaluator):
    case = AKEvaluationCase(user_input="q", actual="Paris", expected=None)
    with pytest.raises(AKMissingInput):
        evaluator.evaluate_by_score(case)


# --- evaluate_by_llm -----------------------------------------------------------------------#


class _FakeScoreResult:
    def __init__(self, value, reason):
        self.value = value
        self.reason = reason


class _FakeGEval:
    """Stand-in for opik.evaluation.metrics.GEval that never calls a real model."""

    instances: ClassVar[list["_FakeGEval"]] = []
    score_value = 1.0
    score_reason = "looks correct"
    score_error: Exception | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeGEval.instances.append(self)

    def score(self, output, **ignored_kwargs):
        self.output = output
        if _FakeGEval.score_error is not None:
            raise _FakeGEval.score_error
        return _FakeScoreResult(_FakeGEval.score_value, _FakeGEval.score_reason)


@pytest.fixture(autouse=True)
def _reset_fake_geval(monkeypatch):
    import agentkernel.test.core.evaluator.opik as ak_opik_module

    _FakeGEval.instances = []
    _FakeGEval.score_value = 1.0
    _FakeGEval.score_reason = "looks correct"
    _FakeGEval.score_error = None
    monkeypatch.setattr(ak_opik_module, "GEval", _FakeGEval)


def test_evaluate_by_llm_missing_expected_raises(evaluator):
    case = AKEvaluationCase(user_input="q", actual="Paris", expected=None)
    with pytest.raises(AKMissingInput):
        evaluator.evaluate_by_llm(case)
    assert _FakeGEval.instances == []  # never got as far as constructing GEval


def test_evaluate_by_llm_empty_actual_fails_without_judging(evaluator):
    case = AKEvaluationCase(user_input="q", actual="", expected="Paris")
    result = evaluator.evaluate_by_llm(case)
    assert result.passed is False
    assert result.score == 0.0
    assert result.metric == "g_eval"
    assert "empty" in result.reason
    assert _FakeGEval.instances == []  # no judge call was attempted


def test_evaluate_by_llm_success(evaluator):
    _FakeGEval.score_value = 0.9
    _FakeGEval.score_reason = "matches expected answer"
    case = AKEvaluationCase(user_input="capital of France?", actual="It's Paris.", expected="Paris")

    result = evaluator.evaluate_by_llm(case)

    assert result.metric == "g_eval"
    assert result.evaluator == "opik"
    assert result.score == 0.9
    assert result.reason == "matches expected answer"
    assert result.passed is True
    (fake,) = _FakeGEval.instances
    assert fake.kwargs["evaluation_criteria"] == _DEFAULT_EVALUATION_CRITERIA
    assert fake.kwargs["track"] is False
    assert fake.kwargs["model"] == "openai/gpt-4o-mini"
    # Regression: the judge must see the question and expected output packed into `output`, since
    # opik's GEval.score() takes only that one string.
    assert "capital of France?" in fake.output
    assert "Paris" in fake.output
    assert "It's Paris." in fake.output


def test_evaluate_by_llm_uses_case_criteria_override(evaluator):
    case = AKEvaluationCase(
        user_input="q",
        actual="a",
        expected="e",
        criteria="Score 1 only if it mentions cats.",
    )
    evaluator.evaluate_by_llm(case)
    (fake,) = _FakeGEval.instances
    assert fake.kwargs["evaluation_criteria"] == "Score 1 only if it mentions cats."


def test_evaluate_by_llm_passed_false_below_case_threshold(evaluator):
    _FakeGEval.score_value = 0.4
    case = AKEvaluationCase(user_input="q", actual="a", expected="e", threshold=0.5)
    result = evaluator.evaluate_by_llm(case)
    assert result.score == 0.4
    assert result.passed is False


def test_evaluate_by_llm_wraps_score_exception(evaluator):
    boom = RuntimeError("provider unreachable")
    _FakeGEval.score_error = boom
    case = AKEvaluationCase(user_input="q", actual="a", expected="e")

    with pytest.raises(AKEvaluationError) as exc_info:
        evaluator.evaluate_by_llm(case)
    assert exc_info.value.__cause__ is boom


def test_llm_model_is_lazily_created_and_cached(evaluator):
    assert evaluator._model is None  # never built by __init__

    case = AKEvaluationCase(user_input="q", actual="a", expected="e")
    evaluator.evaluate_by_llm(case)
    evaluator.evaluate_by_llm(case)

    assert evaluator._model == "openai/gpt-4o-mini"
    assert [f.kwargs["model"] for f in _FakeGEval.instances] == ["openai/gpt-4o-mini", "openai/gpt-4o-mini"]


# --- reference-free metrics -----------------------------------------------------------------#


class _FakeJudge:
    """Stand-in for an Opik judge: replays scripted outcomes, repeating the last one."""

    def __init__(self, model=None, track=True, **kwargs):
        self.model, self.track, self.kwargs = model, track, kwargs
        self.calls: list[dict] = []
        self.outcomes: list = [ScoreResult(name="fake", value=0.8, reason="looks fine")]

    def score(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fake_judges(monkeypatch):
    for metric, (_, kwargs) in list(OpikAKEvaluator._JUDGES.items()):
        monkeypatch.setitem(OpikAKEvaluator._JUDGES, metric, (_FakeJudge, kwargs))


def _run_case(metric, **fields):
    return AKEvaluationCase(
        user_input="Where is my order?",
        actual="It ships Friday.",
        metric=metric,
        transcript="User request:\nWhere is my order?\n\nTool calls:\n1. get_order()\n\nFinal answer:\nIt ships Friday.",
        context=["Shipped, arriving Friday"],
        tool_call_count=1,
        **fields,
    )


def test_metrics_are_selectable_by_name():
    assert OpikAKEvaluator.METRICS == ("task_completion", "tool_correctness", "answer_relevance", "hallucination")


def test_reference_free_metric_needs_no_expected_and_reads_its_inputs(evaluator, fake_judges):
    case = _run_case("task_completion")
    expected = {
        "task_completion": {"output": case.transcript},
        "tool_correctness": {"output": case.transcript},
        "answer_relevance": {"input": case.user_input, "output": case.actual},
        "hallucination": {"input": case.user_input, "output": case.actual, "context": case.context},
    }
    for metric in OpikAKEvaluator.METRICS:
        result = evaluator.evaluate_by_llm(case.model_copy(update={"metric": metric}))
        judge = evaluator._judges[metric]
        assert judge.calls == [expected[metric]]
        assert judge.model == "openai/gpt-4o-mini" and judge.track is False
        assert result.metric == metric and result.evaluator == "opik" and result.mode == "llm" and result.score == 0.8
    assert evaluator._judges["answer_relevance"].kwargs == {"require_context": False}


def test_agent_judges_fall_back_to_input_and_answer_without_a_transcript(evaluator, fake_judges):
    case = _run_case("task_completion").model_copy(update={"transcript": None})
    evaluator.evaluate_by_llm(case)
    assert evaluator._judges["task_completion"].calls == [{"output": "User request:\nWhere is my order?\n\nFinal answer:\nIt ships Friday."}]


def test_pass_rules_flip_for_hallucination(evaluator, fake_judges):
    def judge_with(metric, value):
        evaluator._judges.pop(metric, None)
        evaluator._judge(metric).outcomes = [ScoreResult(name="x", value=value)]
        return evaluator.evaluate_by_llm(_run_case(metric))

    assert judge_with("task_completion", 0.5).passed is True
    assert judge_with("task_completion", 0.4).passed is False
    assert judge_with("hallucination", 0.2).passed is True
    hallucinated = judge_with("hallucination", 0.5)
    assert hallucinated.passed is False and hallucinated.metadata["higher_is_better"] is False


def test_applies_to_skips_tool_correctness_without_tool_calls():
    assert OpikAKEvaluator.applies_to(_run_case("tool_correctness"))
    assert not OpikAKEvaluator.applies_to(_run_case("tool_correctness").model_copy(update={"tool_call_count": 0}))
    assert OpikAKEvaluator.applies_to(_run_case("task_completion").model_copy(update={"tool_call_count": 0}))
    assert OpikAKEvaluator.applies_to(_run_case("tool_correctness").model_copy(update={"tool_call_count": None}))


def test_reference_free_retries_once_then_succeeds(evaluator, fake_judges):
    evaluator._judge("task_completion").outcomes = [RuntimeError("rate limited"), ScoreResult(name="x", value=0.7)]
    result = evaluator.evaluate_by_llm(_run_case("task_completion"))
    assert result.score == 0.7 and result.metadata["attempts"] == 2


def test_reference_free_failure_is_an_unscored_result(evaluator, fake_judges):
    evaluator._judge("answer_relevance").outcomes = [RuntimeError("boom"), ScoreResult(name="x", value=0.0, reason="bad json", scoring_failed=True)]
    result = evaluator.evaluate_by_llm(_run_case("answer_relevance"))
    assert result.score is None and result.passed is None
    assert result.reason == "answer_relevance failed after 2 attempts: bad json"
    assert result.metadata["attempts"] == 2


def test_unknown_metric_raises(evaluator):
    with pytest.raises(AKMissingInput, match="faithfulness"):
        evaluator.evaluate_by_llm(_run_case("faithfulness"))
