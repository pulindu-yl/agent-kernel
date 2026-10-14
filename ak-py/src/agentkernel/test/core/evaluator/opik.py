import logging
import os

# Must be set before any metric is constructed: this stops the SDK from writing traces/spans to
# Opik Cloud (or a self-hosted instance) by default, which would otherwise require an OPIK_API_KEY /
# `opik configure` the AK test framework never asks users to set up. setdefault, not an unconditional
# write, so a user who explicitly wants tracking is respected. `track=False` is also passed to each
# metric below as a second, explicit guard.
os.environ.setdefault("OPIK_TRACK_DISABLE", "True")

from opik.evaluation.metrics import (
    AgentTaskCompletionJudge,
    AgentToolCorrectnessJudge,
    AnswerRelevance,
    GEval,
    Hallucination,
    LevenshteinRatio,
)
from opik.evaluation.metrics.base_metric import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult

from agentkernel.test.config import AKTestConfig

from .base import AKEvaluationCase, AKEvaluationError, AKEvaluationResult, AKEvaluator, AKMissingInput

_log = logging.getLogger("ak.test.opik")

_DEFAULT_TASK_INTRODUCTION = (
    "You are an expert evaluator checking whether an AI system's actual output correctly answers a "
    "user's question, judged against a known-correct expected output."
)

_DEFAULT_EVALUATION_CRITERIA = (
    "Identify the concrete facts the expected output asserts (entities, names, numbers, or other "
    "specific claims), read together with the user input.\n"
    "For each of those facts, check whether the actual output states or clearly implies it — matching "
    "in substance regardless of exact wording, phrasing, synonyms, order, or sentence structure, and "
    "regardless of whether the actual output is a full sentence or a short phrase/keyword.\n"
    "Extra correct, non-contradictory detail or specificity in the actual output beyond what the "
    "expected output states (e.g. naming a more specific type or subcategory of the same thing) must "
    "NOT be penalized — this is not a wording-similarity or length comparison.\n"
    "Score low only if the actual output omits one or more of the required facts, contradicts the "
    "expected output, or asserts something that conflicts with it."
)


class OpikAKEvaluator(AKEvaluator):
    """Scores with Opik: ``evaluate_by_score`` is Levenshtein ratio and ``evaluate_by_llm`` is G-Eval against an expected output.

    Setting ``AKEvaluationCase.metric`` to one of ``METRICS`` makes ``evaluate_by_llm`` run that reference-free
    judge instead, which needs no expected output: ``task_completion`` and ``tool_correctness`` read the case's
    ``transcript``, ``answer_relevance`` reads the input and the answer, and ``hallucination`` also reads the ``context``.
    """

    MAX_ATTEMPTS = 2
    # metric name -> (Opik judge class, constructor kwargs)
    _JUDGES: dict[str, tuple[type[BaseMetric], dict]] = {
        "task_completion": (AgentTaskCompletionJudge, {}),
        "tool_correctness": (AgentToolCorrectnessJudge, {}),
        "answer_relevance": (AnswerRelevance, {"require_context": False}),
        "hallucination": (Hallucination, {}),
    }
    METRICS = tuple(_JUDGES)
    _LOWER_IS_BETTER = {"hallucination"}  # 1 means hallucinated

    def __init__(self, config: AKTestConfig) -> None:
        super().__init__(config)
        self._model: str | None = None  # lazy: score mode never needs it
        self._judges: dict[str, BaseMetric] = {}

    def _llm_model(self) -> str:
        if self._model is None:
            llm = self._config.llm
            self._model = f"{llm.provider}/{llm.model}"
        return self._model

    def evaluate_by_score(self, case: AKEvaluationCase) -> AKEvaluationResult:
        if not case.expected:
            raise AKMissingInput("evaluate_by_score requires AKEvaluationCase.expected")
        metric = LevenshteinRatio(track=False)
        result = metric.score(output=case.actual, reference=case.expected)
        score = float(result.value)
        return AKEvaluationResult(
            metric="levenshtein_ratio",
            evaluator="opik",
            score=score,
            passed=score >= case.threshold,
        )

    @staticmethod
    def higher_is_better(metric: str) -> bool:
        """False for hallucination, where a high score is bad."""
        return metric not in OpikAKEvaluator._LOWER_IS_BETTER

    @staticmethod
    def applies_to(case: AKEvaluationCase) -> bool:
        """Whether the case's metric makes sense for it: tool correctness needs at least one tool call."""
        return case.metric != "tool_correctness" or case.tool_call_count != 0

    def evaluate_by_llm(self, case: AKEvaluationCase) -> AKEvaluationResult:
        if case.metric is not None:
            return self._evaluate_reference_free(case)
        if not case.expected:
            raise AKMissingInput("evaluate_by_llm requires AKEvaluationCase.expected")
        if not case.actual:
            # An empty reply can never satisfy a non-empty expected: report the scored failure so
            # Test.compare raises its normal AssertionError instead of paying for a judge call.
            return AKEvaluationResult(
                metric="g_eval",
                evaluator="opik",
                score=0.0,
                reason="actual output is empty; nothing to judge",
                passed=False,
            )
        metric = GEval(
            task_introduction=_DEFAULT_TASK_INTRODUCTION,
            evaluation_criteria=case.criteria if case.criteria else _DEFAULT_EVALUATION_CRITERIA,
            model=self._llm_model(),
            track=False,
        )
        # Opik's GEval.score() takes a single `output` string (unlike deepeval's LLMTestCase) — the
        # input/expected/actual all have to be packed into it for the judge to have anything to compare.
        formatted_output = f"User input: {case.user_input}\n\nExpected output: {case.expected}\n\nActual output: {case.actual}"
        try:
            result = metric.score(output=formatted_output)
        except Exception as exc:
            raise AKEvaluationError(f"opik g_eval llm-based evaluation failed: {exc}") from exc
        score = float(result.value) if result.value is not None else None
        return AKEvaluationResult(
            metric="g_eval",
            evaluator="opik",
            score=score,
            reason=result.reason,
            passed=score is not None and score >= case.threshold,
        )

    def _evaluate_reference_free(self, case: AKEvaluationCase) -> AKEvaluationResult:
        """Judges the case by ``case.metric``, trying up to MAX_ATTEMPTS times; a failure is an unscored result, never an exception."""
        metric = case.metric
        if metric not in self._JUDGES:
            raise AKMissingInput(f"unknown opik metric {metric!r}; expected one of {', '.join(self.METRICS)}")
        last_error = ""
        attempts = 0
        judged: ScoreResult | None = None
        while attempts < self.MAX_ATTEMPTS and judged is None:
            attempts += 1
            try:
                score_result = self._run_judge(metric, case)
            except Exception as exc:  # Opik and litellm raise many unrelated types; one bad metric never fails the run
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if score_result.scoring_failed or score_result.value is None:
                    last_error = score_result.reason or "scoring failed"
                else:
                    judged = score_result
            if judged is None:
                _log.warning("%s attempt %d failed: %s", metric, attempts, last_error)

        higher_is_better = self.higher_is_better(metric)
        metadata = {"higher_is_better": higher_is_better, "opik_metric": self._JUDGES[metric][0].__name__, "attempts": attempts}
        if judged is None:
            return AKEvaluationResult(
                metric=metric,
                evaluator="opik",
                threshold=case.threshold,
                mode="llm",
                reason=f"{metric} failed after {attempts} attempts: {last_error}",
                metadata=metadata,
            )
        score = float(judged.value)
        return AKEvaluationResult(
            metric=metric,
            evaluator="opik",
            score=score,
            threshold=case.threshold,
            passed=score >= case.threshold if higher_is_better else score < case.threshold,
            mode="llm",
            reason=judged.reason,
            metadata=metadata,
        )

    def _judge(self, metric: str) -> BaseMetric:
        """Builds the metric's Opik judge on first use and reuses it after."""
        if metric not in self._judges:
            judge_class, judge_kwargs = self._JUDGES[metric]
            self._judges[metric] = judge_class(model=self._llm_model(), track=False, **judge_kwargs)
        return self._judges[metric]

    def _run_judge(self, metric: str, case: AKEvaluationCase) -> ScoreResult:
        """Calls the metric's judge once, with the inputs it reads."""
        judge = self._judge(metric)
        if metric in ("task_completion", "tool_correctness"):
            # these judges take one output string, so the transcript carries the request, the tools and the answer
            return judge.score(output=case.transcript or f"User request:\n{case.user_input}\n\nFinal answer:\n{case.actual}")
        if metric == "answer_relevance":
            return judge.score(input=case.user_input, output=case.actual)
        return judge.score(input=case.user_input, output=case.actual, context=case.context)
