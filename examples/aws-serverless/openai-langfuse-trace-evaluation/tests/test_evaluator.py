import gzip
import json
import secrets
import subprocess
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import boto3
import httpx
import pytest
from agentkernel.test import AKTestConfig
from agentkernel.test.core.evaluator import AKEvaluationResult, AKMetricNotSupported
from agentkernel.trace import FetchedSpan, SpanKind, SpanUsage
from botocore.exceptions import ClientError
from moto import mock_aws
from opik.evaluation.metrics.score_result import ScoreResult

from evaluator.api import EvaluationAPI
from evaluator.case import ToolCall, TraceCase, TraceEvaluationCase
from evaluator.job import TraceEvaluationJob
from evaluator.opik_evaluator import OpikTraceEvaluator
from evaluator.store import EvaluationStore, Progress
from simulate_traffic import TrafficSimulator

T0 = datetime(2026, 10, 13, 10, 0, tzinfo=timezone.utc)
SESSION = "session-1"
TABLE = "trace-evaluations"


class Spans:
    """FetchedSpans shaped like the Langfuse OpenAI Agents tree (research/span-formats.md §3)."""

    @staticmethod
    def span(
        trace_id: str,
        span_id: str,
        parent: str | None,
        name: str,
        kind: SpanKind = SpanKind.SPAN,
        start: float = 0,
        end: float | None = 1,
        **fields,
    ) -> FetchedSpan:
        return FetchedSpan(
            provider="langfuse",
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent,
            name=name,
            kind=kind,
            start_time=T0 + timedelta(seconds=start),
            end_time=None if end is None else T0 + timedelta(seconds=end),
            session_id=fields.pop("session_id", SESSION),
            **fields,
        )

    @classmethod
    def request(
        cls,
        trace_id: str = "trace-1",
        start: float = 0,
        output: str | None = '"Your order A-1001 has shipped and arrives Friday."',
        tool: bool = True,
        tool_level: str = "DEFAULT",
        root_end: float | None = 3,
    ) -> list[FetchedSpan]:
        s = start
        spans = [
            cls.span(
                trace_id,
                f"{trace_id}-root",
                None,
                "Agent Kernel OpenAI",
                start=s,
                end=None if root_end is None else s + root_end,
                input='"Where is my order A-1001?"',
                output=output,
            ),
            cls.span(trace_id, f"{trace_id}-wf", f"{trace_id}-root", "Agent workflow", start=s + 0.1, end=s + 2.9),
            cls.span(
                trace_id, f"{trace_id}-triage", f"{trace_id}-wf", "triage", SpanKind.AGENT, start=s + 0.2, end=s + 1
            ),
            cls.span(
                trace_id,
                f"{trace_id}-gen1",
                f"{trace_id}-triage",
                "generation",
                SpanKind.LLM,
                start=s + 0.3,
                end=s + 0.9,
                usage=SpanUsage(input_tokens=100, output_tokens=20, total_tokens=120, cost=0.001),
            ),
            cls.span(
                trace_id, f"{trace_id}-support", f"{trace_id}-wf", "support", SpanKind.AGENT, start=s + 1.1, end=s + 2.8
            ),
            cls.span(
                trace_id,
                f"{trace_id}-gen2",
                f"{trace_id}-support",
                "generation",
                SpanKind.LLM,
                start=s + 1.2,
                end=s + 1.5,
                usage=SpanUsage(input_tokens=200, output_tokens=30, total_tokens=230),
            ),
        ]
        if tool:
            spans.append(
                cls.span(
                    trace_id,
                    f"{trace_id}-tool",
                    f"{trace_id}-support",
                    "get_order_status",
                    SpanKind.TOOL,
                    start=s + 1.6,
                    end=s + 1.7,
                    input='{"order_id": "A-1001"}',
                    output="Shipped, arriving Friday",
                    raw={"level": tool_level},
                )
            )
        return spans


def make_case(trace_id: str = "trace-1", **kwargs) -> TraceCase:
    return TraceCase.from_spans(trace_id, Spans.request(trace_id, **kwargs))


def result(metric: str, score: float | None = 0.9, passed: bool | None = True) -> AKEvaluationResult:
    return AKEvaluationResult(
        metric=metric,
        evaluator="opik",
        score=score,
        threshold=0.5,
        passed=passed,
        mode="llm",
        reason=f"{metric} reason",
        metadata={"higher_is_better": metric != "hallucination", "attempts": 1},
    )


@pytest.fixture
def aws(monkeypatch):
    for name, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-2",
    }.items():
        monkeypatch.setenv(name, value)
    with mock_aws():
        yield


@pytest.fixture
def table(aws):
    # the main.tf schema
    return boto3.resource("dynamodb").create_table(
        TableName=TABLE,
        BillingMode="PAY_PER_REQUEST",
        KeySchema=[{"AttributeName": "trace_id", "KeyType": "HASH"}],
        AttributeDefinitions=[
            {"AttributeName": "trace_id", "AttributeType": "S"},
            {"AttributeName": "session_id", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "session_id-index",
                "KeySchema": [{"AttributeName": "session_id", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "INCLUDE", "NonKeyAttributes": ["progress", "error", "updated_at"]},
            }
        ],
    )


@pytest.fixture
def store(table):
    return EvaluationStore(table)


class TestTraceCase:
    def test_root_input_output_tools_agents_usage_latency(self):
        case = make_case()
        assert case.user_input == "Where is my order A-1001?"  # JSON string unwrapped
        assert case.output == "Your order A-1001 has shipped and arrives Friday."
        assert case.root_name == "Agent Kernel OpenAI"
        assert case.session_id == SESSION
        assert case.tool_calls == [
            ToolCall(name="get_order_status", input='{"order_id": "A-1001"}', output="Shipped, arriving Friday")
        ]
        assert case.agents == ["triage", "support"]
        assert case.usage == SpanUsage(input_tokens=300, output_tokens=50, total_tokens=350, cost=0.001)
        assert case.latency_ms == 3000
        assert case.span_count == 7 and case.spans_missing_parent == 0
        assert case.is_finished

    def test_failed_tool_from_raw_level(self):
        assert make_case(tool_level="ERROR").tool_calls[0].failed

    def test_usage_none_when_no_span_has_usage(self):
        spans = [span.model_copy(update={"usage": None}) for span in Spans.request()]
        assert TraceCase.from_spans("trace-1", spans).usage == SpanUsage()

    def test_non_string_values_are_json_dumped(self):
        spans = Spans.request()
        spans[0] = spans[0].model_copy(update={"input": {"prompt": "hi"}})
        assert TraceCase.from_spans("trace-1", spans).user_input == '{"prompt": "hi"}'

    def test_tool_fallback_through_classifier(self):
        spans = Spans.request(tool=False)
        spans.append(
            Spans.span(
                "trace-1",
                "tool",
                "trace-1-support",
                "get_order_status",
                attributes={"openinference.span.kind": "TOOL"},
                output="x",
            )
        )
        assert [call.name for call in TraceCase.from_spans("trace-1", spans).tool_calls] == ["get_order_status"]

    def test_no_root_or_two_roots_is_none(self):
        spans = Spans.request()
        assert TraceCase.from_spans("trace-1", [s for s in spans if s.parent_span_id is not None]) is None
        assert TraceCase.from_spans("trace-1", spans + [Spans.span("trace-1", "root-2", None, "other")]) is None

    def test_root_without_end_time_is_not_finished(self):
        case = make_case(root_end=None)
        assert not case.is_finished and case.latency_ms is None
        assert not case.is_ready_to_grade(T0 + timedelta(hours=1), timedelta(minutes=2))

    def test_root_without_output_is_not_finished(self):
        assert not make_case(output=None).is_finished

    def test_missing_parents_and_ready(self):
        settle = timedelta(minutes=2)
        case = make_case()
        ended = T0 + timedelta(seconds=3)
        assert not case.is_ready_to_grade(ended + settle - timedelta(seconds=1), settle)
        assert case.is_ready_to_grade(ended + settle + timedelta(seconds=1), settle)

        partial = TraceCase.from_spans("trace-1", [s for s in Spans.request() if s.span_id != "trace-1-support"])
        assert partial.spans_missing_parent == 2  # gen2 and the tool lost their parent
        assert not partial.is_ready_to_grade(ended + timedelta(hours=1), settle)

    def test_transcript_with_and_without_tool_calls(self):
        assert make_case().build_transcript() == (
            "User request:\nWhere is my order A-1001?\n\nTool calls:\n"
            '1. get_order_status({"order_id": "A-1001"}) -> "Shipped, arriving Friday"\n\n'
            "Final answer:\nYour order A-1001 has shipped and arrives Friday."
        )
        assert make_case(tool_level="ERROR").build_transcript().split("\n")[4].endswith(" [failed]")
        assert "Tool calls:\n(none)\n" in make_case(tool=False).build_transcript()

    def test_to_evaluation_case(self):
        evaluation_case = make_case().to_evaluation_case(0.5)
        assert evaluation_case.actual == "Your order A-1001 has shipped and arrives Friday."
        assert evaluation_case.context == ["Shipped, arriving Friday"]
        assert evaluation_case.tool_call_count == 1 and evaluation_case.expected is None
        assert make_case(tool=False).to_evaluation_case(0.5).context is None


class FakeJudge:
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


class TestOpikTraceEvaluator:
    @pytest.fixture
    def config(self):
        return AKTestConfig(llm={"provider": "openai", "model": "gpt-4.1-mini"})

    @pytest.fixture(autouse=True)
    def fake_judges(self, monkeypatch):
        for metric, (_, kwargs) in list(OpikTraceEvaluator._JUDGE_CLASSES.items()):
            monkeypatch.setitem(OpikTraceEvaluator._JUDGE_CLASSES, metric, (FakeJudge, kwargs))

    @staticmethod
    def evaluator(config, metric: str, *outcomes) -> tuple[OpikTraceEvaluator, FakeJudge]:
        evaluator = OpikTraceEvaluator(config, metric)
        judge = evaluator._build_judge()
        if outcomes:
            judge.outcomes = list(outcomes)
        return evaluator, judge

    def test_call_arguments_and_model(self, config):
        case = make_case().to_evaluation_case(0.5)
        expected = {
            "task_completion": {"output": case.transcript},
            "tool_correctness": {"output": case.transcript},
            "answer_relevance": {"input": case.user_input, "output": case.actual},
            "hallucination": {"input": case.user_input, "output": case.actual, "context": case.context},
        }
        for evaluator in OpikTraceEvaluator.for_all_metrics(config):
            judge = evaluator._build_judge()
            evaluator.evaluate_by_llm(case)
            assert judge.calls == [expected[evaluator.metric]]
            assert judge.model == "openai/gpt-4.1-mini" and judge.track is False
        assert OpikTraceEvaluator(config, "answer_relevance")._build_judge().kwargs == {"require_context": False}

    def test_pass_rules(self, config):
        case = make_case().to_evaluation_case(0.5)
        evaluator, _ = self.evaluator(config, "task_completion", ScoreResult(name="x", value=0.5))
        scored = evaluator.evaluate_by_llm(case)
        assert scored.passed and scored.score == 0.5 and scored.metadata["higher_is_better"]
        assert scored.evaluator == "opik" and scored.mode == "llm" and scored.threshold == 0.5

        evaluator, _ = self.evaluator(config, "hallucination", ScoreResult(name="x", value=0.2))
        assert evaluator.evaluate_by_llm(case).passed
        evaluator, _ = self.evaluator(config, "hallucination", ScoreResult(name="x", value=0.5))
        hallucinated = evaluator.evaluate_by_llm(case)
        assert not hallucinated.passed and hallucinated.metadata["higher_is_better"] is False

    def test_applies_to(self, config):
        evaluator = OpikTraceEvaluator(config, "tool_correctness")
        assert evaluator.applies_to(make_case().to_evaluation_case(0.5))
        assert not evaluator.applies_to(make_case(tool=False).to_evaluation_case(0.5))
        assert OpikTraceEvaluator(config, "task_completion").applies_to(make_case(tool=False).to_evaluation_case(0.5))

    def test_retry_then_success(self, config):
        evaluator, _ = self.evaluator(
            config, "task_completion", RuntimeError("rate limited"), ScoreResult(name="x", value=0.7)
        )
        scored = evaluator.evaluate_by_llm(make_case().to_evaluation_case(0.5))
        assert scored.score == 0.7 and scored.metadata["attempts"] == 2

    def test_both_attempts_fail(self, config):
        evaluator, _ = self.evaluator(
            config,
            "answer_relevance",
            RuntimeError("boom"),
            ScoreResult(name="x", value=0.0, reason="bad json", scoring_failed=True),
        )
        failed = evaluator.evaluate_by_llm(make_case().to_evaluation_case(0.5))
        assert failed.score is None and failed.passed is None
        assert failed.reason == "answer_relevance failed after 2 attempts: bad json"
        assert failed.metadata["attempts"] == 2 and failed.metadata["higher_is_better"] is True

        evaluator, _ = self.evaluator(config, "answer_relevance", RuntimeError("boom"))
        assert "RuntimeError: boom" in evaluator.evaluate_by_llm(make_case().to_evaluation_case(0.5)).reason

    def test_score_mode_and_unknown_metric(self, config):
        with pytest.raises(AKMetricNotSupported):
            OpikTraceEvaluator(config, "task_completion").evaluate_by_score(make_case().to_evaluation_case(0.5))
        with pytest.raises(ValueError):
            OpikTraceEvaluator(config, "faithfulness")


class TestEvaluationStore:
    RESULTS = [result("task_completion"), result("answer_relevance"), result("hallucination", score=0.1)]

    def test_save_writes_one_graded_row(self, store, table):
        case = make_case()
        store.save_results(case, self.RESULTS + [result("tool_correctness", score=None, passed=None)], T0)
        item = table.get_item(Key={"trace_id": "trace-1"})["Item"]
        assert item["progress"] == "graded" and item["session_id"] == SESSION and item["provider"] == "langfuse"
        assert item["updated_at"] == item["evaluated_at"] == T0.isoformat()
        assert item["expires_at"] == int((T0 + timedelta(days=90)).timestamp())
        assert (
            item["latency_ms"] == Decimal(3000)
            and item["cost"] == Decimal("0.001")
            and item["total_tokens"] == Decimal(350)
        )
        assert item["tool_calls"] == [
            {
                "name": "get_order_status",
                "input": '{"order_id": "A-1001"}',
                "output": "Shipped, arriving Friday",
                "failed": False,
            }
        ]
        assert (
            item["tool_call_count"] == 1 and item["tool_error_count"] == 0 and item["agents"] == ["triage", "support"]
        )
        assert item["passed"] is True and item["metric_errors"] == 1
        assert item["task_completion"] == {
            "evaluator": "opik",
            "score": Decimal("0.9"),
            "threshold": Decimal("0.5"),
            "passed": True,
            "reason": "task_completion reason",
            "higher_is_better": True,
            "attempts": 1,
        }
        assert "score" not in item["tool_correctness"] and "passed" not in item["tool_correctness"]  # None omitted
        spans = json.loads(gzip.decompress(item["spans_gz"].value))
        assert [span["span_id"] for span in spans] == [span.span_id for span in case.spans]

    def test_no_tool_correctness_column_without_tool_calls(self, store, table):
        store.save_results(make_case(tool=False), self.RESULTS, T0)
        assert "tool_correctness" not in table.get_item(Key={"trace_id": "trace-1"})["Item"]

    def test_failed_metric_sets_passed_false(self, store, table):
        store.save_results(
            make_case(), [result("task_completion", score=0.1, passed=False), result("answer_relevance")], T0
        )
        assert table.get_item(Key={"trace_id": "trace-1"})["Item"]["passed"] is False

    def test_write_progress_writes_summary_and_never_downgrades_graded(self, store, table):
        case = make_case()
        store.write_progress(case, Progress.PENDING, T0)
        item = table.get_item(Key={"trace_id": "trace-1"})["Item"]
        assert (
            item["progress"] == "pending"
            and "task_completion" not in item
            and "spans_gz" not in item
            and "passed" not in item
        )

        store.write_progress(case, Progress.GRADING, T0 + timedelta(minutes=1))
        assert table.get_item(Key={"trace_id": "trace-1"})["Item"]["progress"] == "grading"

        store.save_results(case, self.RESULTS, T0 + timedelta(minutes=2))
        store.write_progress(case, Progress.GRADING, T0 + timedelta(minutes=3))
        assert table.get_item(Key={"trace_id": "trace-1"})["Item"]["progress"] == "graded"

    def test_save_error_leaves_row_grading(self, store, table, monkeypatch):
        case = make_case()
        store.write_progress(case, Progress.GRADING, T0)

        def boom(**kwargs):
            raise ClientError(
                {"Error": {"Code": "ProvisionedThroughputExceededException", "Message": "slow down"}}, "PutItem"
            )

        monkeypatch.setattr(table, "put_item", boom)
        with pytest.raises(ClientError):
            store.save_results(case, self.RESULTS, T0)
        monkeypatch.undo()
        assert table.get_item(Key={"trace_id": "trace-1"})["Item"]["progress"] == "grading"

    def test_truncate(self):
        assert EvaluationStore._truncate("x" * 4096) == "x" * 4096
        assert EvaluationStore._truncate("x" * 5000) == "x" * 4096 + "…[truncated]"
        assert EvaluationStore._truncate(None) is None

    def test_spans_truncated_when_over_budget(self, store, table):
        spans = Spans.request()
        spans[1] = spans[1].model_copy(update={"raw": {"blob": secrets.token_hex(250_000)}})  # ~250 KB gzipped
        spans[2] = spans[2].model_copy(update={"raw": {"blob": secrets.token_hex(250_000)}})
        store.save_results(TraceCase.from_spans("trace-1", spans), self.RESULTS, T0)
        item = table.get_item(Key={"trace_id": "trace-1"})["Item"]
        assert item["spans_truncated"] is True and "spans_gz" not in item

    def test_size_budget_with_many_multibyte_tool_calls(self, store, table):
        spans = Spans.request(tool=False)
        for i in range(100):
            spans.append(
                Spans.span(
                    "trace-1",
                    f"tool-{i}",
                    "trace-1-support",
                    "get_order_status",
                    SpanKind.TOOL,
                    start=2 + i / 1000,
                    input="😀" * 4096,
                    output="😀" * 4096,
                )
            )
        case = TraceCase.from_spans("trace-1", spans)
        store.save_results(case, self.RESULTS, T0)  # moto enforces the 400 KB item limit
        item = table.get_item(Key={"trace_id": "trace-1"})["Item"]
        assert item["tool_calls_truncated"] is True and item["tool_call_count"] == 100
        assert len(item["tool_calls"]) < EvaluationStore.MAX_STORED_TOOL_CALLS
        assert item["progress"] == "graded" and "task_completion" in item

    def test_item_bytes(self):
        item = {"id": "é", "n": Decimal(1), "b": b"abc", "ok": True, "l": ["ab"], "m": {"k": "v"}}
        # names 1-2 bytes; "é" is 2 bytes; numbers 21; list/map 3 + 1 per element
        assert EvaluationStore._item_bytes(item) == (2 + 2) + (1 + 21) + (1 + 3) + (2 + 1) + (1 + 3 + 1 + 2) + (
            1 + 3 + 1 + 1 + 1
        )

    def test_progress_across_chunks(self, store):
        cases = [make_case(f"trace-{i}") for i in range(130)]
        for case in cases:
            store.write_progress(case, Progress.PENDING, T0)
        progress = store.read_progress([case.trace_id for case in cases] + ["unknown"])
        assert len(progress) == 130 and "unknown" not in progress
        assert progress["trace-129"].progress == Progress.PENDING and progress["trace-129"].updated_at == T0

    def test_session_counts(self, store):
        for trace_id, progress in [
            ("a", Progress.GRADING),
            ("b", Progress.GRADING),
            ("c", Progress.PENDING),
            ("d", Progress.UNFINISHED),
        ]:
            store.write_progress(make_case(trace_id), progress, T0)
        store.mark_failed(["b"], "RuntimeError: boom", T0)
        other = make_case("e").model_copy(update={"session_id": "other"})
        store.write_progress(other, Progress.GRADING, T0)

        session = store.count_session_progress(SESSION)
        assert session.counts == {Progress.GRADING: 1, Progress.FAILED: 1, Progress.PENDING: 1, Progress.UNFINISHED: 1}
        assert session.errors == [{"trace_id": "b", "error": "RuntimeError: boom"}]
        assert store.count_session_progress("nobody").counts == {}

    def test_fail_only_touches_grading_rows(self, store, table):
        store.write_progress(make_case("a"), Progress.GRADING, T0)
        store.save_results(make_case("b"), self.RESULTS, T0)
        store.mark_failed(["a", "b", "missing"], "x" * 5000, T0 + timedelta(minutes=1))
        a = table.get_item(Key={"trace_id": "a"})["Item"]
        assert a["progress"] == "failed" and a["error"].endswith("…[truncated]")
        assert a["updated_at"] == (T0 + timedelta(minutes=1)).isoformat()
        assert table.get_item(Key={"trace_id": "b"})["Item"]["progress"] == "graded"
        assert "Item" not in table.get_item(Key={"trace_id": "missing"})


class FakeTrace:
    def __init__(self, spans: list[FetchedSpan] | None = None, error: Exception | None = None):
        self.spans = spans or []
        self.error = error
        self.calls: list[dict] = []

    def fetch(self, **filters):
        self.calls.append(filters)
        if self.error:
            raise self.error
        return self.spans


class FakeEvaluator:
    def __init__(self, metric: str, score: float | None = 0.9, error_for: str | None = None):
        self.metric, self.score, self.error_for = metric, score, error_for

    def applies_to(self, case: TraceEvaluationCase) -> bool:
        return True

    def evaluate_by_llm(self, case: TraceEvaluationCase) -> AKEvaluationResult:
        if self.error_for and self.error_for in case.user_input:
            raise RuntimeError("judge exploded")
        return result(self.metric, score=self.score, passed=None if self.score is None else True)


class TestTraceEvaluationJob:
    NOW = T0 + timedelta(minutes=10)

    @pytest.fixture
    def factory_calls(self):
        return []

    def job(self, store, trace, factory_calls, evaluators=None) -> TraceEvaluationJob:
        def factory():
            factory_calls.append(1)
            return evaluators or [FakeEvaluator("task_completion"), FakeEvaluator("answer_relevance")]

        return TraceEvaluationJob(trace, store, factory, clock=lambda: self.NOW)

    def test_fetch_call(self, store, factory_calls):
        trace = FakeTrace(Spans.request())
        self.job(store, trace, factory_calls).prepare_traces(SESSION)
        assert trace.calls == [{"session_id": SESSION, "last": timedelta(hours=24), "limit": 5000}]

    def test_prepare_table(self, store, table, factory_calls):
        spans = (
            Spans.request("new-ready")
            + Spans.request("new-settling", start=500)  # ends 1.5 minutes before NOW
            + Spans.request("new-no-output", output=None)
            + Spans.request("was-pending")
            + Spans.request("was-grading")
            + Spans.request("was-failed")
            + Spans.request("was-graded")
            + Spans.request("was-unfinished")
        )
        earlier = T0 + timedelta(minutes=1)
        store.write_progress(make_case("was-pending"), Progress.PENDING, earlier)
        store.write_progress(make_case("was-grading"), Progress.GRADING, earlier)
        store.write_progress(make_case("was-failed"), Progress.GRADING, earlier)
        store.mark_failed(["was-failed"], "boom", earlier)
        store.save_results(make_case("was-graded"), [result("task_completion")], earlier)
        store.write_progress(make_case("was-unfinished"), Progress.UNFINISHED, earlier)

        summary = self.job(store, FakeTrace(spans), factory_calls).prepare_traces(SESSION)
        progress = store.read_progress(
            [
                f"{name}"
                for name in (
                    "new-ready",
                    "new-settling",
                    "new-no-output",
                    "was-pending",
                    "was-grading",
                    "was-failed",
                    "was-graded",
                    "was-unfinished",
                )
            ]
        )
        assert {trace_id: p.progress for trace_id, p in progress.items()} == {
            "new-ready": Progress.GRADING,
            "new-settling": Progress.PENDING,
            "new-no-output": Progress.UNFINISHED,
            "was-pending": Progress.GRADING,
            "was-grading": Progress.GRADING,
            "was-failed": Progress.GRADING,
            "was-graded": Progress.GRADED,
            "was-unfinished": Progress.UNFINISHED,
        }
        assert progress["was-grading"].updated_at == self.NOW  # queued again
        assert progress["was-graded"].updated_at == earlier and progress["was-unfinished"].updated_at == earlier
        assert (summary.found, summary.grading, summary.pending, summary.unfinished, summary.already_graded) == (
            8,
            4,
            1,
            2,
            1,
        )
        assert sorted(summary.grading_ids) == ["new-ready", "was-failed", "was-grading", "was-pending"]
        assert summary.spans_fetched == len(spans)
        assert factory_calls == []  # prepare never builds the judges

    def test_no_root_gets_no_row(self, store, table, factory_calls):
        spans = [s for s in Spans.request("rootless") if s.parent_span_id is not None]
        summary = self.job(store, FakeTrace(spans), factory_calls).prepare_traces(SESSION)
        assert summary.no_root == 1 and summary.found == 0
        assert "Item" not in table.get_item(Key={"trace_id": "rootless"})

    def test_grade_only_grading_payload_traces(self, store, table, factory_calls):
        trace = FakeTrace(Spans.request("a") + Spans.request("b", start=1) + Spans.request("c", start=2))
        job = self.job(
            store, trace, factory_calls, [FakeEvaluator("task_completion"), FakeEvaluator("hallucination", score=None)]
        )
        job.prepare_traces(SESSION)
        store.write_progress(make_case("c"), Progress.PENDING, self.NOW)  # moved on meanwhile

        summary = job.grade_traces(SESSION, ["a", "c"])
        assert (summary.graded, summary.failed, summary.metric_errors) == (1, 0, 1)
        assert table.get_item(Key={"trace_id": "a"})["Item"]["progress"] == "graded"
        assert table.get_item(Key={"trace_id": "b"})["Item"]["progress"] == "grading"
        assert table.get_item(Key={"trace_id": "c"})["Item"]["progress"] == "pending"

        job.grade_traces(SESSION, ["b"])
        assert factory_calls == [1]  # built once across two grades

    def test_prepare_raises_on_fetch_error_and_full_result(self, store, factory_calls, monkeypatch):
        with pytest.raises(ConnectionError):
            self.job(store, FakeTrace(error=ConnectionError("down")), factory_calls).prepare_traces(SESSION)
        monkeypatch.setattr(TraceEvaluationJob, "FETCH_LIMIT", 7)
        with pytest.raises(RuntimeError, match="at least 7 spans"):
            self.job(store, FakeTrace(Spans.request()), factory_calls).prepare_traces(SESSION)

    def test_grade_fetch_error_fails_every_grading_row(self, store, table, factory_calls):
        trace = FakeTrace(Spans.request("a") + Spans.request("b", start=1))
        job = self.job(store, trace, factory_calls)
        job.prepare_traces(SESSION)
        trace.error = ConnectionError("down")
        summary = job.grade_traces(SESSION, ["a", "b"])
        assert summary.failed == 2 and summary.graded == 0
        for trace_id in ("a", "b"):
            item = table.get_item(Key={"trace_id": trace_id})["Item"]
            assert item["progress"] == "failed" and item["error"] == "ConnectionError: down"

    def test_grade_error_on_one_trace_continues(self, store, table, factory_calls):
        spans = Spans.request("a") + Spans.request("b", start=1)
        spans[0] = spans[0].model_copy(update={"input": '"explode please"'})
        trace = FakeTrace(spans)
        job = self.job(store, trace, factory_calls, [FakeEvaluator("task_completion", error_for="explode")])
        job.prepare_traces(SESSION)
        summary = job.grade_traces(SESSION, ["a", "b"])
        assert (summary.graded, summary.failed) == (1, 1)
        assert table.get_item(Key={"trace_id": "a"})["Item"]["error"] == "RuntimeError: judge exploded"
        assert table.get_item(Key={"trace_id": "b"})["Item"]["progress"] == "graded"

    def test_grade_missing_trace_fails(self, store, table, factory_calls):
        trace = FakeTrace(Spans.request("a"))
        job = self.job(store, trace, factory_calls)
        job.prepare_traces(SESSION)
        trace.spans = []
        assert job.grade_traces(SESSION, ["a"]).failed == 1
        assert table.get_item(Key={"trace_id": "a"})["Item"]["error"] == "trace not found in the fetch"

    def test_grade_reraises_when_mark_failed_itself_fails(self, store, factory_calls, monkeypatch):
        trace = FakeTrace(Spans.request("a"))
        job = self.job(store, trace, factory_calls)
        job.prepare_traces(SESSION)
        trace.error = ConnectionError("down")

        def broken_mark_failed(*args):
            raise RuntimeError("dynamodb down")

        monkeypatch.setattr(store, "mark_failed", broken_mark_failed)
        with pytest.raises(RuntimeError, match="dynamodb down"):
            job.grade_traces(SESSION, ["a"])


class FakeJob:
    def __init__(self, summary=None, error: Exception | None = None):
        from evaluator.job import GradeSummary, PrepareSummary

        self.summary = summary or PrepareSummary(session_id=SESSION)
        self.error = error
        self.prepared: list[str] = []
        self.graded: list[tuple] = []
        self._grade_summary = GradeSummary(session_id=SESSION, graded=2)

    def prepare_traces(self, session_id):
        self.prepared.append(session_id)
        if self.error:
            raise self.error
        return self.summary

    def grade_traces(self, session_id, trace_ids):
        self.graded.append((session_id, trace_ids))
        return self._grade_summary


class FakeLambdaClient:
    def __init__(self):
        self.invocations: list[dict] = []

    def invoke(self, **kwargs):
        self.invocations.append(kwargs)


class TestEvaluationAPI:
    CONTEXT = SimpleNamespace(function_name="ak-trace-eval")

    @pytest.fixture
    def parts(self, store):
        from evaluator.job import PrepareSummary

        job = FakeJob(PrepareSummary(session_id=SESSION, found=2, grading=1, pending=1, grading_ids=["a"]))
        built = []
        lambda_client = FakeLambdaClient()

        def factory():
            built.append(1)
            return job

        api = EvaluationAPI(store, lambda_client, factory)
        return SimpleNamespace(api=api, job=job, built=built, lambda_client=lambda_client)

    def test_start_prepares_and_invokes_grade(self, parts):
        event = {"httpMethod": "POST", "path": "/api/v1/eval", "body": json.dumps({"session_id": SESSION})}
        code, body = parts.api.start_evaluation(event, self.CONTEXT)
        assert code == 202 and body["found"] == 2 and body["grading"] == 1 and body["grading_ids"] == ["a"]
        [invocation] = parts.lambda_client.invocations
        assert invocation["FunctionName"] == "ak-trace-eval" and invocation["InvocationType"] == "Event"
        grade_request = json.loads(invocation["Payload"])
        assert grade_request["httpMethod"] == "POST" and grade_request["path"] == "/api/v1/grade"
        assert json.loads(grade_request["body"]) == {"session_id": SESSION, "trace_ids": ["a"]}

    def test_start_without_grading_does_not_invoke(self, parts):
        parts.job.summary.grading_ids = []
        assert parts.api.start_evaluation({"body": json.dumps({"session_id": SESSION})}, self.CONTEXT)[0] == 202
        assert parts.lambda_client.invocations == []

    @pytest.mark.parametrize(
        "body", ["not json", json.dumps({}), json.dumps({"session_id": ""}), json.dumps({"session_id": 3}), None]
    )
    def test_start_bad_input(self, parts, body):
        code, response = parts.api.start_evaluation({"body": body}, self.CONTEXT)
        assert code == 400 and "error" in response
        assert parts.job.prepared == [] and parts.lambda_client.invocations == []

    def test_start_propagates_prepare_error(self, parts):
        parts.job.error = RuntimeError("langfuse down")
        with pytest.raises(RuntimeError):
            parts.api.start_evaluation({"body": json.dumps({"session_id": SESSION})}, self.CONTEXT)
        assert parts.lambda_client.invocations == []

    @staticmethod
    def check(api, **params):
        return api.check_progress({"queryStringParameters": params}, TestEvaluationAPI.CONTEXT)

    def test_check_empty_session(self, parts):
        code, body = self.check(parts.api, session_id=SESSION, expected="3")
        assert code == 200
        assert body == {
            "session_id": SESSION,
            "expected": 3,
            "pending": 0,
            "grading": 0,
            "graded": 0,
            "failed": 0,
            "unfinished": 0,
            "errors": [],
            "complete": False,
        }

    def test_check_mixed_session(self, parts, store):
        for trace_id in ("a", "b", "c"):
            store.write_progress(make_case(trace_id), Progress.GRADING, T0)
        store.save_results(make_case("a"), [result("task_completion")], T0)
        store.save_results(make_case("b"), [result("task_completion")], T0)
        store.mark_failed(["c"], "boom", T0)

        body = self.check(parts.api, session_id=SESSION, expected="3")[1]
        assert (body["graded"], body["failed"], body["complete"]) == (2, 1, False)
        assert body["errors"] == [{"trace_id": "c", "error": "boom"}]
        assert self.check(parts.api, session_id=SESSION, expected="2")[1]["complete"] is True

    @pytest.mark.parametrize(
        "params",
        [
            {"expected": "3"},
            {"session_id": SESSION},
            {"session_id": SESSION, "expected": "0"},
            {"session_id": SESSION, "expected": "three"},
        ],
    )
    def test_check_bad_input(self, parts, params):
        assert self.check(parts.api, **params)[0] == 400

    def test_grade(self, parts):
        event = {
            "httpMethod": "POST",
            "path": "/api/v1/grade",
            "body": json.dumps({"session_id": SESSION, "trace_ids": ["a"]}),
        }
        code, body = parts.api.grade_traces(event, self.CONTEXT)
        assert code == 200 and body["graded"] == 2
        parts.api.grade_traces(event, self.CONTEXT)
        assert parts.job.graded == [(SESSION, ["a"]), (SESSION, ["a"])] and parts.built == [1]  # job built once


class TestTrafficSimulator:
    URL = "https://example.com/api/v1/chat"

    @staticmethod
    def transport(requests: list, failures: int = 0, status: int = 502) -> httpx.MockTransport:
        """A fake /chat that records each request and answers the first ``failures`` with ``status``, then 200."""

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            if len(requests) <= failures:
                return httpx.Response(status)
            return httpx.Response(200, json={"result": f"reply {len(requests)}"})

        return httpx.MockTransport(handle)

    def test_send_posts_questions_in_order(self):
        requests: list = []
        replies = TrafficSimulator(self.URL, session_id="s-1", transport=self.transport(requests)).send_questions()
        assert [r["prompt"] for r in requests] == list(TrafficSimulator.QUESTIONS)
        assert {r["session_id"] for r in requests} == {"s-1"} and {r["agent"] for r in requests} == {"triage"}
        assert replies == ["reply 1", "reply 2", "reply 3"]

    def test_generated_session_id(self):
        requests: list = []
        simulator = TrafficSimulator(self.URL, transport=self.transport(requests))
        simulator.send_questions()
        assert {r["session_id"] for r in requests} == {simulator.session_id}

    @pytest.mark.parametrize("status", [429, 502, 503])
    def test_retries_failures_before_the_agent(self, monkeypatch, status):
        """429, 502 and 503 mean the request never reached the agent, so it is sent again."""
        monkeypatch.setattr(TrafficSimulator, "RETRY_WAIT", 0)
        requests: list = []
        replies = TrafficSimulator(
            self.URL, session_id="s-1", transport=self.transport(requests, failures=1, status=status)
        ).send_questions()
        assert len(requests) == 4 and replies[0] == "reply 2"

    @pytest.mark.parametrize("status", [500, 504])
    def test_no_retry_once_the_request_may_be_queued(self, monkeypatch, status):
        """500 and 504 may come after the request is queued, so it is sent once and the error raised."""
        monkeypatch.setattr(TrafficSimulator, "RETRY_WAIT", 0)
        requests: list = []
        simulator = TrafficSimulator(self.URL, transport=self.transport(requests, failures=1, status=status))
        with pytest.raises(httpx.HTTPStatusError):
            simulator.send_questions()
        assert len(requests) == 1

    def test_retries_connect_errors_only(self, monkeypatch):
        """A connect error is retried; a read timeout (the request may have arrived) is raised."""
        monkeypatch.setattr(TrafficSimulator, "RETRY_WAIT", 0)
        attempts = []

        def handle(error: type[httpx.TransportError]):
            def raise_error(request: httpx.Request) -> httpx.Response:
                attempts.append(error)
                if len(attempts) == 1:
                    raise error("failed", request=request)
                return httpx.Response(200, json={"result": "ok"})

            return httpx.MockTransport(raise_error)

        assert TrafficSimulator(self.URL, transport=handle(httpx.ConnectError)).send_questions()[0] == "ok"
        attempts.clear()
        with pytest.raises(httpx.ReadTimeout):
            TrafficSimulator(self.URL, transport=handle(httpx.ReadTimeout)).send_questions()
        assert len(attempts) == 1

    def test_warm_up_uses_a_throwaway_session_and_ignores_failures(self):
        """warm_up sends one request on a warm-up- session and swallows a 504 or a timeout."""
        requests: list = []
        simulator = TrafficSimulator(
            self.URL, session_id="s-1", transport=self.transport(requests, failures=1, status=504)
        )
        simulator.warm_up()
        assert len(requests) == 1 and requests[0]["session_id"].startswith("warm-up-")

        def raise_timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        TrafficSimulator(self.URL, transport=httpx.MockTransport(raise_timeout)).warm_up()

    def test_chat_url_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AK_TEST_ENDPOINT", self.URL)
        assert TrafficSimulator.chat_url_from_env(tmp_path) == self.URL

        monkeypatch.delenv("AK_TEST_ENDPOINT")
        outputs = iter([f"{self.URL}\n", ""])
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=next(outputs)))
        assert TrafficSimulator.chat_url_from_env(tmp_path) == self.URL
        with pytest.raises(RuntimeError, match="empty"):
            TrafficSimulator.chat_url_from_env(tmp_path)
