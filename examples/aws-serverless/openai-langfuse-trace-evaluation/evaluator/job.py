import logging
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable

from agentkernel.test import AKEvaluationResult
from agentkernel.trace import FetchedTrace, Trace
from pydantic import BaseModel

from .case import TraceCase
from .store import EvaluationStore, Progress, TraceProgress

if TYPE_CHECKING:
    from agentkernel.test.core.evaluator.opik import (
        OpikAKEvaluator,
    )  # imported on first grade only: Opik and litellm load slowly

_log = logging.getLogger("ak.example.trace_evaluation")


class PrepareSummary(BaseModel):
    session_id: str
    spans_fetched: int = 0
    found: int = 0  # traces with a root span
    no_root: int = 0  # traces with no single root span: no row
    already_graded: int = 0  # row already graded
    grading: int = 0  # claimed by this call, or still within an earlier call's lease
    pending: int = 0
    unfinished: int = 0
    grading_ids: list[str] = []  # claimed by this call; sent to grade in the invoke payload


class GradeSummary(BaseModel):
    session_id: str
    graded: int = 0
    failed: int = 0
    metric_errors: int = 0


class TraceEvaluationJob:
    """Decides each of a session's traces' progress (prepare) and judges the ready ones (grade)."""

    FETCH_WINDOW = timedelta(hours=24)
    PASS_THRESHOLD = 0.5  # the score a metric must reach to pass
    SETTLE_TIME = timedelta(minutes=2)
    GRADING_LEASE = timedelta(minutes=15)  # above the grading invocation's 10-minute Lambda timeout

    def __init__(
        self,
        trace: Trace,
        store: EvaluationStore,
        evaluator_factory: Callable[[], "OpikAKEvaluator"],
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._trace = trace
        self._store = store
        self._evaluator_factory = evaluator_factory
        self._evaluator: "OpikAKEvaluator | None" = None  # built on the first grade
        self._clock = clock

    @classmethod
    def from_env(cls) -> "TraceEvaluationJob":
        """Builds the job on the configured tracer and the RESULTS_TABLE table, with AK's Opik evaluator."""

        def build_opik_evaluator() -> "OpikAKEvaluator":
            """Builds the evaluator from the AK_TEST__LLM__* settings."""
            from agentkernel.test import AKTestConfig
            from agentkernel.test.core.evaluator.opik import OpikAKEvaluator

            return OpikAKEvaluator(AKTestConfig.get())

        return cls(Trace.get(), EvaluationStore.for_table(os.environ["RESULTS_TABLE"]), build_opik_evaluator)

    def prepare_traces(self, session_id: str) -> PrepareSummary:
        """Gives every found trace of the session its row and progress; returns the traces to grade."""
        now = self._clock()
        traces = self._fetch_session_traces(session_id)
        cases, rootless_traces = self._build_cases(traces)
        summary = PrepareSummary(
            session_id=session_id,
            spans_fetched=sum(len(trace.spans) for trace in traces),
            found=len(cases),
            no_root=rootless_traces,
        )
        progress_by_id = self._store.read_progress([case.trace_id for case in cases])
        for case in cases:
            progress, claimed = self._advance_progress(case, progress_by_id.get(case.trace_id), now)
            match progress:
                case Progress.GRADED:
                    summary.already_graded += 1
                case Progress.GRADING:
                    summary.grading += 1
                    if claimed:
                        summary.grading_ids.append(case.trace_id)
                case Progress.PENDING:
                    summary.pending += 1
                case Progress.UNFINISHED:
                    summary.unfinished += 1
        _log.info(f"prepare: {summary.model_dump_json(exclude={'grading_ids'})}")
        return summary

    def grade_traces(self, session_id: str, trace_ids: list[str]) -> GradeSummary:
        """Judges the given traces that are still grading; a failure marks rows failed instead of raising."""
        summary = GradeSummary(session_id=session_id)
        progress_by_id = self._store.read_progress(trace_ids)
        grading_ids = [
            trace_id
            for trace_id in trace_ids
            if trace_id in progress_by_id and progress_by_id[trace_id].progress == Progress.GRADING
        ]
        if not grading_ids:
            _log.info(f"grade: no trace of session {session_id} is still grading")
            return summary

        try:
            cases, _ = self._build_cases(self._fetch_session_traces(session_id))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            _log.exception(f"grade: fetching session {session_id} failed")
            self._store.mark_failed(grading_ids, error, self._clock())
            summary.failed = len(grading_ids)
            return summary

        case_by_id = {case.trace_id: case for case in cases}
        missing_ids = [trace_id for trace_id in grading_ids if trace_id not in case_by_id]
        if missing_ids:
            self._store.mark_failed(missing_ids, "trace not found in the fetch", self._clock())
            summary.failed += len(missing_ids)

        for case in (case for case in cases if case.trace_id in grading_ids):
            try:
                results = self._run_evaluators(case)
                self._store.save_results(case, results, self._clock())
            except Exception as e:
                _log.exception(f"grade: trace {case.trace_id} failed")
                self._store.mark_failed([case.trace_id], f"{type(e).__name__}: {e}", self._clock())
                summary.failed += 1
                continue
            summary.graded += 1
            summary.metric_errors += sum(1 for result in results if result.score is None)
        _log.info(f"grade: {summary.model_dump_json()}")
        return summary

    def _fetch_session_traces(self, session_id: str) -> list[FetchedTrace]:
        """Fetches the session's traces from the last FETCH_WINDOW."""
        return self._trace.fetch_traces(session_id=session_id, last=self.FETCH_WINDOW)

    @staticmethod
    def _build_cases(traces: list[FetchedTrace]) -> tuple[list[TraceCase], int]:
        """Builds a case per trace that has a root span, oldest first; also returns how many traces had none."""
        cases = [TraceCase.from_trace(trace) for trace in traces if trace.root is not None]
        return sorted(cases, key=lambda case: case.started_at), len(traces) - len(cases)

    def _advance_progress(self, case: TraceCase, current: TraceProgress | None, now: datetime) -> tuple[Progress, bool]:
        """Writes the trace's next progress; returns its progress and whether this call claimed it for grading."""
        next_progress = self._next_progress(case, current, now)
        if next_progress is None:
            return current.progress, False
        if self._store.write_progress(case, next_progress, now, now - self.GRADING_LEASE):
            return next_progress, next_progress == Progress.GRADING
        # a concurrent /eval wrote the row between the read and this write; report what it wrote
        return self._store.read_progress([case.trace_id])[case.trace_id].progress, False

    def _next_progress(self, case: TraceCase, current: TraceProgress | None, now: datetime) -> Progress | None:
        """The progress to write for the trace, or None to leave its row as it is."""
        if current is not None:
            if current.progress in (Progress.GRADED, Progress.UNFINISHED):
                return None
            if current.progress == Progress.GRADING and current.updated_at >= now - self.GRADING_LEASE:
                return None  # an earlier /eval's grading invocation may still be running
        if case.has_settled(now, self.SETTLE_TIME) and not case.is_finished:
            return Progress.UNFINISHED  # the root ended long enough ago and still has no output
        if not case.is_ready_to_grade(now, self.SETTLE_TIME):
            return Progress.PENDING  # still running, recently ended, or partly ingested
        return Progress.GRADING

    def _run_evaluators(self, case: TraceCase) -> list[AKEvaluationResult]:
        """Runs every metric that applies to the trace and returns the results."""
        if self._evaluator is None:
            self._evaluator = self._evaluator_factory()
        evaluation_case = case.to_evaluation_case(self.PASS_THRESHOLD)
        metric_cases = [evaluation_case.model_copy(update={"metric": metric}) for metric in self._evaluator.METRICS]
        return [
            self._evaluator.evaluate_by_llm(metric_case)
            for metric_case in metric_cases
            if self._evaluator.applies_to(metric_case)
        ]
