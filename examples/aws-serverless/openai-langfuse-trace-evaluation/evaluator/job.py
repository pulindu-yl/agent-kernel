import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable

from agentkernel.test import AKEvaluationResult
from agentkernel.trace import FetchedSpan, Trace
from pydantic import BaseModel

from .case import TraceCase
from .store import EvaluationStore, Progress, TraceProgress

if TYPE_CHECKING:
    from .opik_evaluator import OpikTraceEvaluator  # imported on first grade only: Opik and litellm load slowly

_log = logging.getLogger("ak.example.trace_evaluation")


class PrepareSummary(BaseModel):
    session_id: str
    spans_fetched: int = 0
    found: int = 0  # traces with exactly one root span
    no_root: int = 0  # traces with no root span (or several): no row
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
    FETCH_LIMIT = 5_000
    SETTLE_TIME = timedelta(minutes=2)
    GRADING_LEASE = timedelta(minutes=15)  # above the grading invocation's 10-minute Lambda timeout

    def __init__(
        self,
        trace: Trace,
        store: EvaluationStore,
        evaluator_factory: Callable[[], list["OpikTraceEvaluator"]],
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._trace = trace
        self._store = store
        self._evaluator_factory = evaluator_factory
        self._evaluators: list["OpikTraceEvaluator"] | None = None  # built on the first grade
        self._clock = clock

    @classmethod
    def from_env(cls) -> "TraceEvaluationJob":
        """Builds the job on the configured tracer and the RESULTS_TABLE table, with the Opik judges."""

        def build_opik_evaluators() -> list["OpikTraceEvaluator"]:
            """Builds one Opik evaluator per metric, from the AK_TEST__LLM__* settings."""
            from agentkernel.test import AKTestConfig

            from .opik_evaluator import OpikTraceEvaluator

            return OpikTraceEvaluator.for_all_metrics(AKTestConfig.get())

        return cls(Trace.get(), EvaluationStore.for_table(os.environ["RESULTS_TABLE"]), build_opik_evaluators)

    def prepare_traces(self, session_id: str) -> PrepareSummary:
        """Gives every found trace of the session its row and progress; returns the traces to grade."""
        now = self._clock()
        spans = self._fetch_session_spans(session_id)
        cases, rootless_traces = self._build_cases(spans)
        summary = PrepareSummary(
            session_id=session_id, spans_fetched=len(spans), found=len(cases), no_root=rootless_traces
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
            cases, _ = self._build_cases(self._fetch_session_spans(session_id))
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

    def _fetch_session_spans(self, session_id: str) -> list[FetchedSpan]:
        """Fetches the session's spans from the last FETCH_WINDOW; raises when the result may be truncated."""
        spans = self._trace.fetch(session_id=session_id, last=self.FETCH_WINDOW, limit=self.FETCH_LIMIT)
        if len(spans) >= self.FETCH_LIMIT:
            # fetch keeps the newest spans, so older requests would be silently missing
            raise RuntimeError(
                f"session {session_id} has at least {self.FETCH_LIMIT} spans; refusing to grade a truncated session"
            )
        return spans

    @staticmethod
    def _build_cases(spans: list[FetchedSpan]) -> tuple[list[TraceCase], int]:
        """Groups the spans by trace into cases, oldest first; also returns how many traces were not requests."""
        spans_by_trace: dict[str, list[FetchedSpan]] = defaultdict(list)
        for span in spans:
            spans_by_trace[span.trace_id].append(span)
        cases, rootless_traces = [], 0
        for trace_id, trace_spans in spans_by_trace.items():
            case = TraceCase.from_spans(trace_id, trace_spans)
            if case is None:
                rootless_traces += 1
            else:
                cases.append(case)
        return sorted(cases, key=lambda case: case.started_at), rootless_traces

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
        """Runs every evaluator that applies to the trace and returns their results."""
        if self._evaluators is None:
            self._evaluators = self._evaluator_factory()
        evaluation_case = case.to_evaluation_case(self._pass_threshold())
        return [
            evaluator.evaluate_by_llm(evaluation_case)
            for evaluator in self._evaluators
            if evaluator.applies_to(evaluation_case)
        ]

    @staticmethod
    def _pass_threshold() -> float:
        """The score a metric must reach to pass, imported lazily so Opik loads only when grading."""
        from .opik_evaluator import OpikTraceEvaluator

        return OpikTraceEvaluator.THRESHOLD
