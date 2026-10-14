import json
import logging
from datetime import datetime, timedelta
from typing import Any

from agentkernel.test.core.evaluator import AKEvaluationCase
from agentkernel.trace import FetchedSpan, OTelSpanClassifier, SpanKind, SpanUsage
from pydantic import BaseModel

_log = logging.getLogger("ak.example.trace_evaluation")


class ToolCall(BaseModel):
    name: str
    input: str | None = None
    output: str | None = None
    failed: bool = False


class TraceEvaluationCase(AKEvaluationCase):
    """What the judges read: the AK evaluation case plus the trace's transcript."""

    transcript: str
    tool_call_count: int = 0


class TraceCase(BaseModel):
    """The facts of one request (one trace), gathered from its spans."""

    trace_id: str
    session_id: str | None = None
    root_name: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    user_input: str = ""
    output: str = ""
    tool_calls: list[ToolCall] = []
    agents: list[str] = []
    usage: SpanUsage = SpanUsage()  # summed over the trace's spans
    span_count: int
    spans_missing_parent: int = 0  # spans whose parent_span_id is not in the trace
    spans: list[FetchedSpan] = []  # kept for EvaluationStore's spans_gz

    @property
    def is_finished(self) -> bool:
        """The run itself ended: the root span has an end time and an output."""
        return self.ended_at is not None and self.output != ""

    def has_settled(self, now: datetime, settle_time: timedelta) -> bool:
        """The root span ended at least ``settle_time`` ago, so its output and late spans have had time to land."""
        return self.ended_at is not None and self.ended_at <= now - settle_time

    def is_ready_to_grade(self, now: datetime, settle_time: timedelta) -> bool:
        """Finished, settled, and every span's parent is readable."""
        return self.is_finished and self.has_settled(now, settle_time) and self.spans_missing_parent == 0

    @property
    def latency_ms(self) -> int | None:
        """Milliseconds from the root span's start to its end, or None while it hasn't ended."""
        if self.ended_at is None:
            return None
        return int((self.ended_at - self.started_at).total_seconds() * 1000)

    @classmethod
    def from_spans(cls, trace_id: str, spans: list[FetchedSpan]) -> "TraceCase | None":
        """Builds the case, or None when the trace has no single parentless root span (not a request)."""
        root_spans = [span for span in spans if span.parent_span_id is None]
        if len(root_spans) != 1:
            _log.warning(f"trace {trace_id}: {len(root_spans)} root spans, expected 1; not a request")
            return None
        root = root_spans[0]
        if root.start_time is None:
            _log.warning(f"trace {trace_id}: root span has no start time; not a request")
            return None

        spans_oldest_first = FetchedSpan.oldest_first(spans)
        span_ids = {span.span_id for span in spans}
        tool_calls = [
            ToolCall(
                name=span.name or "unknown",
                input=cls._as_text(span.input),
                output=cls._as_text(span.output),
                failed=span.raw.get("level") == "ERROR",
            )
            for span in spans_oldest_first
            if cls._span_kind(span) == SpanKind.TOOL
        ]
        agent_names = [span.name for span in spans_oldest_first if cls._span_kind(span) == SpanKind.AGENT and span.name]
        return cls(
            trace_id=trace_id,
            session_id=root.session_id,
            root_name=root.name,
            started_at=root.start_time,
            ended_at=root.end_time,
            user_input=cls._as_text(root.input) or "",
            output=cls._as_text(root.output) or "",
            tool_calls=tool_calls,
            agents=list(dict.fromkeys(agent_names)),
            usage=cls._sum_usage(spans),
            span_count=len(spans),
            spans_missing_parent=sum(
                1 for span in spans if span.parent_span_id is not None and span.parent_span_id not in span_ids
            ),
            spans=spans_oldest_first,
        )

    def build_transcript(self) -> str:
        """Writes the request, each tool call with its result, and the final answer as one text for the judges."""
        if self.tool_calls:
            tool_call_lines = "\n".join(
                f"{number}. {call.name}({call.input or ''}) -> {json.dumps(call.output)}"
                + (" [failed]" if call.failed else "")
                for number, call in enumerate(self.tool_calls, start=1)
            )
        else:
            tool_call_lines = "(none)"
        return f"User request:\n{self.user_input}\n\nTool calls:\n{tool_call_lines}\n\nFinal answer:\n{self.output}"

    def to_evaluation_case(self, threshold: float) -> TraceEvaluationCase:
        """Builds the case the judges score, with the tool results as context."""
        tool_results = [call.output for call in self.tool_calls if call.output]
        return TraceEvaluationCase(
            user_input=self.user_input,
            actual=self.output,
            context=tool_results or None,
            transcript=self.build_transcript(),
            tool_call_count=len(self.tool_calls),
            threshold=threshold,
        )

    @staticmethod
    def _span_kind(span: FetchedSpan) -> SpanKind:
        """The span's kind, classified from its attributes when Langfuse typed it as a plain span."""
        # Langfuse may type an OpenInference tool or agent span as a plain span
        if span.kind != SpanKind.SPAN:
            return span.kind
        return OTelSpanClassifier.classify(span.attributes, span.name)

    @staticmethod
    def _sum_usage(spans: list[FetchedSpan]) -> SpanUsage:
        """Adds up tokens and cost over the spans; a field stays None when no span reports it."""
        usages = [span.usage for span in spans if span.usage is not None]

        def sum_field(field: str) -> Any:
            """Sums one usage field over the spans that report it."""
            values = [getattr(usage, field) for usage in usages if getattr(usage, field) is not None]
            return sum(values) if values else None

        return SpanUsage(
            input_tokens=sum_field("input_tokens"),
            output_tokens=sum_field("output_tokens"),
            total_tokens=sum_field("total_tokens"),
            cost=sum_field("cost"),
        )

    @staticmethod
    def _as_text(value: Any) -> str | None:
        """Turns a span's input or output into text, unwrapping the JSON-encoded strings Langfuse often returns."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except ValueError:
                return value
            return decoded if isinstance(decoded, str) else value
        return json.dumps(value)
