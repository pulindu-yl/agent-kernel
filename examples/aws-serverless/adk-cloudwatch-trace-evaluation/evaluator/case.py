import json
from datetime import datetime, timedelta
from typing import Any

from agentkernel.test import AKEvaluationCase
from agentkernel.trace import FetchedSpan, FetchedTrace, SpanKind, SpanStatus, SpanUsage
from pydantic import BaseModel


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
    provider: str  # the tracer the spans came from
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
    spans: list[FetchedSpan] = []  # kept for EvaluationStore's spans_gz

    @property
    def is_finished(self) -> bool:
        """The run itself ended: the root span has an end time and an output."""
        return self.ended_at is not None and self.output != ""

    def has_settled(self, now: datetime, settle_time: timedelta) -> bool:
        """The root span ended at least ``settle_time`` ago, so its output and late spans have had time to land."""
        return self.ended_at is not None and self.ended_at <= now - settle_time

    def is_ready_to_grade(self, now: datetime, settle_time: timedelta) -> bool:
        """Finished and settled."""
        return self.is_finished and self.has_settled(now, settle_time)

    @property
    def latency_ms(self) -> int | None:
        """Milliseconds from the root span's start to its end, or None while it hasn't ended."""
        if self.ended_at is None:
            return None
        return int((self.ended_at - self.started_at).total_seconds() * 1000)

    @classmethod
    def from_trace(cls, trace: FetchedTrace) -> "TraceCase":
        """Builds the case from a fetched trace, which must have a root span."""
        root = trace.root
        if root is None or trace.started_at is None:
            raise ValueError(f"trace {trace.trace_id} has no root span with a start time")
        tool_calls = [
            ToolCall(
                name=cls._tool_name(span),
                input=cls._as_text(span.input),
                output=cls._as_text(cls._tool_output(span)),
                failed=cls._tool_failed(span),
            )
            for span in trace.tool_spans
        ]
        agent_names = [name for span in trace.spans if span.kind == SpanKind.AGENT and (name := cls._agent_name(span))]
        return cls(
            trace_id=trace.trace_id,
            provider=root.provider,
            session_id=trace.session_id,
            root_name=root.name,
            started_at=trace.started_at,
            ended_at=trace.ended_at,
            user_input=cls._as_text(trace.user_input) or "",
            output=cls._as_text(trace.output) or "",
            tool_calls=tool_calls,
            agents=list(dict.fromkeys(agent_names)),
            usage=trace.usage,
            span_count=len(trace.spans),
            spans=trace.spans,
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
    def _tool_name(span: FetchedSpan) -> str:
        """The tool's name: OpenInference's tool.name, else the span name."""
        return span.attributes.get("tool.name") or span.name or "unknown"

    @staticmethod
    def _agent_name(span: FetchedSpan) -> str | None:
        """The agent's name, not the span name: OpenInference's agent.name on "agent_run [<name>]", and
        gen_ai.agent.name on the "invoke_agent <name>" span ADK emits for the same run."""
        return span.attributes.get("agent.name") or span.attributes.get("gen_ai.agent.name") or span.name

    @staticmethod
    def _tool_output(span: FetchedSpan) -> Any:
        """The tool's result: ADK records the whole function response {"id", "name", "response": {"result"}}."""
        try:
            output = json.loads(span.output) if isinstance(span.output, str) else span.output
        except ValueError:
            return span.output
        if not isinstance(output, dict) or "response" not in output:
            return span.output
        response = output["response"]
        # ADK wraps a function's non-dict return value as {"result": value}
        if isinstance(response, dict) and set(response) == {"result"}:
            return response["result"]
        return response

    @staticmethod
    def _tool_failed(span: FetchedSpan) -> bool:
        """Whether the tool call raised: the span's status is ERROR."""
        return span.status == SpanStatus.ERROR

    @staticmethod
    def _as_text(value: Any) -> str | None:
        """Turns a span's input or output into text, unwrapping JSON-encoded strings."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except ValueError:
                return value
            return decoded if isinstance(decoded, str) else value
        return json.dumps(value)
