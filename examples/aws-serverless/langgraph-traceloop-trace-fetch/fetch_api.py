import logging
from datetime import timedelta
from typing import Any

from agentkernel.trace import FetchedSpan, SpanKind, Trace
from opentelemetry import trace as trace_api

_log = logging.getLogger("ak.example.traceloop_fetch")


class TraceFetchAPI:
    """GET /fetch: reads a session's spans back from Traceloop with Trace.fetch_spans and returns them as JSON."""

    DEFAULT_MINUTES = 60
    MAX_MINUTES = 24 * 60  # Traceloop's free plan keeps spans for 24 hours
    FETCH_LIMIT = 1_000
    MAX_SPANS_RETURNED = 200  # keeps the response far under Lambda's 6 MB payload limit

    def __init__(self, trace: Any = None):
        self._trace = trace

    def fetch_spans(self, event: dict, context: Any) -> tuple[int, dict]:
        """Query ``session_id`` (required), ``trace_id``, ``kind`` (agent, llm, tool, span) and ``minutes``."""
        params = event.get("queryStringParameters") or {}
        session_id = params.get("session_id")
        if not session_id:
            return 400, {"error": "session_id is required"}
        filters: dict[str, Any] = {"session_id": session_id, "limit": self.FETCH_LIMIT}
        if trace_id := params.get("trace_id"):
            filters["trace_ids"] = [trace_id]
        if kind := params.get("kind"):
            try:
                filters["kinds"] = [SpanKind(kind.lower())]
            except ValueError:
                return 400, {"error": f"kind must be one of {', '.join(k.value for k in SpanKind)}"}
        try:
            minutes = int(params.get("minutes", self.DEFAULT_MINUTES))
        except ValueError:
            minutes = 0
        if not 1 <= minutes <= self.MAX_MINUTES:
            return 400, {"error": f"minutes must be an integer from 1 to {self.MAX_MINUTES}"}
        filters["last"] = timedelta(minutes=minutes)

        spans = self._get_trace().fetch_spans(**filters)
        _log.info(f"fetch {filters}: {len(spans)} spans")
        return 200, {
            "session_id": session_id,
            "span_count": len(spans),
            "trace_count": len({span.trace_id for span in spans}),
            "truncated": len(spans) > self.MAX_SPANS_RETURNED,
            "spans": [self._to_json(span) for span in spans[: self.MAX_SPANS_RETURNED]],
        }

    @staticmethod
    def flush() -> None:
        """Exports the spans Traceloop's batch processor still holds; a frozen Lambda sandbox never would."""
        provider = trace_api.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()

    def _get_trace(self) -> Any:
        """The configured tracer, looked up on first use."""
        if self._trace is None:
            self._trace = Trace.get()
        return self._trace

    @staticmethod
    def _to_json(span: FetchedSpan) -> dict:
        """The span without ``raw``, which repeats everything else as the warehouse returned it."""
        return span.model_dump(mode="json", exclude={"raw"})
