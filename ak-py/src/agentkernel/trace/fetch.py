"""Provider-neutral model for fetching spans back from a tracing provider's cloud.

``TraceQuery`` is the generic filter every tracer translates into its own query language,
``FetchedSpan`` is the shape every tracer returns (with ``SpanUsage`` for its model call's tokens and
cost), and ``OTelSpanClassifier`` maps the raw
OpenTelemetry attribute conventions AK's runners emit (OpenInference, OTel GenAI, Traceloop,
Logfire native) onto one ``SpanKind``. ``TraceFetcher`` holds the paging and merging the built-in
fetchers share.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, ClassVar, Iterable, Optional

from pydantic import BaseModel, Field


class SpanKind(str, Enum):
    """What a span represents, independent of the provider's own vocabulary."""

    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    SPAN = "span"  # anything else: AK wrapper spans, turns, chains, workflows


class TraceQuery(BaseModel):
    """Generic filter for ``Trace.fetch()``; each tracer translates it to its provider's API."""

    start: Optional[datetime] = Field(default=None, description="Inclusive lower bound on span start time; defaults to end - last")
    end: Optional[datetime] = Field(default=None, description="Exclusive upper bound on span start time; defaults to now")
    last: timedelta = Field(default=timedelta(hours=1), description="Look-back window used when start is not given")
    kinds: Optional[list[SpanKind]] = Field(default=None, description="Only return spans of these kinds; None returns all")
    session_id: Optional[str] = Field(default=None, description="Only return spans belonging to this AK session")
    trace_ids: Optional[list[str]] = Field(default=None, description="Only return spans of these traces")
    name: Optional[str] = Field(default=None, description="Only return spans with exactly this name")
    limit: int = Field(default=1000, gt=0, description="Maximum number of spans to return; when more match, the most recent ones are kept")

    def window(self) -> tuple[datetime, datetime]:
        """Resolve the time window to timezone-aware UTC datetimes."""
        end = self._utc(self.end) if self.end else datetime.now(timezone.utc)
        start = self._utc(self.start) if self.start else end - self.last
        if start >= end:
            raise ValueError(f"fetch window start ({start.isoformat()}) must be before end ({end.isoformat()})")
        return start, end

    def accepts(self, kind: SpanKind) -> bool:
        return self.kinds is None or kind in self.kinds

    def matches(self, span: FetchedSpan) -> bool:
        """Whether a fetched span satisfies every filter, for fetchers whose server-side filters are approximate.

        Strict: for each active filter the span must report a value that satisfies it. A span without a start
        time, or without a session id (or name) when that filter is set, is rejected, never assumed to match.
        """
        start, end = self.window()
        if span.start_time is None or not start <= span.start_time < end:
            return False
        if self.session_id is not None and span.session_id != self.session_id:
            return False
        if self.trace_ids is not None and span.trace_id not in self.trace_ids:
            return False
        if self.name is not None and span.name != self.name:
            return False
        return self.accepts(span.kind)

    @staticmethod
    def _utc(value: datetime) -> datetime:
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class SpanUsage(BaseModel):
    """Tokens and cost of the one model call a span records, never a roll-up of its children.

    Summing ``usage`` over a trace's spans gives the trace's total, so values the instrumentation already
    aggregates (Pydantic AI ``gen_ai.aggregated_usage.*``, the OpenAI Agents ``usage`` JSON on turn and
    workflow spans) are deliberately not read.
    """

    input_tokens: Optional[int] = Field(default=None, description="Prompt (input) tokens")
    output_tokens: Optional[int] = Field(default=None, description="Completion (output) tokens")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens; input + output when the span reports no total")
    cost: Optional[float] = Field(default=None, description="Cost in USD, as the provider or instrumentation computed it")

    @classmethod
    def of(cls, input_tokens: Any = None, output_tokens: Any = None, total_tokens: Any = None, cost: Any = None) -> Optional[SpanUsage]:
        """Builds a usage from raw values (numbers or numeric strings), or None when none is known."""
        input_tokens, output_tokens, total_tokens = cls._int(input_tokens), cls._int(output_tokens), cls._int(total_tokens)
        cost = cls._float(cost)
        if input_tokens is None and output_tokens is None and total_tokens is None and cost is None:
            return None
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return cls(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens, cost=cost)

    @classmethod
    def _int(cls, value: Any) -> Optional[int]:
        number = cls._float(value)
        return None if number is None else int(number)

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class FetchedSpan(BaseModel):
    """One span (Langfuse: observation) fetched from a provider, in a provider-neutral envelope.

    ``input``/``output`` are returned as the provider stores them (often JSON strings); ``raw`` keeps the
    provider's full record so later normalisation never loses data.
    """

    provider: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: Optional[str] = None
    kind: SpanKind = SpanKind.SPAN
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    session_id: Optional[str] = None
    input: Any = None
    output: Any = None
    usage: Optional[SpanUsage] = None  # None when the span records no model call
    attributes: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def oldest_first(spans: Iterable[FetchedSpan]) -> list[FetchedSpan]:
        floor = datetime.min.replace(tzinfo=timezone.utc)
        return sorted(spans, key=lambda span: span.start_time or floor)


class OTelSpanClassifier:
    """Classifies raw OTel spans and pulls their input/output, across the conventions AK emits.

    The convention is detected per span, not per provider, because one trace can mix them
    (e.g. Pydantic AI GenAI spans next to Traceloop ``openai.chat`` spans).
    """

    _OPENINFERENCE = {"AGENT": SpanKind.AGENT, "LLM": SpanKind.LLM, "TOOL": SpanKind.TOOL}
    _TRACELOOP = {"agent": SpanKind.AGENT, "tool": SpanKind.TOOL}
    _GEN_AI = {
        "invoke_agent": SpanKind.AGENT,
        "create_agent": SpanKind.AGENT,
        "execute_tool": SpanKind.TOOL,
        "chat": SpanKind.LLM,
        "text_completion": SpanKind.LLM,
        "generate_content": SpanKind.LLM,
    }
    # Logfire's OpenAI Agents instrumentation carries no kind attribute, only message templates
    _LOGFIRE_TEMPLATES = {"Function:": SpanKind.TOOL, "Agent run:": SpanKind.AGENT}

    # (input key, output key) pairs, most specific first
    _IO_KEYS = [
        ("gen_ai.tool.call.arguments", "gen_ai.tool.call.result"),
        ("input.value", "output.value"),
        ("gen_ai.input.messages", "gen_ai.output.messages"),
        ("langfuse.observation.input", "langfuse.observation.output"),
        ("traceloop.entity.input", "traceloop.entity.output"),
        ("input", "output"),
    ]

    _SESSION_KEYS = ["session.id", "session_id", "traceloop.association.properties.session_id"]

    # per-call usage keys, first present wins: OTel GenAI, OpenInference, then older GenAI / Traceloop names
    _INPUT_TOKEN_KEYS = ["gen_ai.usage.input_tokens", "llm.token_count.prompt", "gen_ai.usage.prompt_tokens"]
    _OUTPUT_TOKEN_KEYS = ["gen_ai.usage.output_tokens", "llm.token_count.completion", "gen_ai.usage.completion_tokens"]
    _TOTAL_TOKEN_KEYS = ["gen_ai.usage.total_tokens", "llm.token_count.total", "llm.usage.total_tokens"]
    _COST_KEYS = ["operation.cost"]  # Pydantic AI's per-call cost

    @classmethod
    def classify(cls, attributes: dict[str, Any], name: Optional[str] = None) -> SpanKind:
        if (kind := cls._OPENINFERENCE.get(str(attributes.get("openinference.span.kind", "")).upper())) is not None:
            return kind
        if (kind := cls._TRACELOOP.get(str(attributes.get("traceloop.span.kind", "")).lower())) is not None:
            return kind
        if (kind := cls._GEN_AI.get(str(attributes.get("gen_ai.operation.name", "")))) is not None:
            return kind
        template = str(attributes.get("logfire.msg_template") or name or "")
        for prefix, kind in cls._LOGFIRE_TEMPLATES.items():
            if template.startswith(prefix):
                return kind
        if "LLM" in (attributes.get("logfire.tags") or []):
            return SpanKind.LLM
        return SpanKind.SPAN

    @classmethod
    def io(cls, attributes: dict[str, Any]) -> tuple[Any, Any]:
        for in_key, out_key in cls._IO_KEYS:
            if in_key in attributes or out_key in attributes:
                return attributes.get(in_key), attributes.get(out_key)
        return None, None

    @classmethod
    def usage(cls, attributes: dict[str, Any]) -> Optional[SpanUsage]:
        return SpanUsage.of(
            input_tokens=cls._first(attributes, cls._INPUT_TOKEN_KEYS),
            output_tokens=cls._first(attributes, cls._OUTPUT_TOKEN_KEYS),
            total_tokens=cls._first(attributes, cls._TOTAL_TOKEN_KEYS),
            cost=cls._first(attributes, cls._COST_KEYS),
        )

    @staticmethod
    def _first(attributes: dict[str, Any], keys: list[str]) -> Any:
        return next((attributes[key] for key in keys if attributes.get(key) is not None), None)

    @classmethod
    def session_id(cls, attributes: dict[str, Any]) -> Optional[str]:
        for key in cls._SESSION_KEYS:
            if attributes.get(key):
                return str(attributes[key])
        return None


class TraceFetcher(ABC):
    """The base of the built-in fetchers: the pinned window, the merge and the two paging loops they share.

    Subclasses implement only their provider's requests (``_query_provider``) and their ``_to_span`` mapping.
    Not exported: bring-your-own tracers override ``BaseTrace.fetch`` and only have to pass ``TraceFetchContract``.
    """

    provider: ClassVar[str]  # the tracer's trace.type short name, written to every FetchedSpan.provider
    _log = logging.getLogger("ak.trace.fetch")

    def fetch(self, query: TraceQuery) -> list[FetchedSpan]:
        """
        Fetches the spans matching the query, oldest first. When more than ``limit`` match, the most
        recent ones are kept.
        """
        # matches() resolves window() on every call, so a defaulted end (now) would move between calls
        start, end = query.window()
        query = query.model_copy(update={"start": start, "end": end})
        spans_by_id: dict[tuple[str, str], FetchedSpan] = {}
        for span in self._query_provider(query):
            spans_by_id.setdefault(self._span_key(span), span)
        return FetchedSpan.oldest_first(spans_by_id.values())[-query.limit :]

    @abstractmethod
    def _query_provider(self, query: TraceQuery) -> Iterable[FetchedSpan]:
        """Runs the provider's requests for a query whose window is already pinned."""

    @staticmethod
    def _span_key(span: FetchedSpan) -> tuple[str, str]:
        # OTel span ids are unique only within a trace
        return span.trace_id, span.span_id

    def _fetch_by_cursor(
        self,
        query: TraceQuery,
        fetch_page: Callable[[Optional[str]], tuple[list[Any], Optional[str]]],
        item_to_span: Callable[[Any], FetchedSpan],
    ) -> list[FetchedSpan]:
        """
        Follows a newest-first cursor API, keeping up to ``limit`` spans that ``query.matches``.

        :param fetch_page: Returns ``(items, next_cursor)`` for a cursor (None for the first page).
        :param item_to_span: Maps one item to a span.
        """
        matched_spans: list[FetchedSpan] = []
        cursor: Optional[str] = None
        while True:
            items, next_cursor = fetch_page(cursor)
            for item in items:
                span = item_to_span(item)
                if query.matches(span):
                    matched_spans.append(span)
                    if len(matched_spans) >= query.limit:
                        return matched_spans
            # a repeated cursor would page forever when local filtering keeps the limit from being reached
            if not next_cursor or not items or next_cursor == cursor:
                return matched_spans
            cursor = next_cursor

    def _fetch_by_time_window(
        self,
        query: TraceQuery,
        upper_bound: datetime,
        max_rows_per_request: int,
        fetch_page: Callable[[int, datetime], list[tuple[Optional[datetime], FetchedSpan]]],
    ) -> list[FetchedSpan]:
        """
        Pages backwards in time over an API with a row cap and no cursor, keeping up to ``limit`` spans
        that ``query.matches``.

        Each page's upper bound is inclusive, so the rows at the boundary are read again on the next page
        and dropped as already seen; the page size is grown by that many rows so they never crowd out older ones.

        :param upper_bound: The upper bound of the first page.
        :param max_rows_per_request: The API's row cap per query.
        :param fetch_page: Returns up to ``page_size`` ``(cursor timestamp, span)`` rows at or before
            ``upper_bound``, newest first.
        """
        matched_spans: list[FetchedSpan] = []
        seen_span_keys: set[tuple[str, str]] = set()
        reread_rows = 0  # rows already seen that the next request is expected to return again
        while len(matched_spans) < query.limit:
            page_size = min(max_rows_per_request, query.limit - len(matched_spans) + reread_rows)
            page_rows = fetch_page(page_size, upper_bound)
            new_rows = [(stamp, span) for stamp, span in page_rows if self._span_key(span) not in seen_span_keys]
            seen_span_keys.update(self._span_key(span) for _, span in new_rows)
            for _, span in new_rows:
                # both callers filter by session server side, so spans without their own id belong to it
                span.session_id = span.session_id or query.session_id
                if query.matches(span):
                    matched_spans.append(span)
                    if len(matched_spans) >= query.limit:
                        return matched_spans
            if len(page_rows) < page_size:
                break  # the window is exhausted
            if not new_rows:
                # more already-seen rows came back than expected (e.g. CloudWatch re-reads the whole boundary
                # second): retry with a page big enough to reach past them
                if page_size < max_rows_per_request:
                    reread_rows = page_size
                    continue
                self._log.warning("More than %d spans share one timestamp; stopping the %s fetch early", page_size, self.provider)
                break
            new_row_times = [stamp for stamp, _ in new_rows if stamp is not None]
            if not new_row_times:
                break
            upper_bound = min(new_row_times)
            reread_rows = sum(1 for stamp, _ in page_rows if stamp == upper_bound)
        return matched_spans
