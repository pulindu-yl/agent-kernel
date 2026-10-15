"""Live check against the deployed example: sends the questions, then polls GET /fetch until the spans come back.

Not part of CI. Run with AK_TEST_ENDPOINT set to the agent_invoke_url Terraform output.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict

import httpx
import pytest
import pytest_asyncio

_log = logging.getLogger("ak.example.traceloop_fetch.lambda_test")

pytestmark = pytest.mark.asyncio(loop_scope="session")  # uses a single session for all tests


class TraceFetchClient:
    """Calls the deployed /chat and /fetch routes."""

    QUESTIONS = (
        "Where is my order A-1001?",  # hand-off + tool call
        "Write me a poem about the ocean.",  # off-topic: declined, no tool
    )
    TIMEOUT = 30.0
    CHAT_ATTEMPTS = 3
    RETRY_WAIT = 5.0
    INGESTION_TIMEOUT = 300  # s: Traceloop doesn't document how soon spans become readable
    POLL_INTERVAL = 15  # s between /fetch calls
    MAX_TEXT = 120  # characters of input/output printed per span

    def __init__(self, chat_url: str, session_id: str):
        chat_url = chat_url.rstrip("/")
        if not chat_url.endswith("/chat"):
            raise ValueError(f"expected the agent's /chat URL, got {chat_url}")
        self.chat_url = chat_url
        self.fetch_url = chat_url.removesuffix("/chat") + "/fetch"
        self.session_id = session_id

    async def send_questions(self) -> list[str]:
        """Posts each question as its own request (so its own trace) and returns the replies."""
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            return [await self._send(client, question) for question in self.QUESTIONS]

    async def _send(self, client: httpx.AsyncClient, question: str) -> str:
        """Posts one question, retrying 5xx and timeouts: a cold start can outlast API Gateway's 29 s."""
        payload = {"prompt": question, "session_id": self.session_id, "agent": "triage"}
        for attempt in range(1, self.CHAT_ATTEMPTS + 1):
            try:
                response = await client.post(self.chat_url, json=payload)
            except httpx.TimeoutException:
                if attempt == self.CHAT_ATTEMPTS:
                    raise
                await asyncio.sleep(self.RETRY_WAIT)
                continue
            if response.status_code < 500 or attempt == self.CHAT_ATTEMPTS:
                break
            await asyncio.sleep(self.RETRY_WAIT)
        response.raise_for_status()
        return response.json().get("result", "")

    async def fetch(self, **params) -> dict:
        """Calls GET /fetch for the session with extra query parameters, and returns the body."""
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            response = await client.get(self.fetch_url, params={"session_id": self.session_id, **params})
        assert response.status_code == 200, f"/fetch returned {response.status_code}: {response.text}"
        return response.json()

    async def wait_for_spans(self) -> dict:
        """Polls /fetch until it returns spans and their count stops growing, or the timeout passes."""
        deadline = time.monotonic() + self.INGESTION_TIMEOUT
        previous_count = -1
        while True:
            body = await self.fetch()
            _log.info("/fetch: %d spans in %d traces", body["span_count"], body["trace_count"])
            # stable and non-empty: ingestion of this session has most likely finished
            if body["span_count"] and body["span_count"] == previous_count:
                return body
            if time.monotonic() >= deadline:
                return body
            previous_count = body["span_count"]
            await asyncio.sleep(self.POLL_INTERVAL)

    @classmethod
    def format_spans(cls, spans: list[dict]) -> str:
        """Writes the spans as one indented tree per trace, with each span's kind, name, I/O and tokens."""
        by_trace: dict[str, list[dict]] = defaultdict(list)
        for span in spans:
            by_trace[span["trace_id"]].append(span)
        lines = []
        for trace_id, trace_spans in by_trace.items():
            span_ids = {span["span_id"] for span in trace_spans}
            children: dict[str | None, list[dict]] = defaultdict(list)
            for span in sorted(trace_spans, key=lambda span: span.get("start_time") or ""):
                # a span whose parent wasn't fetched is shown at the top level
                parent = span.get("parent_span_id") if span.get("parent_span_id") in span_ids else None
                children[parent].append(span)
            lines.append(f"trace {trace_id} ({len(trace_spans)} spans, {len(children[None])} at the top level)")
            cls._format_children(children, None, 1, lines)
        return "\n".join(lines)

    @classmethod
    def _format_children(cls, children: dict, parent: str | None, depth: int, lines: list[str]) -> None:
        """Appends the spans under ``parent``, depth first."""
        for span in children.get(parent, []):
            details = [f"[{span['kind']}] {span['name']}"]
            if span.get("input") is not None:
                details.append(f"in={cls._short(span['input'])}")
            if span.get("output") is not None:
                details.append(f"out={cls._short(span['output'])}")
            if span.get("usage"):
                details.append(f"tokens={span['usage'].get('total_tokens')}")
            lines.append("  " * depth + " ".join(details))
            cls._format_children(children, span["span_id"], depth + 1, lines)

    @classmethod
    def _short(cls, value) -> str:
        """The value as one line of at most MAX_TEXT characters."""
        text = value if isinstance(value, str) else json.dumps(value)
        text = " ".join(text.split())
        return json.dumps(text if len(text) <= cls.MAX_TEXT else text[: cls.MAX_TEXT] + "…")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def client():
    """Sends the questions on a fresh session; skips when AK_TEST_ENDPOINT is unset."""
    endpoint = os.getenv("AK_TEST_ENDPOINT")
    if not endpoint:
        pytest.skip("AK_TEST_ENDPOINT is not set; deploy the example first")
    client = TraceFetchClient(endpoint, session_id=f"traceloop-fetch-{uuid.uuid4()}")
    replies = await client.send_questions()
    for question, reply in zip(TraceFetchClient.QUESTIONS, replies):
        _log.info("> %s\n%s", question, reply)
    assert all(replies), f"expected a reply to every question, got {replies}"
    yield client


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def session_body(client):
    """The /fetch body for the session, once Traceloop returns its spans."""
    yield await client.wait_for_spans()


async def test_session_spans_are_fetched(client, session_body):
    """/fetch returns the session's spans; they are printed as one tree per trace."""
    assert session_body["span_count"], (
        f"no spans for session {client.session_id} after {TraceFetchClient.INGESTION_TIMEOUT} s"
    )
    _log.info("session %s:\n%s", client.session_id, TraceFetchClient.format_spans(session_body["spans"]))
    spans = session_body["spans"]
    assert {span["session_id"] for span in spans} == {client.session_id}
    assert {span["provider"] for span in spans} == {"openllmetry"}


async def test_fetch_by_trace_id(client, session_body):
    """/fetch?trace_id=... returns only that trace's spans."""
    if not session_body["span_count"]:
        pytest.skip("the session fetch returned nothing")
    trace_id = session_body["spans"][0]["trace_id"]
    body = await client.fetch(trace_id=trace_id)
    _log.info("trace %s:\n%s", trace_id, TraceFetchClient.format_spans(body["spans"]))
    assert body["span_count"] and {span["trace_id"] for span in body["spans"]} == {trace_id}


async def test_fetch_tool_spans(client, session_body):
    """/fetch?kind=tool returns the get_order_status call (kinds are filtered after the spans arrive)."""
    if not session_body["span_count"]:
        pytest.skip("the session fetch returned nothing")
    body = await client.fetch(kind="tool")
    _log.info("tool spans:\n%s", TraceFetchClient.format_spans(body["spans"]))
    assert body["span_count"] and {span["kind"] for span in body["spans"]} == {"tool"}
