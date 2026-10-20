"""Claude trace wrappers keep an active span while resumed and streaming runs execute."""

from contextlib import contextmanager, nullcontext

import pytest

from agentkernel.core import Session
from agentkernel.core.model import AgentReplyText
from agentkernel.framework.claude.claude import ClaudeRunner
from agentkernel.trace.cloudwatch import claude as cloudwatch_claude
from agentkernel.trace.langfuse import claude as langfuse_claude
from agentkernel.trace.logfire import claude as logfire_claude
from agentkernel.trace.openllmetry import claude as openllmetry_claude


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["langfuse", "logfire", "cloudwatch", "openllmetry"])
async def test_trace_context_covers_resume_and_stream_iteration(monkeypatch, backend):
    active = False

    class FakeSpan:
        def update(self, **kwargs):
            pass

        def set_attribute(self, key, value):
            pass

    @contextmanager
    def span(*args, **kwargs):
        nonlocal active
        assert not active
        active = True
        try:
            yield FakeSpan()
        finally:
            active = False

    class FakeTracer:
        def start_as_current_observation(self, *args, **kwargs):
            return span()

        def span(self, *args, **kwargs):
            return span()

    if backend == "langfuse":
        runner = object.__new__(langfuse_claude.LangFuseClaudeRunner)
        runner._client = FakeTracer()
        monkeypatch.setattr(langfuse_claude, "propagate_attributes", lambda **kwargs: nullcontext())
    elif backend == "logfire":
        runner = object.__new__(logfire_claude.LogfireClaudeRunner)
        monkeypatch.setattr(logfire_claude.logfire, "span", span)
    elif backend == "cloudwatch":
        runner = object.__new__(cloudwatch_claude.CloudWatchClaudeRunner)
        runner._tracer = FakeTracer()
    else:
        runner = object.__new__(openllmetry_claude.OpenLLMetryClaudeRunner)
        monkeypatch.setattr(openllmetry_claude, "TraceloopContext", span)

    async def fake_resume(self, agent, session, requests, decisions, record):
        assert active
        return AgentReplyText(response="done")

    async def fake_stream(self, agent, session, requests):
        assert active
        yield object()

    async def fake_resume_stream(self, agent, session, requests, decisions, record):
        assert active
        yield object()

    monkeypatch.setattr(ClaudeRunner, "resume", fake_resume)
    monkeypatch.setattr(ClaudeRunner, "stream", fake_stream)
    monkeypatch.setattr(ClaudeRunner, "resume_stream", fake_resume_stream)

    session = Session("s")
    assert (await runner.resume(None, session, [], [], None)).response == "done"
    assert not active
    assert len([event async for event in runner.stream(None, session, [])]) == 1
    assert not active
    assert len([event async for event in runner.resume_stream(None, session, [], [], None)]) == 1
    assert not active
