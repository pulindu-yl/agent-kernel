"""
Tests for the tracing backends' Claude runners (trace/{langfuse,logfire,cloudwatch,openllmetry}/claude.py).

Like every framework's traced runner, each wraps `run()` in the backend's span. The Claude runner's own `run` is
replaced, so no SDK client is involved, and the backend's span is a probe that records whether it is open and what was
recorded on it.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentkernel.core.base import Session
from agentkernel.core.model import AgentReplyText, AgentRequestText
from agentkernel.framework.claude.claude import ClaudeRunner


class Probe:
    """Records whether a backend's span is open, and what the run recorded on it."""

    def __init__(self):
        self.open = False
        self.entered = 0
        self.attributes = {}

    @contextmanager
    def span(self):
        self.open = True
        self.entered += 1
        try:
            yield self
        finally:
            self.open = False

    # the span objects the backends record on
    def set_attribute(self, key, value):
        self.attributes[key] = value

    def update(self, **fields):
        self.attributes.update(fields)


@contextmanager
def _nothing(**kwargs):
    yield


def _langfuse(probe):
    import agentkernel.trace.langfuse.claude as module

    client = MagicMock()
    client.start_as_current_observation.side_effect = lambda **kwargs: probe.span()
    with patch.object(module, "ClaudeAgentSDKInstrumentor"):
        runner = module.LangFuseClaudeRunner(client)
    return runner, [patch.object(module, "propagate_attributes", _nothing)]


def _logfire(probe):
    import logfire

    import agentkernel.trace.logfire.claude as module

    with patch.object(logfire, "instrument_claude_agent_sdk"):
        runner = module.LogfireClaudeRunner()
    return runner, [patch.object(logfire, "span", lambda name, **attributes: probe.span())]


def _cloudwatch(probe):
    import agentkernel.trace.cloudwatch.claude as module

    tracer = MagicMock()
    tracer.span.side_effect = lambda name, session: probe.span()
    with patch.object(module, "ClaudeAgentSDKInstrumentor"):
        runner = module.CloudWatchClaudeRunner(tracer)
    return runner, []


def _openllmetry(probe):
    import agentkernel.trace.openllmetry.claude as module

    otel = MagicMock()
    otel.get_tracer.return_value.start_as_current_span.side_effect = lambda name, **kwargs: probe.span()
    return module.OpenLLMetryClaudeRunner(), [patch.object(module, "trace_api", otel), patch.object(module, "TraceloopContext", _nothing)]


BACKENDS = {"langfuse": _langfuse, "logfire": _logfire, "cloudwatch": _cloudwatch, "openllmetry": _openllmetry}


@pytest.fixture(params=sorted(BACKENDS))
def traced(request):
    """(runner, probe) for one backend, with the backend's own span replaced by the probe."""
    probe = Probe()
    runner, patches = BACKENDS[request.param](probe)
    for active in patches:
        active.start()
    yield runner, probe
    for active in patches:
        active.stop()


@pytest.mark.asyncio
async def test_run_is_wrapped_in_the_span_and_records_the_reply(traced):
    runner, probe = traced
    seen_open = []

    async def run(self, agent, session, requests):
        seen_open.append(probe.open)
        return AgentReplyText(response="answer", prompt="hi")

    with patch.object(ClaudeRunner, "run", run):
        reply = await runner.run(object(), Session("s"), [AgentRequestText(prompt="hi")])

    assert str(reply) == "answer"
    assert seen_open == [True] and probe.entered == 1 and not probe.open
    assert "answer" in probe.attributes.values() and "hi" in probe.attributes.values()


@pytest.mark.asyncio
async def test_openllmetry_creates_a_span_of_its_own():
    """TraceloopContext only attaches properties to existing spans; without a span of the runner's own, a Claude run is invisible."""
    import agentkernel.trace.openllmetry.claude as module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    async def run(self, agent, session, requests):
        return AgentReplyText(response="answer", prompt="hi")

    with patch.object(module.trace_api, "get_tracer", lambda name: provider.get_tracer(name)), patch.object(module, "TraceloopContext", _nothing):
        with patch.object(ClaudeRunner, "run", run):
            await module.OpenLLMetryClaudeRunner().run(object(), Session("s"), [AgentRequestText(prompt="hi")])

    (span,) = exporter.get_finished_spans()
    assert span.name == "Agent Kernel Claude"
    assert span.attributes["session_id"] == "s"
    assert span.attributes["traceloop.entity.input"] == "hi" and span.attributes["traceloop.entity.output"] == "answer"
