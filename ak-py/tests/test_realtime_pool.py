import asyncio
import base64
import json
import logging
import re
import threading
import time
import types

import pytest

from agentkernel import Agent, Runner, Session
from agentkernel.core.event import TextDelta
from agentkernel.core.model import AgentReplyText, ExecutionMode, StreamChunk
from agentkernel.core.realtime import RealtimeRunner
from agentkernel.core.runtime import Runtime
from agentkernel.core.session.in_memory import InMemorySessionStore
from agentkernel.core.tool import ToolContext
from agentkernel.pipeline import realtime_pool
from agentkernel.pipeline.envelope import ATTR_INTEGRATION, ATTR_REALTIME, QueueName
from agentkernel.pipeline.realtime_pool import RealtimeConnection, RealtimeConnectionPool
from agentkernel.pipeline.thread_runner import ThreadRunner
from agentkernel.pipeline.transport.base import QueueTransportFactory
from agentkernel.pipeline.transport.in_memory import InMemoryTransport


@pytest.fixture(autouse=True)
def _reset_pipeline_state(monkeypatch):
    monkeypatch.setattr(RealtimeConnectionPool, "_observers", [])
    InMemoryTransport.reset()
    ThreadRunner.shutdown_event.clear()
    yield
    InMemoryTransport.reset()
    RealtimeConnectionPool.reset()
    ThreadRunner.shutdown_event.clear()


class _Runner(Runner):
    def __init__(self):
        super().__init__("test-runner")

    async def run(self, agent, session, requests):
        return AgentReplyText(response="ok")

    async def stream(self, agent, session, requests):
        raise NotImplementedError()
        yield


class _Adapter(RealtimeRunner):
    """Records every call the pool makes; emits nothing on its own."""

    def __init__(self):
        super().__init__("fake")
        self.connected = 0
        self.audio: list[str] = []
        self.texts: list[str] = []
        self.executed: list[tuple[str, str]] = []
        self.tool_results: list[tuple[str, str]] = []
        self.tool_return = "tool-ok"
        self.callback = None

    async def connect(self, session, agent, callback):
        self.connected += 1
        self.callback = callback

    async def append_audio(self, base64_audio):
        self.audio.append(base64_audio)

    async def send_text(self, text):
        self.texts.append(text)

    async def execute_tool(self, name, arguments, context, call_id):
        self.executed.append((name, arguments, call_id))
        return self.tool_return

    async def send_tool_result(self, call_id, result):
        self.tool_results.append((call_id, result))

    async def disconnect(self):
        pass


class _Agent(Agent):
    def __init__(self, name="general", realtime_runner_cls=_Adapter):
        super().__init__(name, _Runner(), realtime_runner_cls=realtime_runner_cls)

    def get_description(self):
        return "fake"

    def get_a2a_card(self):
        return None

    def attach_tool(self, tool):
        pass

    def override_system_prompt(self, prompt):
        pass


def _connection(session_id="s1", agent=None, loop=None, observers=()):
    return RealtimeConnection(
        session_id, agent or _Agent(), Runtime(InMemorySessionStore()), Session(session_id), loop or asyncio.new_event_loop(), observers=observers
    )


def _drain_output(transport):
    """Every output message, in order. The in-memory queue delivers one message per group until it
    is acked, so this drains with acks."""
    consumer = transport.create_consumer(QueueName.OUTPUT)
    drained = []
    while True:
        messages = consumer.fetch(10, 0.2)
        if not messages:
            return drained
        for message in messages:
            drained.append(message)
            consumer.ack(message)


def _kinds(messages):
    return ["done" if json.loads(m.body).get("done") else json.loads(m.body)["event"]["type"] for m in messages]


class TestRealtimeConnection:
    def test_requires_realtime_runner_cls(self):
        with pytest.raises(ValueError, match="no realtime_runner_cls"):
            _connection(agent=_Agent(realtime_runner_cls=None))

    def test_reads_configured_playback_lead(self, monkeypatch):
        """The pacing lead is config-driven so a higher-latency broker can buffer more audio
        (execution.realtime.playback_lead_ms), instead of the old hard-coded 50 ms."""

        class _Cfg:
            class execution:
                class realtime:
                    playback_lead_ms = 250

        monkeypatch.setattr("agentkernel.core.config.AKConfig.get", classmethod(lambda cls: _Cfg))
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        assert _connection()._playback_lead_ms == 250

    @pytest.mark.asyncio
    async def test_emit_stamps_realtime_and_integration(self, monkeypatch):
        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))

        conn = _connection(loop=asyncio.get_running_loop())
        conn.update_delivery_context(request_id="r1", user_id="u1", integration="livekit", reply_context={"session_id": "s1"})
        await conn._emit(StreamChunk(event=TextDelta(message_id="m1", content="QUFB"), delta="QUFB"))

        [message] = transport.create_consumer(QueueName.OUTPUT).fetch(10, 0.5)
        assert message.attributes[ATTR_REALTIME] == "true"
        assert message.attributes[ATTR_INTEGRATION] == "livekit"
        assert message.attributes["request_id"] == "r1"
        assert message.attributes["user_id"] == "u1"
        assert message.attributes["reply_session_id"] == "s1"
        assert message.group_id == "s1"
        body = json.loads(message.body)
        assert body["event"]["type"] == "text_delta"
        assert body["event"]["content"] == "QUFB"
        assert body["delta"] == "QUFB"

    @pytest.mark.asyncio
    async def test_tool_call_delegates_to_adapter(self, monkeypatch):
        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))

        conn = _connection(loop=asyncio.get_running_loop())
        conn.adapter.tool_return = "42"
        await conn._execute_tool_and_reply("call-1", "get_weather", '{"location": "Paris"}')

        assert conn.adapter.executed == [("get_weather", '{"location": "Paris"}', "call-1")]
        assert conn.adapter.tool_results == [("call-1", "42")]


class TestRealtimeAwaitedTools:
    """A tool may await long work, such as another agent's run in a session of its own: the call is busy meanwhile,
    nothing is posted into it, the result goes back through the tool call that asked, and a call closed meanwhile
    cancels the tool and is sent nothing."""

    class _SlowAdapter(_Adapter):
        """``execute_tool`` waits for ``release``, inside a nested run in a session of its own, as a tool asking
        another agent does; it records the contexts it is back in, and whether it was cancelled."""

        def __init__(self):
            super().__init__()
            self.release = asyncio.Event()
            self.started = asyncio.Event()
            self.child = Session("s1-child")
            self.back_in = None
            self.cancelled = False
            self.ignore_cancel = False

        async def execute_tool(self, name, arguments, context, call_id):
            self.executed.append((name, arguments, call_id))
            try:
                async with self.child:
                    with _Agent("child")._activate():
                        self.started.set()
                        await self.release.wait()
                return "answer"
            except asyncio.CancelledError:
                self.cancelled = True
                if not self.ignore_cancel:
                    raise
                await asyncio.sleep(0.05)
                return "late answer"
            finally:
                self.back_in = (Session.current().id, Agent.current().name)

    @pytest.fixture(autouse=True)
    def _transport(self, monkeypatch):
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        monkeypatch.setattr(RealtimeConnectionPool, "_instance", None)

    async def _call(self):
        agent = _Agent(realtime_runner_cls=self._SlowAdapter)
        conn = _connection(agent=agent, loop=asyncio.get_running_loop())
        await conn.connect()
        await asyncio.sleep(0.05)  # the pacing loop sends the starting agent on
        conn._connected_at = time.time() - 10.0
        pool = TestRealtimeCall._pool()
        pool._connections["s1"] = conn
        await conn.handle_framework_event("tool_call", {"call_id": "c1", "name": "research", "arguments": "{}"})
        await asyncio.wait_for(conn.adapter.started.wait(), 1.0)
        return pool, conn

    @pytest.mark.asyncio
    async def test_the_call_is_busy_while_the_tool_waits_and_hears_its_result_once(self):
        from agentkernel.pipeline import RealtimeCall

        pool, conn = await self._call()
        pool._loop = asyncio.get_running_loop()

        assert conn.is_responding and RealtimeCall("s1").quiet_for() is None
        assert await RealtimeCall("s1").send_if_quiet("[Team update] something else") is False
        assert conn.adapter.texts == []  # nothing sent meanwhile, and nothing kept to send later
        assert conn.adapter.tool_results == [] and conn.last_tool_result_at is None

        conn.adapter.release.set()
        await asyncio.sleep(0.05)
        assert conn.adapter.tool_results == [("c1", "answer")]
        assert conn.adapter.back_in == ("s1", "general")  # back in the call's session and agent after the nested run
        assert not conn.session._lock.locked() and not conn.adapter.child._lock.locked()
        assert not conn.is_responding and conn.last_tool_result_at is not None
        await conn.close()

    @pytest.mark.asyncio
    async def test_closing_the_call_cancels_the_waiting_tool_and_sends_it_nothing(self):
        pool, conn = await self._call()

        await conn.close()

        assert conn.adapter.cancelled and conn.adapter.tool_results == []
        assert conn.adapter.back_in == ("s1", "general")
        assert not conn.session._lock.locked() and not conn.adapter.child._lock.locked()
        assert conn._tool_tasks == set() and conn._tools_in_flight == 0 and conn.last_tool_result_at is None
        assert Session.current() is None and Agent.current() is None  # nothing leaked into the pool's loop

    @pytest.mark.asyncio
    async def test_closing_the_call_through_realtime_call_cancels_the_tool(self):
        from agentkernel.pipeline import RealtimeCall

        pool, conn = await self._call()
        pool._loop = asyncio.get_running_loop()

        assert RealtimeCall("s1").close() is True
        await asyncio.sleep(0.1)

        assert conn.adapter.cancelled and conn.adapter.tool_results == [] and conn._tool_tasks == set()

    @pytest.mark.asyncio
    async def test_a_tool_that_ignores_its_cancellation_never_answers_the_closed_call(self, monkeypatch):
        monkeypatch.setattr(realtime_pool, "_TOOL_CANCEL_SECONDS", 0.01)
        pool, conn = await self._call()
        conn.adapter.ignore_cancel = True

        await conn.close()
        await asyncio.sleep(0.1)  # it returns its result after all

        assert conn.adapter.cancelled and conn.adapter.tool_results == [] and conn._tools_in_flight == 0

    @pytest.mark.asyncio
    async def test_a_result_that_comes_after_the_socket_failed_is_not_sent(self):
        pool, conn = await self._call()

        await conn.handle_framework_event("error", {"message": "socket lost"})
        conn.adapter.release.set()
        await asyncio.sleep(0.05)

        assert conn.closed and conn.adapter.tool_results == []
        await conn.close()


class TestRealtimeConnectionPool:
    def test_get_or_create_reuses_one_connection_per_session(self, monkeypatch):
        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))

        pool = RealtimeConnectionPool.initialize()
        thread = threading.Thread(target=pool.start, daemon=True)
        thread.start()
        try:
            runtime = Runtime(InMemorySessionStore())
            agent = _Agent()
            first = pool.get_or_create("s1", agent, runtime, Session("s1"))
            second = pool.get_or_create("s1", agent, runtime, Session("s1"))
            other = pool.get_or_create("s2", agent, runtime, Session("s2"))

            assert first is second
            assert other is not first
            assert first.adapter.connected == 1
            assert other.adapter.connected == 1
        finally:
            ThreadRunner.shutdown_event.set()
            thread.join(timeout=5)


class TestRealtimePoolShutdown:
    """Shutdown closes every connection once, opens none, and never leaves a caller waiting on a dead loop."""

    class _HangingAdapter(_Adapter):
        """Never finishes connecting, like a socket handshake still in flight at shutdown."""

        entered = threading.Event()

        async def connect(self, session, agent, callback):
            type(self).entered.set()
            await asyncio.Event().wait()

    class _CountingAdapter(_Adapter):
        def __init__(self):
            super().__init__()
            self.disconnects = 0

        async def disconnect(self):
            self.disconnects += 1
            await asyncio.sleep(0)

    def test_no_connection_is_opened_once_shutdown_starts(self):
        pool = RealtimeConnectionPool.initialize()
        pool._loop_ready.set()
        ThreadRunner.shutdown_event.set()

        with pytest.raises(RuntimeError, match="shutting down"):
            pool.get_or_create("s1", _Agent(), Runtime(InMemorySessionStore()), Session("s1"))

    def test_a_connect_in_flight_at_shutdown_fails_fast(self, monkeypatch):
        """A consumer waiting on a connect gets an error when the loop stops, not the 15 s timeout."""
        import concurrent.futures

        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        monkeypatch.setattr(realtime_pool, "_SHUTDOWN_GRACE_SECONDS", 0.1)
        self._HangingAdapter.entered.clear()

        pool = RealtimeConnectionPool.initialize()
        pool_thread = threading.Thread(target=pool.start, daemon=True)
        pool_thread.start()

        outcome = {}

        def connect():
            try:
                pool.get_or_create("s1", _Agent(realtime_runner_cls=self._HangingAdapter), Runtime(InMemorySessionStore()), Session("s1"))
            except BaseException as e:
                outcome["error"] = e

        caller = threading.Thread(target=connect, daemon=True)
        caller.start()
        assert self._HangingAdapter.entered.wait(timeout=5)

        ThreadRunner.shutdown_event.set()
        caller.join(timeout=5)
        pool_thread.join(timeout=5)

        assert not caller.is_alive()
        assert not pool_thread.is_alive()
        assert isinstance(outcome.get("error"), concurrent.futures.CancelledError)

    @pytest.mark.asyncio
    async def test_closing_twice_disconnects_once(self, monkeypatch):
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        conn = _connection(agent=_Agent(realtime_runner_cls=self._CountingAdapter), loop=asyncio.get_running_loop())

        await asyncio.gather(conn.close(), conn.close())

        assert conn.adapter.disconnects == 1

    def test_the_runner_drops_input_during_shutdown(self, monkeypatch):
        from unittest.mock import MagicMock

        from agentkernel.pipeline.agent_runner import RealtimeAgentRunner
        from agentkernel.pipeline.envelope import QueueMessage

        transport = InMemoryTransport()
        chat_service = MagicMock()
        runner = RealtimeAgentRunner(transport=transport, chat_service=chat_service)
        ThreadRunner.shutdown_event.set()

        body = json.dumps({"prompt": "", "session_id": "s1", "requests": [{"type": "text", "prompt": "hi"}]})
        runner.process(QueueMessage(body=body, attributes={"request_id": "r1", "integration": "livekit"}, group_id="s1", dedup_id="d1"))

        chat_service.prepare_agent_handler.assert_not_called()
        assert RealtimeConnectionPool.get()._connections == {}


class TestRealtimePoolSweep:
    """The pool's loop releases connections nobody will send to again, and reports when it was blocked."""

    @staticmethod
    def _start(monkeypatch):
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        pool = RealtimeConnectionPool.initialize()
        thread = threading.Thread(target=pool.start, daemon=True)
        thread.start()
        assert pool._loop_ready.wait(timeout=5)
        return pool, thread

    @staticmethod
    def _until(condition, seconds=3.0):
        deadline = time.monotonic() + seconds
        while not condition():
            assert time.monotonic() < deadline, "timed out"
            time.sleep(0.02)

    def test_a_failed_connection_is_released_without_further_input(self, monkeypatch):
        """Before, a failed connection was released only by its session's next input: a caller who had left
        never sent one, and the adapter stayed connected until the process stopped."""
        monkeypatch.setattr(realtime_pool, "_SWEEP_SECONDS", 0.05)
        pool, thread = self._start(monkeypatch)
        agent = _Agent(realtime_runner_cls=TestRealtimePoolShutdown._CountingAdapter)
        conn = pool.get_or_create("s1", agent, Runtime(InMemorySessionStore()), Session("s1"))

        asyncio.run_coroutine_threadsafe(conn.handle_framework_event("error", {"message": "socket closed"}), pool._loop).result(timeout=5)
        self._until(lambda: "s1" not in pool._connections and conn.adapter.disconnects == 1)

        ThreadRunner.shutdown_event.set()
        thread.join(timeout=5)

    def test_the_sweep_closes_the_failed_connection_it_found_not_its_replacement(self):
        """The sweep let go of its lock between finding a failed connection and removing it; a consumer that replaced
        the connection meanwhile had its replacement removed and closed."""
        pool = RealtimeConnectionPool()
        failed = types.SimpleNamespace(closed=True, last_activity_at=time.time())
        replacement = types.SimpleNamespace(closed=False, last_activity_at=time.time())
        pool._connections["s1"] = failed

        class ConsumerReplacesItWhenTheLockIsFree:
            replaced = False

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                if not self.replaced:
                    self.replaced = True
                    pool._connections["s1"] = replacement

        pool._conn_lock = ConsumerReplacesItWhenTheLockIsFree()
        evicted = pool._evict_stale(time.time())

        assert [(session_id, conn) for session_id, conn, _ in evicted] == [("s1", failed)]
        assert pool._connections["s1"] is replacement

    def test_a_blocked_loop_is_reported(self, monkeypatch, caplog):
        monkeypatch.setattr(realtime_pool, "_LOOP_LAG_WARNING_SECONDS", 0.1)
        pool, thread = self._start(monkeypatch)

        # The loop is ready before it runs its watch: block it only once the watch is under way, or the block comes
        # before anything is measured (seen under coverage, which slows the loop's start).
        asyncio.run_coroutine_threadsafe(asyncio.sleep(realtime_pool._TICK_SECONDS), pool._loop).result(timeout=5)
        with caplog.at_level(logging.WARNING, logger="ak.pipeline.realtime_pool"):
            pool._loop.call_soon_threadsafe(time.sleep, 0.4)
            self._until(lambda: "The realtime loop was blocked" in caplog.text)

        ThreadRunner.shutdown_event.set()
        thread.join(timeout=5)


class TestRealtimeRunnerReuse:
    def test_audio_chunks_reuse_connection_without_re_resolving(self, monkeypatch):
        from unittest.mock import MagicMock

        from agentkernel.pipeline.agent_runner import RealtimeAgentRunner
        from agentkernel.pipeline.envelope import QueueMessage

        class _Cfg:
            class execution:
                mode = ExecutionMode.REALTIME
                queues = None

                class realtime:
                    playback_lead_ms = 50

        monkeypatch.setattr("agentkernel.core.config.AKConfig.get", classmethod(lambda cls: _Cfg))

        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))

        handler = types.SimpleNamespace(service=types.SimpleNamespace(agent=_Agent(), runtime=Runtime(InMemorySessionStore()), session=Session("s1")))
        chat_service = MagicMock()
        chat_service.prepare_agent_handler.return_value = handler
        chat_service.process_chat_request.return_value = (200, {})

        pool = RealtimeConnectionPool.initialize()
        thread = threading.Thread(target=pool.start, daemon=True)
        thread.start()
        try:
            runner = RealtimeAgentRunner(transport=transport, chat_service=chat_service)
            body = json.dumps({"prompt": "", "session_id": "s1", "requests": [{"type": "text", "prompt": "hi"}]})
            for i in range(3):
                runner.process(QueueMessage(body=body, attributes={"request_id": f"r{i}", "integration": "livekit"}, group_id="s1", dedup_id=f"d{i}"))

            # Three chunks, one connection: the agent/session is resolved once, not per chunk.
            assert chat_service.prepare_agent_handler.call_count == 1
        finally:
            ThreadRunner.shutdown_event.set()
            thread.join(timeout=5)


class TestFrameworkToolExecution:
    """The adapter must hand each framework the context shape it expects, or tools that read
    their context fail (ADK injects a ``ToolContext``; the OpenAI SDK needs its own)."""

    @pytest.mark.asyncio
    async def test_openai_execute_tool_injects_context(self):
        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner, OpenAIToolBuilder

        def get_weather(location: str) -> str:
            """Get the weather."""
            context = ToolContext.get()
            return f"{location}:{context.session.id if context else 'no-context'}"

        [tool] = OpenAIToolBuilder.bind([get_weather])
        agent = types.SimpleNamespace(agent=types.SimpleNamespace(tools=[tool]), name="general")
        session = Session("s1")
        runtime = Runtime(InMemorySessionStore())

        adapter = OpenAIRealtimeRunner()
        adapter._agent = agent
        adapter._tools = {tool.name: tool}

        result = await adapter.execute_tool("get_weather", '{"location": "Paris"}', ToolContext(runtime, agent, session, []), "call-1")
        assert result == "Paris:s1"

    @pytest.mark.asyncio
    async def test_openai_execute_tool_passes_framework_context_as_run_context(self):
        """Parity with the unary path: ``wrapper.context`` is the session's framework_context, not the AK ToolContext."""
        from agents import RunContextWrapper, function_tool

        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        @function_tool
        def get_cart(wrapper: RunContextWrapper) -> str:
            """Read the cart."""
            return ",".join(wrapper.context["cart"])

        session = Session("s1")
        session.set_framework_context({"cart": ["apple", "pear"]})
        agent = types.SimpleNamespace(agent=types.SimpleNamespace(tools=[get_cart]), name="general")

        adapter = OpenAIRealtimeRunner()
        adapter._agent = agent
        adapter._tools = {get_cart.name: get_cart}

        result = await adapter.execute_tool("get_cart", "{}", ToolContext(Runtime(InMemorySessionStore()), agent, session, []), "call-1")
        assert result == "apple,pear"


class TestRealtimeResampling:
    @pytest.mark.asyncio
    async def test_append_audio_resamples_edge_rate_to_model_rate(self, monkeypatch):
        import base64

        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))
        conn = _connection(loop=asyncio.get_running_loop())
        conn.adapter.input_sample_rate = 16000  # e.g. Gemini's input rate

        # 100 ms of edge-rate (24 kHz) audio; the pool must hand the adapter 16 kHz audio.
        conn.append_audio(base64.b64encode(b"\x00\x00" * 2400).decode())
        for _ in range(50):  # the append is scheduled on the loop; let it run
            if conn.adapter.audio:
                break
            await asyncio.sleep(0.01)

        [converted] = conn.adapter.audio
        assert len(base64.b64decode(converted)) == 1600 * 2  # 100 ms at 16 kHz


class TestRealtimePacing:
    """The connection paces a turn's audio and emits its terminal ``done`` behind it."""

    @pytest.mark.asyncio
    async def test_done_follows_its_audio(self, monkeypatch):
        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))

        conn = _connection(loop=asyncio.get_running_loop())
        await conn.connect()  # starts the connection's shared pacing loop

        delta = "QUFB"  # 4 bytes -> 2 samples -> a tiny duration, so pacing does not sleep
        for _ in range(3):
            await conn.handle_framework_event("audio_delta", {"delta": delta, "message_id": "m1"})
        await conn.handle_framework_event("done", {"status": "completed"})
        await asyncio.sleep(0.2)
        conn._pacing_task.cancel()

        kinds = _kinds(_drain_output(transport))

        # All three audio deltas are emitted, and the terminal event is always the done chunk
        # (it never overtakes its audio).
        assert kinds.count("audio_delta") == 3
        assert kinds[-1] == "done"


class TestRealtimeBargeIn:
    """Barge-in against the pool's paced audio: ``speech_started`` interrupts only while audio is queued."""

    @staticmethod
    def _queued(conn):
        return [item[0] for item in list(conn._audio_queue._queue)]

    @pytest.mark.asyncio
    async def test_speech_started_with_no_queued_audio_is_ignored(self, monkeypatch):
        """The ordinary start of a user's turn must not reach the edge as an interrupt, or it would
        clear playback and drop the transcript of the response that follows."""
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        conn = _connection(loop=asyncio.get_running_loop())

        await conn.handle_framework_event("speech_started", {})

        assert self._queued(conn) == []

    @pytest.mark.asyncio
    async def test_speech_started_over_queued_audio_interrupts_and_keeps_done(self, monkeypatch):
        """The model finished generating but its paced audio is still queued: talking over it is a
        barge-in, and the turn's done still follows so the edge resets its interrupted state."""
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        conn = _connection(loop=asyncio.get_running_loop())

        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        await conn.handle_framework_event("done", {"status": "completed"})
        await conn.handle_framework_event("speech_started", {})

        assert self._queued(conn) == ["interrupt", "done"]
        assert conn._queued_audio == 0

    @pytest.mark.asyncio
    async def test_interrupt_keeps_a_queued_done_behind_it(self, monkeypatch):
        """Regression: the drain used to drop a queued done too, leaving the edge's interrupted state
        set into the next turn, which then lost its transcript."""
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        conn = _connection(loop=asyncio.get_running_loop())

        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        await conn.handle_framework_event("done", {"status": "completed"})
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        await conn.handle_framework_event("interrupt", {})

        assert self._queued(conn) == ["interrupt", "done"]

    @pytest.mark.asyncio
    async def test_emitted_audio_no_longer_counts_as_queued(self, monkeypatch):
        """Once the pacing loop has emitted a turn's audio, a later speech_started is a new turn."""
        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))
        conn = _connection(loop=asyncio.get_running_loop())
        await conn.connect()

        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        await conn.handle_framework_event("done", {"status": "completed"})
        await asyncio.sleep(0.1)
        await conn.handle_framework_event("speech_started", {})
        await asyncio.sleep(0.1)
        conn._pacing_task.cancel()

        assert _kinds(_drain_output(transport)) == ["agent_changed", "audio_delta", "done"]


class TestRealtimeTurnState:
    """Read-only turn state lets code that talks into a call tell a quiet call from a busy one."""

    class _BlockingAdapter(_Adapter):
        """A tool that runs until the test releases it, or raises when told to."""

        def __init__(self):
            super().__init__()
            self.release = asyncio.Event()
            self.fail = False

        async def execute_tool(self, name, arguments, context, call_id):
            await self.release.wait()
            if self.fail:
                raise RuntimeError("tool broke")
            return "done"

    @pytest.fixture(autouse=True)
    def _transport(self, monkeypatch):
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))

    @pytest.mark.asyncio
    async def test_a_new_connection_is_quiet(self):
        conn = _connection(loop=asyncio.get_running_loop())

        assert conn.is_responding is False
        assert conn.last_model_output_at is None
        assert conn.last_tool_result_at is None
        assert conn.last_model_generated_at is None
        assert conn.last_agent_change_at is None
        assert conn.last_caller_speech_at is None
        assert conn.last_edge_text_at is None

    @pytest.mark.asyncio
    async def test_queued_output_is_a_response_until_it_reaches_the_edge(self):
        conn = _connection(loop=asyncio.get_running_loop())
        await conn.connect()
        await asyncio.sleep(0.1)
        assert conn.is_responding is False

        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        await conn.handle_framework_event("done", {"status": "completed"})
        assert conn.is_responding is True

        await asyncio.sleep(0.2)
        conn._pacing_task.cancel()
        assert conn.is_responding is False
        assert conn.last_model_output_at is not None

    @pytest.mark.asyncio
    async def test_control_events_are_not_model_output(self):
        """An interrupt is the caller's doing, so it must not read as the model answering them."""
        conn = _connection(loop=asyncio.get_running_loop())
        await conn.connect()
        await conn.handle_framework_event("interrupt", {})
        await conn.handle_framework_event("done", {"status": "completed"})
        await asyncio.sleep(0.1)
        assert conn.last_model_output_at is None

        await conn.handle_framework_event("transcript_delta", {"delta": "Hi", "message_id": "m1"})
        await asyncio.sleep(0.1)
        conn._pacing_task.cancel()
        assert conn.last_model_output_at is not None

    @pytest.mark.asyncio
    async def test_a_barge_in_ends_the_response_once_its_kept_events_are_out(self):
        conn = _connection(loop=asyncio.get_running_loop())
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        await conn.handle_framework_event("done", {"status": "completed"})
        await conn.handle_framework_event("interrupt", {})
        assert conn.is_responding is True

        await conn.connect()
        await asyncio.sleep(0.2)
        conn._pacing_task.cancel()
        assert conn.is_responding is False

    @pytest.mark.asyncio
    async def test_a_pause_mid_answer_is_still_a_response_until_its_done(self):
        """The model's paced audio can run out while it is still answering: only ``done`` ends the response."""
        conn = _connection(loop=asyncio.get_running_loop())
        await conn.connect()
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        await asyncio.sleep(0.2)
        assert conn._audio_queue.qsize() == 0
        assert conn.is_responding is True

        await conn.handle_framework_event("done", {"status": "completed"})
        await asyncio.sleep(0.1)
        conn._pacing_task.cancel()
        assert conn.is_responding is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ending", ["interrupt", "speech_started"])
    async def test_the_caller_speaking_ends_an_open_response(self, ending):
        conn = _connection(loop=asyncio.get_running_loop())
        await conn.connect()
        await conn.handle_framework_event("transcript_delta", {"delta": "So the plan", "message_id": "m1"})
        await asyncio.sleep(0.1)
        assert conn.is_responding is True

        await conn.handle_framework_event(ending, {})
        await asyncio.sleep(0.1)
        conn._pacing_task.cancel()
        assert conn.is_responding is False

    @pytest.mark.asyncio
    async def test_a_response_whose_done_never_comes_stops_holding_the_call(self, monkeypatch):
        monkeypatch.setattr(realtime_pool, "_UNFINISHED_RESPONSE_SECONDS", 0.05)
        conn = _connection(loop=asyncio.get_running_loop())
        await conn.connect()
        await conn.handle_framework_event("transcript_delta", {"delta": "So the plan", "message_id": "m1"})
        await asyncio.sleep(0.15)
        conn._pacing_task.cancel()
        assert conn.is_responding is False

    @pytest.mark.asyncio
    async def test_a_tool_call_is_a_response_until_its_result_is_sent(self):
        conn = _connection(agent=_Agent(realtime_runner_cls=self._BlockingAdapter), loop=asyncio.get_running_loop())

        await conn.handle_framework_event("tool_call", {"call_id": "c1", "name": "lookup", "arguments": "{}"})
        await asyncio.sleep(0.05)
        assert conn.is_responding is True
        assert conn.adapter.tool_results == []

        conn.adapter.release.set()
        await asyncio.sleep(0.05)
        assert conn.adapter.tool_results == [("c1", "done")]
        assert conn.is_responding is False
        # The result is input the model has still to answer, not output: a caller waiting for a quiet
        # moment must not take the gap before the model's reply for silence.
        assert conn.last_tool_result_at is not None
        assert conn.last_model_output_at is None

    @pytest.mark.asyncio
    async def test_a_failing_tool_call_still_ends(self):
        conn = _connection(agent=_Agent(realtime_runner_cls=self._BlockingAdapter), loop=asyncio.get_running_loop())
        conn.adapter.fail = True
        conn.adapter.release.set()

        await conn._execute_tool_and_reply("c1", "lookup", "{}")

        assert conn.adapter.tool_results == [("c1", "Error: tool broke")]
        assert conn.is_responding is False
        assert conn.last_tool_result_at is not None

    @pytest.mark.asyncio
    async def test_the_caller_starting_to_speak_is_timestamped(self):
        conn = _connection(loop=asyncio.get_running_loop())

        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        assert conn.last_caller_speech_at is None

        await conn.handle_framework_event("speech_started", {})
        first = conn.last_caller_speech_at
        assert first is not None

        await asyncio.sleep(0.01)
        await conn.handle_framework_event("interrupt", {})
        assert conn.last_caller_speech_at > first

    @pytest.mark.asyncio
    async def test_a_long_answer_keeps_the_caller_heard(self):
        """Live: the caller's turn start was the only timestamp, so a 20 s answer read as silence after 15 s."""
        conn = _connection(loop=asyncio.get_running_loop())
        await conn.handle_framework_event("speech_started", {})
        started = conn.last_caller_speech_at

        await asyncio.sleep(0.01)
        await conn.handle_framework_event("speech_heard", {})

        assert conn.last_caller_speech_at > started
        assert conn.is_caller_speaking is False  # this adapter cannot tell when the caller stops
        assert list(conn._audio_queue._queue) == []

    @pytest.mark.asyncio
    async def test_an_adapter_that_reports_the_end_of_speech_marks_the_caller_speaking_until_then(self):
        class _VADAdapter(_Adapter):
            reports_speech_stopped = True

        conn = _connection(agent=_Agent(realtime_runner_cls=_VADAdapter), loop=asyncio.get_running_loop())

        await conn.handle_framework_event("speech_started", {})
        assert conn.is_caller_speaking is True
        started = conn.last_caller_speech_at

        await asyncio.sleep(0.01)
        await conn.handle_framework_event("speech_stopped", {})
        assert conn.is_caller_speaking is False
        assert conn.last_caller_speech_at > started

        await conn.handle_framework_event("interrupt", {})
        assert conn.is_caller_speaking is True
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        assert conn.is_caller_speaking is False  # the model answering ends it, should a stop never arrive

    @pytest.mark.asyncio
    async def test_what_the_model_generates_is_timestamped_before_it_is_paced_out(self):
        """A handoff's tool call runs while its agent's words are still queued: whether that agent said
        anything is known from when the words were generated, not from when they reached the edge."""
        conn = _connection(loop=asyncio.get_running_loop())

        await conn.handle_framework_event("agent_changed", {"agent": "billing", "previous_agent": "supervisor"})
        changed = conn.last_agent_change_at
        assert changed is not None and conn.last_model_generated_at is None

        await asyncio.sleep(0.01)
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        assert conn.last_model_generated_at > changed  # the new agent has spoken
        assert conn.last_model_output_at is None  # though nothing has been paced out yet

    @pytest.mark.asyncio
    async def test_text_from_the_edge_is_timestamped_and_the_applications_own_is_not(self):
        """Typed chat is the caller's input, which the model owes a reply to; an application's own update is not."""
        conn = _connection(loop=asyncio.get_running_loop())

        conn.send_text("[Team update] J1 finished.")
        await asyncio.sleep(0)
        assert conn.last_edge_text_at is None

        conn.send_text("Here is the link: example.com", from_edge=True)
        await asyncio.sleep(0)
        assert conn.last_edge_text_at is not None
        assert conn.last_caller_speech_at is None  # typed, not spoken
        assert conn.adapter.texts == ["[Team update] J1 finished.", "Here is the link: example.com"]

    def test_turn_state_is_readable_from_another_thread(self):
        loop = asyncio.new_event_loop()
        conn = _connection(loop=loop)
        try:
            loop.run_until_complete(conn.handle_framework_event("audio_delta", {"delta": "QUFB"}))
            seen = {}
            reader = threading.Thread(target=lambda: seen.update(responding=conn.is_responding, speech=conn.last_caller_speech_at))
            reader.start()
            reader.join(timeout=5)
            assert seen == {"responding": True, "speech": None}
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_a_call_whose_socket_is_being_recovered_is_not_ready(self):
        conn = _connection(loop=asyncio.get_running_loop())
        await conn.connect()
        assert conn.is_ready is True

        await conn.handle_framework_event("done", {"status": "reconnecting"})
        assert conn.is_ready is False

        await conn.handle_framework_event("reconnected", {"agent": "general", "seconds": 1, "seeded": False})
        assert conn.is_ready is True
        assert conn.last_reconnect_at is None  # nothing was replayed, so nothing is owed

        await conn.handle_framework_event("done", {"status": "reconnecting"})
        await conn.handle_framework_event("reconnected", {"agent": "general", "seconds": 1, "seeded": True})
        assert conn.last_reconnect_at is not None  # the model owes a reply to the replayed conversation
        conn._pacing_task.cancel()


class TestRealtimeAgentChanged:
    """The agent speaking is announced in order with the output it divides, so an edge never credits
    one agent's words to another."""

    @staticmethod
    def _queued(conn):
        return [item[0] for item in list(conn._audio_queue._queue)]

    @pytest.mark.asyncio
    async def test_the_starting_agent_is_announced_before_any_output(self, monkeypatch):
        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))
        conn = _connection(agent=_Agent(name="supervisor"), loop=asyncio.get_running_loop())
        await conn.connect()

        await conn.handle_framework_event("audio_delta", {"delta": "QUFB", "message_id": "m1"})
        await conn.handle_framework_event("done", {"status": "completed"})
        await asyncio.sleep(0.1)
        conn._pacing_task.cancel()

        messages = _drain_output(transport)
        assert _kinds(messages) == ["agent_changed", "audio_delta", "done"]
        assert json.loads(messages[0].body)["event"] == {"type": "agent_changed", "agent": "supervisor", "agents": ["supervisor"]}

    @pytest.mark.asyncio
    async def test_the_transcript_and_a_handoff_stay_in_order_with_the_audio(self, monkeypatch):
        """Regression: the transcript used to bypass the paced queue, so the new agent's words could
        reach the edge while the previous agent's audio was still queued, ahead of the handoff."""
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        conn = _connection(loop=asyncio.get_running_loop())

        await conn.handle_framework_event("audio_delta", {"delta": "QUFB", "message_id": "m1"})
        await conn.handle_framework_event("transcript_delta", {"delta": "Transferring you.", "message_id": "m1"})
        await conn.handle_framework_event("done", {"status": "completed"})
        await conn.handle_framework_event("agent_changed", {"agent": "billing", "previous_agent": "supervisor"})
        await conn.handle_framework_event("transcript_delta", {"delta": "Billing here.", "message_id": "m2"})
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB", "message_id": "m2"})

        assert self._queued(conn) == ["audio_delta", "transcript_delta", "done", "agent_changed", "transcript_delta", "audio_delta"]

    @pytest.mark.asyncio
    async def test_a_handoff_is_emitted_between_the_agents_output(self, monkeypatch):
        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))
        conn = _connection(agent=_Agent(name="supervisor"), loop=asyncio.get_running_loop())
        await conn.connect()

        await conn.handle_framework_event("audio_delta", {"delta": "QUFB", "message_id": "m1"})
        await conn.handle_framework_event("done", {"status": "completed"})
        await conn.handle_framework_event("agent_changed", {"agent": "billing", "previous_agent": "supervisor"})
        await conn.handle_framework_event("transcript_delta", {"delta": "Billing here.", "message_id": "m2"})
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB", "message_id": "m2"})
        await conn.handle_framework_event("done", {"status": "completed"})
        await asyncio.sleep(0.1)
        conn._pacing_task.cancel()

        messages = _drain_output(transport)
        assert _kinds(messages) == ["agent_changed", "audio_delta", "done", "agent_changed", "text_delta", "audio_delta", "done"]
        handoff, transcript = json.loads(messages[3].body), json.loads(messages[4].body)
        assert handoff["event"] == {"type": "agent_changed", "agent": "billing", "previous_agent": "supervisor"}
        assert transcript["delta"] == "Billing here."

    @pytest.mark.asyncio
    async def test_an_interrupt_drops_the_unheard_transcript_but_keeps_the_handoff(self, monkeypatch):
        """The user will not hear the dropped audio, so its words are dropped with it; talking over
        an agent does not undo the handoff that made it the speaker."""
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        conn = _connection(loop=asyncio.get_running_loop())

        await conn.handle_framework_event("agent_changed", {"agent": "billing", "previous_agent": "supervisor"})
        await conn.handle_framework_event("transcript_delta", {"delta": "Your invoice", "message_id": "m2"})
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB", "message_id": "m2"})
        await conn.handle_framework_event("interrupt", {})

        assert self._queued(conn) == ["interrupt", "agent_changed"]

    def test_a_new_connection_is_routed_before_it_connects(self, monkeypatch):
        """Its first output (the starting agent) is emitted as soon as it connects, before the caller
        could set a delivery context afterwards, so get_or_create must apply one first."""
        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))

        pool = RealtimeConnectionPool.initialize()
        thread = threading.Thread(target=pool.start, daemon=True)
        thread.start()
        try:
            pool.get_or_create(
                "s1",
                _Agent(),
                Runtime(InMemorySessionStore()),
                Session("s1"),
                request_id="r1",
                integration="livekit",
                reply_context={"session_id": "s1"},
            )
            [message] = _drain_output(transport)
        finally:
            ThreadRunner.shutdown_event.set()
            thread.join(timeout=5)

        assert json.loads(message.body)["event"] == {"type": "agent_changed", "agent": "general", "agents": ["general"]}
        assert message.attributes[ATTR_INTEGRATION] == "livekit"
        assert message.attributes["request_id"] == "r1"
        assert message.attributes["reply_session_id"] == "s1"


class TestOpenAIBargeIn:
    """The OpenAI adapter reports ``interrupt`` only while a response is in flight."""

    @staticmethod
    def _adapter():
        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        adapter = OpenAIRealtimeRunner()
        adapter.events = []

        async def callback(event_type, data):
            adapter.events.append(event_type)

        adapter._callback = callback
        return adapter

    @staticmethod
    def _event(event_type, **fields):
        return types.SimpleNamespace(type=event_type, **fields)

    @pytest.mark.asyncio
    async def test_speech_started_outside_a_response_is_reported_as_speech_started(self):
        adapter = self._adapter()

        await adapter._handle_message(self._event("input_audio_buffer.speech_started"))

        assert adapter.events == ["speech_started"]

    @pytest.mark.asyncio
    async def test_speech_started_during_a_response_is_an_interrupt(self):
        adapter = self._adapter()

        await adapter._handle_message(self._event("response.created"))
        await adapter._handle_message(self._event("input_audio_buffer.speech_started"))

        assert adapter.events == ["interrupt"]

    @pytest.mark.asyncio
    async def test_speech_started_after_response_done_is_not_an_interrupt(self):
        adapter = self._adapter()

        await adapter._handle_message(self._event("response.created"))
        await adapter._handle_message(self._event("response.done", response=types.SimpleNamespace(status="completed", output=[])))
        await adapter._handle_message(self._event("input_audio_buffer.speech_started"))

        assert adapter.events == ["done", "speech_started"]

    @pytest.mark.asyncio
    async def test_the_end_of_the_users_speech_is_reported(self):
        """Server VAD signals it, so the pool can tell a long answer from silence."""
        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        adapter = self._adapter()

        await adapter._handle_message(self._event("input_audio_buffer.speech_started"))
        await adapter._handle_message(self._event("input_audio_buffer.speech_stopped"))

        assert adapter.events == ["speech_started", "speech_stopped"]
        assert OpenAIRealtimeRunner.reports_speech_stopped is True


class TestOpenAIRealtimeSession:
    """Session setup and the tool round trip follow the Realtime API's documented flow."""

    class _Connection:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

        async def __aiter__(self):
            # An async generator, as the SDK's connection iterates.
            return
            yield

    @classmethod
    def _runner(cls):
        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        runner = OpenAIRealtimeRunner()
        runner._connection = cls._Connection()
        runner.events = []

        async def callback(event_type, data):
            runner.events.append((event_type, data))

        runner._callback = callback
        return runner

    @staticmethod
    def _response_done(status, *calls):
        output = [types.SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=args) for call_id, name, args in calls]
        return types.SimpleNamespace(type="response.done", response=types.SimpleNamespace(status=status, output=output))

    @pytest.mark.asyncio
    async def test_function_calls_are_dispatched_from_the_completed_response(self):
        runner = self._runner()

        # The arguments event precedes response.done; acting on it would race the active response.
        await runner._handle_message(types.SimpleNamespace(type="response.function_call_arguments.done", call_id="c1", name="a", arguments="{}"))
        await runner._handle_message(self._response_done("completed", ("c1", "a", '{"x": 1}'), ("c2", "b", "")))

        assert [(kind, data.get("call_id")) for kind, data in runner.events] == [("tool_call", "c1"), ("tool_call", "c2"), ("done", None)]
        assert runner.events[0][1]["arguments"] == '{"x": 1}'
        assert runner.events[1][1]["arguments"] == "{}"

    @pytest.mark.asyncio
    async def test_a_cancelled_response_dispatches_no_function_calls(self):
        runner = self._runner()

        await runner._handle_message(self._response_done("cancelled", ("c1", "a", "{}")))

        assert [kind for kind, _ in runner.events] == ["done"]

    @pytest.mark.asyncio
    async def test_the_follow_up_response_is_requested_once_after_the_last_output(self):
        runner = self._runner()
        await runner._handle_message(self._response_done("completed", ("c1", "a", "{}"), ("c2", "b", "{}")))

        await runner.send_tool_result("c1", "one")
        assert [payload["type"] for payload in runner._connection.sent] == ["conversation.item.create"]

        await runner.send_tool_result("c2", "two")
        assert [payload["type"] for payload in runner._connection.sent] == ["conversation.item.create", "conversation.item.create", "response.create"]

    @staticmethod
    def _patch_socket(monkeypatch):
        connection = TestOpenAIRealtimeSession._Connection()

        class _ConnectManager:
            async def __aenter__(self):
                return connection

            async def __aexit__(self, *exc):
                return False

        class _Client:
            realtime = types.SimpleNamespace(connect=lambda model: _ConnectManager())

            async def close(self):
                pass

        monkeypatch.setattr("openai.AsyncOpenAI", lambda *a, **k: _Client())
        return connection

    @staticmethod
    async def _connect(sdk_agent, session=None):
        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        runner = OpenAIRealtimeRunner()

        async def callback(event_type, data):
            pass

        await runner.connect(session or Session("s1"), types.SimpleNamespace(name="general", agent=sdk_agent), callback)
        await runner.disconnect()
        return runner

    @pytest.mark.asyncio
    async def test_callable_instructions_are_resolved_with_the_framework_context(self, monkeypatch):
        from agents import Agent as SDKAgent

        connection = self._patch_socket(monkeypatch)
        session = Session("s1")
        session.set_framework_context({"name": "Ada"})
        sdk_agent = SDKAgent(name="general", model="gpt-realtime", instructions=lambda ctx, agent: f"Greet {ctx.context['name']}")

        await self._connect(sdk_agent, session)

        assert connection.sent[0]["session"]["instructions"] == "Greet Ada"

    @pytest.mark.asyncio
    async def test_no_instructions_leaves_the_server_default(self, monkeypatch):
        from agents import Agent as SDKAgent

        connection = self._patch_socket(monkeypatch)

        await self._connect(SDKAgent(name="general", model="gpt-realtime"))

        assert "instructions" not in connection.sent[0]["session"]

    @pytest.mark.asyncio
    async def test_only_function_tools_are_offered_and_others_are_reported(self, monkeypatch, caplog):
        from agents import Agent as SDKAgent
        from agents import WebSearchTool

        from agentkernel.framework.openai.openai import OpenAIToolBuilder

        def get_weather(location: str) -> str:
            """Get the weather."""
            return location

        connection = self._patch_socket(monkeypatch)
        sdk_agent = SDKAgent(name="general", model="gpt-realtime", tools=[*OpenAIToolBuilder.bind([get_weather]), WebSearchTool()])

        with caplog.at_level("WARNING"):
            await self._connect(sdk_agent)

        assert [tool["name"] for tool in connection.sent[0]["session"]["tools"]] == ["get_weather"]
        assert "not a function tool" in caplog.text


class TestOpenAIRealtimeToolGuards:
    """The runner calls tools directly, so tools whose checks it cannot apply are not offered, and only
    offered tools run, within their timeout."""

    @staticmethod
    def _tool(func, **changes):
        import dataclasses

        from agentkernel.framework.openai.openai import OpenAIToolBuilder

        [tool] = OpenAIToolBuilder.bind([func])
        return dataclasses.replace(tool, **changes) if changes else tool

    @staticmethod
    def _offered(*tools):
        from agents import RunContextWrapper

        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        sdk_agent = types.SimpleNamespace(tools=list(tools))
        return asyncio.run(OpenAIRealtimeRunner()._offered_tools(sdk_agent, RunContextWrapper(context=None)))

    @staticmethod
    def plain(x: str) -> str:
        """A tool with no extra checks."""
        return x

    def test_tools_needing_approval_or_guardrails_are_not_offered(self, caplog):
        def approve(x: str) -> str:
            """Needs approval."""
            return x

        def guarded(x: str) -> str:
            """Has an input guardrail."""
            return x

        def checked(x: str) -> str:
            """Has an output guardrail."""
            return x

        with caplog.at_level("WARNING"):
            offered = self._offered(
                self._tool(self.plain),
                self._tool(approve, needs_approval=True),
                self._tool(guarded, tool_input_guardrails=[object()]),
                self._tool(checked, tool_output_guardrails=[object()]),
            )

        assert list(offered) == ["plain"]
        assert caplog.text.count("needs approval or has tool guardrails") == 3

    def test_disabled_tools_are_not_offered(self):
        def off(x: str) -> str:
            """Disabled."""
            return x

        def off_by_rule(x: str) -> str:
            """Disabled by a callable."""
            return x

        def on_async(x: str) -> str:
            """Enabled by an async callable."""
            return x

        async def enabled(ctx, agent):
            return True

        offered = self._offered(
            self._tool(off, is_enabled=False),
            self._tool(off_by_rule, is_enabled=lambda ctx, agent: False),
            self._tool(on_async, is_enabled=enabled),
        )

        assert list(offered) == ["on_async"]

    @pytest.mark.asyncio
    async def test_a_tool_that_was_not_offered_does_not_run(self):
        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        ran = []

        def secret(x: str) -> str:
            """Never offered."""
            ran.append(x)
            return x

        agent = types.SimpleNamespace(agent=types.SimpleNamespace(tools=[self._tool(secret)]), name="general")
        runner = OpenAIRealtimeRunner()
        runner._agent = agent

        result = await runner.execute_tool("secret", '{"x": "1"}', ToolContext(Runtime(InMemorySessionStore()), agent, Session("s1"), []), "call-1")

        assert result == "Error: Tool secret not found"
        assert ran == []

    @pytest.mark.asyncio
    async def test_the_tool_timeout_is_enforced(self):
        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        async def slow() -> str:
            """Takes too long."""
            await asyncio.sleep(1)
            return "late"

        tool = self._tool(slow, timeout_seconds=0.05)
        agent = types.SimpleNamespace(agent=types.SimpleNamespace(tools=[tool]), name="general")
        runner = OpenAIRealtimeRunner()
        runner._agent = agent
        runner._tools = {tool.name: tool}

        result = await runner.execute_tool("slow", "{}", ToolContext(Runtime(InMemorySessionStore()), agent, Session("s1"), []), "call-1")

        assert "timed out" in result


class _ScriptedRealtimeSocket:
    """A realtime socket that replays scripted server events in order.

    A callable in the script is a wait: replay pauses until it returns True over the client events
    sent so far, the way a real model answers only after the runner has. Once the script is spent the
    socket stays open. Iterated through an async generator, as the SDK's connection is."""

    def __init__(self, script):
        self.script = list(script)
        self.sent = []
        self._sent = asyncio.Event()

    async def send(self, payload):
        self.sent.append(payload)
        self._sent.set()

    async def __aiter__(self):
        while self.script:
            item = self.script.pop(0)
            if not callable(item):
                yield item
                continue
            while not item(self.sent):
                self._sent.clear()
                await self._sent.wait()
        await asyncio.Event().wait()


class TestOpenAIRealtimeHandoffs:
    """Native handoffs run over the one socket and through the tool path, as the SDK's own realtime
    session runs them: the session moves to the agent the handoff returned when its result is sent,
    and the change is reported in order."""

    @staticmethod
    def _graph(**billing_options):
        from agents import Agent as SDKAgent

        supervisor = SDKAgent(name="supervisor", model="gpt-realtime", instructions="Route the caller.")
        billing = SDKAgent(name="billing", instructions="Answer billing questions.", handoffs=[supervisor], **billing_options)
        supervisor.handoffs = [billing]
        return supervisor, billing

    @staticmethod
    async def _runner(active):
        from agents import RunContextWrapper

        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        runner = OpenAIRealtimeRunner()
        runner._connection = TestOpenAIRealtimeSession._Connection()
        runner._model = "gpt-realtime"
        runner._run_context = RunContextWrapper(context=None)
        runner.events = []
        runner.answered = set()

        async def callback(event_type, data):
            runner.events.append((event_type, data))

        runner._callback = callback
        await runner._activate(active)
        return runner

    @staticmethod
    def _context():
        return ToolContext(Runtime(InMemorySessionStore()), _Agent(), Session("s1"), [])

    @classmethod
    async def _answer(cls, runner):
        """Do what the pool does with each reported call: run it, then send its result."""
        for kind, data in list(runner.events):
            if kind != "tool_call" or data["call_id"] in runner.answered:
                continue
            runner.answered.add(data["call_id"])
            try:
                result = await runner.execute_tool(data["name"], data["arguments"], cls._context(), data["call_id"])
            except Exception as e:
                result = f"Error: {e}"
            await runner.send_tool_result(data["call_id"], result)

    @staticmethod
    def _done(*calls, status="completed"):
        return TestOpenAIRealtimeSession._response_done(status, *calls)

    @staticmethod
    def _reported(runner):
        return [event for event in runner.events if event[0] != "tool_call"]

    @pytest.mark.asyncio
    async def test_enabled_handoffs_are_offered_beside_the_tools(self, monkeypatch):
        supervisor, _ = self._graph()
        connection = TestOpenAIRealtimeSession._patch_socket(monkeypatch)

        await TestOpenAIRealtimeSession._connect(supervisor)

        [offered] = connection.sent[0]["session"]["tools"]
        assert offered["name"] == "transfer_to_billing"
        assert offered["description"].startswith("Handoff to the billing agent")

    def test_the_team_is_every_agent_reachable_through_handoffs(self):
        """Followed through handoff() wrappers too, nearest first: the room shows exactly these agents."""
        from agents import Agent as SDKAgent
        from agents import handoff

        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        supervisor, billing = self._graph()
        supervisor.handoffs = [billing, handoff(SDKAgent(name="tech_support", handoffs=[supervisor]))]

        team = OpenAIRealtimeRunner().team(types.SimpleNamespace(name="supervisor", agent=supervisor))

        assert team == ["supervisor", "billing", "tech_support"]

    def test_a_statically_disabled_handoff_is_not_part_of_the_team(self):
        """It is never offered to the model, so no participant is opened for it."""
        from agents import Agent as SDKAgent
        from agents import handoff

        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        supervisor, billing = self._graph()
        supervisor.handoffs = [billing, handoff(SDKAgent(name="sales"), is_enabled=False)]

        team = OpenAIRealtimeRunner().team(types.SimpleNamespace(name="supervisor", agent=supervisor))

        assert team == ["supervisor", "billing"]

    @pytest.mark.asyncio
    async def test_a_handoff_call_is_dispatched_like_a_tool_call(self):
        supervisor, _ = self._graph()
        runner = await self._runner(supervisor)

        await runner._handle_message(self._done(("c1", "transfer_to_billing", "{}")))

        assert runner.events == [
            ("tool_call", {"call_id": "c1", "name": "transfer_to_billing", "arguments": "{}"}),
            ("done", {"status": "completed"}),
        ]
        assert runner._active is supervisor
        assert runner._connection.sent == []

    @pytest.mark.asyncio
    async def test_a_handoff_points_the_session_at_the_new_agent(self):
        supervisor, billing = self._graph()
        runner = await self._runner(supervisor)

        await runner._handle_message(self._done(("c1", "transfer_to_billing", "{}")))
        await self._answer(runner)

        assert runner._active is billing
        assert self._reported(runner) == [("done", {"status": "completed"}), ("agent_changed", {"agent": "billing", "previous_agent": "supervisor"})]
        update, output, follow_up = runner._connection.sent
        assert update["type"] == "session.update"
        assert update["session"]["instructions"] == "Answer billing questions."
        assert [tool["name"] for tool in update["session"]["tools"]] == ["transfer_to_supervisor"]
        assert output["item"] == {"type": "function_call_output", "call_id": "c1", "output": '{"assistant": "billing"}'}
        assert follow_up == {"type": "response.create"}

    @pytest.mark.asyncio
    async def test_the_handoff_callback_runs_once(self):
        from agents import handoff

        calls = []
        supervisor, billing = self._graph()
        supervisor.handoffs = [handoff(billing, on_handoff=lambda ctx: calls.append(ctx))]
        runner = await self._runner(supervisor)

        await runner._handle_message(self._done(("c1", "transfer_to_billing", "{}")))
        await self._answer(runner)

        assert len(calls) == 1
        assert runner._active is billing

    @pytest.mark.asyncio
    async def test_an_agent_with_no_instructions_does_not_keep_the_previous_ones(self):
        supervisor, billing = self._graph()
        billing.instructions = None
        runner = await self._runner(supervisor)

        await runner._handle_message(self._done(("c1", "transfer_to_billing", "{}")))
        await self._answer(runner)

        assert runner._connection.sent[0]["session"]["instructions"] == ""

    @pytest.mark.asyncio
    async def test_a_tool_called_beside_a_handoff_runs_as_the_agent_that_called_it(self):
        from agents import function_tool
        from agents.tool_context import ToolContext as SDKToolContext

        @function_tool
        def whoami(ctx: SDKToolContext) -> str:
            """Name the agent running this tool."""
            return ctx.agent.name

        supervisor, billing = self._graph()
        supervisor.tools = [whoami]
        runner = await self._runner(supervisor)
        await runner._handle_message(self._done(("c1", "whoami", "{}"), ("c2", "transfer_to_billing", "{}")))

        # The pool may finish the handoff first; billing has no whoami tool.
        await runner.send_tool_result("c2", await runner.execute_tool("transfer_to_billing", "{}", self._context(), "c2"))
        assert runner._active is billing
        # The follow-up response waits for the tool's output too.
        assert {"type": "response.create"} not in runner._connection.sent

        result = await runner.execute_tool("whoami", "{}", self._context(), "c1")
        await runner.send_tool_result("c1", result)

        assert result == "supervisor"
        assert runner._connection.sent[-1] == {"type": "response.create"}

    @pytest.mark.asyncio
    async def test_a_specialist_hands_back_to_the_supervisor(self):
        supervisor, billing = self._graph()
        runner = await self._runner(supervisor)

        await runner._handle_message(self._done(("c1", "transfer_to_billing", "{}")))
        await self._answer(runner)
        await runner._handle_message(self._done(("c2", "transfer_to_supervisor", "{}")))
        await self._answer(runner)

        assert runner._active is supervisor
        assert runner.events[-1] == ("agent_changed", {"agent": "supervisor", "previous_agent": "billing"})

    @pytest.mark.asyncio
    async def test_a_failed_handoff_keeps_the_agent(self):
        from agents import handoff

        def refuse(ctx):
            raise RuntimeError("billing is closed")

        supervisor, billing = self._graph()
        supervisor.handoffs = [handoff(billing, on_handoff=refuse)]
        runner = await self._runner(supervisor)

        await runner._handle_message(self._done(("c1", "transfer_to_billing", "{}")))
        await self._answer(runner)

        assert runner._active is supervisor
        assert self._reported(runner) == [("done", {"status": "completed"})]
        output, follow_up = runner._connection.sent
        assert output["item"]["output"] == "Error: billing is closed"
        assert follow_up == {"type": "response.create"}

    @pytest.mark.asyncio
    async def test_an_agent_that_cannot_be_activated_keeps_the_call_where_it_was(self):
        """Regression: a failure resolving the new agent escaped the tool task, leaving the call with
        no output and no follow-up response. Now the model is told and the agent stays."""

        def broken(ctx, agent):
            raise RuntimeError("no instructions today")

        supervisor, billing = self._graph()
        billing.instructions = broken
        runner = await self._runner(supervisor)

        await runner._handle_message(self._done(("c1", "transfer_to_billing", "{}")))
        await self._answer(runner)

        assert runner._active is supervisor
        assert self._reported(runner) == [("done", {"status": "completed"})]
        output, follow_up = runner._connection.sent
        assert output["item"]["output"] == "Error: the handoff to billing failed: no instructions today"
        assert follow_up == {"type": "response.create"}
        assert list(runner._handoffs) == ["transfer_to_billing"]

    @pytest.mark.asyncio
    async def test_only_the_first_handoff_of_a_response_is_performed(self):
        from agents import Agent as SDKAgent
        from agents import handoff

        invoked = []
        supervisor, billing = self._graph()
        supervisor.handoffs = [billing, handoff(SDKAgent(name="tech_support"), on_handoff=lambda ctx: invoked.append(ctx))]
        runner = await self._runner(supervisor)

        await runner._handle_message(self._done(("c1", "transfer_to_billing", "{}"), ("c2", "transfer_to_tech_support", "{}")))
        await self._answer(runner)

        assert runner._active is billing
        assert invoked == []
        outputs = {p["item"]["call_id"]: p["item"]["output"] for p in runner._connection.sent if p["type"] == "conversation.item.create"}
        assert outputs["c2"] == "Error: only one handoff is performed per turn; this one, to tech_support, was not"
        assert runner._connection.sent[-1] == {"type": "response.create"}

    @pytest.mark.asyncio
    async def test_a_cancelled_response_performs_no_handoff(self):
        supervisor, _ = self._graph()
        runner = await self._runner(supervisor)

        await runner._handle_message(self._done(("c1", "transfer_to_billing", "{}"), status="cancelled"))

        assert runner._active is supervisor
        assert runner.events == [("done", {"status": "cancelled"})]

    @pytest.mark.asyncio
    async def test_a_disabled_handoff_is_not_offered(self):
        from agents import Agent as SDKAgent
        from agents import handoff

        supervisor, billing = self._graph()
        supervisor.handoffs = [billing, handoff(SDKAgent(name="sales"), is_enabled=False)]

        runner = await self._runner(supervisor)

        assert list(runner._handoffs) == ["transfer_to_billing"]

    @pytest.mark.asyncio
    async def test_a_handoff_that_filters_the_history_fails_before_connecting(self, monkeypatch):
        """Checked across the whole graph, through handoff() wrappers, before any socket opens."""
        from agents import handoff

        supervisor, billing = self._graph()
        supervisor.handoffs = [handoff(billing)]
        billing.handoffs = [handoff(supervisor, input_filter=lambda data: data)]
        connection = TestOpenAIRealtimeSession._patch_socket(monkeypatch)

        with pytest.raises(ValueError, match="filters or nests the history"):
            await TestOpenAIRealtimeSession._connect(supervisor)

        assert connection.sent == []

    @pytest.mark.asyncio
    async def test_a_specialist_declaring_another_model_is_reported_at_connect(self, monkeypatch, caplog):
        supervisor, _ = self._graph(model="gpt-4o")
        TestOpenAIRealtimeSession._patch_socket(monkeypatch)

        with caplog.at_level("WARNING"):
            await TestOpenAIRealtimeSession._connect(supervisor)

        assert "Agent 'billing' declares model gpt-4o" in caplog.text

    @pytest.mark.asyncio
    async def test_a_conversation_moves_to_a_specialist_and_back(self, monkeypatch):
        """End to end through the pool: the specialist's own tool runs once, the handoff's callback runs
        in Agent Kernel's session scope as a tool does, and every speaker change reaches the output in
        order with the speech around it."""
        from agents import handoff

        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner, OpenAIToolBuilder

        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))

        looked_up = []
        handed_off_in = []

        def get_invoice(account_id: str) -> str:
            """Look up the caller's latest invoice."""
            looked_up.append(account_id)
            return f"{account_id} owes $42"

        supervisor, billing = self._graph(tools=OpenAIToolBuilder.bind([get_invoice]))
        supervisor.handoffs = [handoff(billing, on_handoff=lambda ctx: handed_off_in.append(Session.current().id))]

        def answered(call_id):
            return lambda sent: any(p.get("item", {}).get("call_id") == call_id for p in sent)

        def event(event_type, **fields):
            return types.SimpleNamespace(type=event_type, **fields)

        socket = _ScriptedRealtimeSocket(
            [
                event("response.created"),
                event("response.output_audio_transcript.delta", delta="Connecting you to billing.", item_id="m1"),
                event("response.output_audio.delta", delta="QUFB", item_id="m1"),
                self._done(("c1", "transfer_to_billing", "{}")),
                answered("c1"),
                event("response.created"),
                event("response.output_audio_transcript.delta", delta="Billing here.", item_id="m2"),
                event("response.output_audio.delta", delta="QUFB", item_id="m2"),
                self._done(("c2", "get_invoice", '{"account_id": "A1"}')),
                answered("c2"),
                event("response.created"),
                self._done(("c3", "transfer_to_supervisor", "{}")),
                answered("c3"),
            ]
        )

        class _ConnectManager:
            async def __aenter__(self):
                return socket

            async def __aexit__(self, *exc):
                return False

        class _Client:
            realtime = types.SimpleNamespace(connect=lambda model: _ConnectManager())

            async def close(self):
                pass

        monkeypatch.setattr("openai.AsyncOpenAI", lambda *a, **k: _Client())

        agent = _Agent(name="supervisor", realtime_runner_cls=OpenAIRealtimeRunner)
        agent.agent = supervisor
        conn = _connection(agent=agent, loop=asyncio.get_running_loop())
        await conn.connect()
        for _ in range(100):
            if not socket.script:
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        await conn.close()

        assert looked_up == ["A1"]
        assert handed_off_in == ["s1"]
        tool_output = next(p for p in socket.sent if p.get("item", {}).get("call_id") == "c2")
        assert tool_output["item"]["output"] == "A1 owes $42"
        assert [p["session"].get("instructions") for p in socket.sent if p["type"] == "session.update"] == [
            "Route the caller.",
            "Answer billing questions.",
            "Route the caller.",
        ]

        bodies = [json.loads(m.body) for m in _drain_output(transport)]
        # The conversation opens by naming its whole team, for the edge to show.
        assert bodies[0]["event"]["agents"] == ["supervisor", "billing"]
        speech = [
            (body["event"]["type"], body["event"].get("agent") or body.get("delta"))
            for body in bodies
            if body.get("event", {}).get("type") in ("agent_changed", "text_delta")
        ]
        assert speech == [
            ("agent_changed", "supervisor"),
            ("text_delta", "Connecting you to billing."),
            ("agent_changed", "billing"),
            ("text_delta", "Billing here."),
            ("agent_changed", "supervisor"),
        ]


class TestRealtimeRunnerCleanup:
    """Both framework runners release what they opened the same way."""

    @staticmethod
    def _runners():
        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        return [OpenAIRealtimeRunner]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("index", [0], ids=["openai"])
    async def test_disconnect_waits_for_the_listener_to_end(self, index):
        runner = self._runners()[index]()
        listener = asyncio.create_task(asyncio.Event().wait())
        runner._listen_task = listener

        await runner.disconnect()

        assert listener.done()
        assert runner._listen_task is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("index", [0], ids=["openai"])
    async def test_stopping_the_listener_closes_the_sockets_event_stream(self, index):
        """Regression: stopped while handling an event, the listener left the SDK's event generator
        half-run for the loop's finalizer to close from another task, which raced the loop's shutdown
        ("aclose(): asynchronous generator is already running"). The listener must close it itself."""
        closed_by = []

        class _Socket:
            async def _events(self):
                try:
                    yield types.SimpleNamespace(type="unhandled", server_content=None, tool_call=None)
                    await asyncio.Event().wait()
                finally:
                    closed_by.append(asyncio.current_task())

            def __aiter__(self):
                return self._events()

            def receive(self):
                return self._events()

        runner = self._runners()[index]()
        runner._connection = _Socket()
        handling = asyncio.Event()

        async def handle(message):
            handling.set()
            await asyncio.Event().wait()

        runner._handle_message = handle
        listener = asyncio.create_task(runner._listen())
        runner._listen_task = listener
        await asyncio.wait_for(handling.wait(), timeout=1)

        await runner.disconnect()
        for _ in range(3):  # let a finalizer-scheduled close run, were the stream left to one
            await asyncio.sleep(0)

        assert closed_by == [listener]

    @pytest.mark.asyncio
    async def test_a_socket_that_fails_to_open_closes_everything_it_opened(self, monkeypatch):
        from agents import Agent as SDKAgent

        from agentkernel.framework.openai.openai import OpenAIRealtimeRunner

        closed = []

        class _RefusedSocket:
            async def __aenter__(self):
                raise ConnectionError("socket refused")

            async def __aexit__(self, *exc):
                closed.append("socket")

        class _OpenAIClient:
            realtime = types.SimpleNamespace(connect=lambda model: _RefusedSocket())

            async def close(self):
                closed.append("openai client")

        monkeypatch.setattr("openai.AsyncOpenAI", lambda *a, **k: _OpenAIClient())

        async def callback(event_type, data):
            pass

        runner = OpenAIRealtimeRunner()
        with pytest.raises(ConnectionError):
            await runner.connect(Session("s1"), types.SimpleNamespace(name="general", agent=SDKAgent(name="general", model="gpt-realtime")), callback)
        assert runner._cm is None
        # The client is closed; a socket that never opened is not.
        assert closed == ["openai client"]


class TestRealtimeFailureHandling:
    """A dead model socket must be surfaced to the edge and evicted, not written to forever."""

    @pytest.mark.asyncio
    async def test_error_event_marks_closed_and_emits_terminal_error_chunk(self, monkeypatch):
        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))

        conn = _connection(loop=asyncio.get_running_loop())
        await conn.connect()
        await conn.handle_framework_event("error", {"message": "socket boom"})

        assert conn.closed is True
        # The starting agent may or may not have been announced before the failure; the error is last.
        body = json.loads(_drain_output(transport)[-1].body)
        assert body["error"] == "socket boom"
        assert body["done"] is True

    def test_get_connection_evicts_a_closed_connection(self, monkeypatch):
        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))

        pool = RealtimeConnectionPool.initialize()
        thread = threading.Thread(target=pool.start, daemon=True)
        thread.start()
        try:
            runtime = Runtime(InMemorySessionStore())
            agent = _Agent()
            conn = pool.get_or_create("s1", agent, runtime, Session("s1"))
            conn.closed = True

            assert pool.get_connection("s1") is None
            # The next chunk reconnects rather than reusing the dead socket.
            fresh = pool.get_or_create("s1", agent, runtime, Session("s1"))
            assert fresh is not conn
        finally:
            ThreadRunner.shutdown_event.set()
            thread.join(timeout=5)

    class _SlowConnectAdapter(_Adapter):
        """Connects only after the consumer stopped waiting; records being cancelled and disconnected."""

        instances: list = []

        def __init__(self):
            super().__init__()
            self.cancelled = threading.Event()
            self.disconnected = threading.Event()
            type(self).instances.append(self)

        async def connect(self, session, agent, callback):
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

        async def disconnect(self):
            self.disconnected.set()

    def test_a_connect_that_times_out_is_stopped_and_closed(self, monkeypatch):
        """The connect would otherwise go on on the loop and leave a socket that no eviction ever reaches."""
        import concurrent.futures

        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        monkeypatch.setattr(realtime_pool, "_CONNECT_TIMEOUT_SECONDS", 0.1)
        self._SlowConnectAdapter.instances = []

        pool = RealtimeConnectionPool.initialize()
        thread = threading.Thread(target=pool.start, daemon=True)
        thread.start()
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                pool.get_or_create("s1", _Agent(realtime_runner_cls=self._SlowConnectAdapter), Runtime(InMemorySessionStore()), Session("s1"))

            (adapter,) = self._SlowConnectAdapter.instances
            assert adapter.cancelled.wait(timeout=2)
            assert adapter.disconnected.wait(timeout=2)
            assert pool._connections == {}
        finally:
            ThreadRunner.shutdown_event.set()
            thread.join(timeout=5)

    @pytest.mark.asyncio
    async def test_a_tool_result_the_model_never_got_is_logged_and_owes_no_reply(self, monkeypatch, caplog):
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        conn = _connection(loop=asyncio.get_running_loop())

        async def broken(call_id, result):
            raise ConnectionError("socket gone")

        conn.adapter.send_tool_result = broken
        with caplog.at_level(logging.ERROR, logger="ak.pipeline.realtime_pool"):
            await conn.handle_framework_event("tool_call", {"call_id": "c1", "name": "lookup", "arguments": "{}"})
            await asyncio.gather(*conn._tool_tasks)

        assert "Sending the result of tool lookup to session s1 failed" in caplog.text
        assert conn.last_tool_result_at is None and conn._tools_in_flight == 0


class TestLiveKitGatewayErrorChunk:
    @pytest.mark.asyncio
    async def test_error_chunk_is_surfaced_and_clears_partial_transcript(self):
        from agentkernel.integration.livekit.adapter import LiveKitEdgeGateway

        gateway = object.__new__(LiveKitEdgeGateway)
        gateway._transcript = ["half a sentence"]
        gateway._interrupted = False
        gateway.audio_source = None
        gateway._loop = None
        gateway.room = None

        delivered = []

        async def fake_deliver_error(message, reply_context):
            delivered.append(message)

        gateway.deliver_error = fake_deliver_error

        await gateway.deliver_chunk(StreamChunk(error="model failed", done=True), {})
        assert delivered == ["model failed"]
        assert gateway._transcript == []


class TestRealtimePermanentFailure:
    def test_permanent_failure_reaches_the_live_edge_as_an_error_chunk(self, monkeypatch):
        """End to end: an input that exhausts its retries must be delivered by the registered edge's
        ``deliver_chunk`` error path, not routed to a webhook outbound adapter (which a stateful
        edge has none of), so the room is told instead of left silent."""
        from unittest.mock import MagicMock

        from agentkernel.integration.adapter.registry import StatefulEdgeRegistry
        from agentkernel.pipeline.agent_runner import RealtimeAgentRunner
        from agentkernel.pipeline.envelope import QueueMessage
        from agentkernel.pipeline.response_handler import ResponseHandler

        class _Input:
            max_receive_count = 3

        class _Queues:
            input = _Input()

        class _Cfg:
            class execution:
                mode = ExecutionMode.REALTIME
                queues = _Queues()

                class realtime:
                    playback_lead_ms = 50

        monkeypatch.setattr("agentkernel.core.config.AKConfig.get", classmethod(lambda cls: _Cfg))

        class _Edge:
            ERROR_MESSAGE = "sorry"

            def __init__(self):
                self.chunks = []

            async def deliver_chunk(self, chunk, reply_context):
                self.chunks.append((chunk, reply_context))

        edge = _Edge()
        monkeypatch.setattr(StatefulEdgeRegistry, "get", classmethod(lambda cls, session_id: edge if session_id == "s1" else None))

        transport = InMemoryTransport()
        runner = RealtimeAgentRunner(transport=transport, chat_service=MagicMock())
        message = QueueMessage(
            body="{}", attributes={"request_id": "r0", "integration": "livekit", "reply_session_id": "s1"}, group_id="s1", dedup_id="d0"
        )

        runner.on_permanent_failure(message)

        [out] = transport.create_consumer(QueueName.OUTPUT).fetch(10, 0.5)
        assert out.attributes[ATTR_REALTIME] == "true"
        assert out.attributes[ATTR_INTEGRATION] == "livekit"
        assert out.group_id == "s1"

        ResponseHandler(transport=transport).process(out)

        [(chunk, reply_context)] = edge.chunks
        assert chunk.error == "Failed to process message after 3 retries"
        assert chunk.done is True
        assert reply_context == {"session_id": "s1"}


class TestResponseHandlerRealtimeRouting:
    """Realtime output goes to the live edge registered in this process, never to IntegrationDelivery's
    webhook adapters, which a stateful edge has none of."""

    class _Edge:
        ERROR_MESSAGE = "sorry"

        def __init__(self):
            self.errors = []

        async def deliver_error(self, message, reply_context):
            self.errors.append((message, reply_context))

    @staticmethod
    def _message():
        from agentkernel.pipeline.envelope import QueueMessage

        body = json.dumps(StreamChunk(done=True).model_dump(exclude_none=True))
        attributes = {"request_id": "r1", ATTR_INTEGRATION: "livekit", ATTR_REALTIME: "true", "reply_session_id": "s1"}
        return QueueMessage(body=body, attributes=attributes, group_id="s1", dedup_id="d1")

    def test_a_chunk_for_an_edge_that_is_gone_raises_for_retry(self, monkeypatch):
        from agentkernel.integration.adapter.registry import StatefulEdgeRegistry
        from agentkernel.pipeline.response_handler import ResponseHandler

        monkeypatch.setattr(StatefulEdgeRegistry, "get", classmethod(lambda cls, session_id: None))
        handler = ResponseHandler(transport=InMemoryTransport())
        handler._integration_delivery = None  # the webhook path must not be reached

        with pytest.raises(ValueError, match="No active realtime edge"):
            handler.process(self._message())

    def test_a_permanent_failure_tells_the_live_edge(self, monkeypatch):
        from agentkernel.integration.adapter.registry import StatefulEdgeRegistry
        from agentkernel.pipeline.response_handler import ResponseHandler

        class _Cfg:
            class execution:
                mode = ExecutionMode.REALTIME

                class queues:
                    class output:
                        max_receive_count = 3

        monkeypatch.setattr("agentkernel.core.config.AKConfig.get", classmethod(lambda cls: _Cfg))
        edge = self._Edge()
        monkeypatch.setattr(StatefulEdgeRegistry, "get", classmethod(lambda cls, session_id: edge if session_id == "s1" else None))
        handler = ResponseHandler(transport=InMemoryTransport())
        handler._integration_delivery = None  # the webhook path must not be reached

        handler.on_permanent_failure(self._message())

        assert edge.errors == [("sorry", {"session_id": "s1"})]


class TestBrokerRealtimeUserGate:
    def test_realtime_chunk_without_user_id_is_not_rejected_on_a_broker(self, monkeypatch):
        """Realtime delivers through the integration adapter, not WebSocket, so the broker
        user_id requirement that guards the STREAM path must not apply to it."""
        from unittest.mock import MagicMock

        from agentkernel.pipeline.agent_runner import RealtimeAgentRunner
        from agentkernel.pipeline.envelope import QueueMessage

        class _Cfg:
            class execution:
                mode = ExecutionMode.REALTIME
                queues = None

                class realtime:
                    playback_lead_ms = 50

        monkeypatch.setattr("agentkernel.core.config.AKConfig.get", classmethod(lambda cls: _Cfg))
        monkeypatch.setattr(QueueTransportFactory, "resolve_type", staticmethod(lambda *a, **k: "kafka"))

        transport = InMemoryTransport()
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: transport))

        handler = types.SimpleNamespace(service=types.SimpleNamespace(agent=_Agent(), runtime=Runtime(InMemorySessionStore()), session=Session("s1")))
        chat_service = MagicMock()
        chat_service.prepare_agent_handler.return_value = handler
        chat_service.process_chat_request.return_value = (200, {})

        pool = RealtimeConnectionPool.initialize()
        thread = threading.Thread(target=pool.start, daemon=True)
        thread.start()
        try:
            runner = RealtimeAgentRunner(transport=transport, chat_service=chat_service)
            body = json.dumps({"prompt": "", "session_id": "s1", "requests": [{"type": "text", "prompt": "hi"}]})
            # No user_id: on a broker transport this used to raise for realtime and drop every chunk.
            runner.process(QueueMessage(body=body, attributes={"request_id": "r0", "integration": "livekit"}, group_id="s1", dedup_id="d0"))
            assert chat_service.prepare_agent_handler.call_count == 1
        finally:
            ThreadRunner.shutdown_event.set()
            thread.join(timeout=5)


def _quiet_connection(**state):
    """A connection connected at 100.0, quiet and ready unless ``state`` (its private turn-state fields) says otherwise."""
    conn = _connection()
    conn._connected_at = 100.0
    for name, value in state.items():
        setattr(conn, name, value)
    return conn


class TestRealtimeQuietCall:
    """When a call is quiet, read from what the connection observed, and the one inference it makes: a reply still owed."""

    @pytest.mark.parametrize("busy", [{"_tools_in_flight": 1}, {"_caller_speaking": True}, {"_recovering": True}, {"_connected_at": None}])
    def test_a_busy_recovering_or_unready_call_is_not_quiet(self, busy):
        assert _quiet_connection(**busy).quiet_since(now=200.0) is None

    def test_output_still_to_play_is_not_quiet(self):
        conn = _quiet_connection()
        conn._audio_queue.put_nowait(("audio_delta", {"delta": "QUFB"}))

        assert conn.quiet_since(now=200.0) is None

    @pytest.mark.parametrize("given", ["_last_caller_speech_at", "_last_edge_text_at", "_last_tool_result_at", "_last_reconnect_at", "_last_post_at"])
    def test_the_model_is_taken_to_owe_a_reply_until_it_generates_one(self, given):
        conn = _quiet_connection(**{given: 110.0})

        assert conn.awaiting_reply(now=120.0) and conn.quiet_since(now=120.0) is None
        conn._last_model_generated_at = conn._last_model_output_at = 121.0
        assert not conn.awaiting_reply(now=122.0) and conn.quiet_since(now=122.0) == 121.0

    def test_an_input_never_answered_stops_holding_the_call_without_being_taken_for_answered(self):
        """Noise or a silent turn must not hold the call for ever; the pool stops waiting, nothing is marked answered."""
        conn = _quiet_connection(_last_caller_speech_at=110.0)
        expired = 110.0 + realtime_pool._REPLY_WAIT_SECONDS + 0.1

        assert not conn.awaiting_reply(now=expired) and conn.quiet_since(now=expired) == 110.0
        assert conn.last_model_generated_at is None

    def test_words_generated_before_a_tool_result_are_no_answer_to_it(self):
        """Audio generated before a handoff or a tool result was taken for the answer to it once it had played out,
        and text sent then cut off the answer the model still owed."""
        conn = _quiet_connection(_last_model_generated_at=105.0, _last_tool_result_at=106.0, _last_model_output_at=107.0)

        assert conn.quiet_since(now=110.0) is None
        conn._last_model_generated_at = conn._last_model_output_at = 111.0
        assert conn.quiet_since(now=112.0) == 111.0

    def test_a_handover_starts_a_new_quiet_stretch(self):
        assert _quiet_connection(_last_agent_change_at=119.0).quiet_since(now=120.0) == 119.0


class TestRealtimeCall:
    """Application code reads and talks into a live call through ``RealtimeCall``, never through the connection's internals."""

    @pytest.fixture(autouse=True)
    def _transport(self, monkeypatch):
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))

    @staticmethod
    def _pool():
        pool = RealtimeConnectionPool()
        RealtimeConnectionPool._instance = pool
        try:
            pool._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass  # a test with no loop of its own reads the call's state only
        return pool

    @staticmethod
    def _connect(pool, session_id="s1", connected_ago=5.0):
        conn = _connection(session_id, loop=asyncio.get_running_loop())
        conn._connected_at = time.time() - connected_ago
        pool._connections[session_id] = conn
        return conn

    @pytest.mark.asyncio
    async def test_text_is_sent_once_the_call_has_been_quiet_a_moment_and_is_then_owed_a_reply(self):
        from agentkernel.pipeline import RealtimeCall

        pool, call = self._pool(), RealtimeCall("s1")
        assert await call.send_if_quiet("[Team update] J1 finished.") is False  # no call yet
        conn = self._connect(pool, connected_ago=0.5)
        assert await call.send_if_quiet("[Team update] J1 finished.") is False  # connected half a second ago

        conn._connected_at = time.time() - 5.0
        assert await call.send_if_quiet("[Team update] J1 finished.") is True
        assert conn.adapter.texts == ["[Team update] J1 finished."]
        assert conn.last_edge_text_at is None and conn.last_post_at is not None  # the application's words, not the caller's
        # The model owes the text a reply: a second one waits for it.
        assert await call.send_if_quiet("[Team update] J2 finished.") is False and call.quiet_for() is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("busy", ["caller", "output", "tool", "recovering"])
    async def test_a_declined_send_keeps_nothing(self, busy):
        from agentkernel.pipeline import RealtimeCall

        pool, call = self._pool(), RealtimeCall("s1")
        conn = self._connect(pool)
        if busy == "caller":
            conn._last_caller_speech_at = time.time()
        elif busy == "output":
            conn._audio_queue.put_nowait(("audio_delta", {"delta": "QUFB"}))
        elif busy == "tool":
            conn._tools_in_flight = 1
        else:
            conn._recovering = True

        assert await call.send_if_quiet("[Team update] J1 finished.") is False
        conn._last_caller_speech_at, conn._tools_in_flight, conn._recovering = None, 0, False
        if busy == "output":
            conn._audio_queue.get_nowait()
        await asyncio.sleep(0.05)
        assert conn.adapter.texts == []  # nothing waits in AK to be sent later

    @pytest.mark.asyncio
    async def test_text_for_one_quiet_stretch_is_not_sent_once_the_call_moved_on(self):
        from agentkernel.pipeline import RealtimeCall

        pool, call = self._pool(), RealtimeCall("s1")
        conn = self._connect(pool)
        stretch = call.quiet_since()

        conn._last_agent_change_at = conn._last_model_generated_at = conn._last_model_output_at = time.time() - 3.0
        assert call.quiet_since() != stretch
        assert await call.send_if_quiet("Take the next step.", quiet_since=stretch) is False
        assert await call.send_if_quiet("Take the next step.", quiet_since=call.quiet_since()) is True
        assert conn.adapter.texts == ["Take the next step."]

    @pytest.mark.asyncio
    async def test_a_failing_send_raises_and_owes_nothing(self):
        from agentkernel.pipeline import RealtimeCall

        pool, call = self._pool(), RealtimeCall("s1")
        conn = self._connect(pool)

        async def broken(text):
            raise ConnectionError("socket gone")

        conn.adapter.send_text = broken
        with pytest.raises(ConnectionError):
            await call.send_if_quiet("[Team update] J1 finished.")
        assert conn.last_post_at is None

    @pytest.mark.asyncio
    async def test_a_cancelled_send_owes_nothing(self):
        """An application that gives up on a send (its meeting ended) must not leave the call waiting for a reply."""
        from agentkernel.pipeline import RealtimeCall

        pool, call = self._pool(), RealtimeCall("s1")
        conn = self._connect(pool)
        started = asyncio.Event()

        async def hanging(text):
            started.set()
            await asyncio.Event().wait()

        conn.adapter.send_text = hanging
        send = asyncio.create_task(call.send_if_quiet("[Team update] J1 finished."))
        await asyncio.wait_for(started.wait(), 1.0)
        send.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send
        assert conn.last_post_at is None and call.quiet_for() is not None

    def test_a_send_from_another_event_loop_runs_on_the_pools(self, monkeypatch):
        """The check and the send run on the pool's loop; a web handler awaiting it from its own loop never blocks it."""
        from agentkernel.pipeline import RealtimeCall

        monkeypatch.setattr(RealtimeConnectionPool, "_instance", None)
        pool = RealtimeConnectionPool.initialize()
        thread = threading.Thread(target=pool.start, daemon=True)
        thread.start()
        try:
            conn = pool.get_or_create("s1", _Agent(), Runtime(InMemorySessionStore()), Session("s1"))
            conn._connected_at = time.time() - 5.0
            ran_on = []
            send = conn.adapter.send_text

            async def recorded(text):
                ran_on.append(asyncio.get_running_loop())
                await send(text)

            conn.adapter.send_text = recorded
            assert asyncio.run(RealtimeCall("s1").send_if_quiet("[Team update] J1 finished.")) is True
            assert ran_on == [pool._loop] and conn.adapter.texts == ["[Team update] J1 finished."]
        finally:
            ThreadRunner.shutdown_event.set()
            thread.join(timeout=5)

    @pytest.mark.asyncio
    async def test_what_the_call_is_doing(self):
        from agentkernel.pipeline import RealtimeCall

        pool, call = self._pool(), RealtimeCall("s1")
        assert (call.connected, call.quiet_for(), call.quiet_since(), call.agent_has_spoken) == (False, None, None, None)

        conn = self._connect(pool, connected_ago=3.0)
        assert call.quiet_for() == pytest.approx(3.0, abs=0.5)
        assert call.agent_has_spoken is False

        await conn.handle_framework_event("agent_changed", {"agent": "billing", "previous_agent": "general"})
        assert call.agent_has_spoken is False  # the new agent has not said a word
        await asyncio.sleep(0.01)
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB"})
        assert call.agent_has_spoken is True
        assert call.quiet_for() is None  # its words are still waiting to be played

    @pytest.mark.asyncio
    async def test_without_a_pool_nothing_is_reached(self, monkeypatch):
        from agentkernel.pipeline import RealtimeCall

        monkeypatch.setattr(RealtimeConnectionPool, "_instance", None)
        call = RealtimeCall("s1")
        assert await call.send_if_quiet("J1 finished.") is False
        assert (call.connected, call.quiet_for(), call.close()) == (False, None, False)

    def test_close_ends_the_sessions_connection(self, monkeypatch):
        from agentkernel.pipeline import RealtimeCall

        closed = []

        class Conn:
            closed = False
            is_ready = False

            async def close(self):
                closed.append(True)

        pool = RealtimeConnectionPool()
        loop = asyncio.new_event_loop()
        pool._loop = loop
        conn = Conn()
        pool._connections["s1"] = conn
        monkeypatch.setattr(RealtimeConnectionPool, "_instance", pool)
        try:
            assert RealtimeCall("s1").close() is True
            loop.run_until_complete(asyncio.sleep(0.01))
        finally:
            loop.close()

        assert closed == [True] and conn.closed
        assert "s1" not in pool._connections
        assert RealtimeCall("s1").close() is False

    @pytest.mark.asyncio
    async def test_nothing_is_sent_while_the_socket_is_being_recovered(self):
        """Text handed to a recovering adapter waits for its next socket, then cuts off the recovered agent."""
        from agentkernel.pipeline import RealtimeCall

        pool, call = self._pool(), RealtimeCall("s1")
        conn = self._connect(pool)

        await conn.handle_framework_event("done", {"status": "reconnecting"})
        conn._audio_queue.get_nowait()  # the turn's done, which the pacing loop would have sent on
        assert await call.send_if_quiet("[Team update] J1 finished.") is False and call.connected is False

        await conn.handle_framework_event("reconnected", {"agent": "general", "seconds": 1, "seeded": False})
        assert await call.send_if_quiet("[Team update] J1 finished.") is True
        assert conn.adapter.texts == ["[Team update] J1 finished."]


class TestRealtimeToolScope:
    """The scope an adapter whose framework runs its own tools opens for each call (``open_tool_scope``)."""

    class _NativeAdapter(_Adapter):
        runs_tools = True

    @pytest.fixture(autouse=True)
    def _transport(self, monkeypatch):
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))

    def _connection(self):
        return _connection(agent=_Agent(realtime_runner_cls=self._NativeAdapter), loop=asyncio.get_running_loop())

    @pytest.mark.asyncio
    async def test_a_tool_runs_in_the_calls_scope_until_its_task_ends(self):
        conn = self._connection()
        seen = []

        async def tool():
            ctx = await conn.adapter._open_tool_scope()
            seen.append((ctx.session.id, Session.current().id, Agent.current().name, conn.session._lock.locked(), conn.is_responding))

        await asyncio.create_task(tool())
        await asyncio.sleep(0.01)  # the holder lets the lock go once the tool's task has ended

        assert seen == [("s1", "s1", "general", True, True)]
        assert not conn.session._lock.locked() and conn._tools_in_flight == 0 and conn._scope_holders == set()
        assert conn.last_tool_result_at is not None

    @pytest.mark.asyncio
    async def test_a_tool_cancelled_while_the_session_is_busy_leaves_no_holder_behind(self):
        """Review: a tool cancelled while something else held the session lock left its holder waiting for it."""
        conn = self._connection()
        await conn.session._lock.acquire()  # something else has the session
        waiting = asyncio.create_task(conn.adapter._open_tool_scope())
        await asyncio.sleep(0.01)
        assert len(conn._scope_holders) == 1

        waiting.cancel()
        await asyncio.wait({waiting})
        await asyncio.sleep(0.01)

        assert conn._scope_holders == set() and conn._tools_in_flight == 0 and conn._tool_tasks == set()
        conn.session._lock.release()
        await asyncio.sleep(0.01)
        assert not conn.session._lock.locked()  # the holder never took it afterwards
        assert conn.last_tool_result_at is None

    @pytest.mark.asyncio
    async def test_a_holder_ended_before_it_had_the_lock_fails_the_tool_instead_of_hanging(self):
        conn = self._connection()
        await conn.session._lock.acquire()
        waiting = asyncio.create_task(conn.adapter._open_tool_scope())
        await asyncio.sleep(0.01)

        for holder in list(conn._scope_holders):
            holder.cancel()  # as at the loop's shutdown
        done, _ = await asyncio.wait({waiting}, timeout=1)

        assert done and isinstance(waiting.exception(), RuntimeError)
        await asyncio.sleep(0)
        assert conn._tools_in_flight == 0
        conn.session._lock.release()


class TestRealtimeObservers:
    """Realtime observers see the call: who connected, what each side said, who has the call."""

    class _Recorder:
        def __init__(self, fail=False):
            self.events = []
            self.fail = fail

        async def on_event(self, session, agent, event):
            if self.fail:
                raise RuntimeError("observer broke")
            self.events.append((event.type, event.agent, event.text, event.interrupted))

        def name(self):
            return "recorder"

    @pytest.fixture(autouse=True)
    def _transport(self, monkeypatch):
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))

    @pytest.mark.asyncio
    async def test_the_observers_follow_a_call_from_connect_to_close(self):
        recorder = self._Recorder()
        conn = _connection(agent=_Agent(name="general"), loop=asyncio.get_running_loop(), observers=[recorder])

        await conn.connect()
        await asyncio.sleep(0.05)
        await conn.handle_framework_event("caller_transcript", {"text": " I need a login page. "})
        await conn.handle_framework_event("transcript_delta", {"delta": "Sure, ", "message_id": "m1"})
        await conn.handle_framework_event("transcript_delta", {"delta": "a login page.", "message_id": "m1"})
        await conn.handle_framework_event("done", {"status": "completed"})
        await asyncio.sleep(0.05)
        await conn.handle_framework_event("agent_changed", {"agent": "billing", "previous_agent": "general"})
        await conn.handle_framework_event("transcript_delta", {"delta": "Billing here, your invoice", "message_id": "m2"})
        await asyncio.sleep(0.05)
        await conn.handle_framework_event("audio_delta", {"delta": "QUFB", "message_id": "m2"})
        await conn.handle_framework_event("interrupt", {})
        await asyncio.sleep(0.05)
        conn.send_text("example.com/login", from_edge=True)
        conn.send_text("[Team update] J1 finished.")  # the application's own: not reported
        await asyncio.sleep(0.05)
        await conn.close()

        assert recorder.events == [
            ("connected", "general", "", False),
            ("agent_changed", "general", "", False),
            ("caller_said", None, "I need a login page.", False),
            ("agent_said", "general", "Sure, a login page.", False),
            ("agent_changed", "billing", "", False),
            ("agent_said", "billing", "Billing here, your invoice", True),
            ("edge_text", None, "example.com/login", False),
            ("disconnected", "billing", "", False),
        ]

    @pytest.mark.asyncio
    async def test_a_failing_observer_is_logged_and_the_next_still_runs(self, caplog):
        recorder = self._Recorder()
        conn = _connection(loop=asyncio.get_running_loop(), observers=[self._Recorder(fail=True), recorder])

        with caplog.at_level(logging.ERROR, logger="ak.pipeline.realtime_pool"):
            await conn.connect()
            await conn.close()

        assert [event[0] for event in recorder.events] == ["connected", "disconnected"]
        assert "Realtime observer _Recorder failed on connected" in caplog.text

    @pytest.mark.asyncio
    async def test_a_failed_socket_is_reported_disconnected_once(self):
        recorder = self._Recorder()
        conn = _connection(loop=asyncio.get_running_loop(), observers=[recorder])
        await conn.connect()

        await conn.handle_framework_event("error", {"message": "socket closed"})
        await conn.close()

        assert [event[0] for event in recorder.events].count("disconnected") == 1

    @pytest.mark.asyncio
    async def test_text_from_the_edge_on_another_thread_reaches_the_observers(self):
        recorder = self._Recorder()
        conn = _connection(loop=asyncio.get_running_loop(), observers=[recorder])
        await conn.connect()

        typed = threading.Thread(target=lambda: conn.send_text("example.com", from_edge=True))
        typed.start()
        while typed.is_alive():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        await conn.close()

        assert ("edge_text", None, "example.com", False) in recorder.events

    @pytest.mark.asyncio
    async def test_a_call_without_observers_queues_nothing(self):
        conn = _connection(loop=asyncio.get_running_loop())
        await conn.connect()

        await conn.handle_framework_event("caller_transcript", {"text": "Hello"})

        assert conn._observer_events is None and conn._observer_task is None
        await conn.close()

    @pytest.mark.asyncio
    async def test_a_reconnect_reaches_the_observers(self):
        recorder = self._Recorder()
        conn = _connection(loop=asyncio.get_running_loop(), observers=[recorder])
        await conn.connect()

        await conn.handle_framework_event("reconnected", {"agent": "billing", "seconds": 2, "seeded": True})
        await conn.close()

        assert ("reconnected", "billing", "", False) in recorder.events

    def test_observers_registered_for_the_process_watch_the_calls_connected_after(self, monkeypatch):
        from agentkernel.pipeline import RealtimeCall

        monkeypatch.setattr(RealtimeConnectionPool, "_instance", None)
        recorder, dropped = self._Recorder(), self._Recorder()
        RealtimeCall.observe(recorder)
        RealtimeCall.observe(recorder)  # once only
        RealtimeCall.observe(dropped)
        RealtimeCall.unobserve(dropped)
        pool = RealtimeConnectionPool.initialize()
        thread = threading.Thread(target=pool.start, daemon=True)
        thread.start()
        try:
            conn = pool.get_or_create("s1", _Agent(), Runtime(InMemorySessionStore()), Session("s1"))
            assert conn._observers == [recorder]
        finally:
            ThreadRunner.shutdown_event.set()
            thread.join(timeout=5)

        kinds = [event[0] for event in recorder.events]
        assert kinds[0] == "connected" and kinds[-1] == "disconnected" and dropped.events == []


class TestRealtimeCallFacts:
    """``RealtimeCall`` tells when the caller, a handoff and a tool result last moved the call."""

    @pytest.fixture(autouse=True)
    def _transport(self, monkeypatch):
        monkeypatch.setattr(QueueTransportFactory, "create", staticmethod(lambda *a, **k: InMemoryTransport()))
        monkeypatch.setattr(RealtimeConnectionPool, "_instance", None)

    @pytest.mark.asyncio
    async def test_the_facts_follow_the_call(self):
        from agentkernel.pipeline import RealtimeCall

        pool = TestRealtimeCall._pool()
        call = RealtimeCall("s1")
        assert (call.last_caller_input_at, call.last_agent_change_at, call.last_tool_result_at) == (None, None, None)

        conn = TestRealtimeCall._connect(pool)
        await conn.handle_framework_event("speech_started", {})
        spoke = call.last_caller_input_at
        assert spoke is not None
        await asyncio.sleep(0.01)
        conn.send_text("example.com", from_edge=True)
        assert call.last_caller_input_at > spoke  # typed text counts as the caller's input too
        conn.send_text("[Team update] J1 finished.")  # the application's own words do not
        assert call.last_caller_input_at == conn.last_edge_text_at

        await conn.handle_framework_event("agent_changed", {"agent": "billing", "previous_agent": "general"})
        assert call.last_agent_change_at == conn.last_agent_change_at is not None

        await conn._execute_tool_and_reply("c1", "lookup", "{}")
        assert call.last_tool_result_at == conn.last_tool_result_at is not None

    @pytest.mark.asyncio
    async def test_the_facts_are_read_while_the_socket_is_recovered(self):
        from agentkernel.pipeline import RealtimeCall

        pool = TestRealtimeCall._pool()
        conn = TestRealtimeCall._connect(pool)
        await conn.handle_framework_event("speech_started", {})
        await conn.handle_framework_event("done", {"status": "reconnecting"})

        call = RealtimeCall("s1")
        assert call.connected is False
        assert call.last_caller_input_at == conn.last_caller_speech_at

    def test_a_closed_connection_or_no_pool_has_no_facts(self):
        from agentkernel.pipeline import RealtimeCall

        call = RealtimeCall("s1")
        assert call.last_caller_input_at is None  # no pool

        pool = TestRealtimeCall._pool()
        conn = types.SimpleNamespace(closed=True, last_caller_speech_at=1.0, last_edge_text_at=None)
        pool._connections["s1"] = conn
        assert call.last_caller_input_at is None
