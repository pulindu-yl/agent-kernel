"""GatewayRunner: hosting a stateful gateway adapter, and its outbound registration."""

from typing import Dict

import pytest

from agentkernel.core.model import AgentReply
from agentkernel.integration.adapter.base import Source, StatefulEdgeAdapter
from agentkernel.integration.adapter.gateway import GatewayRunner
from agentkernel.integration.adapter.registry import StatefulEdgeRegistry


class FakeStatefulEdgeAdapter(StatefulEdgeAdapter):
    name = "fakegateway"

    def __init__(self):
        self.started = False
        self.session_id = "test-session-123"

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        pass

    async def deliver(self, reply: AgentReply, reply_context: Dict[str, str]) -> None:
        pass

    async def deliver_chunk(self, chunk, reply_context: Dict[str, str]) -> None:
        pass

    async def deliver_error(self, message: str, reply_context: Dict[str, str]) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_registry():
    StatefulEdgeRegistry.reset()
    yield
    StatefulEdgeRegistry.reset()


def test_rejects_a_non_realtime_source():
    class WebhookGateway(FakeStatefulEdgeAdapter):
        source = Source.WEBHOOK

    with pytest.raises(ValueError, match="not GatewayRunner"):
        GatewayRunner(WebhookGateway())


def test_adapter_property_is_the_hosted_adapter():
    adapter = FakeStatefulEdgeAdapter()
    assert GatewayRunner(adapter).adapter is adapter


def test_start_registers_and_unregisters_the_outbound_adapter():
    adapter = FakeStatefulEdgeAdapter()
    was_registered = False

    async def fake_start():
        adapter.started = True
        nonlocal was_registered
        # Check if it's registered WHILE running
        was_registered = StatefulEdgeRegistry.get(adapter.session_id) is adapter

    adapter.start = fake_start
    GatewayRunner(adapter).start()

    assert adapter.started is True
    assert was_registered is True
    # Verify it cleans up properly after the blocking loop exits
    assert StatefulEdgeRegistry.get(adapter.session_id) is None


class StayingGateway(FakeStatefulEdgeAdapter):
    """A gateway that stays in its room until stopped, as a live one does."""

    def __init__(self, session_id="room-1", fail=False):
        super().__init__()
        self.session_id = session_id
        self.fail = fail
        self.stopped = False
        self._left = None

    async def start(self) -> None:
        import asyncio

        if self.fail:
            raise RuntimeError("could not join the room")
        self._left = asyncio.Event()
        self.started = True
        await self._left.wait()

    async def stop(self) -> None:
        self.stopped = True
        if self._left is not None:
            self._left.set()


def _wait(condition, timeout=3.0):
    import time

    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.01)


class TestGatewayRunnerStop:
    def test_stop_from_another_thread_ends_a_running_gateway(self):
        import threading

        adapter = StayingGateway()
        runner = GatewayRunner(adapter)
        host = threading.Thread(target=runner.start, daemon=True)
        host.start()
        _wait(lambda: adapter.started)
        assert StatefulEdgeRegistry.get("room-1") is adapter

        assert runner.stop(timeout=5) is True

        host.join(timeout=1)
        assert adapter.stopped and not host.is_alive()
        assert StatefulEdgeRegistry.get("room-1") is None

    def test_a_gateway_stopped_before_it_starts_never_joins(self):
        adapter = StayingGateway()
        runner = GatewayRunner(adapter)

        assert runner.stop(timeout=1) is True  # nothing was running
        runner.start()  # returns at once

        assert adapter.started is False
        assert StatefulEdgeRegistry.get("room-1") is None


class TestGatewayPool:
    def test_a_session_gets_one_gateway_until_it_is_closed(self):
        from agentkernel.integration.adapter import GatewayPool

        built = []

        def factory(session_id):
            built.append(session_id)
            return StayingGateway(session_id)

        pool = GatewayPool(factory)
        gateway = pool.open("room-1")
        _wait(lambda: gateway.started)

        assert pool.open("room-1") is gateway and built == ["room-1"]
        assert pool.is_open("room-1") and pool.get("room-1") is gateway

        assert pool.close("room-1", timeout=5) is True
        assert gateway.stopped and not pool.is_open("room-1") and pool.get("room-1") is None
        assert pool.close("room-1") is False

        reopened = pool.open("room-1")
        assert reopened is not gateway and built == ["room-1", "room-1"]
        pool.close_all(timeout=5)
        assert not pool.is_open("room-1")

    def test_a_gateway_leaving_slowly_does_not_unregister_its_replacement(self):
        import asyncio
        import threading

        from agentkernel.integration.adapter import GatewayPool

        release = threading.Event()

        class SlowToLeave(StayingGateway):
            async def stop(self):
                while not release.is_set():
                    await asyncio.sleep(0.01)
                await super().stop()

        built = []

        def factory(session_id):
            built.append((SlowToLeave if not built else StayingGateway)(session_id))
            return built[-1]

        pool = GatewayPool(factory)
        old = pool.open("room-1")
        _wait(lambda: old.started)
        old_runner = pool._runners["room-1"]
        pool.close("room-1", timeout=0)
        new = pool.open("room-1")
        _wait(lambda: new.started and StatefulEdgeRegistry.get("room-1") is new)

        release.set()
        assert old_runner.stop(timeout=5)

        assert StatefulEdgeRegistry.get("room-1") is new
        pool.close_all(timeout=5)
        assert StatefulEdgeRegistry.get("room-1") is None

    def test_a_gateway_that_fails_to_start_is_forgotten(self, caplog):
        import logging

        from agentkernel.integration.adapter import GatewayPool

        pool = GatewayPool(lambda session_id: StayingGateway(session_id, fail=True))
        with caplog.at_level(logging.ERROR, logger="ak.integration.gateway_pool"):
            pool.open("room-1")
            _wait(lambda: not pool.is_open("room-1"))

        assert "The gateway of session room-1 failed" in caplog.text

    def test_a_factory_error_opens_nothing(self):
        from agentkernel.integration.adapter import GatewayPool

        def factory(session_id):
            raise ValueError("no LiveKit settings")

        pool = GatewayPool(factory)
        with pytest.raises(ValueError, match="no LiveKit settings"):
            pool.open("room-1")
        assert not pool.is_open("room-1")

    def test_a_gateway_built_for_another_session_is_refused(self):
        from agentkernel.integration.adapter import GatewayPool

        pool = GatewayPool(lambda session_id: StayingGateway("room-2"))
        with pytest.raises(ValueError, match="for session 'room-2', not 'room-1'"):
            pool.open("room-1")
        assert not pool.is_open("room-1")
