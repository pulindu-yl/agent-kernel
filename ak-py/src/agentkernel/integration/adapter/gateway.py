import logging
import threading
from typing import Callable, Optional

from ...core.util.async_bridge import run_async_sync
from .base import Source, StatefulEdgeAdapter
from .registry import StatefulEdgeRegistry


class GatewayRunner:
    """Hosts a stateful :class:`StatefulEdgeAdapter` (spec #524 §7).

    The bidirectional sibling of :class:`WebhookRESTRequestHandler` and ``PollerRunner``: a
    gateway owns a long-lived platform connection, feeds the input queue with what it reads, and
    delivers replies back through that same connection. ``start()`` runs the adapter's own loop
    on an already-constructed instance.

    A gateway must run in the same process as the Response Handler: it registers itself as the
    platform's outbound adapter as it starts, so replies route back to its live connection.

    It runs until the process shuts down, or until :meth:`stop` is called from any thread, such as
    when an application ends the meeting a room was opened for (``GatewayPool``).
    """

    _log = logging.getLogger("ak.integration.gateway_runner")
    GRACEFUL_STOP_TIMEOUT_SECONDS: float = 10.0
    """Bound on the adapter's own teardown before it is force-cancelled."""

    def __init__(self, adapter: StatefulEdgeAdapter):
        """
        :param adapter: The gateway adapter to host.
        :raises ValueError: If the adapter's ``source`` is not ``Source.REALTIME``.
        """
        if adapter.source is not Source.REALTIME:
            raise ValueError(
                f"{type(adapter).__name__} is a {adapter.source} adapter: "
                "host it with WebhookRESTRequestHandler or PollerRunner, not GatewayRunner"
            )
        self._adapter = adapter
        self._stop_requested = threading.Event()
        self._started = threading.Event()
        self._finished = threading.Event()

    @property
    def adapter(self) -> StatefulEdgeAdapter:
        """The hosted adapter (IOHandler names its thread after it)."""
        return self._adapter

    def start(self) -> None:
        """Register the outbound adapter, then run the gateway's blocking loop.

        The registration is what lets the Response Handler resolve this live connection by the
        adapter's ``session_id``. It happens here rather than in the adapter's constructor so the side
        effect is tied to being hosted, not to merely existing.
        """
        self._started.set()
        try:
            if self._stop_requested.is_set():
                self._log.info(f"Gateway {self._adapter.name} was stopped before it started")
                return
            session_id = getattr(self._adapter, "session_id", None)
            StatefulEdgeRegistry.register(self._adapter, session_id=session_id)
            try:
                run_async_sync(self._run_until_shutdown())
            finally:
                if session_id:
                    # Only its own entry: a gateway opened for the same session since this one was stopped may
                    # already have registered in its place.
                    StatefulEdgeRegistry.unregister(session_id, adapter=self._adapter)
        finally:
            self._finished.set()

    def stop(self, timeout: Optional[float] = None) -> bool:
        """Stop the gateway; callable from any thread, before or after :meth:`start`.

        The runner stops its adapter on the adapter's own loop, as at shutdown, and ``start()`` returns. A runner
        stopped before it starts never starts its adapter.

        :param timeout: Seconds to wait for ``start()`` to return; ``None`` waits as long as it takes. Ignored
            when it never started.
        :return: Whether ``start()`` has returned, or never ran.
        """
        self._stop_requested.set()
        if not self._started.is_set():
            return True
        return self._finished.wait(timeout)

    async def _run_until_shutdown(self) -> None:
        import asyncio

        from ...pipeline.thread_runner import ThreadRunner

        task = asyncio.create_task(self._adapter.start())

        while not ThreadRunner.shutdown_event.is_set() and not self._stop_requested.is_set():
            if task.done():
                task.result()  # raise if exception occurred
                return
            await asyncio.sleep(0.5)

        reason = "it was stopped" if self._stop_requested.is_set() else "the shutdown event"
        self._log.info(f"Gateway {self._adapter.name} stopping: {reason}")

        try:
            await self._adapter.stop()
        except Exception:
            self._log.exception(f"Error calling stop() on Gateway {self._adapter.name}")

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self.GRACEFUL_STOP_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            self._log.warning(f"Gateway {self._adapter.name} did not stop within {self.GRACEFUL_STOP_TIMEOUT_SECONDS}s; cancelling")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class GatewayPool:
    """One gateway per session, opened when a session needs one and closed when it ends.

    For an application that gives each meeting a room of its own: ``open(meeting_id)`` builds the gateway with
    ``factory(meeting_id)`` and hosts it on a thread of its own, unless it is already open, and ``close`` stops it.
    A gateway whose ``start()`` fails is logged and forgotten, so the next ``open`` starts a new one. Like
    ``GatewayRunner``, it runs in the process of the Response Handler. Thread-safe.
    """

    _log = logging.getLogger("ak.integration.gateway_pool")
    STOP_TIMEOUT_SECONDS: float = 10.0
    """How long ``close`` waits for a gateway to leave by default."""

    def __init__(self, factory: Callable[[str], StatefulEdgeAdapter]):
        """
        :param factory: Builds the gateway for a session: a ``StatefulEdgeAdapter`` whose ``session_id`` is that
            session, such as ``lambda sid: LiveKitEdgeGateway(session_id=sid, agent_name="pm")``.
        """
        self._factory = factory
        self._lock = threading.Lock()
        self._runners: dict[str, GatewayRunner] = {}

    def open(self, session_id: str) -> StatefulEdgeAdapter:
        """The session's gateway, started on a thread of its own if it is not open yet.

        :raises ValueError: The factory built a gateway for another session, or one ``GatewayRunner`` cannot host.
        :raises Exception: Whatever the factory raised, such as missing settings; nothing is opened then.
        """
        with self._lock:
            runner = self._runners.get(session_id)
            if runner is not None:
                return runner.adapter
            adapter = self._factory(session_id)
            if getattr(adapter, "session_id", None) != session_id:
                raise ValueError(f"The gateway factory built a gateway for session {getattr(adapter, 'session_id', None)!r}, not {session_id!r}")
            runner = GatewayRunner(adapter)
            self._runners[session_id] = runner
        threading.Thread(target=self._host, args=(session_id, runner), name=f"gateway-{session_id}", daemon=True).start()
        return adapter

    def get(self, session_id: str) -> Optional[StatefulEdgeAdapter]:
        """The session's open gateway, or ``None``."""
        with self._lock:
            runner = self._runners.get(session_id)
        return runner.adapter if runner is not None else None

    def is_open(self, session_id: str) -> bool:
        """Whether the session has a gateway open."""
        with self._lock:
            return session_id in self._runners

    def close(self, session_id: str, timeout: float = STOP_TIMEOUT_SECONDS) -> bool:
        """Stop the session's gateway; returns whether it had one. Waits up to ``timeout`` for it to leave."""
        with self._lock:
            runner = self._runners.pop(session_id, None)
        if runner is None:
            return False
        if not runner.stop(timeout):
            self._log.warning(f"The gateway of session {session_id} did not stop within {timeout}s")
        return True

    def close_all(self, timeout: float = STOP_TIMEOUT_SECONDS) -> None:
        """Stop every open gateway."""
        with self._lock:
            session_ids = list(self._runners)
        for session_id in session_ids:
            self.close(session_id, timeout)

    def _host(self, session_id: str, runner: GatewayRunner) -> None:
        """Run one gateway until it stops; a gateway that fails is forgotten, so the next ``open`` starts a new one."""
        try:
            runner.start()
        except Exception:
            self._log.exception(f"The gateway of session {session_id} failed")
        finally:
            with self._lock:
                if self._runners.get(session_id) is runner:
                    del self._runners[session_id]
