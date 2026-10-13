"""Speaking into the live call at the right moment.

Text sent into a call starts a model response, and on Gemini Live it cuts off any response being
generated, so it must wait until nobody is talking: the model is not mid-response, it has answered
everything it was last given (the caller's words, text from the client's page, a tool result, this
link's own post), and nothing has happened for a moment. ``CallLink`` reads that from the
connection's turn state (``is_responding``, ``is_caller_speaking``, ``last_model_output_at``,
``last_caller_speech_at``, ``last_edge_text_at``, ``last_tool_result_at``), which Agent Kernel's
realtime pool exposes.
"""

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional, Union

from agentkernel.pipeline.realtime_pool import RealtimeConnection, RealtimeConnectionPool

_log = logging.getLogger("software_team.call")


class CallLink:
    """Talks into one live call: posts the team's updates when nobody is speaking, and nudges a stalled team.

    A watcher thread decides once per tick from the connection's turn state. The call is quiet
    when the model is not mid-response, owes no reply, and nothing has happened for
    ``QUIET_SECONDS``. The model owes a reply to whatever it was last given until its output is
    later than it, or until the reply has been waited for long enough: an input it never answered
    (noise, a silent turn) must not hold the call forever.
    """

    PREFIX = "[Team update, not from the client]"
    NUDGE = (
        "Nobody is speaking. If you just named a teammate, the call was not handed over: they cannot hear you, "
        "so call the transfer tool now. Otherwise take the next step yourself, without repeating what you already "
        "said: hand over, ask the client a question, or start the next job."
    )
    QUIET_SECONDS = 1.5
    # How long the caller's words, the client's typed text or a post may go unanswered before they
    # stop holding the call.
    REPLY_TIMEOUT_SECONDS = 15.0
    # The same for a tool result. Live, 8 s was too short: a reply sometimes started 10 s after a
    # tool's result (an agent just handed the call reads the whole conversation first), and the
    # update posted into that gap cut it off.
    TOOL_REPLY_TIMEOUT_SECONDS = 15.0
    NUDGE_AFTER_SECONDS = 5.0
    # How long the agents must have been silent before what they said counts as heard, and how long
    # a tool result's reply is waited for before then.
    HEARD_AFTER_SECONDS = 1.0
    HEARD_TOOL_REPLY_SECONDS = 4.0
    MAX_NUDGES_IN_A_ROW = 2
    POLL_SECONDS = 0.25
    MAX_PENDING = 20

    def __init__(
        self,
        session_id: str,
        should_nudge: Callable[[], bool] = lambda: False,
        connection: Optional[Callable[[], Optional[RealtimeConnection]]] = None,
        clock: Callable[[], float] = time.time,
        on_reconnect: Callable[[], None] = lambda: None,
    ):
        self.session_id = session_id
        self.should_nudge = should_nudge
        self.on_reconnect = on_reconnect
        self._connection = connection or self._pool_connection
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: deque[tuple[Union[str, Callable[[], str]], Optional[Callable[[], bool]]]] = deque()
        self._last_post_at: Optional[float] = None
        self._nudges_in_a_row = 0
        self._last_nudge_at: Optional[float] = None
        self._seen: Optional[RealtimeConnection] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def post(self, text: Union[str, Callable[[], str]], valid: Optional[Callable[[], bool]] = None) -> None:
        """Queue an update for the call; it is spoken into the call at the next quiet moment.

        :param text: The update, or a function that writes it when it is posted: an update that
            describes the project must say where it stands then, not when it was queued (live, a
            queued "the Worker delivers the project" was still posted after the delivery).
        :param valid: Checked just before posting; an update that no longer applies (a question
            that has timed out, a notice a newer one replaced) is dropped instead of posted.
        """
        with self._lock:
            if len(self._pending) >= self.MAX_PENDING:
                dropped, _ = self._pending.popleft()
                _log.warning(
                    f"Too many updates waiting for call {self.session_id}; dropped the oldest: {self._text(dropped)}"
                )
            self._pending.append((text, valid))
        self.start()

    @staticmethod
    def _text(text: Union[str, Callable[[], str]]) -> str:
        return text() if callable(text) else text

    @property
    def last_post_at(self) -> Optional[float]:
        """When this link last posted into the call (an update or a nudge); ``None`` before that."""
        return self._last_post_at

    def agent_spoke_since_handover(self) -> bool:
        """Whether the agent now holding the call has said anything since it was handed the call.

        Read from when the model generated its words, not when they were played: a handoff runs
        while the words before it are still queued. Without a live call there is nothing to check.
        """
        connection = self._connection()
        if connection is None or connection.closed:
            return True
        generated, changed = connection.last_model_generated_at, connection.last_agent_change_at
        return generated is not None and (changed is None or generated > changed)

    def heard(self) -> bool:
        """Whether everything the agents have said so far has been heard: no audio still waiting to play, none for a moment.

        The model generates its words faster than they are spoken, and Agent Kernel plays them at
        speaking pace, but a tool runs as soon as the model calls it, while the words before the call
        are still waiting to be played. A tool's result is given a moment to be answered too, as the
        model may say its words after the call. Without a live call there is nothing to wait for.
        """
        connection = self._connection()
        if connection is None or connection.closed:
            return True
        if connection.is_responding:
            return False
        now = self._clock()
        output, result = connection.last_model_output_at, connection.last_tool_result_at
        if result is not None and (output is None or output < result) and now - result < self.HEARD_TOOL_REPLY_SECONDS:
            return False
        return output is None or now - output >= self.HEARD_AFTER_SECONDS

    def caller_spoke_at(self) -> Optional[float]:
        """When the client last spoke or typed into the call; ``None`` before that."""
        connection = self._connection()
        return self._caller_input_at(connection) if connection is not None else None

    def start(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._watch, name=f"call-link-{self.session_id}", daemon=True)
                self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def tick(self) -> Optional[str]:
        """Post one queued update, else a nudge, if the call is quiet; return what was posted."""
        connection = self._connection()
        if connection is None or connection.closed:
            return None
        self._notice_reconnect(connection)
        now = self._clock()
        if not self._is_quiet(connection, now):
            return None

        text = self._next_pending()
        if text is None:
            if not self._nudge_due(connection, now):
                return None
            text = self.NUDGE
            self._nudges_in_a_row += 1
            self._last_nudge_at = now
        connection.send_text(f"{self.PREFIX} {text}")
        self._last_post_at = now
        _log.info(f"Posted into call {self.session_id}: {text}")
        return text

    def _next_pending(self) -> Optional[str]:
        with self._lock:
            while self._pending:
                text, valid = self._pending.popleft()
                if valid is None or valid():
                    return self._text(text)
                _log.info(f"Dropped an update for call {self.session_id} that no longer applies: {self._text(text)}")
        return None

    def _notice_reconnect(self, connection: RealtimeConnection) -> None:
        """A new model connection replaced the one this link was watching: its posts and nudges start over."""
        if connection is self._seen:
            return
        previous, self._seen = self._seen, connection
        if previous is None:
            return
        self._last_post_at = self._last_nudge_at = None
        self._nudges_in_a_row = 0
        _log.info(f"Call {self.session_id} has a new model connection")
        self.on_reconnect()

    @staticmethod
    def _caller_input_at(connection: RealtimeConnection) -> Optional[float]:
        return max(
            (t for t in (connection.last_caller_speech_at, connection.last_edge_text_at) if t is not None), default=None
        )

    def _owes_reply(self, connection: RealtimeConnection, now: float) -> bool:
        """Whether the model has still to answer something it was given, and is still being waited for."""
        output = connection.last_model_output_at
        given = (
            (connection.last_caller_speech_at, self.REPLY_TIMEOUT_SECONDS),
            (connection.last_edge_text_at, self.REPLY_TIMEOUT_SECONDS),
            (self._last_post_at, self.REPLY_TIMEOUT_SECONDS),
            (connection.last_tool_result_at, self.TOOL_REPLY_TIMEOUT_SECONDS),
        )
        return any(at is not None and (output is None or output < at) and now - at < wait for at, wait in given)

    def _latest(self, connection: RealtimeConnection) -> Optional[float]:
        times = (
            connection.last_model_output_at,
            connection.last_caller_speech_at,
            connection.last_edge_text_at,
            connection.last_tool_result_at,
            self._last_post_at,
        )
        return max((t for t in times if t is not None), default=None)

    def _is_quiet(self, connection: RealtimeConnection, now: float) -> bool:
        if connection.is_responding or connection.is_caller_speaking or self._owes_reply(connection, now):
            return False
        latest = self._latest(connection)
        return latest is None or now - latest >= self.QUIET_SECONDS

    def _nudge_due(self, connection: RealtimeConnection, now: float) -> bool:
        spoke = self._caller_input_at(connection)
        if self._last_nudge_at is not None and spoke is not None and spoke > self._last_nudge_at:
            self._nudges_in_a_row = 0
        if self._nudges_in_a_row >= self.MAX_NUDGES_IN_A_ROW or not self.should_nudge():
            return False
        latest = self._latest(connection)
        return latest is not None and now - latest >= self.NUDGE_AFTER_SECONDS

    def _watch(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                _log.exception(f"Call link for {self.session_id} failed a tick")
            self._stop.wait(self.POLL_SECONDS)

    def _pool_connection(self) -> Optional[RealtimeConnection]:
        pool = RealtimeConnectionPool.get()
        return pool.get_connection(self.session_id) if pool is not None else None
