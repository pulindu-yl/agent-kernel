import asyncio
import base64
import inspect
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from fastapi import APIRouter, WebSocket

from ...core.config import AKConfig
from ...core.model import AgentReply, AgentRequestText, StreamChunk
from ...core.realtime import EDGE_SAMPLE_RATE
from ..adapter.base import StatefulEdgeAdapter
from ..adapter.registry import StatefulEdgeRegistry

_log = logging.getLogger("ak.integration.browser")

INTEGRATION_NAME = "browser"
# Cap on inbound requests staged for the sender task (~6 s of audio), as on the LiveKit edge: it only
# bites when the broker is slow or down, where dropping a frame beats growing the queue without bound.
_MAX_PENDING_REQUESTS = 64


class BrowserVoiceEdge(StatefulEdgeAdapter):
    """One browser's live call over a WebSocket, bridged to the Agent Kernel queue, with no media server.

    The browser sends its microphone and typed words; the edge plays the agents' voices and tells the browser
    who is speaking and what each agent said. One socket carries both directions, in order, so a change of
    speaker reaches the browser exactly where the new agent's audio starts.

    The wire protocol:

    - **Binary frames**, both ways: 16-bit little-endian mono PCM at ``EDGE_SAMPLE_RATE`` (24 kHz). Up, the
      microphone; down, the agent speaking, paced by the realtime pool.
    - **Text frames**, JSON objects with a ``type``. From the browser: ``{"type": "hello", "name": ...}`` (the
      caller joined: the agent greets them by this name) and ``{"type": "text", "text": ...}`` (typed words).
      To the browser:

      - ``{"type": "agent", "agent": ..., "agents": [...]}``: the agent speaking from here on; ``agents``, every
        agent the conversation can hand off to, on the first one only.
      - ``{"type": "said", "agent": ..., "text": ..., "at": ...}``: what an agent said, once its turn ends; ``at``, the
        server's time (epoch seconds), so a page can order it among words it gets from elsewhere.
      - ``{"type": "interrupt"}``: the caller barged in, so drop the audio still queued.
      - ``{"type": "error", "message": ...}``: the turn failed.

    Served by ``BrowserVoiceEndpoint``, which creates one edge per connection and registers it under its session,
    so the Response Handler routes the session's output back to this socket.
    """

    name = INTEGRATION_NAME
    # What the agent is told when the caller joins; ``{name}`` is the name the browser sent. The LiveKit edge's
    # wording, so an agent prompt written for one edge works with the other.
    GREETING = "A user named {name} just connected their microphone. Say a very short hello to them and confirm you are connected!"
    # Longest caller name and typed message passed on: the name goes into the agent's prompt.
    MAX_NAME = 40
    MAX_TEXT = 4000
    # Messages waiting to be sent to the browser. The pool paces audio, so this stays small; a browser that
    # stops reading fills it, and is disconnected rather than buffered without bound.
    MAX_OUTBOX = 2000
    # When the call ends, how long what the caller said last may take to reach the input queue.
    FLUSH_SECONDS = 2.0

    def __init__(self, websocket: WebSocket, session_id: str, agent_name: Optional[str] = None):
        """
        :param websocket: The accepted connection.
        :param session_id: The session the call runs in.
        :param agent_name: The agent the session is bound to; ``None`` selects the runner's default agent.
        """
        super().__init__()
        self._websocket = websocket
        self.session_id = session_id
        self.agent = agent_name or None
        # Microphone audio is accumulated to one input-queue message of this many bytes, as on the LiveKit edge.
        self._input_batch_bytes = int(EDGE_SAMPLE_RATE * 2 * (AKConfig.get().execution.realtime.input_batch_ms / 1000))

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending: Optional[asyncio.Queue] = None
        self._outbox: Optional[asyncio.Queue] = None
        self._receiver: Optional[asyncio.Task] = None
        self._closed = False
        # Output state, kept by the Response Handler thread that delivers this session's chunks in order.
        self._speaker: Optional[str] = None
        self._transcript: list[str] = []
        self._interrupted = False

    # -- lifecycle ---------------------------------------------------------------------------

    async def start(self) -> None:
        """Bridge the socket until it closes, or ``stop`` is called."""
        self._loop = asyncio.get_running_loop()
        self._init_producer()
        self._pending = asyncio.Queue(maxsize=_MAX_PENDING_REQUESTS)
        self._outbox = asyncio.Queue(maxsize=self.MAX_OUTBOX)
        drain = asyncio.create_task(self._drain_requests())
        sender = asyncio.create_task(self._send_outbox())
        self._receiver = asyncio.create_task(self._receive())
        try:
            # Not awaited directly: ``stop`` cancels the receiver, which must not cancel this caller too.
            await asyncio.wait({self._receiver})
        finally:
            self._closed = True
            sender.cancel()
            # What the caller said last still reaches the queue, in order, behind what was staged before it.
            if not self._pending.full():
                self._pending.put_nowait(None)
            await asyncio.wait({drain}, timeout=self.FLUSH_SECONDS)
            drain.cancel()
            await asyncio.gather(drain, sender, return_exceptions=True)
            _log.info(f"Browser call for session {self.session_id} ended")

    async def stop(self, code: int = 1000, reason: str = "") -> None:
        """Close the socket and end ``start``.

        :param code: The WebSocket close code the browser receives.
        :param reason: The close reason the browser receives.
        """
        if self._closed:
            return
        self._closed = True
        try:
            await self._websocket.close(code=code, reason=reason)
        except Exception:
            pass  # already closed by the browser
        if self._receiver is not None:
            self._receiver.cancel()

    # -- inbound: browser -> queue ---------------------------------------------------------------

    async def _receive(self) -> None:
        buffer = bytearray()
        while True:
            message = await self._websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            audio, text = message.get("bytes"), message.get("text")
            if audio:
                if self._input_batch_bytes == 0:
                    self._stage_request(self._enqueue_audio(audio))
                    continue
                buffer.extend(audio)
                if len(buffer) >= self._input_batch_bytes:
                    self._stage_request(self._enqueue_audio(bytes(buffer)))
                    buffer.clear()
            elif text is not None:
                self._on_text(text)
        if buffer:
            self._stage_request(self._enqueue_audio(bytes(buffer)))

    def _on_text(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            _log.debug(f"Ignored a text frame that is not JSON on session {self.session_id}")
            return
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        if kind == "hello":
            name = " ".join(str(message.get("name") or "").split())[: self.MAX_NAME] or "there"
            _log.info(f"Browser caller '{name}' joined session {self.session_id}")
            self._stage_request(AgentRequestText(prompt=self.GREETING.format(name=name)))
        elif kind == "text":
            text = str(message.get("text") or "").strip()[: self.MAX_TEXT]
            if text:
                self._stage_request(AgentRequestText(prompt=text))
        else:
            _log.debug(f"Ignored a browser message of type {kind!r} on session {self.session_id}")

    def _stage_request(self, request) -> None:
        if self._pending is None:
            return
        try:
            self._pending.put_nowait(self._enqueue(request))
        except asyncio.QueueFull:
            _log.warning(f"Realtime input queue is full for session {self.session_id}; dropping a frame (broker slow or unreachable?)")

    async def _drain_requests(self) -> None:
        """Send staged requests to the input queue in order, off the socket's loop (``enqueue`` may talk to a broker),
        until ``None`` marks the end of the call."""
        assert self._pending is not None
        while True:
            inbound = await self._pending.get()
            if inbound is None:
                return
            try:
                await asyncio.to_thread(self._producer.enqueue, self.name, inbound)
            except Exception as e:
                _log.error(f"Failed to enqueue request for session {self.session_id}: {e}")

    # -- outbound: queue -> browser --------------------------------------------------------------

    async def deliver_chunk(self, chunk: StreamChunk, reply_context: Dict[str, str]) -> None:
        """Translate one streamed chunk to what the browser receives.

        Audio plays as it comes; an agent's words are sent as one ``said`` message when its turn ends, or when the
        call moves to another agent; a barge-in drops the audio queued in the browser and the half-said words.
        """
        if chunk.error:
            _log.error(f"Turn failed on session {self.session_id}: {chunk.error}")
            self._transcript.clear()
            self._interrupted = False
            self._send({"type": "interrupt"})
            await self.deliver_error(self.ERROR_MESSAGE, reply_context)
            return

        event = chunk.event
        if event is not None:
            if event.type == "audio_delta":
                audio = base64.b64decode(event.content)
                if audio:
                    self._send(audio)
            elif event.type == "text_delta":
                self._transcript.append(event.content)
            elif event.type == "interrupt":
                self._interrupted = True
                self._transcript.clear()
                self._send({"type": "interrupt"})
            elif event.type == "agent_changed":
                self._flush_said()
                self._speaker = event.agent
                message: Dict[str, Any] = {"type": "agent", "agent": event.agent}
                if event.agents:
                    message["agents"] = list(event.agents)
                self._send(message)

        if chunk.done:
            if self._interrupted:
                self._interrupted = False
                self._transcript.clear()
            else:
                self._flush_said()

    async def deliver(self, reply: AgentReply, reply_context: Dict[str, str]) -> None:
        """Send a completed reply as the speaking agent's words (the non-streamed fallback)."""
        text = getattr(reply, "response", "") or str(reply)
        if text:
            self._send({"type": "said", "agent": self._speaker, "text": text, "at": time.time()})

    async def deliver_error(self, message: str, reply_context: Dict[str, str]) -> None:
        """Tell the browser a turn failed, so the caller is never left in silence."""
        self._send({"type": "error", "message": message})

    def _flush_said(self) -> None:
        text = "".join(self._transcript).strip()
        self._transcript.clear()
        if text:
            _log.info(f"AI response: {text}")
            self._send({"type": "said", "agent": self._speaker, "text": text, "at": time.time()})

    def _send(self, message: Union[bytes, Dict[str, Any]]) -> None:
        """Queue ``message`` for the browser. Thread-safe: chunks are delivered from the Response Handler's thread."""
        loop = self._loop
        if self._closed or loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._put, message)
        except RuntimeError:
            pass  # the socket's loop closed while this was delivered

    def _put(self, message: Union[bytes, Dict[str, Any]]) -> None:
        if self._closed or self._outbox is None:
            return
        try:
            self._outbox.put_nowait(message)
        except asyncio.QueueFull:
            _log.warning(f"Browser for session {self.session_id} is not reading its call; disconnecting it")
            asyncio.ensure_future(self.stop(1013, "not reading"))

    async def _send_outbox(self) -> None:
        assert self._outbox is not None
        while True:
            message = await self._outbox.get()
            try:
                if isinstance(message, bytes):
                    await self._websocket.send_bytes(message)
                else:
                    await self._websocket.send_text(json.dumps(message))
            except Exception:
                # The browser went away: the receiver sees the disconnect and ends the call.
                return


class BrowserVoiceEndpoint:
    """Serves browser voice calls on a WebSocket route: one ``BrowserVoiceEdge`` per connection.

    ``session_for`` decides which session a connection joins, such as a meeting id taken from the route's path,
    after any check the application needs (the origin, a login, that the meeting exists); ``None`` refuses the
    connection. A new connection for a session that already has one (the page reloaded, a second tab) replaces it.
    Sessions come and go with their connections, so one process serves many calls at once with nothing started
    in advance.

    .. code-block:: python

        endpoint = BrowserVoiceEndpoint(lambda ws: ws.path_params["meeting"], agent_name="pm")
        RESTAPI.add(endpoint.router("/calls/{meeting}"))
        IOHandler.run()
    """

    # Close codes the browser receives: refused by ``session_for``, and replaced by a newer connection.
    REFUSED = 4403
    REPLACED = 4409

    def __init__(
        self,
        session_for: Callable[[WebSocket], Union[Optional[str], Awaitable[Optional[str]]]],
        agent_name: Optional[str] = None,
        edge_class: type = BrowserVoiceEdge,
    ):
        """
        :param session_for: The session a connection joins, or ``None`` to refuse it; may be a coroutine function.
        :param agent_name: The agent every session is bound to; ``None`` selects the runner's default agent.
        :param edge_class: The edge to serve each connection with, such as a ``BrowserVoiceEdge`` subclass.
        """
        self._session_for = session_for
        self._agent_name = agent_name
        self._edge_class = edge_class
        self._edges: Dict[str, BrowserVoiceEdge] = {}

    def router(self, path: str = "/voice") -> APIRouter:
        """A router serving the WebSocket at ``path``, to add with ``RESTAPI.add``."""
        router = APIRouter()
        router.add_api_websocket_route(path, self.serve)
        return router

    def connected(self, session_id: str) -> bool:
        """Whether a browser is connected to the session now."""
        return session_id in self._edges

    async def disconnect(self, session_id: str) -> None:
        """Close the session's browser connection, if there is one, as when the application ends a meeting."""
        edge = self._edges.get(session_id)
        if edge is not None:
            await edge.stop(1000, "ended")

    async def serve(self, websocket: WebSocket) -> None:
        """Serve one connection until it closes."""
        session_id = self._session_for(websocket)
        if inspect.isawaitable(session_id):
            session_id = await session_id
        if not session_id:
            await websocket.close(code=self.REFUSED)
            return
        await websocket.accept()
        edge = self._edge_class(websocket, session_id, agent_name=self._agent_name)
        previous = self._edges.get(session_id)
        self._edges[session_id] = edge
        if previous is not None:
            await previous.stop(self.REPLACED, "replaced by a newer connection")
        StatefulEdgeRegistry.register(edge, session_id=session_id)
        try:
            await edge.start()
        finally:
            if self._edges.get(session_id) is edge:
                del self._edges[session_id]
            if StatefulEdgeRegistry.get(session_id) is edge:
                StatefulEdgeRegistry.unregister(session_id)
