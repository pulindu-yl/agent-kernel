import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from livekit.agents import ModelSettings, llm
from pydantic import ValidationError

from agentkernel.core.config import AKConfig
from agentkernel.core.model import AgentReplyText, AgentRequestImage, AgentRequestText, StreamChunk
from agentkernel.integration.livekit import livekit_handler
from agentkernel.integration.livekit.livekit_handler import AgentKernelVoiceAgent, LiveKitLLM


def _chat_context(text: str = "hello") -> llm.ChatContext:
    context = llm.ChatContext.empty()
    context.add_message(role="user", content=text)
    return context


class _FakeService:
    def __init__(self, *, chunks=None, reply="buffered", select_agent=True):
        self.agent = SimpleNamespace() if select_agent else None
        self._select_agent = select_agent
        self.chunks = chunks or []
        self.reply = reply
        self.selected = None
        self.stream_requests = None
        self.run_requests = None
        self.stream_drained = False
        self.stream_closed = False

    def select(self, *, name, session_id):
        self.selected = (name, session_id)
        if self._select_agent:
            self.agent = SimpleNamespace()

    async def run_multi(self, requests):
        self.run_requests = requests
        return AgentReplyText(response=self.reply)

    async def stream_multi(self, requests):
        self.stream_requests = requests
        try:
            for chunk in self.chunks:
                yield chunk
            self.stream_drained = True
        finally:
            self.stream_closed = True


async def _collect(agent: AgentKernelVoiceAgent, text: str = "hello") -> list[str]:
    return [item async for item in agent.llm_node(_chat_context(text), [], ModelSettings())]


@pytest.mark.asyncio
async def test_llm_node_streams_each_delta_and_fully_drains():
    service = _FakeService(chunks=[StreamChunk(delta="Hel"), StreamChunk(delta=""), StreamChunk(delta="lo"), StreamChunk(done=True)])
    agent = AgentKernelVoiceAgent(
        agent_name="assistant",
        session_id="livekit:RM_1",
        service=service,
    )

    assert await _collect(agent) == ["Hel", "lo"]
    assert service.stream_drained is True
    assert service.stream_closed is True
    assert service.run_requests is None
    assert len(service.stream_requests) == 1
    assert isinstance(service.stream_requests[0], AgentRequestText)
    assert service.stream_requests[0].prompt == "hello"


@pytest.mark.asyncio
async def test_llm_node_speaks_stream_error_once_and_skips_done():
    service = _FakeService(
        chunks=[
            StreamChunk(error="That request is blocked.", done=True),
            StreamChunk(error="That request is blocked.", done=True),
        ]
    )
    agent = AgentKernelVoiceAgent(
        agent_name="assistant",
        session_id="livekit:RM_1",
        streaming_mode="streaming",
        service=service,
    )

    assert await _collect(agent) == ["That request is blocked."]
    assert service.stream_drained is True


@pytest.mark.asyncio
async def test_buffered_mode_uses_run_multi():
    service = _FakeService(reply="safe buffered reply")
    agent = AgentKernelVoiceAgent(
        agent_name="assistant",
        session_id="livekit:RM_1",
        streaming_mode="buffered",
        service=service,
    )

    assert await _collect(agent) == ["safe buffered reply"]
    assert service.run_requests is not None
    assert service.stream_requests is None


@pytest.mark.asyncio
async def test_streaming_mode_reports_not_implemented_without_buffered_retry():
    class _UnsupportedService(_FakeService):
        async def stream_multi(self, requests):
            self.stream_requests = requests
            raise NotImplementedError("runner does not support streaming")
            yield  # pragma: no cover - keeps this method an async generator

    service = _UnsupportedService(reply="must not be used")
    agent = AgentKernelVoiceAgent(
        agent_name="assistant",
        session_id="livekit:RM_1",
        streaming_mode="streaming",
        service=service,
    )

    assert await _collect(agent) == [livekit_handler._STREAMING_UNSUPPORTED_MESSAGE]
    assert service.stream_requests is not None
    assert service.run_requests is None


def test_livekit_streaming_mode_config_defaults_and_validation(monkeypatch):
    monkeypatch.setenv("AK_CONFIG_PATH_OVERRIDE", "/nonexistent/config.yaml")
    monkeypatch.delenv("AK_LIVEKIT__STREAMING_MODE", raising=False)

    assert AKConfig().livekit.streaming_mode == "streaming"
    assert AKConfig().livekit.worker_name == "agent-kernel-worker"
    assert AKConfig(livekit={"streaming_mode": "streaming"}).livekit.streaming_mode == "streaming"
    assert AKConfig(livekit={"streaming_mode": "buffered"}).livekit.streaming_mode == "buffered"

    with pytest.raises(ValidationError):
        AKConfig(livekit={"streaming_mode": "auto"})
    with pytest.raises(ValidationError):
        AKConfig(livekit={"streaming_mode": "fallback"})
    with pytest.raises(ValidationError):
        AKConfig(livekit={"worker_name": ""})


def test_voice_agent_rejects_unknown_streaming_mode():
    with pytest.raises(ValueError, match="either 'streaming' or 'buffered'"):
        AgentKernelVoiceAgent(
            agent_name="assistant",
            session_id="livekit:RM_1",
            streaming_mode="auto",
            service=_FakeService(),
        )


@pytest.mark.asyncio
async def test_llm_node_builds_prompt_and_optional_image_request(monkeypatch):
    service = _FakeService(reply="vision reply")
    agent = AgentKernelVoiceAgent(
        agent_name="assistant",
        session_id="livekit:RM_1",
        frame_holder={"frame": object()},
        streaming_mode="buffered",
        service=service,
    )
    monkeypatch.setattr(livekit_handler, "_consume_video_frame", lambda holder: "encoded-frame")

    assert await _collect(agent, "what can you see?") == ["vision reply"]
    assert service.run_requests == [
        AgentRequestText(prompt="what can you see?"),
        AgentRequestImage(image_data="encoded-frame", mime_type="image/jpeg", name="webcam_frame"),
    ]


@pytest.mark.asyncio
async def test_closing_llm_node_closes_agent_kernel_stream():
    service = _FakeService(chunks=[StreamChunk(delta="first"), StreamChunk(delta="second")])
    agent = AgentKernelVoiceAgent(
        agent_name="assistant",
        session_id="livekit:RM_1",
        streaming_mode="streaming",
        service=service,
    )

    generation = agent.llm_node(_chat_context(), [], ModelSettings())
    assert await anext(generation) == "first"
    await generation.aclose()

    assert service.stream_closed is True
    assert service.stream_drained is False


@pytest.mark.asyncio
async def test_stream_failure_after_first_delta_does_not_append_apology():
    class _FailingService(_FakeService):
        async def stream_multi(self, requests):
            self.stream_requests = requests
            try:
                yield StreamChunk(delta="partial reply")
                raise RuntimeError("stream failed")
            finally:
                self.stream_closed = True

    service = _FailingService()
    agent = AgentKernelVoiceAgent(
        agent_name="assistant",
        session_id="livekit:RM_1",
        streaming_mode="streaming",
        service=service,
    )

    assert await _collect(agent) == ["partial reply"]
    assert service.stream_closed is True


@pytest.mark.asyncio
async def test_stream_cancellation_propagates_and_closes_upstream():
    class _CancelledService(_FakeService):
        async def stream_multi(self, requests):
            self.stream_requests = requests
            try:
                raise asyncio.CancelledError()
                yield  # pragma: no cover - keeps this method an async generator
            finally:
                self.stream_closed = True

    service = _CancelledService()
    agent = AgentKernelVoiceAgent(
        agent_name="assistant",
        session_id="livekit:RM_1",
        streaming_mode="streaming",
        service=service,
    )

    with pytest.raises(asyncio.CancelledError):
        await _collect(agent)
    assert service.stream_closed is True


@pytest.mark.asyncio
async def test_legacy_livekit_llm_chat_uses_proper_livekit_stream():
    service = _FakeService(chunks=[StreamChunk(delta="legacy "), StreamChunk(delta="stream"), StreamChunk(done=True)])
    bridge = LiveKitLLM(
        "assistant",
        "livekit:RM_legacy",
        streaming_mode="streaming",
        service=service,
    )

    async with bridge.chat(chat_ctx=_chat_context()) as stream:
        chunks = [chunk async for chunk in stream]

    assert [chunk.delta.content for chunk in chunks] == ["legacy ", "stream"]
    assert len({chunk.id for chunk in chunks}) == 1
    assert service.stream_drained is True
    assert service.stream_closed is True


@pytest.mark.asyncio
async def test_missing_agent_returns_safe_message_without_running():
    service = _FakeService(select_agent=False)
    agent = AgentKernelVoiceAgent(
        agent_name="missing",
        session_id="livekit:RM_1",
        service=service,
    )

    assert await _collect(agent) == [livekit_handler._NO_AGENT_MESSAGE]
    assert service.selected == ("missing", "livekit:RM_1")
    assert service.run_requests is None
    assert service.stream_requests is None


def test_session_id_prefers_room_sid_and_can_scope_to_participant():
    room = SimpleNamespace(sid="RM_123", name="reusable-room")

    assert livekit_handler._session_id_for_room(room) == "livekit:RM_123"
    assert livekit_handler._session_id_for_room(room, "user-7") == "livekit:RM_123:user-7"


def test_session_id_falls_back_to_room_name():
    room = SimpleNamespace(sid="", name="fallback-room")

    assert livekit_handler._session_id_for_room(room) == "livekit:fallback-room"


def test_participant_identity_prefers_dispatched_participant():
    ctx = SimpleNamespace(
        job=SimpleNamespace(
            participant=SimpleNamespace(identity="dispatched-user"),
            metadata=json.dumps({"participant_identity": "metadata-user"}),
        )
    )

    assert livekit_handler._participant_identity(ctx) == "dispatched-user"


def test_participant_identity_falls_back_to_dispatch_metadata():
    ctx = SimpleNamespace(
        job=SimpleNamespace(
            participant=None,
            metadata=json.dumps({"participant_identity": "metadata-user"}),
        )
    )

    assert livekit_handler._participant_identity(ctx) == "metadata-user"


@pytest.mark.parametrize("metadata", ["[]", "null", '"plain text"'])
def test_participant_identity_ignores_non_object_metadata(metadata):
    ctx = SimpleNamespace(job=SimpleNamespace(participant=None, metadata=metadata))

    assert livekit_handler._participant_identity(ctx) is None


def test_token_dispatch_includes_worker_and_participant_identity(monkeypatch):
    config = SimpleNamespace(
        livekit=SimpleNamespace(
            url="wss://livekit.example",
            api_key="key",
            api_secret="secret",
            worker_name="agent-kernel-worker",
        )
    )
    monkeypatch.setattr(livekit_handler.Config, "get", classmethod(lambda cls: config))

    class _FakeAccessToken:
        latest = None

        def __init__(self, api_key, api_secret):
            self.api_key = api_key
            self.api_secret = api_secret
            self.room_config = None
            _FakeAccessToken.latest = self

        def with_identity(self, identity):
            self.identity = identity
            return self

        def with_name(self, name):
            self.name = name
            return self

        def with_grants(self, grants):
            self.grants = grants
            return self

        def with_room_config(self, room_config):
            self.room_config = room_config
            return self

        def to_jwt(self):
            return "token"

    monkeypatch.setattr(livekit_handler.api, "AccessToken", _FakeAccessToken)
    handler = livekit_handler.AgentLiveKitRequestHandler()
    router = handler.get_router()
    token_endpoint = next(route.endpoint for route in router.routes if route.path == "/livekit/token")

    assert token_endpoint(room="support-room", identity="user-7") == {"token": "token"}
    dispatch = _FakeAccessToken.latest.room_config.agents[0]
    assert dispatch.agent_name == "agent-kernel-worker"
    assert json.loads(dispatch.metadata) == {"participant_identity": "user-7"}


@pytest.mark.asyncio
async def test_token_endpoint_rejects_requests_when_worker_has_stopped(monkeypatch):
    config = SimpleNamespace(
        livekit=SimpleNamespace(
            url="wss://livekit.example",
            api_key="key",
            api_secret="secret",
            worker_name="agent-kernel-worker",
        )
    )
    monkeypatch.setattr(livekit_handler.Config, "get", classmethod(lambda cls: config))
    handler = livekit_handler.AgentLiveKitRequestHandler()
    handler._worker_started = True
    handler._worker_task = asyncio.create_task(asyncio.sleep(0))
    await handler._worker_task
    router = handler.get_router()
    token_endpoint = next(route.endpoint for route in router.routes if route.path == "/livekit/token")

    with pytest.raises(HTTPException) as error:
        token_endpoint(room="support-room", identity="user-7")

    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_worker_failure_is_observed_and_logged(caplog):
    async def _fail():
        raise RuntimeError("worker disconnected")

    task = asyncio.create_task(_fail())
    await asyncio.sleep(0)
    handler = object.__new__(livekit_handler.AgentLiveKitRequestHandler)
    handler._log = logging.getLogger("test.livekit.worker")

    with caplog.at_level(logging.ERROR, logger="test.livekit.worker"):
        handler._worker_done(task)

    assert "LiveKit background worker failed" in caplog.text
    assert "worker disconnected" in caplog.text
