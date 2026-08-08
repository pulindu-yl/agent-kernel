import asyncio
import base64
import io
import json
import logging
import uuid
from collections.abc import AsyncIterable
from typing import Awaitable, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException
from livekit import agents, api, rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, ModelSettings, WorkerOptions, WorkerType, llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.plugins import deepgram, openai, silero
from PIL import Image

from ...api import RESTRequestHandler
from ...core import AgentService, Config
from ...core.model import AgentRequest, AgentRequestImage, AgentRequestText

logger = logging.getLogger("ak.api.livekit")

_NO_AGENT_MESSAGE = "No Agent Kernel agent is available to handle this request."
_INTERNAL_ERROR_MESSAGE = "I'm sorry, I encountered an internal error while processing your request."
_STREAMING_UNSUPPORTED_MESSAGE = "This agent does not support streaming. Set livekit.streaming_mode to buffered."
_VALID_STREAMING_MODES = frozenset({"streaming", "buffered"})


def _validate_streaming_mode(streaming_mode: str) -> str:
    if streaming_mode not in _VALID_STREAMING_MODES:
        raise ValueError("LiveKit streaming_mode must be either 'streaming' or 'buffered'")
    return streaming_mode


class LiveKitLLM(llm.LLM):
    """LiveKit LLM adapter retained for ``chat()`` compatibility."""

    def __init__(
        self,
        agent_name: str | None = None,
        session_id: str | None = None,
        frame_holder: Optional[dict] = None,
        *,
        streaming_mode: str = "streaming",
        service: AgentService | None = None,
    ) -> None:
        super().__init__()
        self._agent_name = agent_name
        self._session_id = session_id
        self._frame_holder = frame_holder
        self._streaming_mode = _validate_streaming_mode(streaming_mode)
        self._service = service

    @property
    def model(self) -> str:
        return "agent-kernel"

    @property
    def provider(self) -> str:
        return "agent-kernel"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict] = NOT_GIVEN,
    ) -> llm.LLMStream:
        del parallel_tool_calls, tool_choice, extra_kwargs
        if not self._agent_name or not self._session_id:
            raise RuntimeError("LiveKitLLM.chat() requires agent_name and session_id; use AgentKernelVoiceAgent for the default pipeline")
        return _LiveKitCompatibilityStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class _LiveKitCompatibilityStream(llm.LLMStream):
    """Expose Agent Kernel through LiveKit's ``LLMStream`` API."""

    def __init__(
        self,
        parent_llm: LiveKitLLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(parent_llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._parent_llm = parent_llm
        self._chunk_id = f"agent-kernel-{uuid.uuid4()}"

    async def _run(self) -> None:
        assert self._parent_llm._agent_name is not None
        assert self._parent_llm._session_id is not None
        voice_agent = AgentKernelVoiceAgent(
            agent_name=self._parent_llm._agent_name,
            session_id=self._parent_llm._session_id,
            frame_holder=self._parent_llm._frame_holder,
            streaming_mode=self._parent_llm._streaming_mode,
            service=self._parent_llm._service,
        )
        async for content in voice_agent.llm_node(self.chat_ctx, self.tools, ModelSettings()):
            self._event_ch.send_nowait(llm.ChatChunk(id=self._chunk_id, delta=llm.ChoiceDelta(content=content, role="assistant")))


def _session_id_for_room(room: rtc.Room, participant_identity: str | None = None) -> str:
    """Build a session ID from the room SID and optional participant identity."""

    room_id = getattr(room, "sid", None) or room.name
    return f"livekit:{room_id}:{participant_identity}" if participant_identity else f"livekit:{room_id}"


def _participant_identity(ctx: JobContext) -> str | None:
    job = getattr(ctx, "job", None)
    participant = getattr(job, "participant", None)
    if participant and getattr(participant, "identity", None):
        return participant.identity

    metadata = getattr(job, "metadata", None)
    if metadata:
        try:
            dispatch_metadata = json.loads(metadata)
            if not isinstance(dispatch_metadata, dict):
                logger.debug("Ignoring non-object LiveKit job metadata")
                return None
            identity = dispatch_metadata.get("participant_identity")
            return identity if isinstance(identity, str) and identity else None
        except (TypeError, json.JSONDecodeError):
            logger.warning("Invalid LiveKit job metadata")
    return None


class AgentKernelVoiceAgent(Agent):
    """LiveKit voice agent whose LLM node delegates to Agent Kernel."""

    def __init__(
        self,
        *,
        agent_name: str,
        session_id: str,
        frame_holder: Optional[dict] = None,
        streaming_mode: str = "streaming",
        service: AgentService | None = None,
    ) -> None:
        super().__init__(
            instructions="Route user turns to Agent Kernel.",
            llm=LiveKitLLM(),
        )
        self._agent_name = agent_name
        self._session_id = session_id
        self._frame_holder = frame_holder
        self._streaming_mode = _validate_streaming_mode(streaming_mode)
        self._service = service or AgentService()

    def _select_service(self) -> bool:
        if not self._service.agent:
            self._service.select(name=self._agent_name, session_id=self._session_id)
        return self._service.agent is not None

    def _requests_for_turn(self, user_message: str) -> list[AgentRequest]:
        requests: list[AgentRequest] = [AgentRequestText(prompt=user_message)]
        frame_data = _consume_video_frame(self._frame_holder)
        if frame_data:
            requests.append(AgentRequestImage(image_data=frame_data, mime_type="image/jpeg", name="webcam_frame"))
        return requests

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        # Agent Kernel handles tools and model settings.
        del tools, model_settings

        user_message = _latest_user_message(chat_ctx)
        if not user_message:
            yield "I did not hear anything."
            return

        output_emitted = False
        try:
            if not self._select_service():
                logger.warning("No agent available for name: %s", self._agent_name)
                output_emitted = True
                yield _NO_AGENT_MESSAGE
                return

            requests = self._requests_for_turn(user_message)
            if self._streaming_mode == "buffered":
                response = str(await self._service.run_multi(requests))
                output_emitted = True
                yield response
                return

            stream = self._service.stream_multi(requests)
            error_spoken = False
            try:
                async for chunk in stream:
                    if chunk.error and not error_spoken:
                        error_spoken = True
                        output_emitted = True
                        yield chunk.error
                    if chunk.delta:
                        output_emitted = True
                        yield chunk.delta
            finally:
                await stream.aclose()
        except asyncio.CancelledError:
            raise
        except NotImplementedError:
            logger.warning(
                "Agent %s does not support streaming; set livekit.streaming_mode to buffered",
                self._agent_name,
            )
            if not output_emitted:
                yield _STREAMING_UNSUPPORTED_MESSAGE
        except Exception:
            logger.exception("Error handling LiveKit turn")
            if not output_emitted:
                yield _INTERNAL_ERROR_MESSAGE


def _latest_user_message(chat_ctx: llm.ChatContext) -> str:
    for message in reversed(chat_ctx.messages()):
        if message.role == "user" and message.text_content:
            return message.text_content
    return ""


def _consume_video_frame(frame_holder: Optional[dict]) -> str | None:
    if not frame_holder or not frame_holder.get("frame"):
        return None

    try:
        frame = frame_holder.pop("frame")
        rgba_frame = frame.convert(rtc.VideoBufferType.RGBA)
        image = Image.frombytes("RGBA", (rgba_frame.width, rgba_frame.height), rgba_frame.data)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        logger.debug("Encoded LiveKit video frame")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        logger.exception("Failed to encode a LiveKit video frame")
        return None


async def _default_entrypoint(ctx: JobContext) -> None:
    """Run the default LiveKit STT -> Agent Kernel -> TTS pipeline."""

    config = Config.get().livekit
    logger.info("Connecting to LiveKit room %s", ctx.room.name)

    if config.vision_enabled:
        await ctx.connect(auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL)
        logger.info("Subscribing to audio and video tracks")
    else:
        await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    if not config.agent:
        logger.warning("No LiveKit agent configured; set livekit.agent")

    if config.stt_provider == "openai":
        stt_plugin = openai.STT()
    else:
        stt_plugin = deepgram.STT()

    if config.tts_provider == "elevenlabs":
        from livekit.plugins import elevenlabs

        tts_plugin = elevenlabs.TTS()
    elif config.tts_provider == "google":
        from livekit.plugins import google

        tts_plugin = google.TTS()
    else:
        tts_plugin = openai.TTS()

    frame_holder = {} if config.vision_enabled else None
    voice_agent = AgentKernelVoiceAgent(
        agent_name=config.agent,
        session_id=_session_id_for_room(ctx.room, _participant_identity(ctx)),
        frame_holder=frame_holder,
        streaming_mode=config.streaming_mode,
    )
    session = AgentSession(vad=silero.VAD.load(), stt=stt_plugin, tts=tts_plugin)
    await session.start(agent=voice_agent, room=ctx.room)

    if frame_holder is not None:
        cleanup_video = _start_video_capture(ctx.room, frame_holder)
        ctx.add_shutdown_callback(cleanup_video)


def _start_video_capture(room: rtc.Room, frame_holder: dict) -> Callable[[], Awaitable[None]]:
    """Capture the latest video frame and return a job-shutdown callback."""

    video_stream: rtc.VideoStream | None = None
    reader_task: asyncio.Task | None = None

    def _reader_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        if error := task.exception():
            logger.error("LiveKit video capture task failed", exc_info=(type(error), error, error.__traceback__))

    def _create_stream(track: rtc.Track) -> None:
        nonlocal video_stream, reader_task
        if reader_task and not reader_task.done():
            reader_task.cancel()
        if video_stream is not None:
            video_stream.close()

        stream = rtc.VideoStream(track)
        video_stream = stream

        async def _read_stream() -> None:
            async for event in stream:
                frame_holder["frame"] = event.frame

        reader_task = asyncio.create_task(_read_stream())
        reader_task.add_done_callback(_reader_done)

    for participant in room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                _create_stream(publication.track)
                logger.info("Capturing video from existing track")
                break
        if video_stream is not None:
            break

    @room.on("track_subscribed")
    def _on_track_subscribed(track: rtc.Track, publication, participant) -> None:
        del publication, participant
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            _create_stream(track)
            logger.info("Capturing video from subscribed track")

    async def _cleanup() -> None:
        room.off("track_subscribed", _on_track_subscribed)
        if reader_task and not reader_task.done():
            reader_task.cancel()
            await asyncio.gather(reader_task, return_exceptions=True)
        if video_stream is not None:
            video_stream.close()

    return _cleanup


class AgentLiveKitRequestHandler(RESTRequestHandler):
    """Expose LiveKit tokens and run the LiveKit worker with the REST API."""

    def __init__(
        self,
        entrypoint_fnc: Optional[Callable[[JobContext], Awaitable[None]]] = None,
        auth_dependency: Optional[Callable] = None,
    ) -> None:
        self._log = logger
        self._entrypoint = entrypoint_fnc or _default_entrypoint
        self._auth_dependency = auth_dependency
        self._worker_task: asyncio.Task | None = None
        self._server: AgentServer | None = None
        self._worker_started = False

        config = Config.get().livekit
        self.url = config.url
        self.api_key = config.api_key
        self.api_secret = config.api_secret
        self.worker_name = config.worker_name

    def _worker_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        if error := task.exception():
            self._log.error("LiveKit background worker failed", exc_info=(type(error), error, error.__traceback__))

    def get_router(self) -> APIRouter:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(router: APIRouter):
            del router
            if not self._worker_started:
                self._worker_started = True
                self._log.info("Starting LiveKit background worker")

                kwargs = {"port": 0}
                if self.url:
                    kwargs["ws_url"] = self.url
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.api_secret:
                    kwargs["api_secret"] = self.api_secret

                worker_options = WorkerOptions(
                    agent_name=self.worker_name,
                    entrypoint_fnc=self._entrypoint,
                    worker_type=WorkerType.ROOM,
                    **kwargs,
                )
                self._server = AgentServer.from_server_options(worker_options)
                self._worker_task = asyncio.create_task(self._server.run())
                self._worker_task.add_done_callback(self._worker_done)

            try:
                yield
            finally:
                if self._server:
                    self._log.info("Shutting down LiveKit background worker")
                    try:
                        await self._server.aclose()
                    except Exception:
                        self._log.exception("Failed to close LiveKit worker")
                if self._worker_task:
                    if not self._worker_task.done():
                        self._worker_task.cancel()
                    try:
                        await self._worker_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        # _worker_done already logged this exception.
                        pass

        router = APIRouter(prefix="/livekit", tags=["LiveKit Integration"], lifespan=lifespan)
        dependencies = [Depends(self._auth_dependency)] if self._auth_dependency else []

        @router.get("/token", dependencies=dependencies)
        def get_token(room: str, identity: str):
            if not self.api_key or not self.api_secret:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "LiveKit API key or secret not configured. Set them in config.yaml under "
                        "'livekit' or via AK_LIVEKIT__API_KEY and AK_LIVEKIT__API_SECRET."
                    ),
                )

            if self._worker_started and (self._worker_task is None or self._worker_task.done()):
                raise HTTPException(status_code=503, detail="The LiveKit worker is not running. Check the server logs before issuing tokens.")

            token = api.AccessToken(self.api_key, self.api_secret)
            token.with_identity(identity)
            token.with_name(identity)
            token.with_grants(api.VideoGrants(room_join=True, room=room))
            dispatch_metadata = json.dumps({"participant_identity": identity})
            token.with_room_config(api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=self.worker_name, metadata=dispatch_metadata)]))
            return {"token": token.to_jwt()}

        return router
