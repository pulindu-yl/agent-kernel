"""Start here: registers the team and serves it in a LiveKit room, with the client's page and the project board.

    uv run server.py

Every agent is a Google ADK agent (agents.py). The voice agents run on Gemini Live; the Coder and the Reviewer run
on a text model through LiteLLM, OpenAI's ``gpt-4.1`` unless ``TEAM_TEXT_MODEL`` says otherwise. The client joins
from the page at http://localhost:8000 (panel/, built with ``npm run build``), which shows the call beside the
project board.

Contents:

1. Room tokens for the client's page and the board
2. The project board: a room participant that shows the project live
3. The client's page and its routes
4. Start here: register the team and serve it
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, Optional

from agentkernel.api import RESTAPI
from agentkernel.core.config import AKConfig
from agentkernel.integration.adapter import GatewayRunner
from agentkernel.integration.livekit import LiveKitEdgeGateway
from agentkernel.pipeline import IOHandler
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from livekit import api, rtc
from pydantic import BaseModel

from agents import BackgroundTeam, VoiceTeam
from project import ProjectFiles, ProjectState
from team import ClientApproval, ClientQuestionWatcher, LogLineWatcher, ModelWaitWatcher, TeamCall, TeamCalls

# ====================================================================================================
# 1. Room tokens for the client's page and the board
#
# Tokens for joining the call's LiveKit room, made from the settings the gateway already uses.
#
# Two participants join besides the agents: the project board (``ProjectBoard``, below), which only sends
# data, and the client's page (panel/), which talks and listens.
# ====================================================================================================


class RoomAccess:
    """Makes LiveKit tokens for one room, from ``AK_LIVEKIT__URL``, ``AK_LIVEKIT__API_KEY`` and ``AK_LIVEKIT__API_SECRET``."""

    BOARD_IDENTITY = "project-board"
    CLIENT_IDENTITY = "client"
    # The gateway names its agent participants agent-<name>; a client must never look like one.
    AGENT_PREFIX = "agent-"
    MAX_NAME = 40

    def __init__(self, room: str, url: str, api_key: str, api_secret: str):
        self.room = room
        self.url = url
        self._api_key = api_key
        self._api_secret = api_secret

    @classmethod
    def from_config(cls, room: str) -> "RoomAccess":
        settings = AKConfig.get().livekit
        return cls(room, settings.url, settings.api_key, settings.api_secret)

    @property
    def configured(self) -> bool:
        """Whether tokens can be made: the URL, key and secret are all set."""
        return bool(self.url and self._api_key and self._api_secret)

    def board_token(self) -> str:
        """For the project board: it sends data only, with no microphone, and subscribes to nothing."""
        grants = api.VideoGrants(
            room_join=True, room=self.room, can_publish=False, can_subscribe=False, can_publish_data=True
        )
        return self._token(self.BOARD_IDENTITY, "Project board", grants)

    def client_token(self, name: str = "") -> str:
        """For the client's page: it publishes the microphone, hears the agents, and can type.

        Joins under the client's name (``client_identity``), which the gateway greets them by. Typed
        messages go to the room chat, which the gateway passes to the agent with the call as the
        client's words: for links, names and anything easier written than said.
        """
        grants = api.VideoGrants(
            room_join=True, room=self.room, can_publish=True, can_subscribe=True, can_publish_data=True
        )
        display = self.display_name(name)
        return self._token(self.client_identity(display), display, grants)

    @classmethod
    def display_name(cls, name: str) -> str:
        return " ".join(name.split())[: cls.MAX_NAME] or "Client"

    @classmethod
    def client_identity(cls, display_name: str) -> str:
        """A room identity from the client's name, never one an agent or the board uses.

        It keeps the name's capitals: the gateway greets the caller by their identity ("A user named
        Pulindu just connected"), so the team calls them by it.
        """
        identity = re.sub(r"[^A-Za-z0-9]+", "-", display_name).strip("-") or cls.CLIENT_IDENTITY
        if identity.lower().startswith(cls.AGENT_PREFIX) or identity.lower() == cls.BOARD_IDENTITY:
            identity = f"{cls.CLIENT_IDENTITY}-{identity}"
        return identity

    def _token(self, identity: str, name: str, grants: api.VideoGrants) -> str:
        return (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(identity)
            .with_name(name)
            .with_grants(grants)
            .to_jwt()
        )


# ====================================================================================================
# 2. The project board: a room participant that shows the project live
#
# The project board: one more participant in the call's room, showing the client the project as it changes.
#
# The voice agents talk; the board shows what they recorded. It joins the room as "Project board",
# with no microphone, and sends the project as JSON on its own topic, ``ak.project``:
#
# - to everyone, whenever the project changes (at most once a second);
# - to each participant that joins, so a page opened or reloaded mid-call starts full.
#
# The client's page (panel/) draws it beside the call. The board never reaches the agents: Agent
# Kernel's LiveKit gateway takes only audio and the chat topics as the client's input, and the board
# sends neither.
# ====================================================================================================

_log = logging.getLogger("software_team.board")


class Criterion(BaseModel):
    id: str
    text: str
    # "pass", "fail" or "not covered" once acceptance is checked; "not checked" before.
    result: str


class RequirementCard(BaseModel):
    id: str
    title: str
    criteria: list[Criterion]


class DecisionCard(BaseModel):
    id: str
    title: str
    choice: str
    challenge: str


class TaskCard(BaseModel):
    id: str
    title: str
    status: str


class FindingCard(BaseModel):
    id: str
    text: str
    status: str
    # Why the Architect ruled as they did: to fix, or waived.
    reason: str


class JobCard(BaseModel):
    id: str
    kind: str
    seconds: int
    activity: list[str]


class StepCard(BaseModel):
    name: str
    # "done", "current" (the first step not done) or "todo".
    state: str


class LogLine(BaseModel):
    at: float
    who: str
    what: str


class BoardSnapshot(BaseModel):
    """Everything the client's page shows, built from the project's records. No LLM, so it shows only what was recorded."""

    project: str
    goal: str
    phase: str
    requirements_confirmed: bool
    design_written: bool
    # The requirements and the design are ready for the client to approve, with a button.
    approval_pending: bool
    # The Architect approved the code as it is now, after the code review.
    code_approved: bool
    delivered: bool
    folder: str
    steps: list[StepCard]
    waiting_for_you: list[str]
    requirements: list[RequirementCard]
    decisions: list[DecisionCard]
    tasks: list[TaskCard]
    job: Optional[JobCard]
    findings: list[FindingCard]
    tests: str
    files: list[str]
    recent: list[LogLine]
    updated_at: float

    PHASES: ClassVar[dict[str, str]] = {
        "develop": "Building",
        "review": "Reviewing the code",
        "acceptance": "Checking acceptance",
        "deliver": "Delivering",
    }
    ASKED_PREFIX: ClassVar[str] = "Asked the client: "
    MAX_ACTIVITY: ClassVar[int] = 5
    MAX_RECENT: ClassVar[int] = 15

    @classmethod
    def of(cls, call: TeamCall, now: float) -> "BoardSnapshot":
        state = call.store.state
        running = call.jobs.running()
        results = {r.criterion_id: r.result for r in state.acceptance}

        if state.delivered:
            phase = "Delivered"
        elif running is not None:
            phase = cls.PHASES.get(running.kind, "Working")
        elif not state.requirements:
            phase = "Getting started"
        elif ClientApproval(call).pending():
            phase = "Waiting for your approval"
        elif not (state.requirements_confirmed and state.design_written):
            phase = "Agreeing what to build"
        elif any(s.state == "current" and s.name == "Review" for s in cls._steps(call, state)):
            phase = "Code review"
        else:
            phase = "Building"

        job = None
        if running is not None:
            started = running.started_at or now
            entries = call.activity.since(started)[-cls.MAX_ACTIVITY :]
            job = JobCard(
                id=running.id,
                kind=running.kind,
                seconds=int(now - started),
                activity=[f"{e.who} {e.text}" for e in entries],
            )

        return cls(
            # Until the PM names it, the store is named after the call's room.
            project=state.name if call.store.started else "New project",
            goal=state.goal,
            phase=phase,
            requirements_confirmed=state.requirements_confirmed,
            design_written=state.design_written,
            approval_pending=ClientApproval(call).pending(),
            delivered=state.delivered,
            folder=str(call.store.folder),
            steps=cls._steps(call, state),
            waiting_for_you=cls._waiting(call, state),
            requirements=[
                RequirementCard(
                    id=r.id,
                    title=r.title,
                    criteria=[
                        Criterion(id=r.criterion_id(i), text=c, result=results.get(r.criterion_id(i), "not checked"))
                        for i, c in enumerate(r.acceptance_criteria)
                    ],
                )
                for r in state.requirements
            ],
            decisions=[
                DecisionCard(id=d.id, title=d.title, choice=d.choice, challenge=d.challenge) for d in state.decisions
            ],
            tasks=[TaskCard(id=t.id, title=t.title, status=t.status) for t in state.tasks],
            job=job,
            findings=[FindingCard(id=f.id, text=f.text, status=f.status, reason=f.reason) for f in state.findings],
            code_approved=call.gates.blocker("check_acceptance") is None,
            tests=state.test_summary or "",
            files=ProjectFiles(call).list(),
            recent=[LogLine(at=e.at, who=e.who, what=e.what) for e in state.log[-cls.MAX_RECENT :]],
            updated_at=now,
        )

    @staticmethod
    def _steps(call: TeamCall, state: ProjectState) -> list[StepCard]:
        """The SDLC steps, each done or not, from the same checks the process gates use."""
        gates = call.gates
        built = bool(state.tasks) and all(t.status == "done" for t in state.tasks)
        done = {
            "Requirements": bool(state.requirements) and (state.design_written or state.requirements_confirmed),
            "Design": state.design_written,
            "Approval": state.requirements_confirmed,
            "Build": built,
            "Review": built and gates.blocker("check_acceptance") is None,
            "Acceptance": state.delivered or (built and gates.blocker("deliver") is None),
            "Delivered": state.delivered,
        }
        current = next((name for name, finished in done.items() if not finished), None)
        return [
            StepCard(name=name, state="done" if finished else "current" if name == current else "todo")
            for name, finished in done.items()
        ]

    @classmethod
    def _waiting(cls, call: TeamCall, state: ProjectState) -> list[str]:
        """What the team needs from the client: the question just asked, open questions, and assumptions to confirm."""
        waiting = []
        if call.awaiting_client():
            asked = [p.what for p in state.progress if p.what.startswith(cls.ASKED_PREFIX)]
            if asked:
                waiting.append(asked[-1].removeprefix(cls.ASKED_PREFIX))
        waiting += [q.text for q in state.questions if not q.answer]
        waiting += [f"Confirm: {a.text}" for a in state.assumptions if a.needs_settling and not a.settled]
        return waiting


class ProjectBoard:
    """Joins the call's room as the project board and keeps every page in the room up to date.

    Runs on a thread and event loop of its own, like the background jobs, so it never touches the
    realtime pool's loop. A project change only schedules a send; the board sends the latest
    snapshot at most every ``MIN_INTERVAL_SECONDS``.
    """

    TOPIC = "ak.project"
    # The client's own words, as the model heard them (see ClientWordsWatcher).
    CLIENT_WORDS_TOPIC = "ak.client"
    MIN_INTERVAL_SECONDS = 1.0
    # The gateway's agent participants are named agent-<name>; they have no use for the board.
    AGENT_PREFIX = "agent-"

    def __init__(self, call: TeamCall, access: RoomAccess):
        self._call = call
        self._access = access
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._room: Any = None
        self._changed: Optional[asyncio.Event] = None
        self._thread: Optional[threading.Thread] = None
        call.store.on_change(self.changed)

    def start(self) -> None:
        """Join the room in the background. Without LiveKit settings there is no board, and the call works as before."""
        if not self._access.configured:
            _log.warning("No LiveKit URL, API key or secret: the project board is off")
            return
        self.prepare_livekit()
        self._thread = threading.Thread(target=self._run, name="project-board", daemon=True)
        self._thread.start()

    @staticmethod
    def prepare_livekit() -> None:
        """Create the LiveKit SDK's native client now, on the calling thread. Call it before the gateway starts.

        The SDK creates that client on first use, without a lock. Live, the board's thread and the
        gateway's thread first used it at the same moment and created two; LiveKit then sent its events
        to the second, so the gateway's room never heard it had connected, and the SDK ended the process.

        ``FfiClient`` is not public SDK API, so a release that moves it must not stop the server: the
        board then starts without this, and the race can come back.
        """
        try:
            from livekit.rtc._ffi_client import FfiClient

            FfiClient.instance  # noqa: B018 (the property creates the client)
        except (ImportError, AttributeError) as e:
            _log.warning(
                f"Could not create the LiveKit SDK client ahead of the board ({e!r}); starting the board anyway"
            )

    def changed(self) -> None:
        """A project change listener: schedules a send on the board's loop. Safe from any thread."""
        loop, changed = self._loop, self._changed
        if loop is not None and changed is not None:
            loop.call_soon_threadsafe(changed.set)

    def snapshot(self) -> BoardSnapshot:
        return BoardSnapshot.of(self._call, time.time())

    async def serve(self, room: Any) -> None:
        """Send the snapshot now, then again after each change, at most every ``MIN_INTERVAL_SECONDS``.

        Changes made while one send waits out the interval are covered by the next send, which
        carries the latest state.
        """
        self._loop = asyncio.get_running_loop()
        self._room = room
        self._changed = asyncio.Event()
        await self.send(room)
        while True:
            await self._changed.wait()
            self._changed.clear()
            await self.send(room)
            await asyncio.sleep(self.MIN_INTERVAL_SECONDS)

    async def send(self, room: Any, to: Optional[list[str]] = None) -> None:
        """Send the latest snapshot to ``to``, or to every viewer. A failed send is logged; the next change sends again.

        The agents are left out: their participants run in this process too, and each would only log
        that it has no use for the topic.
        """
        viewers = to if to is not None else self.viewers(room)
        if not viewers:
            return
        try:
            await room.local_participant.send_text(
                self.snapshot().model_dump_json(), topic=self.TOPIC, destination_identities=viewers
            )
        except Exception:
            _log.exception("The project board could not send the project")

    def client_said(self, text: str) -> None:
        """Show the client their own words on their page. Safe from any thread; dropped before the board joins."""
        loop, room = self._loop, self._room
        if loop is not None and room is not None:
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.send_client_words(room, text)))

    async def send_client_words(self, room: Any, text: str) -> None:
        viewers = self.viewers(room)
        if not viewers:
            return
        try:
            await room.local_participant.send_text(
                json.dumps({"text": text}), topic=self.CLIENT_WORDS_TOPIC, destination_identities=viewers
            )
        except Exception:
            _log.exception("The project board could not send the client's words")

    @classmethod
    def viewers(cls, room: Any) -> list[str]:
        """Everyone in the room but the agents: the client's page, and anyone else watching."""
        return [identity for identity in room.remote_participants if not identity.startswith(cls.AGENT_PREFIX)]

    def _run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        room = rtc.Room()

        @room.on("participant_connected")
        def joined(participant: rtc.RemoteParticipant) -> None:
            # A page that just joined has missed the earlier sends.
            if not participant.identity.startswith(self.AGENT_PREFIX):
                asyncio.ensure_future(self.send(room, [participant.identity]))

        try:
            await room.connect(self._access.url, self._access.board_token())
        except Exception:
            _log.exception("The project board could not join the room")
            return
        _log.info(f"The project board joined {self._access.room}")
        await self.serve(room)


class ClientWordsWatcher(LogLineWatcher):
    """Shows the client their own words: Agent Kernel's Gemini adapter logs each of the caller's turns as
    the model heard it ("Caller said: ..."), and the board passes it to the client's page.

    Reading a log line is the example's shortcut, as ``ModelWaitWatcher`` reads the OpenAI client's retry
    lines: Agent Kernel does not send the caller's words to the edge. Only on Gemini, which transcribes
    what it hears; the line is logged when the model has answered the turn, so it reaches the page
    before the agent's own words, which the gateway publishes once their audio has played.
    """

    LOGGERS = ("ak.adk.realtime",)
    PREFIX = "Caller said: "

    def __init__(self, board: ProjectBoard):
        super().__init__()
        self._board = board

    def line(self, message: str) -> None:
        if message.startswith(self.PREFIX):
            self._board.client_said(message.removeprefix(self.PREFIX))


# ====================================================================================================
# 3. The client's page and its routes
#
# The client's page, served by the example's own server on port 8000.
#
# The page (panel/, a small React app) joins the call and draws the project board beside it. This
# serves its built files and gives it a LiveKit token, as the ``examples/api/agui`` example serves its
# UI; and it lets the page view and download the project's files, and share files with the team.
# ====================================================================================================


class ClientPage:
    """Routes for the client's page: the page itself, its token, and the project's files.

    ``GET /`` and ``GET /assets/...`` serve the built page; ``GET /token`` gives it a room token;
    ``GET /project/files``, ``GET /project/files/{path}`` and ``GET /project/download`` list, read
    and zip the project folder; ``POST /project/share`` saves a file the client shares;
    ``POST /project/approval`` takes the client's approval of the requirements and the design, or
    their wish to change something first.

    None of them asks who is calling, which is fine on your own machine but not on a public server:
    put real authentication in front of them before deploying.
    """

    DIST = Path(__file__).parent / "panel" / "dist"
    BUILD_HINT = (
        "<h1>The client page is not built</h1>"
        "<p>From <code>panel/</code> run <code>npm install &amp;&amp; npm run dev</code> and open "
        '<a href="http://localhost:5173">http://localhost:5173</a>, or run <code>npm run build</code> to serve it here.</p>'
    )

    def __init__(self, access: RoomAccess, calls: TeamCalls, dist: Path = DIST):
        self._access = access
        self._calls = calls
        self._dist = dist

    def router(self) -> APIRouter:
        router = APIRouter()
        router.add_api_route("/", self.index, methods=["GET"], response_class=HTMLResponse)
        router.add_api_route("/assets/{filename}", self.asset, methods=["GET"], include_in_schema=False)
        router.add_api_route("/token", self.token, methods=["GET"])
        router.add_api_route("/project/files", self.files, methods=["GET"])
        router.add_api_route("/project/files/{path:path}", self.file, methods=["GET"])
        router.add_api_route("/project/download", self.download, methods=["GET"])
        router.add_api_route("/project/share", self.share, methods=["POST"])
        router.add_api_route("/project/approval", self.approval, methods=["POST"])
        return router

    def index(self) -> HTMLResponse:
        """The built page, or how to build it."""
        entry = self._dist / "index.html"
        if not entry.is_file():
            return HTMLResponse(self.BUILD_HINT, status_code=503)
        return HTMLResponse(entry.read_text(encoding="utf-8"))

    def asset(self, filename: str) -> FileResponse:
        """A file of the built page, by exact name only."""
        root = self._dist / "assets"
        match = next((p for p in root.iterdir() if p.name == filename and p.is_file()), None) if root.is_dir() else None
        if match is None:
            raise HTTPException(status_code=404, detail="No such asset")
        return FileResponse(match)

    def token(self, name: str = "") -> dict:
        """Where the room is and a token to join it as the client, under their name (``?name=Pulindu``)."""
        if not self._access.configured:
            raise HTTPException(
                status_code=503, detail="Set AK_LIVEKIT__URL, AK_LIVEKIT__API_KEY and AK_LIVEKIT__API_SECRET"
            )
        display = RoomAccess.display_name(name)
        # Kept for a renewed model connection, which has lost the greeting that named the client.
        self._calls.get(self._access.room).client_name = display
        return {
            "url": self._access.url,
            "room": self._access.room,
            "name": display,
            "token": self._access.client_token(display),
        }

    def files(self) -> dict:
        """The files in the project folder."""
        return {"files": self._project().list()}

    def file(self, path: str) -> FileResponse:
        """One file of the project, to view or save."""
        try:
            return FileResponse(self._project().path(path))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No such file in the project")

    def download(self) -> Response:
        """The whole project as a zip file."""
        project = self._project()
        return Response(
            project.zip(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{project.folder.name}.zip"'},
        )

    async def share(self, request: Request, name: str, by: str = "The client") -> dict:
        """Save a file the client shares (the request body is the file) and tell the team about it."""
        try:
            saved = self._project().share(name, await request.body(), RoomAccess.display_name(by))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"saved": saved}

    def approval(self, decision: str, by: str = "The client") -> dict:
        """The client's answer to the requirements and the design: ``?decision=approve`` or ``?decision=change``."""
        actions = {"approve": ClientApproval.approve, "change": ClientApproval.ask_for_change}
        if decision not in actions:
            raise HTTPException(status_code=400, detail="decision is approve or change")
        try:
            actions[decision](ClientApproval(self._calls.get(self._access.room)), RoomAccess.display_name(by))
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"decision": decision}

    def _project(self) -> ProjectFiles:
        return ProjectFiles(self._calls.get(self._access.room))


# ====================================================================================================
# 4. Start here: register the team and serve it
# ====================================================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
)
# The gateway enqueues the caller's microphone audio ten times a second, and the producer logs every chunk,
# which buries the lines worth reading (transfers, tool calls, the team's posts, the agents' replies).
logging.getLogger("ak.integration.producer").setLevel(logging.WARNING)
# LiteLLM logs every model call of the Coder and the Reviewer, twice; their tool calls are logged as activity instead.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class SoftwareTeamServer:
    """Loads the team and serves it in a LiveKit room.

    ``TEAM_TEXT_MODEL`` overrides the background agents' model (any LiteLLM ``provider/model``);
    ``TEAM_OUTPUT_DIR`` is where each project's folder is written (default ``./output``).
    """

    ROOM = "room_01"

    def __init__(self, output_root: Path, text_model: Optional[str] = None):
        self.text_model = text_model
        self.calls = TeamCalls(output_root)

    @classmethod
    def from_env(cls) -> "SoftwareTeamServer":
        return cls(Path(os.getenv("TEAM_OUTPUT_DIR", "./output")), os.getenv("TEAM_TEXT_MODEL") or None)

    def load(self) -> None:
        """Register the voice team and the background agents with Agent Kernel."""
        ModelWaitWatcher.install(self.calls)
        VoiceTeam(self.calls).load()
        BackgroundTeam(self.calls, self.text_model).load()

    def run(self) -> None:
        self.load()
        access = RoomAccess.from_config(self.ROOM)
        RESTAPI.add(ClientPage(access, self.calls).router())
        board = ProjectBoard(self.calls.get(self.ROOM), access)
        board.start()
        ClientWordsWatcher.install(board)
        ClientQuestionWatcher.install(self.calls.get(self.ROOM))
        gateway = LiveKitEdgeGateway(session_id=self.ROOM, agent_name="pm")
        IOHandler.run(gateways=[GatewayRunner(gateway)])


if __name__ == "__main__":
    SoftwareTeamServer.from_env().run()
