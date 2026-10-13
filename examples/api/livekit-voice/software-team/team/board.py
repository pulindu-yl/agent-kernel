"""The project board: one more participant in the call's room, showing the client the project as it changes.

The voice agents talk; the board shows what they recorded. It joins the room as "Project board",
with no microphone, and sends the project as JSON on its own topic, ``ak.project``:

- to everyone, whenever the project changes (at most once a second);
- to each participant that joins, so a page opened or reloaded mid-call starts full.

The client's page (panel/) draws it beside the call. The board never reaches the agents: Agent
Kernel's LiveKit gateway takes only audio and the chat topics as the client's input, and the board
sends neither.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Any, ClassVar, Optional

from livekit import rtc
from pydantic import BaseModel

from team.approval import ClientApproval
from team.log_watch import LogLineWatcher
from team.models import ProjectState
from team.project_files import ProjectFiles
from team.room_access import RoomAccess
from team.state import TeamCall

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
