"""Everything the team keeps for one call, and how a tool finds it."""

import threading
import time
from pathlib import Path
from typing import Optional

from agentkernel.core.base import Session

from team.approval import ClientApproval
from team.call_link import CallLink
from team.jobs import DevJobRunner
from team.progress import ActivityFeed
from team.project import ProjectStore
from team.rules import HandoffGuard, ProcessGates


class CoderQuestion:
    """The Coder's one open question to the team, and the PM's answer to it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._question: Optional[str] = None
        self._answer: Optional[str] = None
        self._answered = threading.Event()

    @property
    def open(self) -> Optional[str]:
        with self._lock:
            return self._question

    def ask(self, question: str) -> None:
        with self._lock:
            self._question, self._answer = question, None
            self._answered.clear()

    def wait(self, timeout: float) -> Optional[str]:
        """Block until the question is answered or ``timeout`` passes; the question closes either way."""
        self._answered.wait(timeout)
        with self._lock:
            answer, self._question = self._answer, None
            return answer

    def answer(self, text: str) -> Optional[str]:
        """Answer the open question; returns it, or ``None`` when there is none."""
        with self._lock:
            if self._question is None:
                return None
            self._answer = text
            self._answered.set()
            return self._question


class TeamCall:
    """Everything the team keeps for one call: the project, the background jobs, and the call's coordination.

    The background agents run in their own AK session, ``<call session>-dev``, so a long run never
    holds the call's session lock, which every voice tool call takes.
    """

    DEV_SUFFIX = "-dev"
    VOICE_ROLES = ("pm", "worker", "architect")
    NAMES = {"pm": "PM", "worker": "Worker", "architect": "Architect"}
    # The agent the LiveKit gateway serves, which a new model connection starts with.
    ENTRY_AGENT = "pm"
    # Long enough for the Worker and the Architect to talk a technical question through.
    ASK_TEAM_TIMEOUT_SECONDS = 120.0

    def __init__(self, session_id: str, output_root: Path):
        self.session_id = session_id
        self.store = ProjectStore(output_root, name=session_id)
        self.activity = ActivityFeed(self.store)
        self.link = CallLink(session_id, should_nudge=self.should_nudge, on_reconnect=self.reconnected)
        self.guard = HandoffGuard(self.link.caller_spoke_at, recorded_at=self.moved_on_at)
        self.jobs = DevJobRunner(self)
        self.coder_question = CoderQuestion()
        self.gates = ProcessGates(self.store)
        self._client_asked_at: Optional[float] = None
        # Who has the call (the PM opens it), since when, and how many handovers there have been.
        self.holder = self.ENTRY_AGENT
        self.held_since = 0.0
        self.handovers = 0
        # The client's name, as their page joined the call: a renewed model connection has lost it.
        self.client_name: Optional[str] = None
        # Counts the model connections renewed, so only the latest one's notice is posted.
        self._reconnects = 0

    @property
    def dev_session_id(self) -> str:
        return f"{self.session_id}{self.DEV_SUFFIX}"

    def notify(self, text: str) -> None:
        """Tell the voice team something, at the next quiet moment of the call."""
        self.link.post(text)

    def handed_over(self, to: str) -> None:
        """The call was handed to ``to``."""
        self.holder = to
        self.held_since = time.time()
        self.handovers += 1

    def moved_on_at(self) -> Optional[float]:
        """When the team last moved on without the client: a voice agent recorded something, or a job finished.

        A finished job is news the team must act on, so the handoff it needs (to the Architect to
        approve the code, say) is never refused as part of a loop.
        """
        recorded = self.store.last_recorded_at(self.VOICE_ROLES)
        finished = max((j.finished_at for j in self.store.state.jobs if j.finished_at), default=None)
        return max((t for t in (recorded, finished) if t is not None), default=None)

    def reconnected(self) -> None:
        """The model connection was replaced (it failed, or went idle): the new one starts at the PM and remembers nothing.

        The project is still recorded, so the PM is told where it stands rather than starting the
        meeting over. Only the latest renewal's notice is posted, and it is written when it is
        posted: live, failed attempts to reconnect queued thirteen notices, each saying "the Worker
        delivers the project" long after it had, and the Worker delivered thirteen times.
        """
        self.holder = self.ENTRY_AGENT
        self.held_since = time.time()
        self._reconnects += 1
        renewal = self._reconnects
        self.link.post(self._reconnect_notice, valid=lambda: renewal == self._reconnects)

    def _reconnect_notice(self) -> str:
        client = f" The client is {self.client_name}." if self.client_name else ""
        return (
            "The model connection was renewed, so the conversation before this point is gone, but the project is "
            f"recorded.{client} Where it stands: {self.gates.next_step(self.ENTRY_AGENT)} Carry on from there without "
            "greeting the client again; call get_project_state for the details."
        )

    def job_blocker(self) -> Optional[str]:
        """Why the call cannot be handed over while background work runs, or ``None``.

        The agent that starts a job keeps the call until its result comes. Live, while a review ran
        the call went back and forth between the Worker and the PM, each telling the client the
        same status, until the loop guard refused the one handover that was needed. A handover
        still goes through when something new came since the agent got the call: the client spoke
        or typed, or a team update arrived (a question or an assumption from the Coder, a file).
        """
        state = self.store.state
        job = next((j for j in state.jobs if j.status in ("queued", "running")), None)
        if job is None or self.coder_question.open:
            return None
        news = (self.link.caller_spoke_at(), self.link.last_post_at)
        if any(at is not None and at > self.held_since for at in news):
            return None
        return (
            f"Not handed over: {job.id} ({job.kind}) is running, and nothing needs a teammate until its result comes "
            "as a team update, so you keep the call. If you have told the client what the team is doing, say nothing more."
        )

    def ask_client(self) -> None:
        """Mark a question to the client as waiting for their answer, so the team is not nudged meanwhile."""
        self._client_asked_at = time.time()

    def client_answered(self) -> None:
        """The client answered the question waiting for them without a word (a button on their page)."""
        self._client_asked_at = None

    def awaiting_client(self) -> bool:
        """Whether a question to the client is waiting: asked, and the client has not spoken or typed since."""
        if self._client_asked_at is None:
            return False
        spoke = self.link.caller_spoke_at()
        return spoke is None or spoke < self._client_asked_at

    def should_nudge(self) -> bool:
        """Nudge only a meeting under way that waits on the team.

        Not before the client has spoken or typed, while a job runs, while a question or the plan's
        approval waits for the client, or once the project is delivered.
        """
        state = self.store.state
        return (
            self.link.caller_spoke_at() is not None
            and not self.jobs.busy
            and not self.awaiting_client()
            and not ClientApproval(self).pending()
            and not state.delivered
        )

    def ask_team(self, question: str) -> Optional[str]:
        """The Coder's question to the team; blocks the job thread until it is answered or the timeout passes.

        Posted only while it is still open: a question that timed out before the call was quiet
        enough to hear it is dropped, not discussed after the Coder has moved on.
        """
        self.coder_question.ask(question)
        self.link.post(
            f"The Coder asks: {question} The Coder is waiting: answer it with answer_coder. A business question is the "
            "PM's, from the requirements, asking the client only if it is a decision they have not made. A technical one "
            "is talked through: the Worker gives the Architect their view in one turn, the Architect decides and answers. "
            "Deciding out loud is not enough: the Coder hears only answer_coder.",
            valid=lambda: self.coder_question.open == question,
        )
        return self.coder_question.wait(self.ASK_TEAM_TIMEOUT_SECONDS)


class TeamCalls:
    """The calls in progress, one ``TeamCall`` per call session, found from the session a tool runs in."""

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self._lock = threading.Lock()
        self._calls: dict[str, TeamCall] = {}

    def get(self, session_id: str) -> TeamCall:
        """The call a session belongs to: the call's own session or its background ``-dev`` session.

        A call's link starts watching it as soon as the call is first seen, so a team that goes
        silent is nudged even before anything has been posted into the call.
        """
        call_id = session_id.removesuffix(TeamCall.DEV_SUFFIX)
        with self._lock:
            if call_id not in self._calls:
                call = TeamCall(call_id, self.output_root)
                call.link.start()
                self._calls[call_id] = call
            return self._calls[call_id]

    def current(self) -> TeamCall:
        session = Session.current()
        if session is None:
            raise RuntimeError("No session is active; team tools run inside an agent run")
        return self.get(session.id)
