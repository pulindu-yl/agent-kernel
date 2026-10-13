"""How the call is run: the process gates and the handoff guard, speaking into the live call, the background jobs, the
activity feed, the client's approval, and one call's state.

Contents:

1. Log watching (a shared helper for the watchers below)
2. Process rules: the SDLC gates (ProcessGates) and the handoff loop guard (HandoffGuard)
3. Speaking into the live call at a quiet moment (CallLink)
4. Background jobs: the Coder and the Reviewer, one job at a time (DevJobRunner)
5. Activity: what the background agents are doing, for the board
6. The client's approval, from the button on their page
7. Noticing a question put to the client
8. One call's state, tying the above together (TeamCall, TeamCalls)
"""

import asyncio
import io
import json
import logging
import queue
import re
import tarfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, Union

from agentkernel.core.base import Session
from agentkernel.core.runtime import Runtime
from agentkernel.core.service import AgentService
from agentkernel.pipeline.realtime_pool import RealtimeConnection, RealtimeConnectionPool
from agentkernel.sandbox import ExecutionManager, SandboxResult
from pydantic import BaseModel

from project import Job, ProjectState, ProjectStore

# ====================================================================================================
# 1. Log watching (a shared helper for the watchers below)
#
# Reading a line another library logs, as the example's shortcut for an event it does not otherwise get.
# ====================================================================================================

_watch_log = logging.getLogger("software_team.log_watch")


class LogLineWatcher(logging.Handler):
    """A logging handler that reacts to lines from the loggers in ``LOGGERS``.

    Three parts of the example learn of something only from a log line: the caller's words
    (``Caller said: ...``), each agent turn the gateway played (``AI response: ...``) and the model
    client's rate-limit retries. That depends on the wording of those lines and on the loggers
    emitting INFO, so ``install`` warns when a logger is set above INFO, which would silence the
    watcher without an error.
    """

    LOGGERS: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)

    @classmethod
    def install(cls, *args: Any) -> "LogLineWatcher":
        """Create the watcher with ``args`` and attach it to each of its loggers."""
        watcher = cls(*args)
        for name in cls.LOGGERS:
            logger = logging.getLogger(name)
            logger.addHandler(watcher)
            if not logger.isEnabledFor(logging.INFO):
                _watch_log.warning(
                    f"{cls.__name__} reads INFO lines from '{name}', which is set to log less; it will see nothing"
                )
        return watcher

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.line(record.getMessage())
        except Exception:
            # Never let reading a log line break the code that logged it.
            self.handleError(record)

    def line(self, message: str) -> None:
        """React to one line from a watched logger."""
        raise NotImplementedError


# ====================================================================================================
# 2. Process rules: the SDLC gates (ProcessGates) and the handoff loop guard (HandoffGuard)
#
# The rules that keep the meeting on track, whatever the models do.
#
# - ``ProcessGates``: the order the team works in (requirements, design, build, review, acceptance,
#   delivery). A tool asked to skip ahead answers "Not ..." with what is missing and who should act.
# - ``HandoffGuard``: stops the voice agents handing the call to each other forever.
# ====================================================================================================

_rules_log = logging.getLogger("software_team.rules")


class ProcessGates:
    """The order the team works in, checked before each step, so a model that skips ahead is told what is missing.

    The prompts ask for this order; a voice model does not always keep it (an Architect handing over
    before any design, a Worker building before the client confirmed anything). Each check returns
    why a step cannot happen yet, naming who should act, or ``None`` when it may.
    """

    STEPS = (
        "confirm_requirements",
        "write_design",
        "create_tasks",
        "start_development",
        "request_review",
        "rule_on_finding",
        "approve_code",
        "check_acceptance",
        "deliver",
    )

    # The design is debated over rounds (the Architect's challenge, the Worker's answer): at least
    # MIN_DESIGN_ROUNDS before it is written, so agreement is earned, and at most MAX_DESIGN_ROUNDS.
    MIN_DESIGN_ROUNDS = 2
    MAX_DESIGN_ROUNDS = 4
    # The code review is a conversation too: the Architect puts the findings to the Worker, at least
    # once before ruling on them, and at most MAX_REVIEW_ROUNDS times.
    MAX_REVIEW_ROUNDS = 2

    def __init__(self, store: "ProjectStore"):
        self._store = store

    def blocker(self, step: str, items: Optional[list[str]] = None) -> Optional[str]:
        if step not in self.STEPS:
            raise ValueError(f"Unknown step '{step}'")
        return getattr(self, f"_{step}")(self._store.state, list(items or []))

    def debate(self) -> Optional[str]:
        """Which debate an Architect-to-Worker handover now belongs to: "design", "review" or ``None``."""
        state = self._store.state
        if not state.design_written:
            return "design" if state.requirements else None
        if any(f.status == "open" for f in state.findings) and self.last_done(state, "review") is not None:
            return "review"
        return None

    def handover_blocker(self, sender: Optional[str], receiver: Optional[str]) -> Optional[str]:
        """Why the call cannot be handed from ``sender`` to ``receiver`` now, or ``None``: a debate has had its last round."""
        if (sender, receiver) != ("architect", "worker"):
            return None
        state = self._store.state
        debate = self.debate()
        if debate == "design" and state.design_rounds >= self.MAX_DESIGN_ROUNDS:
            return (
                f"Not handed over: the design has had {self.MAX_DESIGN_ROUNDS} rounds of debate, the most there are. "
                "Settle what is left yourself, record the decisions, call write_design and hand over to the PM."
            )
        if debate == "review" and state.review_rounds >= self.MAX_REVIEW_ROUNDS:
            return (
                f"Not handed over: the review findings have had {self.MAX_REVIEW_ROUNDS} rounds of discussion. Rule on "
                "each yourself with rule_on_finding: fix it, or waive it with the reason."
            )
        return None

    # What the Worker is told when the Architect hands them the call in a debate. Only the Architect
    # hears about rounds: told "round 1 of 2 to 4", the Worker spoke four answers in one turn, acting
    # out every round alone. Worded as facts, like every brief: the agent taking the call reads the
    # transfer's result as quoted data (ADK fences it), and its instructions are its own prompt.
    WORKER_IN_DEBATE = (
        "the Architect is {what} with you and is waiting for your answer to the points in their last turn. It is "
        "one answer per handover, and the call then goes back to the Architect."
    )

    def next_step(self, receiver: Optional[str] = None) -> str:
        """Where the project stands and who acts next, in one or two sentences.

        Told to the agent the call is handed to (``receiver``), so it knows why it has the call even
        when the teammate's words did not say so.
        """
        state = self._store.state
        if receiver == "worker" and self.debate() == "design" and state.design_rounds > 0:
            return self.WORKER_IN_DEBATE.format(what="debating the design")
        if receiver == "worker" and self.debate() == "review":
            return self.WORKER_IN_DEBATE.format(what="reviewing the code")
        if state.delivered:
            return "the project is delivered. The Worker tells the client what is still open."
        if not state.requirements:
            return "no requirements are recorded yet. The PM gathers them with the client."
        if not state.design_written:
            rounds = state.design_rounds
            if rounds == 0:
                return (
                    "the design is not written yet. The Architect proposes it to the Worker and they debate it, "
                    f"{self.MIN_DESIGN_ROUNDS} to {self.MAX_DESIGN_ROUNDS} rounds, then the Architect calls write_design."
                )
            if rounds >= self.MAX_DESIGN_ROUNDS:
                return (
                    f"the design debate is in round {rounds}, the last. The Worker answers; the Architect then settles "
                    "what is left, records the decisions and calls write_design."
                )
            more = (
                "at least one more round"
                if rounds < self.MIN_DESIGN_ROUNDS
                else "more rounds only if a point is still open"
            )
            return (
                f"the design debate is in round {rounds} of {self.MIN_DESIGN_ROUNDS} to {self.MAX_DESIGN_ROUNDS}, with "
                f"{more}. The Architect and the Worker settle it, then the Architect calls write_design."
            )
        if not state.requirements_confirmed:
            return (
                "the client has not approved the requirements and the design yet. The PM asks them to look them over on "
                "their page and approve them."
            )
        if not state.tasks:
            return "the requirements are confirmed and the design is written. The Worker creates the tasks and starts development."
        running = next((j for j in state.jobs if j.status in ("queued", "running")), None)
        if running:
            # Worded so nobody acts: told "the team tells the client where things stand", each agent
            # handed the call said the same status again and handed it on.
            return f"{running.id} ({running.kind}) is running, and nothing is needed from anyone until its result comes as a team update."
        unfinished = [t.id for t in state.tasks if t.status != "done"]
        if unfinished:
            return f"tasks {', '.join(unfinished)} are not done. The Worker starts development for them."
        for step in ("request_review", "check_acceptance", "deliver"):  # check_acceptance covers the review
            blocked = self.blocker(step)
            if blocked:
                return blocked.removeprefix("Not started: ")
        return "everything is built, reviewed and accepted. The Worker delivers the project."

    @staticmethod
    def _confirm_requirements(state: ProjectState, items: list[str]) -> Optional[str]:
        if not state.requirements:
            return "Not confirmed: no requirements are recorded yet. Record each one with record_requirement first."
        if not state.design_written:
            # The client approves what will be built and how together, on their page, once the design is written.
            return (
                "Not confirmed: the design is not written yet. The Architect settles it with the Worker and calls "
                "write_design; then the client looks over the requirements and the design together and approves them."
            )
        return None

    @classmethod
    def _write_design(cls, state: ProjectState, items: list[str]) -> Optional[str]:
        missing = []
        if state.design_rounds < cls.MIN_DESIGN_ROUNDS:
            missing.append(
                f"a real debate with the Worker, at least {cls.MIN_DESIGN_ROUNDS} rounds and so far {state.design_rounds} "
                "(press on the weakest part of their answer and hand them the call again)"
            )
        if not state.components:
            missing.append("at least one component (record_component)")
        if not state.decisions:
            missing.append("at least one decision with the options rejected (record_decision), agreed with the Worker")
        unchallenged = [d.id for d in state.decisions if not d.challenge.strip()]
        if unchallenged:
            missing.append(
                f"the challenge behind decisions {', '.join(unchallenged)}: the objection you raised to the Worker and how "
                "it was answered (record the decision again, with the same title and the challenge)"
            )
        return f"Not written: the design needs {' and '.join(missing)}." if missing else None

    @staticmethod
    def last_done(state: ProjectState, kind: str) -> Optional[float]:
        """When the last job of ``kind`` finished successfully, or ``None``."""
        return max(
            (j.finished_at for j in state.jobs if j.kind == kind and j.status == "done" and j.finished_at), default=None
        )

    @staticmethod
    def _planning(state: ProjectState, step: str) -> Optional[str]:
        missing = []
        if not state.requirements_confirmed:
            missing.append(
                "the client has not approved the requirements and the design (the PM asks them to, on their page)"
            )
        if not state.design_written:
            missing.append(
                "the design is not written (the Architect discusses it with the Worker and calls write_design)"
            )
        if not missing:
            return None
        return f"Not {step}: {'; and '.join(missing)}. Hand over to whoever is next."

    def _create_tasks(self, state: ProjectState, items: list[str]) -> Optional[str]:
        return self._planning(state, "created")

    def _start_development(self, state: ProjectState, items: list[str]) -> Optional[str]:
        status = {f.id: f.status for f in state.findings}
        if items and all(item in status for item in items):
            # Fixing review findings: only those the Architect ruled to fix, after the review conversation.
            not_to_fix = [item for item in items if status[item] != "to fix"]
            if not_to_fix:
                return (
                    f"Not started: findings {', '.join(not_to_fix)} are not ruled to be fixed. The Architect puts each "
                    "finding to the Worker and rules on it with rule_on_finding first; only findings ruled to fix are built."
                )
            return None
        blocked = self._planning(state, "started")
        if blocked:
            return blocked
        if not state.tasks:
            return "Not started: there are no tasks yet. Create them with create_tasks first."
        return None

    @staticmethod
    def _request_review(state: ProjectState, items: list[str]) -> Optional[str]:
        if not any(j.kind == "develop" and j.status == "done" for j in state.jobs):
            return "Not started: nothing has been built yet. The Worker starts development first."
        return None

    @classmethod
    def _review_settled(cls, state: ProjectState) -> Optional[str]:
        """What the code review still needs before the Architect can approve the code, or ``None``."""
        reviewed, built = cls.last_done(state, "review"), cls.last_done(state, "develop")
        if reviewed is None:
            return "Not started: the code has not been reviewed yet. The Architect requests a review first."
        if built is not None and reviewed < built:
            return "Not started: the latest build has not been reviewed. The Architect requests a review first."
        unruled = [f.id for f in state.findings if f.status == "open"]
        if unruled:
            return (
                f"Not started: review findings {', '.join(unruled)} are still open, waiting for the Architect's ruling "
                f"(code review round {state.review_rounds} of at most {cls.MAX_REVIEW_ROUNDS}). The Architect asks the "
                "Worker why the code does each, the Worker answers, and the Architect rules on each with rule_on_finding."
            )
        to_fix = [f.id for f in state.findings if f.status == "to fix"]
        if to_fix:
            return f"Not started: findings {', '.join(to_fix)} are to be fixed. The Worker fixes them with start_development first."
        # A fix counts only once the Reviewer has checked it; a waived finding stays as it is.
        unchecked = [f.id for f in state.findings if f.status not in ("resolved", "waived")]
        if unchecked:
            return (
                f"Not started: the fixes for {', '.join(unchecked)} have not been re-reviewed. The Architect calls "
                "request_review with their ids first."
            )
        return None

    def _rule_on_finding(self, state: ProjectState, items: list[str]) -> Optional[str]:
        if state.review_rounds == 0:
            return (
                "Not ruled: put the findings to the Worker first, as questions about their code (why is it done this "
                "way?), and rule once they have answered."
            )
        return None

    @classmethod
    def _approve_code(cls, state: ProjectState, items: list[str]) -> Optional[str]:
        return cls._review_settled(state)

    @classmethod
    def _check_acceptance(cls, state: ProjectState, items: list[str]) -> Optional[str]:
        blocked = cls._review_settled(state)
        if blocked:
            return blocked
        # The approval counts for the code as it was then: a later build or review needs a new one.
        latest = max((t for t in (cls.last_done(state, "develop"), cls.last_done(state, "review")) if t), default=0.0)
        if state.code_approved_at is None or state.code_approved_at < latest:
            return (
                "Not started: the Architect has not approved the latest code. The Architect calls approve_code first."
            )
        return None

    @classmethod
    def _deliver(cls, state: ProjectState, items: list[str]) -> Optional[str]:
        built = cls.last_done(state, "develop")
        if state.delivered_at is not None and (built is None or built < state.delivered_at):
            # Live: thirteen deliveries of the same code, each passing every other check.
            return "Not started: the project is already delivered, and nothing was built since. The Worker tells the client where it is."
        unfinished = [t.id for t in state.tasks if t.status != "done"]
        if unfinished:
            return (
                f"Not started: tasks {', '.join(unfinished)} are not done. The Worker starts development for them "
                "with start_development first."
            )
        if not state.acceptance or state.acceptance_checked_at is None:
            return "Not started: acceptance has not been checked yet. The PM runs check_acceptance first."
        if built is not None and built > state.acceptance_checked_at:
            return "Not started: the code changed after the last acceptance check. The PM runs check_acceptance again first."
        results = {r.criterion_id: r.result for r in state.acceptance}
        criteria = [r.criterion_id(i) for r in state.requirements for i in range(len(r.acceptance_criteria))]
        unchecked = [c for c in criteria if c not in results]
        if unchecked:
            return (
                f"Not started: criteria {', '.join(unchecked)} were added after the last acceptance check. The client "
                "confirms the changed requirements with the PM, the Worker builds them, and the PM checks acceptance again."
            )
        failing = [c for c in criteria if results[c] != "pass"]
        if failing:
            return f"Not started: criteria {', '.join(failing)} do not pass. The Worker fixes them first."
        unsettled = [a.id for a in state.assumptions if a.needs_settling and not a.settled]
        if unsettled:
            return (
                f"Not started: the Coder's assumptions {', '.join(unsettled)} are not settled. The PM checks a business "
                "one with the client, the Architect decides a technical one, and either calls settle_assumption."
            )
        return None


class HandoffGuard:
    """Stops the voice agents handing the call back and forth forever.

    At most ``limit`` handoffs in a row with nothing to show for them: the count starts again when the
    client speaks or types, or when the team records something (a requirement, a decision, a task; not
    an activity note), so a design debate that keeps agreeing things is never cut off. A refused handoff sends the agent back to the
    client. How often the team hands over is otherwise left to the prompts (one complete turn per
    handover) and the handover brief, not to a limit.

    A refusal is for the agent, not the client: it says so, because an agent told only "refused"
    explained the limit to the client.
    """

    LIMIT = 6
    # Worded so the agent does not repeat itself: told "tell the client in two sentences where the
    # team has got to", an agent that had just done so said it again.
    REFUSAL = (
        "Handoff refused: the team has handed over {count} times without recording anything or hearing from the client, "
        "so you keep the call. If you have not yet told the client where the team has got to, tell them in one or two "
        "sentences and ask what they think; if you have, say nothing more. This is internal: never mention handovers, "
        "limits or this message to the client."
    )

    def __init__(
        self,
        caller_spoke_at: Callable[[], Optional[float]],
        limit: int = LIMIT,
        clock: Callable[[], float] = time.time,
        recorded_at: Callable[[], Optional[float]] = lambda: None,
    ):
        self._caller_spoke_at = caller_spoke_at
        self._recorded_at = recorded_at
        self._limit = limit
        self._clock = clock
        self._lock = threading.Lock()
        self._count = 0
        self._counting_since = 0.0

    def allow(self) -> bool:
        """Count one handoff; ``False`` once ``limit`` were made in a row with nothing to show for them."""
        with self._lock:
            progress = [t for t in (self._caller_spoke_at(), self._recorded_at()) if t is not None]
            if progress and max(progress) > self._counting_since:
                self._count = 0
                self._counting_since = self._clock()
            if self._count >= self._limit:
                _rules_log.info(f"Refused a handoff after {self._count} in a row")
                return False
            self._count += 1
            return True

    def refusal(self) -> str:
        """Why a handoff was refused, and what to do instead."""
        return self.REFUSAL.format(count=self._limit)


# ====================================================================================================
# 3. Speaking into the live call at a quiet moment (CallLink)
#
# Speaking into the live call at the right moment.
#
# Text sent into a call starts a model response, and on Gemini Live it cuts off any response being
# generated, so it must wait until nobody is talking: the model is not mid-response, it has answered
# everything it was last given (the caller's words, text from the client's page, a tool result, this
# link's own post), and nothing has happened for a moment. ``CallLink`` reads that from the
# connection's turn state (``is_responding``, ``is_caller_speaking``, ``last_model_output_at``,
# ``last_caller_speech_at``, ``last_edge_text_at``, ``last_tool_result_at``), which Agent Kernel's
# realtime pool exposes.
# ====================================================================================================

_call_log = logging.getLogger("software_team.call")


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
                _call_log.warning(
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
        _call_log.info(f"Posted into call {self.session_id}: {text}")
        return text

    def _next_pending(self) -> Optional[str]:
        with self._lock:
            while self._pending:
                text, valid = self._pending.popleft()
                if valid is None or valid():
                    return self._text(text)
                _call_log.info(
                    f"Dropped an update for call {self.session_id} that no longer applies: {self._text(text)}"
                )
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
        _call_log.info(f"Call {self.session_id} has a new model connection")
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
                _call_log.exception(f"Call link for {self.session_id} failed a tick")
            self._stop.wait(self.POLL_SECONDS)

    def _pool_connection(self) -> Optional[RealtimeConnection]:
        pool = RealtimeConnectionPool.get()
        return pool.get_connection(self.session_id) if pool is not None else None


# ====================================================================================================
# 4. Background jobs: the Coder and the Reviewer, one job at a time (DevJobRunner)
#
# Background work for one call: the Coder and the Reviewer, run one job at a time off the call's event loop.
# ====================================================================================================

_jobs_log = logging.getLogger("software_team.jobs")


class BackgroundAgentError(RuntimeError):
    """A background agent's run failed: the model or the framework, not the project's code."""


class JobRefused(RuntimeError):
    """A job was not queued: the same kind keeps failing, and the client has not spoken since."""


class DevJobRunner:
    """Runs one call's background jobs in order, each to the end, on a thread and event loop of its own.

    The realtime pool's loop paces the call's audio, so a background agent never runs on it: model
    calls and sandbox waits would stall the voice. Jobs run in the call's ``-dev`` AK session, so the
    Coder and the Reviewer share one sandbox workspace and, holding that session's lock, take turns.

    Test results come from running the tests here, after the agent is done, not from what the agent
    says about them.

    A failure no retry can fix (the model is unavailable, the sandbox is down) must not turn into a
    loop of the voice team restarting the job: after ``MAX_FAILURES_IN_A_ROW`` failures of one kind,
    that kind is refused until the client speaks or types.

    A background agent's word is not taken for its work: a develop job checks the Coder reported
    its tasks, and a review or acceptance job that the Reviewer finished without submitting asks it
    once more, then fails, rather than reporting a clean review or a check that never happened.
    """

    MAX_FAILURES_IN_A_ROW = 2
    # Live: every build ended with the same 2 failing tests, and the Worker started another every ten
    # seconds, 13 times, telling the client each time. After this many, the team must change something.
    STUCK_AFTER = 3
    # Live: the review was done on the board while the Architect was still saying it would ask for one.
    # A job waits until the words announcing it have been heard, at most this long.
    HEARD_TIMEOUT_SECONDS = 20.0
    HEARD_POLL_SECONDS = 0.2
    # A framework runner reports a failed run as a reply worded by agentkernel's
    # ``user_facing_error_message``, not as an exception.
    AGENT_ERROR_PREFIXES = ("Error:", "An unexpected error occurred")

    KINDS = ("develop", "review", "acceptance", "deliver")
    # The process step each kind of job is, checked again when the job starts (``ProcessGates``, above).
    STEPS = {
        "develop": "start_development",
        "review": "request_review",
        "acceptance": "check_acceptance",
        "deliver": "deliver",
    }
    TEST_COMMAND = (
        "pip install -q pytest >/dev/null 2>&1; "
        "if [ -f requirements.txt ]; then pip install -q -r requirements.txt >/dev/null 2>&1; fi; "
        "python -m pytest -q 2>&1"
    )
    MAX_FILES = 300
    MAX_FILE_BYTES = 512_000
    LIST_COMMAND = (
        "find . -type f -not -path '*/.*' -not -path '*/__pycache__/*' -not -path '*/venv/*' "
        f"-not -path '*/node_modules/*' -not -name '*.pyc' | head -n {MAX_FILES}"
    )
    # The workspace packed into one archive in the sandbox, so delivery takes one download instead
    # of one per file. Python, not tar: the sandbox always has it (the team builds Python), and
    # GNU and BSD tar match exclusions differently. The same files as LIST_COMMAND.
    ARCHIVE = ".ak-delivery.tar.gz"
    PACK_CODE = f"""
import os, tarfile
count = 0
with tarfile.open({ARCHIVE!r}, "w:gz") as archive:
    for root, dirs, files in os.walk("."):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "venv", "node_modules"))
        for name in sorted(files):
            path = os.path.join(root, name)
            if name.startswith(".") or name.endswith(".pyc") or count >= {MAX_FILES} or os.path.getsize(path) > {MAX_FILE_BYTES}:
                continue
            archive.add(path, arcname=os.path.relpath(path, "."))
            count += 1
print(count)
"""
    WAIT_SECONDS = 600.0
    # Live: the Coder (GPT-4.1) built everything and passed its tests but reported no task, so none
    # counted as done and the project was delivered with every task "in progress". Asked once more.
    # The Coder starts each run with only its brief (no earlier runs), so it checks the code first.
    REPORT_TASKS = (
        "You finished without reporting your tasks. Check the code and the tests in the sandbox, then for each of "
        '{ids}, call report_progress with its task id and task_status "done" if it is built and tested, or '
        '"in progress" if not. Then reply in one sentence.'
    )
    NOT_SUBMITTED = "\n\nLast time you finished without calling {tool}, so nothing was recorded. Call {tool} this time."

    def __init__(self, call: "TeamCall", service_factory: Callable[[], AgentService] = AgentService):
        self._call = call
        self._service_factory = service_factory
        self._queue: "queue.Queue[tuple[str, str, list[str]]]" = queue.Queue()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._active: Optional[str] = None
        self._job_tests = ""

    def running(self) -> Optional[Job]:
        """The job running now, as the project records it, or ``None``."""
        return next((job for job in self._call.store.state.jobs if job.status == "running"), None)

    def running_kind(self) -> Optional[str]:
        job = self.running()
        return job.kind if job is not None else None

    @property
    def busy(self) -> bool:
        """Whether a job is running or waiting to run."""
        return self._active is not None or not self._queue.empty()

    def submit(self, kind: str, items: Optional[list[str]] = None) -> str:
        """Queue a job and return its id; it starts once the jobs before it are done.

        :raises JobRefused: The same job is already queued or running, or the last jobs of this kind
            all failed and the client has not spoken since, or the last builds are stuck.
        """
        if kind not in self.KINDS:
            raise ValueError(f"Unknown job kind '{kind}'")
        refusal = self.refusal(kind, items)
        if refusal is not None:
            raise JobRefused(refusal)
        items = list(items or [])
        job = self._call.store.add_job(kind, " ".join(items))
        self._queue.put((job.id, kind, items))
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._serve, name=f"dev-jobs-{self._call.session_id}", daemon=True
                )
                self._thread.start()
        return job.id

    def refusal(self, kind: str, items: Optional[list[str]] = None) -> Optional[str]:
        """Why a job of this kind (on ``items``) must not start now, or ``None`` when it may."""
        return (
            self._already_under_way(kind, items)
            or self._failing_runs(kind)
            or (self._stuck_builds() if kind == "develop" else None)
        )

    def _already_under_way(self, kind: str, items: Optional[list[str]]) -> Optional[str]:
        """The same job is already queued or running: asking again must not queue it twice.

        Live, a repeated instruction made the Worker call deliver thirteen times, and thirteen
        delivery jobs queued, each passing its checks.
        """
        detail = " ".join(items or [])
        same = next(
            (
                j
                for j in self._call.store.state.jobs
                if j.kind == kind and j.detail == detail and j.status in ("queued", "running")
            ),
            None,
        )
        if same is None:
            return None
        return f"Not started: {same.id} ({kind}) is already under way, and its result comes as a team update."

    def _failing_runs(self, kind: str) -> Optional[str]:
        """The last jobs of this kind all failed to run (the model or the sandbox), and the client has not spoken or typed since."""
        recent = [job for job in self._call.store.state.jobs if job.kind == kind][-self.MAX_FAILURES_IN_A_ROW :]
        if len(recent) < self.MAX_FAILURES_IN_A_ROW or any(job.status != "failed" for job in recent):
            return None
        spoke = self._call.link.caller_spoke_at()
        if spoke is not None and spoke > (recent[-1].finished_at or 0.0):
            return None
        return (
            f"Not started: the last {self.MAX_FAILURES_IN_A_ROW} {kind} jobs failed ({self._short(recent[-1].summary, 200)}). "
            "Starting it again will not help. Tell the client in one sentence what failed, and start it again only if they ask."
        )

    def _stuck_builds(self) -> Optional[str]:
        """The last builds all ended with the same failing tests, and nothing has changed since: another build will end the same way.

        Something changes when a voice agent records something (a decision, a revised requirement; not
        an activity note such as a question put to the client) or the client speaks or types.
        """
        builds = [j for j in self._call.store.state.jobs if j.kind == "develop" and j.status == "done"][
            -self.STUCK_AFTER :
        ]
        if len(builds) < self.STUCK_AFTER:
            return None
        results = {re.sub(r"\s+in [0-9.]+s$", "", j.tests) for j in builds}
        result = results.pop() if len(results) == 1 else ""
        if not re.search(r"\b(failed|errors?)\b", result):
            return None
        last = builds[-1].finished_at or 0.0
        changed = (self._call.store.last_recorded_at(self._call.VOICE_ROLES), self._call.link.caller_spoke_at())
        if any(at is not None and at > last for at in changed):
            return None
        return (
            f"Not started: the last {self.STUCK_AFTER} builds ended with the same failing tests ({result}), so another "
            "build will end the same way. The Architect reads which tests fail (get_progress lists them) and records a "
            "different approach with record_decision, or the PM asks the client whether to drop or simplify that part "
            "and records it; then development can start again."
        )

    def _serve(self) -> None:
        # One thread and event loop for the call's life, never closed between jobs: LiteLLM keeps a
        # logging task on the loop of its first call, and closing that loop left the task pending
        # ("Task was destroyed but it is pending", "Event loop is closed").
        loop = asyncio.new_event_loop()
        try:
            while True:
                job_id, kind, items = self._queue.get()
                self._active = job_id
                try:
                    loop.run_until_complete(self.run(job_id, kind, items))
                finally:
                    self._active = None
        finally:
            loop.close()

    async def _until_heard(self) -> None:
        """Wait until the client has heard the agent announce the job, so the board does not run ahead of the call.

        A voice agent starts a job with a tool call, which runs as soon as the model makes it, while
        the agent's words before it are still being played; a short job (a review takes seconds) was
        done before the client heard it start.
        """
        deadline = time.monotonic() + self.HEARD_TIMEOUT_SECONDS
        while not self._call.link.heard() and time.monotonic() < deadline:
            await asyncio.sleep(self.HEARD_POLL_SECONDS)

    async def run(self, job_id: str, kind: str, items: list[str]) -> None:
        """Run one job and report how it went, to the project and, if it is still on, to the call.

        The process step is checked again here, not only when the job was requested: a job waits in
        the queue, and what it was allowed to do may have changed meanwhile (a delivery queued
        behind a new build must not ship code nobody checked).
        """
        store = self._call.store
        self._job_tests = ""
        await self._until_heard()
        blocked = self._call.gates.blocker(self.STEPS[kind], items)
        if blocked:
            store.update_job(job_id, "skipped", blocked)
            self._call.notify(f"{job_id} ({kind}) did not start, because things changed while it waited. {blocked}")
            return
        store.update_job(job_id, "running")
        try:
            summary, next_step = await getattr(self, f"_{kind}")(items)
        except Exception as e:
            reason = self._short(str(e), 300)
            _jobs_log.error(f"Job {job_id} ({kind}) failed: {reason}")
            store.update_job(job_id, "failed", reason)
            self._call.notify(
                f"{job_id} ({kind}) failed: {reason}. This is a problem in the team's systems, not in the code, so do not "
                "start the job again. Tell the client in one plain sentence what went wrong, and let them say how to go on."
            )
            return
        store.update_job(job_id, "done", summary, tests=self._job_tests)
        self._call.notify(f"{job_id} ({kind}) finished. {summary} {next_step}")

    # Jobs

    async def _develop(self, items: list[str]) -> tuple[str, str]:
        store = self._call.store
        state = store.state
        findings = [f for f in state.findings if f.id in items and f.status == "to fix"]
        tasks = [t for t in state.tasks if t.id in items]
        if not tasks and not findings:
            tasks = [t for t in state.tasks if t.status != "done"]
        for task in tasks:
            store.set_task_status(task.id, "in progress", "coder")

        work = []
        if tasks:
            work.append("Implement these tasks:\n" + "\n".join(f"- {t.id} {t.title}: {t.details}" for t in tasks))
        if findings:
            work.append(
                "Fix these review findings, and add a test for each:\n"
                + "\n".join(f"- {f.id}: {f.text}" for f in findings)
            )
        try:
            await self.share_files_with_the_coder()
            reply = await self._ask(
                "coder", "\n\n".join(work) or "Make sure the existing code meets the requirements and its tests pass."
            )
            statuses = {t.id: t.status for t in store.state.tasks}
            if tasks and all(statuses.get(t.id) == "in progress" for t in tasks):
                await self._ask("coder", self.REPORT_TASKS.format(ids=", ".join(t.id for t in tasks)))
        except BackgroundAgentError:
            # Nothing was built: the tasks are where they were.
            for task in tasks:
                store.set_task_status(task.id, task.status, "jobs")
            raise

        passed, summary = await self.run_tests()
        # A task is done only when the Coder says so and the tests pass: passing tests alone do not
        # show that every task was built.
        reported = {t.id: t.status for t in store.state.tasks}
        done, unfinished = [], []
        for task in tasks:
            if reported.get(task.id) == "done" and passed:
                done.append(task.id)
            else:
                unfinished.append(task.id)
                if reported.get(task.id) == "done":
                    store.set_task_status(task.id, "needs work", "tests")
        for finding in findings:
            store.set_finding_status(finding.id, "fixed", "coder")

        progress = (
            f" Tasks done: {', '.join(done) or 'none'}; not finished: {', '.join(unfinished)}." if unfinished else ""
        )
        if not passed:
            failures = self._call.store.failing_tests()
            named = f" ({'; '.join(failures[:3])})" if failures else ""
            next_step = (
                f"Next: the Worker starts development again to fix the failing tests{named}; the Coder is given them. "
                "If the same tests keep failing, the Architect decides a different approach first."
            )
        elif unfinished:
            next_step = f"Next: the Worker starts development again for {', '.join(unfinished)} with start_development."
        elif findings:
            fixed = ", ".join(f.id for f in findings)
            next_step = f"Next: the Architect re-checks the fixes for {fixed} with request_review and those ids."
        else:
            next_step = "Next: the Architect reviews it with request_review."
        return f"The Coder says: {self._short(reply)} Tests: {summary}.{progress}", next_step

    async def _review(self, items: list[str]) -> tuple[str, str]:
        store = self._call.store
        before = {f.id for f in store.state.findings}
        recheck = [f for f in store.state.findings if f.id in items]
        if recheck:
            work = (
                "Re-review only these findings. Mark each one that is now fixed as resolved; report any that are not:\n"
                + "\n".join(f"- {f.id}: {f.text}" for f in recheck)
            )
        else:
            work = "Review all the code against the requirements and the design."
        await self._ask_until_submitted(
            "reviewer",
            f"{work}\n\nSubmit your result with submit_review.",
            "submit_review",
            lambda: store.state.review_submitted_at,
        )

        # A fix the Reviewer did not confirm is still to be fixed: the Architect already ruled on it.
        status = {f.id: f.status for f in store.state.findings}
        for finding in recheck:
            if status[finding.id] == "fixed":
                store.set_finding_status(finding.id, "to fix", "reviewer")

        state = store.state
        new = [f for f in state.findings if f.id not in before]
        unruled = [f.id for f in state.findings if f.status == "open"]
        to_fix = [f.id for f in state.findings if f.status == "to fix"]
        resolved = [f.id for f in state.findings if f.id in items and f.status == "resolved"]
        summary = (
            f"Review: {len(new)} new findings ({'; '.join(f'{f.id} {f.text}' for f in new) or 'none'}); "
            f"resolved: {', '.join(resolved) or 'none'}; still to fix: {', '.join(to_fix) or 'none'}."
        )
        if unruled:
            next_step = (
                f"Next: the code review. The Architect puts {', '.join(unruled)} to the Worker as questions about their "
                "code (why is it done this way?), the Worker answers each, and the Architect rules on each with "
                "rule_on_finding: fix it, or waive it when the Worker's reason holds."
            )
        elif to_fix:
            next_step = f"Next: the Worker fixes {', '.join(to_fix)} again with start_development and those ids."
        else:
            next_step = (
                "Next: the Architect approves the code with approve_code and hands over to the PM to check acceptance."
            )
        return summary, next_step

    async def _acceptance(self, items: list[str]) -> tuple[str, str]:
        criteria = self._call.store.criteria()
        work = (
            "Check acceptance: map every criterion below to the test that covers it and run the tests.\n"
            + "\n".join(f"- {criterion_id}: {text}" for criterion_id, text in criteria)
        )
        await self._ask_until_submitted(
            "reviewer",
            f"{work}\n\nSubmit the result with submit_acceptance.",
            "submit_acceptance",
            lambda: self._call.store.state.acceptance_checked_at,
        )

        results = self._call.store.state.acceptance
        failing = [r for r in results if r.result != "pass"]
        passed = len(results) - len(failing)
        summary = f"Acceptance: {passed} of {len(criteria)} criteria pass" + (
            f"; not passing: {', '.join(f'{r.criterion_id} ({r.result})' for r in failing)}." if failing else "."
        )
        next_step = (
            "Next: the Worker delivers with deliver."
            if not failing and passed
            else "Next: the Worker fixes what is not passing."
        )
        return summary, next_step

    async def _deliver(self, items: list[str]) -> tuple[str, str]:
        store = self._call.store
        passed, summary = await self.run_tests()
        copied = await self.copy_workspace(store.folder / "code")
        if copied == 0:
            return (
                f"Not delivered: no code could be copied out of the sandbox into {store.folder}. Tests: {summary}.",
                "Next: the Worker tells the client delivery failed, and tries deliver again once.",
            )
        if not passed:
            # The code is copied for the client to see, but the project is not called delivered.
            return (
                f"Not delivered: the tests fail ({summary}). The code is copied to {store.folder} as unfinished work.",
                "Next: the Worker tells the client, and fixes the failing tests with start_development.",
            )
        store.mark_delivered()
        return (
            f"Delivered to {store.folder}: {copied} files, with the requirements, design, review and acceptance documents. Tests: {summary}.",
            "Next: the Worker tells the client where it is and what is left open.",
        )

    # Sandbox

    async def share_files_with_the_coder(self) -> None:
        """Put the files the client shared into the sandbox, under ``shared/``, for the Coder to use."""
        folder = self._call.store.folder / "shared"
        for shared in self._call.store.state.shared_files:
            path = folder / shared.name
            if path.is_file():
                content = path.read_bytes()
                await self._sandbox(
                    lambda manager, name=shared.name, data=content: manager.upload(f"shared/{name}", data)
                )

    async def run_tests(self) -> tuple[bool, str]:
        """Run the project's tests in the sandbox and record the output; returns whether they passed and a one-line summary."""
        result = await self._sandbox(lambda manager: manager.execute(command=self.TEST_COMMAND, wait=self.WAIT_SECONDS))
        if not isinstance(result, SandboxResult):
            return False, "the test run did not finish"
        output = (result.stdout or "") + (result.stderr or "")
        summary = self.test_summary(output)
        self._call.store.set_test_output(output, summary)
        self._job_tests = summary
        return result.exit_code == 0, summary

    async def copy_workspace(self, destination: Path) -> int:
        """Copy the sandbox workspace's files into ``destination``; returns how many were copied.

        One archive, packed in the sandbox and downloaded once; file by file only if that fails.
        """
        try:
            copied = await self._copy_archive(destination)
        except Exception as e:
            _jobs_log.warning(f"Could not copy the workspace as one archive ({e}); copying it file by file")
            copied = None
        return copied if copied is not None else await self._copy_files(destination)

    async def _copy_archive(self, destination: Path) -> Optional[int]:
        """Pack the workspace in the sandbox, download it, and unpack it into ``destination``; ``None`` if it could not be packed."""
        packed = await self._sandbox(lambda manager: manager.execute(code=self.PACK_CODE, wait=self.WAIT_SECONDS))
        if not isinstance(packed, SandboxResult) or packed.exit_code != 0:
            return None
        content = await self._sandbox(lambda manager: manager.download(self.ARCHIVE))
        await self._sandbox(lambda manager: manager.execute(command=f"rm -f {self.ARCHIVE}", wait=self.WAIT_SECONDS))
        if not content:
            return None
        root = destination.resolve()
        copied = 0
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            for member in archive.getmembers():
                target = (root / member.name).resolve()
                if not member.isfile() or member.size > self.MAX_FILE_BYTES or not target.is_relative_to(root):
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())
                copied += 1
        return copied

    async def _copy_files(self, destination: Path) -> int:
        """Copy the workspace file by file: list it, then download each file."""
        listing = await self._sandbox(
            lambda manager: manager.execute(command=self.LIST_COMMAND, wait=self.WAIT_SECONDS)
        )
        if not isinstance(listing, SandboxResult):
            return 0
        root = destination.resolve()
        copied = 0
        for line in listing.stdout.splitlines():
            path = line.strip().removeprefix("./")
            if not path:
                continue
            target = (root / path).resolve()
            if not target.is_relative_to(root):
                continue
            content = await self._sandbox(lambda manager, path=path: manager.download(path))
            # An empty file is still copied: a package's empty __init__.py is part of the code.
            if len(content) > self.MAX_FILE_BYTES:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            copied += 1
        return copied

    @staticmethod
    def test_summary(output: str) -> str:
        """pytest's closing line (``12 passed in 0.4s``), or ``no test results``.

        Only a counted result counts ("3 passed", "1 failed", "2 errors", "no tests ran"): a bare word
        would match a traceback line, such as one naming starlette's ``errors.py``.
        """
        for line in reversed(output.strip().splitlines()):
            text = line.strip(" =")
            if re.search(r"\b\d+ (passed|failed|errors?)\b|\bno tests ran\b", text):
                return text
        return "no test results"

    async def _sandbox(self, operation: Callable[[ExecutionManager], Any]) -> Any:
        """Run a sandbox operation in the ``-dev`` session, where the background agents' workspace lives."""
        manager = ExecutionManager.get()
        if manager is None:
            raise RuntimeError("The sandbox is not enabled: set sandbox.enabled in config.yaml")
        sessions = Runtime.current().sessions()
        session = sessions.load(self._call.dev_session_id)
        async with session:
            result = await operation(manager)
        sessions.store(session)
        return result

    # Agents

    async def _ask_until_submitted(
        self, agent_name: str, work: str, tool: str, submitted_at: Callable[[], Optional[float]]
    ) -> None:
        """Run a background agent whose work counts only once it calls ``tool``; ask once more if it does not.

        :raises BackgroundAgentError: It finished twice without calling ``tool``.
        """
        before = submitted_at()
        await self._ask(agent_name, work)
        if submitted_at() != before:
            return
        await self._ask(agent_name, work + self.NOT_SUBMITTED.format(tool=tool))
        if submitted_at() == before:
            raise BackgroundAgentError(f"the {agent_name} finished twice without calling {tool}")

    async def _ask(self, agent_name: str, work: str) -> str:
        """Run a background agent on the project brief and a piece of work; returns its reply.

        :raises BackgroundAgentError: The run failed, which the runner reports as an error reply.
        """
        service = self._service_factory()
        service.select(self._call.dev_session_id, agent_name)
        if service.agent is None:
            raise RuntimeError(f"No agent named '{agent_name}' is loaded")
        reply = await service.run(f"{self._call.store.brief()}\n\n{work}")
        if reply.strip().startswith(self.AGENT_ERROR_PREFIXES):
            raise BackgroundAgentError(f"the {agent_name} could not run: {reply.strip()}")
        return reply

    @staticmethod
    def _short(text: str, limit: int = 400) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."


# ====================================================================================================
# 5. Activity: what the background agents are doing, for the board
#
# Making background work observable: what the Coder and the Reviewer are doing, on the board and in the log.
#
# - ``ActivityFeed``: every tool call they make, in plain words, from the frameworks' own hooks.
# - ``ModelWaitWatcher``: the model making them wait (a rate limit).
#
# Neither speaks into the call. Every post into a live call makes the voice agent answer it, so
# running work was narrated over and over in near-identical words; the client follows it on the
# board instead, and an agent asked "how is it going?" answers from ``get_progress``.
# ====================================================================================================

_activity_log = logging.getLogger("software_team.activity")


class Activity(BaseModel):
    at: float
    who: str
    text: str
    # "action" for a tool call, "wait" for the model making the agent wait.
    kind: str = "action"


class ActivityFeed:
    """Every tool call the Coder and the Reviewer make, in plain words, without relying on them to report it.

    Fed by the background agents' ADK ``after_tool_callback`` and by :class:`ModelWaitWatcher`. Each entry is also recorded as
    project progress, which the board, ``get_progress`` and ``log.md`` show.
    """

    # Tools that already record what they did, or that do nothing worth telling.
    QUIET_TOOLS = {
        "report_progress",
        "ask_team",
        "record_assumption",
        "submit_review",
        "submit_acceptance",
        "list_sandbox_sessions",
        "new_sandbox_session",
        "destroy_sandbox_session",
        "check_sandbox_task",
    }

    def __init__(self, store: ProjectStore, clock: Callable[[], float] = time.time):
        self._store = store
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: list[Activity] = []

    def record_tool(self, who: str, tool: str, args: dict, result: Any) -> Optional[Activity]:
        """Record a finished tool call, if it is worth telling; returns the entry."""
        text = self.describe(tool, args or {}, result)
        if text is None:
            return None
        return self._add(Activity(at=self._clock(), who=who, text=text))

    def record_wait(self, who: str, seconds: float) -> Activity:
        """Record the model making a background agent wait before a retry (usually a rate limit)."""
        text = f"is waiting {max(1, round(seconds))} s for the model's rate limit"
        return self._add(Activity(at=self._clock(), who=who, text=text, kind="wait"))

    def since(self, at: float) -> list[Activity]:
        """Entries recorded after ``at``, oldest first."""
        with self._lock:
            return [entry for entry in self._entries if entry.at > at]

    @property
    def last(self) -> Optional[Activity]:
        with self._lock:
            return self._entries[-1] if self._entries else None

    @classmethod
    def describe(cls, tool: str, args: dict, result: Any) -> Optional[str]:
        """A tool call in plain words, or ``None`` for one not worth telling."""
        if tool in cls.QUIET_TOOLS:
            return None
        path = args.get("path", "a file")
        if tool == "write_sandbox_file":
            return f"wrote {path}"
        if tool == "read_sandbox_file":
            return f"read {path}"
        if tool == "run_code":
            return "ran some Python"
        if tool == "run_command":
            command = str(args.get("command", ""))
            if "pytest" in command:
                return f"ran the tests: {DevJobRunner.test_summary(cls._output(result))}"
            if "pip install" in command:
                return "installed the dependencies"
            return f"ran `{command[:60]}`"
        return f"used {tool}"

    @staticmethod
    def _output(result: Any) -> str:
        """The command output in a sandbox tool's result, which is a JSON string (or already a dict)."""
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                return result
        if isinstance(result, dict):
            return f"{result.get('stdout', '')}{result.get('stderr', '')}"
        return str(result)

    def _add(self, entry: Activity) -> Activity:
        with self._lock:
            self._entries.append(entry)
        self._store.add_progress(entry.text, entry.who)
        _activity_log.info(f"{entry.who} {entry.text}")
        return entry


class ModelWaitWatcher(LogLineWatcher):
    """Turns the model client's retry log lines into activity, so a rate-limit wait shows on the board.

    LiteLLM reaches OpenAI through the ``openai`` client, which retries a rate-limited request on its
    own and only logs it ("Retrying request to /chat/completions in 2.0 seconds"), in the background
    agent's run (so the session tells which call it belongs to). Matching its wording is best-effort:
    a line that does not match, or another provider's, is simply not shown.
    """

    LOGGERS = ("openai._base_client",)
    RETRY = re.compile(r"Retrying .*? in ([0-9.]+) seconds")

    def __init__(self, calls: "TeamCalls"):
        super().__init__()
        self._calls = calls

    def line(self, message: str) -> None:
        match = self.RETRY.search(message)
        if match is None:
            return
        try:
            call = self._calls.current()
        except RuntimeError:
            return  # a retry outside any agent run belongs to no call
        who = "coder" if call.jobs.running_kind() == "develop" else "reviewer"
        call.activity.record_wait(who, float(match.group(1)))


# ====================================================================================================
# 6. The client's approval, from the button on their page
#
# The client's approval of what the team will build, given with a button on their page.
# ====================================================================================================


class ClientApproval:
    """Is this what we build? The client answers on their page, with the requirements and the design in front of them.

    Once the Architect has written the design, the page shows the requirements and the design with
    two buttons. Checking documents by ear is hard, and a click is a clear answer: "Looks good" records
    the approval at once, as the PM's ``confirm_requirements`` does after a spoken yes, and tells the
    team; "I'd like a change" tells the team to ask what. Either way the voice agents hear it as a team
    update, so the meeting carries on from it.
    """

    def __init__(self, call: "TeamCall"):
        self._call = call

    def pending(self) -> bool:
        """Whether the requirements and the design are ready and waiting for the client's approval."""
        state = self._call.store.state
        return bool(state.requirements) and state.design_written and not state.requirements_confirmed

    def approve(self, by: str) -> None:
        """Record the client's approval and tell the team.

        :raises ValueError: Nothing is waiting for approval.
        """
        self._check_pending()
        self._call.store.confirm_requirements(by, how="on the client's page")
        self._call.client_answered()
        self._call.notify(
            f"{by} approved the requirements and the design on their page, so the requirements are confirmed: do not "
            "call confirm_requirements. Thank them in one sentence, and hand the call to the Worker to start the build."
        )

    def ask_for_change(self, by: str) -> None:
        """Tell the team the client wants something changed first.

        :raises ValueError: Nothing is waiting for approval.
        """
        self._check_pending()
        self._call.client_answered()
        self._call.notify(
            f'{by} pressed "I\'d like a change" on their page: they want to change something before approving. The PM '
            "asks them, in one easy sentence, what they would like to change, and records it."
        )

    def _check_pending(self) -> None:
        if not self.pending():
            state = self._call.store.state
            if state.requirements_confirmed:
                raise ValueError("The requirements are already approved")
            raise ValueError(
                "There is nothing to approve yet: the team is still working out the requirements and the design"
            )


# ====================================================================================================
# 7. Noticing a question put to the client
#
# Knowing when the team is waiting for the client's answer, from what the agents say.
# ====================================================================================================


class ClientQuestionWatcher(LogLineWatcher):
    """Notices an agent's turn that ends by asking the client something, and marks the team as waiting for the answer.

    The team must know when the client owes an answer: no "nobody is speaking" nudge then, and the
    board shows the question under "Waiting for you". The agents used to call an ``ask_client``
    tool around each question, but a tool answering in the middle of a spoken turn makes Gemini
    Live start talking again: live, the PM asked the same question three times. So an agent just
    asks, and this reads the question from its words.

    The LiveKit gateway logs each agent turn once its audio has played ("AI response: ..."; an
    interrupted turn is not logged). Reading that line is the example's shortcut, as
    ``ClientWordsWatcher`` reads the client's. A question put to a teammate is not one for the client:
    it addresses them by name ("Worker, how would you ...?", "... what do you think, Architect?"),
    and the turn ends with a handover. A question that merely mentions a role ("should the Architect
    plan for phones too?") is still the client's.
    """

    LOGGERS = ("ak.integration.livekit",)
    PREFIX = "AI response: "
    SENTENCE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
    TEAMMATE = re.compile(r"^(worker|architect|pm)\b|\b(worker|architect|pm)\s*\?$", re.IGNORECASE)

    def __init__(self, call: "TeamCall"):
        super().__init__()
        self._call = call
        self._handovers_seen = 0

    def line(self, message: str) -> None:
        if message.startswith(self.PREFIX):
            self.agent_said(message.removeprefix(self.PREFIX))

    @classmethod
    def closing_question(cls, text: str) -> Optional[str]:
        """The turn's last sentence, if it is a question; ``None`` otherwise."""
        sentences = [s.strip() for s in cls.SENTENCE.findall(text) if s.strip()]
        return sentences[-1] if sentences and sentences[-1].endswith("?") else None

    def agent_said(self, text: str) -> None:
        """One agent turn, as it was heard: if it ended by asking the client something, the team now waits for the answer."""
        handovers = self._call.handovers
        handed_over, self._handovers_seen = handovers > self._handovers_seen, handovers
        question = self.closing_question(text)
        if handed_over or question is None or self.TEAMMATE.search(question):
            return
        self._call.ask_client()
        self._call.store.add_progress(f"Asked the client: {question}", self._call.holder)


# ====================================================================================================
# 8. One call's state, tying the above together (TeamCall, TeamCalls)
#
# Everything the team keeps for one call, and how a tool finds it.
# ====================================================================================================


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
