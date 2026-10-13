"""The rules that keep the meeting on track, whatever the models do.

- ``ProcessGates``: the order the team works in (requirements, design, build, review, acceptance,
  delivery). A tool asked to skip ahead answers "Not ..." with what is missing and who should act.
- ``HandoffGuard``: stops the voice agents handing the call to each other forever.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

from team.models import ProjectState

if TYPE_CHECKING:
    from team.project import ProjectStore

_log = logging.getLogger("software_team.rules")


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
                _log.info(f"Refused a handoff after {self._count} in a row")
                return False
            self._count += 1
            return True

    def refusal(self) -> str:
        """Why a handoff was refused, and what to do instead."""
        return self.REFUSAL.format(count=self._limit)
