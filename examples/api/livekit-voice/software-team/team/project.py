"""What the team knows about the project, kept in one place every agent reads and writes."""

import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from team.documents import DocumentWriter, ProjectDocuments
from team.models import (
    AcceptanceResult,
    Component,
    Decision,
    Finding,
    Job,
    LogEntry,
    Note,
    ProjectState,
    Requirement,
    SharedFile,
    Task,
)

_logger = logging.getLogger("software_team.project")


class ProjectStore:
    """One call's project: what the team agreed and did. Every change is written to the documents.

    Shared by the voice tools (on the realtime pool's loop) and the background jobs (on their own
    thread), so every read and write holds one lock. ``ProjectDocuments`` writes the markdown, on a
    ``DocumentWriter`` thread, so a change never waits for files to be written.

    Each project gets a folder of its own: a folder an earlier call left under the same name is
    never reused, so nothing from another project ends up in this one's documents or delivery.
    """

    # The last lines of a failing test run the background agents are given, and how many failing
    # tests the voice agents are told about.
    FAILURE_TAIL_LINES = 40
    MAX_FAILURES_TOLD = 5

    def __init__(self, output_root: Path, name: str):
        self._lock = threading.RLock()
        # Taken before ``_lock`` wherever both are held: a write and a folder move never overlap.
        self._render_lock = threading.Lock()
        self._output_root = output_root
        self._state = ProjectState(name=name)
        # Until the project is named, records go to a folder named after the call, so nothing
        # recorded before start() is lost; start() moves it.
        self._folder = self._free_folder(self.slug(name))
        self._started = False
        self._listeners: list[Callable[[], None]] = []
        self._writer = DocumentWriter(self._write_documents)

    @property
    def folder(self) -> Path:
        """Where the documents are written: the project's folder, or the call's until the project is named."""
        return self._folder

    @property
    def started(self) -> bool:
        """Whether the project has been named with ``start``."""
        return self._started

    @property
    def state(self) -> ProjectState:
        """A copy of the project state, safe to read while the team keeps working."""
        with self._lock:
            return self._state.model_copy(deep=True)

    def on_change(self, listener: Callable[[], None]) -> None:
        """Call ``listener`` once the documents show a change, such as the project board showing it to the client.

        It runs on the document writer's thread, once per batch of changes, so it must return at
        once: schedule the work, do not do it. A listener that raises is logged and never fails a change.
        """
        self._listeners.append(listener)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until the documents show every change made so far; returns whether they did in time."""
        return self._writer.flush(timeout)

    @staticmethod
    def slug(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"

    def _free_folder(self, slug: str) -> Path:
        """This project's folder for ``slug``: its current one if it already has that name, else one no other project uses."""
        current = getattr(self, "_folder", None)
        if current is not None and re.fullmatch(rf"{re.escape(slug)}(-\d+)?", current.name):
            return current
        folder, n = self._output_root / slug, 1
        while folder.exists():
            n += 1
            folder = self._output_root / f"{slug}-{n}"
        return folder

    # Recording

    def start(self, name: str, goal: str, by: str) -> Path:
        """Name the project; its documents move to a folder named after it, as on a second start."""
        with self._render_lock, self._lock:
            folder = self._free_folder(self.slug(name))
            if folder != self._folder and self._folder.exists():
                shutil.move(str(self._folder), str(folder))
            self._folder = folder
            self._started = True
            self._state.name = name
            self._state.goal = goal
            self._log(by, f"Started the project '{name}': {goal}")
            return folder

    def add_requirement(self, title: str, description: str, acceptance_criteria: list[str], by: str) -> Requirement:
        with self._lock:
            self._state.requirements_issued += 1
            requirement = Requirement(
                id=f"R{self._state.requirements_issued}",
                title=title,
                description=description,
                acceptance_criteria=acceptance_criteria,
                by=by,
            )
            self._state.requirements.append(requirement)
            self._state.requirements_confirmed = False
            self._log(by, f"Recorded requirement {requirement.id}: {title}")
            return requirement

    def revise_requirement(
        self, requirement_id: str, title: str, description: str, acceptance_criteria: list[str], by: str
    ) -> Requirement:
        """Change a requirement the client corrected. It keeps its id; the client confirms the requirements
        again, and its old acceptance results no longer count."""
        with self._lock:
            requirement = self._find(self._state.requirements, requirement_id)
            requirement.title, requirement.description = title, description
            requirement.acceptance_criteria = acceptance_criteria
            self._state.requirements_confirmed = False
            self._forget_acceptance(requirement_id)
            self._log(by, f"Revised requirement {requirement_id}: {title}")
            return requirement

    def drop_requirement(self, requirement_id: str, reason: str, by: str) -> Requirement:
        """Remove a requirement the client does not want; the client approves the requirements again."""
        with self._lock:
            requirement = self._find(self._state.requirements, requirement_id)
            self._state.requirements.remove(requirement)
            self._state.requirements_confirmed = False
            self._forget_acceptance(requirement_id)
            self._log(by, f"Dropped requirement {requirement_id} ({requirement.title}): {reason}")
            return requirement

    def add_shared_file(self, name: str, size: int, by: str) -> SharedFile:
        """Record a file the client shared; one shared again under the same name replaces it."""
        with self._lock:
            shared = SharedFile(name=name, size=size, by=by, at=time.time())
            self._state.shared_files = [f for f in self._state.shared_files if f.name != name] + [shared]
            self._log(by, f"Shared the file {name} ({size} bytes)")
            return shared

    def add_assumption(self, text: str, by: str, needs_settling: bool = False) -> Note:
        with self._lock:
            note = Note(id=f"A{len(self._state.assumptions) + 1}", text=text, by=by, needs_settling=needs_settling)
            self._state.assumptions.append(note)
            self._log(by, f"Assumed {note.id}: {text}")
            return note

    def settle_assumption(self, assumption_id: str, outcome: str, by: str) -> Note:
        with self._lock:
            note = self._find(self._state.assumptions, assumption_id)
            note.settled = outcome
            self._log(by, f"Settled {assumption_id}: {outcome}")
            return note

    def add_question(self, text: str, by: str) -> Note:
        with self._lock:
            note = Note(id=f"Q{len(self._state.questions) + 1}", text=text, by=by)
            self._state.questions.append(note)
            self._log(by, f"Opened question {note.id}: {text}")
            return note

    def answer_question(self, question_id: str, answer: str, by: str) -> Note:
        with self._lock:
            note = self._find(self._state.questions, question_id)
            note.answer = answer
            self._log(by, f"Answered {question_id}: {answer}")
            return note

    def add_decision(
        self, title: str, choice: str, reason: str, rejected: list[str], by: str, challenge: str = ""
    ) -> Decision:
        """Record a decision; one with the same title as an earlier one replaces it, keeping its id."""
        with self._lock:
            earlier = next((d for d in self._state.decisions if d.title.strip().lower() == title.strip().lower()), None)
            decision = Decision(
                id=earlier.id if earlier else f"D{len(self._state.decisions) + 1}",
                title=title,
                choice=choice,
                reason=reason,
                rejected=rejected,
                by=by,
                challenge=challenge,
            )
            if earlier:
                self._state.decisions[self._state.decisions.index(earlier)] = decision
            else:
                self._state.decisions.append(decision)
            self._reopen_design()
            self._log(by, f"{'Revised' if earlier else 'Decided'} {decision.id}: {title}: {choice}")
            return decision

    def add_component(self, name: str, responsibility: str, by: str) -> Component:
        with self._lock:
            component = Component(name=name, responsibility=responsibility, by=by)
            self._state.components = [c for c in self._state.components if c.name != name] + [component]
            self._reopen_design()
            self._log(by, f"Added component {name}: {responsibility}")
            return component

    def count_design_round(self, by: str) -> int:
        """Count a round of the design debate (the Architect handed the Worker the design to answer); returns it."""
        with self._lock:
            self._state.design_rounds += 1
            self._log(by, f"Design debate: round {self._state.design_rounds}")
            return self._state.design_rounds

    def _reopen_design(self) -> None:
        """The design changed: it must be written again, and a design changed after it was written is debated again."""
        if self._state.design_written:
            self._state.design_rounds = 0
        self._state.design_written = False

    def confirm_requirements(self, by: str, how: str = "with the client") -> None:
        with self._lock:
            self._state.requirements_confirmed = True
            self._log(by, f"Confirmed the requirements {how}")

    def write_design(self, by: str) -> None:
        with self._lock:
            self._state.design_written = True
            self._log(by, "Wrote the design")

    def add_tasks(self, tasks: list[str], by: str) -> list[Task]:
        """Add tasks, each written as ``title: details`` (details optional); returns the ones added.

        A task with the title of one already there is not added again: a voice turn that repeats
        itself must not double the work.
        """
        with self._lock:
            added = []
            for text in tasks:
                title, _, details = text.partition(":")
                if self.find_task(title) is not None:
                    continue
                task = Task(id=f"T{len(self._state.tasks) + 1}", title=title.strip(), details=details.strip())
                self._state.tasks.append(task)
                added.append(task)
            if added:
                self._log(by, f"Created tasks {', '.join(task.id for task in added)}")
            return added

    def find_task(self, title: str) -> Optional[Task]:
        """The task with this title (ignoring case and spacing), or ``None``."""
        key = " ".join(title.split()).lower()
        with self._lock:
            return next((t for t in self._state.tasks if " ".join(t.title.split()).lower() == key), None)

    def set_task_status(self, task_id: str, status: str, by: str) -> Task:
        with self._lock:
            task = self._find(self._state.tasks, task_id)
            task.status = status
            self._log(by, f"Task {task_id} is {status}")
            return task

    def add_progress(self, text: str, by: str) -> None:
        """Note an activity. It is not a record: it does not count as the team moving on (``last_recorded_at``)."""
        with self._lock:
            self._state.progress.append(LogEntry(at=time.time(), who=by, what=text, kind="progress"))
            self._log(by, f"Progress: {text}", kind="progress")

    def add_findings(self, findings: list[str], by: str) -> list[Finding]:
        """Record new review findings; the code review conversation on them starts at round 0."""
        with self._lock:
            added = [Finding(id=f"F{len(self._state.findings) + i + 1}", text=text) for i, text in enumerate(findings)]
            self._state.findings.extend(added)
            if added:
                self._state.review_rounds = 0
                self._log(by, f"Found {', '.join(f.id for f in added)}")
            return added

    def set_finding_status(self, finding_id: str, status: str, by: str, reason: str = "") -> Finding:
        with self._lock:
            finding = self._find(self._state.findings, finding_id)
            finding.status = status
            if reason:
                finding.reason = reason
            self._log(by, f"Finding {finding_id} is {status}" + (f": {reason}" if reason else ""))
            return finding

    def count_review_round(self, by: str) -> int:
        """Count a round of the code review conversation (the Architect put the findings to the Worker); returns it."""
        with self._lock:
            self._state.review_rounds += 1
            self._log(by, f"Code review: round {self._state.review_rounds}")
            return self._state.review_rounds

    def record_review(self, verdict: str, by: str) -> None:
        """Note that the Reviewer submitted a review, with its overall verdict."""
        with self._lock:
            self._state.review_submitted_at = time.time()
            self.add_progress(f"Review: {verdict}", by)

    def approve_code(self, summary: str, by: str) -> None:
        with self._lock:
            self._state.code_approved_at = time.time()
            self._log(by, f"Approved the code: {summary}")

    def add_build_note(self, choice: str, reason: str, by: str) -> Note:
        """Record a choice the Coder made that the design did not settle, and why."""
        with self._lock:
            note = Note(id=f"N{len(self._state.build_notes) + 1}", text=f"{choice} ({reason})", by=by)
            self._state.build_notes.append(note)
            self._log(by, f"Build note {note.id}: {note.text}")
            return note

    def set_acceptance(self, results: list[AcceptanceResult], by: str) -> None:
        with self._lock:
            self._state.acceptance = results
            self._state.acceptance_checked_at = time.time()
            passed = sum(1 for r in results if r.result == "pass")
            self._log(by, f"Checked acceptance: {passed} of {len(results)} criteria pass")

    def set_test_output(self, output: str, summary: str) -> None:
        with self._lock:
            self._state.test_output = output
            self._state.test_summary = summary
            self._log("tests", f"Test run: {summary}")

    def add_job(self, kind: str, detail: str) -> Job:
        with self._lock:
            job = Job(id=f"J{len(self._state.jobs) + 1}", kind=kind, detail=detail)
            self._state.jobs.append(job)
            self._log("jobs", f"Queued {job.id} ({kind}) {detail}".strip())
            return job

    def update_job(self, job_id: str, status: str, summary: str = "", tests: str = "") -> Job:
        with self._lock:
            job = self._find(self._state.jobs, job_id)
            job.status = status
            if tests:
                job.tests = tests
            if summary:
                job.summary = summary
            if status == "running":
                job.started_at = time.time()
            elif status in ("done", "failed", "skipped"):
                job.finished_at = time.time()
            self._log("jobs", f"{job.id} ({job.kind}) is {status}{': ' + summary if summary else ''}")
            return job

    def mark_delivered(self) -> None:
        with self._lock:
            self._state.delivered = True
            self._state.delivered_at = time.time()
            self._log("jobs", "Delivered the project")

    # Reading

    def criteria(self) -> list[tuple[str, str]]:
        """Every acceptance criterion as ``(id, text)``, ids like ``R1.2``."""
        with self._lock:
            return [
                (r.criterion_id(i), c) for r in self._state.requirements for i, c in enumerate(r.acceptance_criteria)
            ]

    def failing_tests(self) -> list[str]:
        """The failing tests of the last test run, from pytest's short summary (``FAILED ...``, ``ERROR ...``)."""
        with self._lock:
            output = self._state.test_output
        return [line.strip() for line in output.splitlines() if line.startswith(("FAILED ", "ERROR "))]

    def brief(self) -> str:
        """The project as a background agent needs it: requirements, decisions, design, tasks, open findings and failing tests."""
        state = self.state
        lines = [f"Project: {state.name}. Goal: {state.goal}", "", "Requirements:"]
        for r in state.requirements:
            lines.append(f"- {r.id} {r.title}: {r.description}")
            lines.extend(f"  - {r.criterion_id(i)}: {c}" for i, c in enumerate(r.acceptance_criteria))
        if state.assumptions:
            lines += ["", "Assumptions:"] + [f"- {a.id}: {a.text}" for a in state.assumptions]
        if state.shared_files:
            lines += ["", "Files the client shared (in the sandbox under shared/):"]
            lines += [f"- shared/{f.name}" for f in state.shared_files]
        answered = [q for q in state.questions if q.answer]
        if answered:
            lines += ["", "Answered questions:"] + [f"- {q.id}: {q.text} -> {q.answer}" for q in answered]
        if state.components:
            lines += ["", "Components:"] + [f"- {c.name}: {c.responsibility}" for c in state.components]
        if state.decisions:
            lines += ["", "Decisions:"] + [f"- {d.id} {d.title}: {d.choice} ({d.reason})" for d in state.decisions]
        if state.tasks:
            lines += ["", "Tasks:"] + [f"- {t.id} [{t.status}] {t.title}: {t.details}" for t in state.tasks]
        if state.build_notes:
            lines += ["", "Build notes (choices made while coding):"] + [
                f"- {n.id}: {n.text}" for n in state.build_notes
            ]
        to_fix = [f for f in state.findings if f.status == "to fix"]
        if to_fix:
            lines += ["", "Review findings to fix:"] + [f"- {f.id}: {f.text}" for f in to_fix]
        waived = [f for f in state.findings if f.status == "waived"]
        if waived:
            lines += ["", "Review findings the Architect waived (leave them as they are, do not report them again):"]
            lines += [f"- {f.id}: {f.text} (waived: {f.reason})" for f in waived]
        if self._failed(state.test_summary):
            tail = "\n".join(state.test_output.strip().splitlines()[-self.FAILURE_TAIL_LINES :])
            lines += ["", f"The last test run failed ({state.test_summary}). The end of its output:", tail]
        return "\n".join(lines)

    @staticmethod
    def _failed(summary: str) -> bool:
        return bool(re.search(r"\b(failed|errors?)\b", summary))

    def _failures_told(self) -> str:
        """The failing tests, for the voice agents, or "" when the last run passed."""
        failures = self.failing_tests()
        if not failures:
            return ""
        shown = "; ".join(failures[: self.MAX_FAILURES_TOLD])
        more = len(failures) - self.MAX_FAILURES_TOLD
        return f"Failing tests: {shown}" + (f" (and {more} more)" if more > 0 else "") + "."

    def overview(self) -> str:
        """A short spoken-friendly summary of where the project stands."""
        state = self.state
        if not state.requirements and not state.goal:
            return "No project has been started yet."
        done = sum(1 for t in state.tasks if t.status == "done")
        open_questions = [q for q in state.questions if not q.answer]
        unruled = [f for f in state.findings if f.status == "open"]
        to_fix = [f for f in state.findings if f.status in ("to fix", "fixed")]
        waived = [f for f in state.findings if f.status == "waived"]
        parts = [
            f"Project {state.name}: {state.goal}." if state.goal else f"Project {state.name}.",
            f"{len(state.requirements)} requirements ({'confirmed' if state.requirements_confirmed else 'not yet confirmed'} with the client): "
            + "; ".join(f"{r.id} {r.title}" for r in state.requirements)
            + ".",
            f"{len(state.decisions)} design decisions; design {'written' if state.design_written else 'not written yet'}.",
            f"{done} of {len(state.tasks)} tasks done." if state.tasks else "No tasks yet.",
        ]
        if state.assumptions:
            parts.append("Assumptions: " + "; ".join(a.text for a in state.assumptions) + ".")
        unsettled = [a for a in state.assumptions if a.needs_settling and not a.settled]
        if unsettled:
            parts.append(
                "The Coder's assumptions still to settle: " + "; ".join(f"{a.id} {a.text}" for a in unsettled) + "."
            )
        if open_questions:
            parts.append("Open questions: " + "; ".join(q.text for q in open_questions) + ".")
        if state.build_notes:
            parts.append("The Coder's build notes: " + "; ".join(f"{n.id} {n.text}" for n in state.build_notes) + ".")
        if unruled:
            parts.append(
                "Review findings waiting for the Architect's ruling: "
                + "; ".join(f"{f.id} {f.text}" for f in unruled)
                + "."
            )
        if to_fix:
            parts.append("Review findings to fix: " + "; ".join(f"{f.id} [{f.status}] {f.text}" for f in to_fix) + ".")
        if waived:
            parts.append("Waived findings: " + "; ".join(f"{f.id} {f.text} ({f.reason})" for f in waived) + ".")
        if state.code_approved_at:
            parts.append("The Architect approved the code.")
        if state.test_summary:
            parts.append(f"Last test run: {state.test_summary}.")
        failures = self._failures_told()
        if failures:
            parts.append(failures)
        return " ".join(parts)

    def progress_report(self) -> str:
        """What the background work is doing, from the recorded jobs and progress."""
        state = self.state
        running = [j for j in state.jobs if j.status == "running"]
        queued = [j for j in state.jobs if j.status == "queued"]
        finished = [j for j in state.jobs if j.status in ("done", "failed")]
        parts = []
        if running:
            job = running[0]
            parts.append(
                f"{job.id} ({job.kind}) has been running for {int(time.time() - (job.started_at or time.time()))} seconds."
            )
        if queued:
            parts.append(f"Queued: {', '.join(f'{j.id} ({j.kind})' for j in queued)}.")
        if state.progress:
            parts.append("Latest progress: " + " / ".join(p.what for p in state.progress[-3:]) + ".")
        if finished:
            parts.append(
                f"Last finished: {finished[-1].id} ({finished[-1].kind}, {finished[-1].status}): {finished[-1].summary}"
            )
        if state.tasks:
            parts.append("Tasks: " + "; ".join(f"{t.id} {t.status}" for t in state.tasks) + ".")
        if state.test_summary:
            parts.append(f"Last test run: {state.test_summary}.")
        failures = self._failures_told()
        if failures:
            parts.append(failures)
        return " ".join(parts) or "No development work has started yet."

    def last_recorded_at(self, roles: tuple[str, ...]) -> Optional[float]:
        """When one of ``roles`` last recorded something (a record, not an activity note), from the log; ``None`` if never."""
        with self._lock:
            return max((e.at for e in self._state.log if e.who in roles and e.kind == "record"), default=None)

    # Internals

    def _write_documents(self) -> None:
        """Rewrite the documents from the current state, then tell the listeners. Runs on the writer's thread."""
        with self._render_lock:
            with self._lock:
                state, folder = self._state.model_copy(deep=True), self._folder
            ProjectDocuments.write(state, folder)
        for listener in self._listeners:
            try:
                listener()
            except Exception:
                _logger.exception("A project change listener failed")

    def _log(self, who: str, what: str, kind: str = "record") -> None:
        """Every change ends here: it is logged, and the documents are rewritten and the listeners told, off this thread."""
        self._state.log.append(LogEntry(at=time.time(), who=who, what=what, kind=kind))
        self._writer.request()

    def _forget_acceptance(self, requirement_id: str) -> None:
        self._state.acceptance = [
            a for a in self._state.acceptance if not a.criterion_id.startswith(f"{requirement_id}.")
        ]

    @staticmethod
    def _find(items: list, item_id: str):
        for item in items:
            if item.id == item_id:
                return item
        raise KeyError(f"No {item_id} in this project")
