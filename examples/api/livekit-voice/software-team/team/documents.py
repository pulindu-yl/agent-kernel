"""The project documents the client gets, written from the team's records after every change.

Plain templating, no LLM, so the documents say exactly what was recorded:

- ``requirements.md``: what the client wants, with acceptance criteria, assumptions and open questions;
- ``design.md``: the components and the decisions, with the options rejected and why;
- ``status.md``: tasks, background jobs, review findings, acceptance results and the outcome;
- ``log.md``: who recorded what, and when;
- ``test-results.txt``: the last raw pytest output, once tests have run.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from team.models import Note, ProjectState

_log = logging.getLogger("software_team.documents")


class DocumentWriter:
    """Runs ``write`` on a thread of its own after each request, so a change returns at once.

    A voice tool records a change on the realtime pool's event loop, which paces the call's audio;
    rewriting four documents there, or waiting for a background job's rewrite to finish, would hold
    the loop up. Requests made while a write runs are covered by one more write, which reads the
    latest state. The thread ends after ``IDLE_SECONDS`` without a request and starts again on the
    next one.
    """

    IDLE_SECONDS = 5.0

    def __init__(self, write: Callable[[], None], name: str = "project-documents"):
        self._write = write
        self._name = name
        self._cond = threading.Condition()
        self._requested = 0
        self._written = 0
        self._thread: Optional[threading.Thread] = None

    def request(self) -> None:
        """Ask for a write that covers every change made so far. Safe from any thread."""
        with self._cond:
            self._requested += 1
            self._cond.notify_all()
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                self._thread.start()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every write requested so far is done; returns whether it was in time."""
        with self._cond:
            target = self._requested
            return self._cond.wait_for(lambda: self._written >= target, timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                if not self._cond.wait_for(lambda: self._written < self._requested, self.IDLE_SECONDS):
                    self._thread = None
                    return
                target = self._requested
            try:
                self._write()
            except Exception:
                _log.exception("Could not write the project documents")
            with self._cond:
                self._written = target
                self._cond.notify_all()


class ProjectDocuments:
    """Writes the project's documents into its folder."""

    FILES = ("requirements.md", "design.md", "status.md", "log.md")

    @classmethod
    def write(cls, state: ProjectState, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "requirements.md").write_text(cls.requirements(state), encoding="utf-8")
        (folder / "design.md").write_text(cls.design(state), encoding="utf-8")
        (folder / "status.md").write_text(cls.status(state), encoding="utf-8")
        (folder / "log.md").write_text(cls.log(state), encoding="utf-8")
        if state.test_output:
            (folder / "test-results.txt").write_text(state.test_output, encoding="utf-8")

    @classmethod
    def requirements(cls, state: ProjectState) -> str:
        confirmed = "Confirmed with the client" if state.requirements_confirmed else "Draft, not yet confirmed"
        lines = [f"# {state.name}: requirements", "", f"**Goal:** {state.goal}", "", f"**Status:** {confirmed}", ""]
        for r in state.requirements:
            lines += [f"## {r.id}. {r.title}", "", r.description, "", "Acceptance criteria:", ""]
            lines += [f"- **{r.criterion_id(i)}** {c}" for i, c in enumerate(r.acceptance_criteria)]
            lines += ["", f"_Recorded by {r.by}_", ""]
        lines += ["## Assumptions", ""]
        lines += [f"- **{a.id}** {a.text} _({a.by}{cls._settling(a)})_" for a in state.assumptions] or ["None."]
        if state.shared_files:
            lines += ["", "## Files shared by the client", ""] + [
                f"- `shared/{f.name}` ({f.size} bytes)" for f in state.shared_files
            ]
        lines += ["", "## Open questions", ""]
        lines += [
            f"- **{q.id}** {q.text}" + (f" **Answer:** {q.answer}" if q.answer else "") for q in state.questions
        ] or ["None."]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _settling(note: Note) -> str:
        if not note.needs_settling:
            return ""
        return f"; settled: {note.settled}" if note.settled else "; not settled yet"

    @staticmethod
    def design(state: ProjectState) -> str:
        written = "Written by the Architect" if state.design_written else "In discussion"
        lines = [f"# {state.name}: technical design", "", f"**Status:** {written}", ""]
        if state.design_rounds:
            lines += [f"**Debated:** {state.design_rounds} rounds between the Architect and the Worker", ""]
        lines += ["## Components", ""]
        lines += [f"- **{c.name}**: {c.responsibility}" for c in state.components] or ["None yet."]
        if state.build_notes:
            lines += ["", "## Build notes", "", "Choices the Coder made while building, with the reason:", ""]
            lines += [f"- **{n.id}** {n.text}" for n in state.build_notes]
        lines += ["", "## Decisions", ""]
        for d in state.decisions:
            lines += [f"### {d.id}. {d.title}", "", f"**Choice:** {d.choice}", "", f"**Why:** {d.reason}", ""]
            if d.challenge:
                lines += [f"**Challenged:** {d.challenge}", ""]
            if d.rejected:
                lines += ["**Rejected:**", ""] + [f"- {option}" for option in d.rejected] + [""]
            lines += [f"_Decided by {d.by}_", ""]
        return "\n".join(lines) + "\n"

    @staticmethod
    def status(state: ProjectState) -> str:
        done = sum(1 for t in state.tasks if t.status == "done")
        passing = sum(1 for a in state.acceptance if a.result == "pass")
        open_findings = [f for f in state.findings if f.status in ("open", "to fix", "fixed")]
        waived = [f for f in state.findings if f.status == "waived"]
        lines = [
            f"# {state.name}: status",
            "",
            f"- **Delivered:** {'yes' if state.delivered else 'no'}",
            f"- **Tasks done:** {done} of {len(state.tasks)}",
            f"- **Tests:** {state.test_summary or 'not run'}",
            f"- **Review findings:** {len(state.findings)} ({len(open_findings)} open, {len(waived)} waived)",
            f"- **Code approved by the Architect:** {'yes' if state.code_approved_at else 'no'}",
            f"- **Acceptance criteria passing:** {passing} of {len(state.acceptance)}",
            "",
            "## Tasks",
            "",
            "| Task | Status | Details |",
            "|---|---|---|",
        ]
        lines += [f"| {t.id} {t.title} | {t.status} | {t.details} |" for t in state.tasks]
        lines += ["", "## Background jobs", "", "| Job | Kind | Status | Result |", "|---|---|---|---|"]
        lines += [f"| {j.id} | {j.kind} {j.detail} | {j.status} | {j.summary} |" for j in state.jobs]
        lines += ["", "## Review findings", ""]
        lines += [
            f"- **{f.id}** [{f.status}] {f.text}" + (f" _(the Architect: {f.reason})_" if f.reason else "")
            for f in state.findings
        ] or ["None."]
        lines += ["", "## Acceptance", "", "| Criterion | Test | Result |", "|---|---|---|"]
        lines += [f"| **{a.criterion_id}** {a.criterion} | {a.test} | {a.result} |" for a in state.acceptance]
        open_questions = [q for q in state.questions if not q.answer]
        if open_findings or open_questions:
            lines += ["", "## Still open", ""]
            lines += [f"- {f.id}: {f.text}" for f in open_findings] + [f"- {q.id}: {q.text}" for q in open_questions]
        return "\n".join(lines) + "\n"

    @staticmethod
    def log(state: ProjectState) -> str:
        lines = [f"# {state.name}: log", ""]
        lines += [f"- {time.strftime('%H:%M:%S', time.localtime(e.at))} **{e.who}**: {e.what}" for e in state.log]
        return "\n".join(lines) + "\n"
