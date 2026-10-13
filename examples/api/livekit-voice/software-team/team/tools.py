"""The team's tools: short ones for the voice agents, and the background agents' own."""

import asyncio
from typing import Callable

from team.jobs import JobRefused
from team.models import AcceptanceResult
from team.state import TeamCall, TeamCalls


class TeamTools:
    """One role's tools, as plain methods a framework's ToolBuilder binds.

    One instance per role, so each record says who made it. Every tool finds its call from the
    session it runs in: a voice tool in the call's session, a background tool in the call's
    ``-dev`` session. Voice tools return at once; the job tools only queue background work, whose
    result comes back into the call as a team update.

    A tool's reply states what happened and never tells the agent what to say: how to speak is the
    agents' prompt's job. A reply that asked for a sentence ("tell the client the team started
    building it") made an agent that had already said so say it again.
    """

    # What the Coder may call a task's state, mapped to the two the team tracks.
    TASK_STATUSES = {
        "done": "done",
        "completed": "done",
        "complete": "done",
        "finished": "done",
        "in progress": "in progress",
        "started": "in progress",
        "working": "in progress",
    }

    VOICE_TOOLS = {
        "worker": ["create_tasks", "start_development", "deliver"],
        "pm": [
            "start_project",
            "settle_assumption",
            "revise_requirement",
            "drop_requirement",
            "record_requirement",
            "record_assumption",
            "record_question",
            "confirm_requirements",
            "check_acceptance",
        ],
        "architect": [
            "record_decision",
            "record_component",
            "write_design",
            "request_review",
            "rule_on_finding",
            "approve_code",
            "settle_assumption",
        ],
    }
    # Any voice agent may answer the Coder: the Architect a technical question, the PM a business one.
    SHARED_VOICE_TOOLS = ["get_project_state", "get_progress", "answer_coder"]
    BACKGROUND_TOOLS = {
        "coder": ["report_progress", "ask_team", "record_assumption", "record_build_note"],
        "reviewer": ["submit_review", "submit_acceptance"],
    }

    def __init__(self, calls: TeamCalls, role: str):
        if role not in self.VOICE_TOOLS and role not in self.BACKGROUND_TOOLS:
            raise ValueError(f"Unknown role '{role}'")
        self._calls = calls
        self.role = role

    def tools(self) -> list[Callable]:
        """The tools this role is given."""
        if self.role in self.VOICE_TOOLS:
            names = self.VOICE_TOOLS[self.role] + self.SHARED_VOICE_TOOLS
        else:
            names = self.BACKGROUND_TOOLS[self.role]
        return [getattr(self, name) for name in names]

    def _call(self) -> TeamCall:
        return self._calls.current()

    def _start_job(self, step: str, kind: str, started: str, items: list[str] | None = None) -> str:
        """Queue a background job; ``started`` is the reply, with ``{job_id}`` in it.

        A step the process does not allow yet, or a job refused after repeated failures, is not
        queued; the reason is the reply instead, so the voice agent can say it and act on it.
        """
        call = self._call()
        blocked = call.gates.blocker(step, items)
        if blocked:
            return blocked
        try:
            return started.format(job_id=call.jobs.submit(kind, items))
        except JobRefused as refused:
            return str(refused)

    # Every voice agent

    def get_project_state(self) -> str:
        """Get where the project stands: the requirements, assumptions, open questions, decisions, tasks, findings and last test run."""
        return self._call().store.overview()

    def get_progress(self) -> str:
        """Get what the background developers are doing right now: the running job, the latest progress and the last result."""
        call = self._call()
        report = call.store.progress_report()
        question = call.coder_question.open
        return f"{report} The Coder is waiting for an answer to: {question}" if question else report

    # Worker

    def create_tasks(self, tasks: list[str]) -> str:
        """Create the development tasks, once the requirements are confirmed and the design is written.

        :param tasks: One entry per task, written as "short title: what it covers, with the requirement ids".
        """
        call = self._call()
        blocked = call.gates.blocker("create_tasks")
        if blocked:
            return blocked
        existing = [task for text in tasks if (task := call.store.find_task(text.partition(":")[0])) is not None]
        added = call.store.add_tasks(tasks, self.role)
        if not added:
            ids = ", ".join(t.id for t in call.store.state.tasks)
            return f"Not created: these tasks already exist ({ids}); start_development builds the unfinished ones."
        reply = "Created " + "; ".join(f"{t.id} {t.title}" for t in added) + "."
        if existing:
            reply += " Already there, so not added again: " + "; ".join(f"{t.id} {t.title}" for t in existing) + "."
        return reply

    def start_development(self, item_ids: list[str]) -> str:
        """Start the background Coder on tasks, or on review findings to fix. Returns at once; the result arrives as a team update.

        :param item_ids: Task ids (such as T1) and/or finding ids (such as F2). Empty means every unfinished task.
        """
        return self._start_job(
            "start_development",
            "develop",
            "Started development job {job_id}. Its progress shows on the client's board; its result comes as a team update.",
            item_ids,
        )

    def deliver(self) -> str:
        """Deliver the finished project: copy the code and the test results into the project folder next to the documents."""
        return self._start_job(
            "deliver",
            "deliver",
            "Started delivery job {job_id}; its result comes as a team update.",
        )

    # PM

    def start_project(self, name: str, goal: str) -> str:
        """Start the project once the client has said what they want. Call it before recording anything else.

        :param name: A short name for the project, such as "login system".
        :param goal: One sentence on what the client wants to achieve.
        """
        store = self._call().store
        state = store.state
        if store.started and state.requirements_confirmed:
            # Live: the model connection dropped, the new one remembered nothing, and the PM started over,
            # renaming a confirmed project. The records are still here.
            return (
                f"Not started: the project '{state.name}' is already under way, with requirements the client "
                "confirmed. If the conversation just started over, it is the same project: call get_project_state "
                "and continue it."
            )
        folder = store.start(name, goal, self.role)
        return f"Project '{name}' started. Its documents are written to {folder}."

    def record_requirement(self, title: str, description: str, acceptance_criteria: list[str]) -> str:
        """Record a business requirement the client stated, with criteria a test could check.

        :param title: A short name, such as "Password reset".
        :param description: What the client needs, in one or two sentences.
        :param acceptance_criteria: Testable statements, such as "A reset link expires after 30 minutes".
        """
        requirement = self._call().store.add_requirement(title, description, acceptance_criteria, self.role)
        criteria = ", ".join(requirement.criterion_id(i) for i in range(len(acceptance_criteria)))
        return f"Recorded {requirement.id} ({title}) with criteria {criteria}."

    def revise_requirement(
        self, requirement_id: str, title: str, description: str, acceptance_criteria: list[str]
    ) -> str:
        """Change a requirement the client corrected, at once: saying it will be corrected changes nothing.

        :param requirement_id: The requirement's id, such as "R1".
        :param title: The corrected short name.
        :param description: What the client needs, in one or two sentences.
        :param acceptance_criteria: The corrected testable statements.
        """
        store = self._call().store
        if not any(r.id == requirement_id for r in store.state.requirements):
            return f"Not revised: there is no requirement {requirement_id}. Call get_project_state to see them."
        requirement = store.revise_requirement(requirement_id, title, description, acceptance_criteria, self.role)
        criteria = ", ".join(requirement.criterion_id(i) for i in range(len(acceptance_criteria)))
        return f"Revised {requirement_id} ({title}) with criteria {criteria}. The client approves the plan again, on their page or out loud."

    def drop_requirement(self, requirement_id: str, reason: str) -> str:
        """Remove a requirement the client does not want.

        :param requirement_id: The requirement's id, such as "R1".
        :param reason: Why, in one sentence, such as "The client sells cosmetics, not software".
        """
        store = self._call().store
        if not any(r.id == requirement_id for r in store.state.requirements):
            return f"Not dropped: there is no requirement {requirement_id}. Call get_project_state to see them."
        store.drop_requirement(requirement_id, reason, self.role)
        return f"Dropped {requirement_id}. The client approves the plan again, on their page or out loud."

    def record_assumption(self, assumption: str) -> str:
        """Record an assumption the team made because the client left it to the team or it was not covered.

        :param assumption: What the team assumed, in one sentence.
        """
        call = self._call()
        # The Coder decided something no one in the call agreed, so it must be settled before delivery.
        by_coder = self.role == "coder"
        note = call.store.add_assumption(assumption, self.role, needs_settling=by_coder)
        if by_coder:
            call.notify(
                f"The Coder assumed {note.id}: {assumption} If this is a business rule, the PM tells the client in one "
                "sentence and asks them to confirm it; if they want something else, the PM records it with "
                "record_requirement. If it is technical, the Architect decides. Either way, call settle_assumption with "
                f"{note.id} and the outcome: the project cannot be delivered until it is settled."
            )
        return f"Recorded assumption {note.id}."

    def settle_assumption(self, assumption_id: str, outcome: str) -> str:
        """Settle an assumption the Coder made on its own: confirmed, or replaced by a requirement.

        :param assumption_id: The assumption's id, such as "A2".
        :param outcome: How it was settled, in one sentence, such as "The client confirmed it" or "Replaced by R4: cancellations up to 48 hours before".
        """
        call = self._call()
        if not any(a.id == assumption_id for a in call.store.state.assumptions):
            return f"Not settled: there is no assumption {assumption_id}. Call get_project_state to see them."
        call.store.settle_assumption(assumption_id, outcome, self.role)
        return f"Settled {assumption_id}."

    def record_question(self, question: str) -> str:
        """Record a question that stays open for now.

        :param question: The open question, in one sentence.
        """
        note = self._call().store.add_question(question, self.role)
        return f"Recorded open question {note.id}."

    def confirm_requirements(self) -> str:
        """Mark the requirements as confirmed, only after the client has agreed to them."""
        call = self._call()
        blocked = call.gates.blocker("confirm_requirements")
        if blocked:
            return blocked
        call.store.confirm_requirements(self.role)
        return f"Requirements confirmed. {call.store.overview()}"

    def answer_coder(self, answer: str) -> str:
        """Answer the background Coder's open question.

        :param answer: The answer, from the requirements or from the client.
        """
        call = self._call()
        question = call.coder_question.answer(answer)
        if question is None:
            return "The Coder has no open question."
        note = call.store.add_question(question, "coder")
        call.store.answer_question(note.id, answer, self.role)
        return "Answer passed to the Coder."

    def check_acceptance(self) -> str:
        """Check every acceptance criterion against the real tests. Returns at once; the result arrives as a team update."""
        return self._start_job(
            "check_acceptance",
            "acceptance",
            "Started acceptance check {job_id}; its result comes as a team update.",
        )

    # Architect

    def record_decision(self, title: str, choice: str, reason: str, challenge: str, rejected_options: list[str]) -> str:
        """Record a technical decision the team agreed on, after debating it with the Worker. Recording one with the same title again replaces it.

        :param title: What was decided, such as "Password reset tokens".
        :param choice: The chosen approach.
        :param reason: Why it was chosen.
        :param challenge: The objection you raised to the Worker's proposal and how it was answered, such as "Two users could book the same slot at once: a unique index on the slot makes the second booking fail".
        :param rejected_options: The options considered and rejected, each with why, such as "User id in the link: guessable".
        """
        store = self._call().store
        decision = store.add_decision(title, choice, reason, rejected_options, self.role, challenge=challenge)
        return f"Recorded decision {decision.id}."

    def record_component(self, name: str, responsibility: str) -> str:
        """Record a main component of the architecture.

        :param name: The component, such as "Auth service".
        :param responsibility: What it is responsible for, in one sentence.
        """
        self._call().store.add_component(name, responsibility, self.role)
        return f"Recorded component {name}."

    def write_design(self) -> str:
        """Mark the design as written, once the components and decisions are agreed."""
        call = self._call()
        blocked = call.gates.blocker("write_design")
        if blocked:
            return blocked
        call.store.write_design(self.role)
        state = call.store.state
        return f"Design written to {call.store.folder}/design.md: {len(state.components)} components, {len(state.decisions)} decisions."

    def request_review(self, finding_ids: list[str]) -> str:
        """Start the background Reviewer on the real code and tests. Returns at once; the findings arrive as a team update.

        :param finding_ids: Finding ids (such as F1) to re-check after a fix. Empty means review all the code.
        """
        return self._start_job(
            "request_review",
            "review",
            "Started review job {job_id}; the findings come as a team update.",
            finding_ids,
        )

    def rule_on_finding(self, finding_id: str, ruling: str, reason: str) -> str:
        """Rule on a review finding once the Worker has answered your question about it: fix it, or waive it when their reason holds.

        :param finding_id: The finding, such as "F2".
        :param ruling: "fix" or "waive".
        :param reason: Why, in one sentence, such as "Two people can book one slot" or, for a waiver, "The unique index from D1 already stops double booking".
        """
        call = self._call()
        finding = next((f for f in call.store.state.findings if f.id == finding_id), None)
        if finding is None:
            return f"Not ruled: there is no finding {finding_id}. Call get_project_state to see them."
        if finding.status != "open":
            return f"Not ruled: {finding_id} is already {finding.status}."
        ruling = ruling.strip().lower()
        if ruling not in ("fix", "waive"):
            return 'Not ruled: the ruling is "fix" or "waive".'
        blocked = call.gates.blocker("rule_on_finding")
        if blocked:
            return blocked
        call.store.set_finding_status(finding_id, "to fix" if ruling == "fix" else "waived", self.role, reason)
        left = [f.id for f in call.store.state.findings if f.status == "open"]
        return f"Ruled on {finding_id}." + (f" Still to rule on: {', '.join(left)}." if left else "")

    def approve_code(self, summary: str) -> str:
        """Approve the code once the review is settled: every finding fixed and re-checked, or waived.

        :param summary: How the review went, in one sentence, such as "Approved after one fix (F1, double booking); F2 waived: not in the requirements".
        """
        call = self._call()
        blocked = call.gates.blocker("approve_code")
        if blocked:
            return blocked
        call.store.approve_code(summary, self.role)
        return "Code approved."

    # Coder

    def report_progress(self, message: str, task_id: str, task_status: str) -> str:
        """Report progress to the team.

        :param message: What you just did or found, in one sentence.
        :param task_id: The task it is about (such as T1), or "" for none.
        :param task_status: The task's new status ("in progress" or "done"), or "" to leave it.
        """
        store = self._call().store
        store.add_progress(message, self.role)
        if task_id and task_status:
            status = self.TASK_STATUSES.get(" ".join(task_status.lower().replace("_", " ").replace("-", " ").split()))
            if status is None:
                return f'Noted, but a task\'s status is "in progress" or "done", not "{task_status}": report it again.'
            try:
                store.set_task_status(task_id, status, self.role)
            except KeyError:
                return f"Noted, but there is no task {task_id}."
        return "Noted."

    async def ask_team(self, question: str) -> str:
        """Ask the team one short question when a requirement is unclear. Waits up to two minutes for the answer.

        :param question: The question, in one sentence.
        """
        answer = await asyncio.to_thread(self._call().ask_team, question)
        if answer is None:
            return "No answer within two minutes: decide yourself, record it with record_assumption, and carry on."
        return f"The team says: {answer}"

    def record_build_note(self, choice: str, reason: str) -> str:
        """Record a choice you made that the design did not settle, and why, so the team can explain the code in the review.

        :param choice: What you chose, such as "Prices are stored in cents".
        :param reason: Why, such as "Adding them up never rounds wrongly".
        """
        note = self._call().store.add_build_note(choice, reason, self.role)
        return f"Noted {note.id}."

    # Reviewer

    def submit_review(self, verdict: str, new_findings: list[str], resolved_finding_ids: list[str]) -> str:
        """Submit a code review.

        :param verdict: One sentence on the overall state of the code.
        :param new_findings: Each issue worth fixing, in one sentence naming the file; empty if none.
        :param resolved_finding_ids: Ids of earlier findings (such as F1) that are now fixed; empty if none.
        """
        store = self._call().store
        added = store.add_findings(new_findings, self.role)
        # Only a fix the Coder made can be confirmed: an open finding waits for the Architect's
        # ruling, and a waived one stays waived.
        status = {f.id: f.status for f in store.state.findings}
        resolved = [finding_id for finding_id in resolved_finding_ids if status.get(finding_id) == "fixed"]
        for finding_id in resolved:
            store.set_finding_status(finding_id, "resolved", self.role)
        store.record_review(verdict, self.role)
        reply = f"Recorded {len(added)} new findings and {len(resolved)} resolved."
        kept = [
            finding_id for finding_id in resolved_finding_ids if finding_id in status and finding_id not in resolved
        ]
        if kept:
            reply += f" Not resolved: {', '.join(kept)}, which the Coder has not fixed; they stay as they are."
        return reply

    def submit_acceptance(self, results: list[str]) -> str:
        """Submit the acceptance check.

        :param results: One entry per criterion, written as "criterion id | test name | pass or fail", such as "R1.1 | tests/test_auth.py::test_signup | pass".
        """
        store = self._call().store
        criteria = dict(store.criteria())
        recorded = []
        for entry in results:
            parts = [part.strip() for part in entry.split("|")]
            if len(parts) != 3 or parts[0] not in criteria:
                continue
            criterion_id, test, result = parts
            recorded.append(
                AcceptanceResult(
                    criterion_id=criterion_id, criterion=criteria[criterion_id], test=test, result=self._result(result)
                )
            )
        covered = {r.criterion_id for r in recorded}
        recorded += [
            AcceptanceResult(criterion_id=c, criterion=text, test="none", result="not covered")
            for c, text in criteria.items()
            if c not in covered
        ]
        store.set_acceptance(recorded, self.role)
        return f"Recorded {len(recorded)} acceptance results."

    @staticmethod
    def _result(text: str) -> str:
        """An acceptance result as the team tracks it: "pass" or "fail" (so "passed" counts), else the words given."""
        word = text.strip().lower()
        if word.startswith("pass"):
            return "pass"
        if word.startswith("fail"):
            return "fail"
        return word
