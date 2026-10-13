"""Background work for one call: the Coder and the Reviewer, run one job at a time off the call's event loop."""

import asyncio
import io
import logging
import queue
import re
import tarfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from agentkernel.core.runtime import Runtime
from agentkernel.core.service import AgentService
from agentkernel.sandbox import ExecutionManager, SandboxResult

from team.models import Job

if TYPE_CHECKING:
    from team.state import TeamCall

_log = logging.getLogger("software_team.jobs")


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
    # The process step each kind of job is, checked again when the job starts (team/rules.py).
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
            _log.error(f"Job {job_id} ({kind}) failed: {reason}")
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
            _log.warning(f"Could not copy the workspace as one archive ({e}); copying it file by file")
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
