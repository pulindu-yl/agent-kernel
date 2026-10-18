"""Background jobs for live realtime calls (``RealtimeCall.run_in_background``).

A voice agent starts long work with a short tool and keeps talking; the work runs here, off the realtime pool's
loop (which paces every call's audio), and its result is said into the call at a quiet moment. A job asks agents in
an AK session of its own, never the call's, whose lock every voice tool call takes.
"""

import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Optional, Sequence

from ...core.model import AgentReply, AgentRequest, AgentRequestText
from ...core.service import AgentService
from ..thread_runner import ThreadRunner

if TYPE_CHECKING:
    from .pool import RealtimeCall

_log = logging.getLogger("ak.pipeline.realtime.jobs")

JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]


class BackgroundJobRefused(Exception):
    """A job was not started; its text says why, and is for the model (a tool returns it as its result)."""


class BackgroundJobCancelled(Exception):
    """The job was cancelled: the call ended (``RealtimeCall.close()``) or the host began shutting down before its work
    started, the host cancelled it at shutdown, or its work raised ``asyncio.CancelledError``. Given to the job's
    ``on_error``; its text is posted only in the last case, the call still being on."""


class BackgroundJob:
    """One piece of background work for a live call: what it is, how far it got, and what it can do.

    ``work`` receives the job: it asks agents with :meth:`ask` and says things into the call with :meth:`post`. Made
    by ``RealtimeCall.run_in_background``, never directly.
    """

    # A job's AK session is the call's session, this, and the job's id.
    SESSION_INFIX = "-job-"

    def __init__(
        self,
        runner: "BackgroundJobRunner",
        session_id: str,
        job_id: str,
        name: str,
        arguments: dict[str, Any],
        work: Callable[["BackgroundJob"], Awaitable[Optional[str]]],
        on_error: Optional[Callable[["BackgroundJob", Exception], Optional[str]]],
        wait_until_heard: bool,
    ):
        self._runner = runner
        self.session_id = session_id
        self.id = job_id
        self.name = name
        self.arguments = arguments
        self.status: JobStatus = "queued"
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        # What ``work`` raised, for a failed job.
        self.error: Optional[Exception] = None
        self._work = work
        self._on_error = on_error
        self._wait_until_heard = wait_until_heard
        # Set, under the runner's lock, when the call ended: what the job posts afterwards is for a call that is over.
        self._closed = False

    @property
    def job_session_id(self) -> str:
        """The AK session the job's agents run in: ``<call session>-job-<job id>``. A later job with the same id
        continues it."""
        return f"{self.session_id}{self.SESSION_INFIX}{self.id}"

    @classmethod
    def call_session_id(cls, session_id: str) -> str:
        """The call's session for a job's session (``<call>-job-<id>``), so a tool a background agent calls finds its
        call; any other session id is returned as it is."""
        call, infix, _ = session_id.partition(cls.SESSION_INFIX)
        return call if infix else session_id

    async def ask(self, agent: str, prompt: str | Sequence[AgentRequest]) -> AgentReply:
        """Run ``agent`` in the job's own session and return its reply, as the runtime returns it.

        The job's asks share the session, so a second ask continues the first's conversation. A framework error
        comes back as error text, as on every other surface: detect a failed run through the framework's own signal
        during the run (such as the OpenAI Agents SDK's ``RunHooks.on_agent_end``), and raise from ``work`` to fail
        the job.

        :param agent: A registered agent's name.
        :param prompt: The text to send, or the requests as they are.
        :raises ValueError: No agent named ``agent`` is loaded.
        """
        AgentService.ensure_agent_available(agent)
        service = AgentService()
        service.select(self.job_session_id, agent)
        requests = [AgentRequestText(prompt=prompt)] if isinstance(prompt, str) else list(prompt)
        return await service.run_multi(requests)

    def post(self, text: str, *, key: Optional[str] = None, expires_after: float = 600.0) -> bool:
        """Say ``text`` into the job's call at its next quiet moment (``RealtimeCall.post_when_idle``).

        Dropped once the call has ended (``RealtimeCall.close()``) or the host is shutting down, so a job finishing
        late never speaks into the session's next call. Returns whether the call is connected and the text queued.
        """
        return self._runner.post(self, text, key=key, expires_after=expires_after)

    def __repr__(self) -> str:
        return f"BackgroundJob(id={self.id!r}, name={self.name!r}, session_id={self.session_id!r}, status={self.status!r})"


class _SessionJobs:
    """One session's jobs: the queue, the one running, and whether a worker is serving them."""

    def __init__(self) -> None:
        self.queued: list[BackgroundJob] = []
        self.running: Optional[BackgroundJob] = None
        self.serving = False


class BackgroundJobRunner:
    """Runs every live call's background jobs, on one event loop of its own, in this process.

    The realtime host runs it as it runs the pool: ``initialize()`` makes it, and ``get_task()`` is the
    ``ThreadRunner`` task whose ``start()`` runs the loop on a thread of its own and supervises it until shutdown.
    Never the pool's loop, which paces every call's audio, and never closed while the host runs, as a model client may
    keep tasks on the loop of its first call. A session's jobs run one at a time, in the order they were submitted;
    other sessions' run beside them. A job must not block the loop: blocking work goes to ``asyncio.to_thread``.

    Thread-safe; reached through ``RealtimeCall``. The ``RealtimeConnectionPool`` singleton pattern.
    """

    # A job that waits to be heard starts once the call has been quiet this long, or after the timeout.
    HEARD_SECONDS = 1.0
    HEARD_TIMEOUT_SECONDS = 20.0
    HEARD_POLL_SECONDS = 0.2
    # Cleanup budgets at host shutdown: running work may finish within the first, then is cancelled and may clean up
    # within the second. Not a guaranteed shutdown time: Python cannot stop application code that blocks the loop.
    SHUTDOWN_GRACE_SECONDS = 5.0
    CANCEL_CLEANUP_SECONDS = 5.0
    # How long the pool waits, on a consumer thread, for the job loop before a call's first connection opens.
    READY_TIMEOUT_SECONDS = 10.0
    _SUPERVISE_SECONDS = 0.5
    _SHUTDOWN_POLL_SECONDS = 0.05
    _LOOP_JOIN_SECONDS = 1.0
    _TERMINAL = ("done", "failed", "cancelled")

    _instance: Optional["BackgroundJobRunner"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def initialize(cls) -> "BackgroundJobRunner":
        """The process's runner, made on the first call; it starts nothing until its host runs ``start()``."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def current(cls) -> Optional["BackgroundJobRunner"]:
        """The process's runner, or ``None`` when no realtime host made one."""
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Test isolation: stop the runner (the production shutdown) and forget it."""
        with cls._instance_lock:
            runner, cls._instance = cls._instance, None
        if runner is not None:
            runner.stop(timeout=runner.SHUTDOWN_GRACE_SECONDS + runner.CANCEL_CLEANUP_SECONDS + 5.0)

    @classmethod
    def _forget(cls, runner: "BackgroundJobRunner") -> None:
        with cls._instance_lock:
            if cls._instance is runner:
                cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionJobs] = {}
        # Closed, under the lock, once shutdown begins: no job is submitted and no worker restarted after that.
        self._accepting = True
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._started = threading.Event()
        self._finished = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    # Hosting

    def get_task(self) -> ThreadRunner.Task:
        """The ``ThreadRunner.Task`` that runs this runner, beside the realtime pool's."""
        return ThreadRunner.Task(execution_function=self.start, thread_name="background-jobs", stop_all_on_failure=True, graceful=True)

    def start(self) -> None:
        """Blocking: run the job loop on a thread of its own, and supervise it until shutdown or :meth:`stop`.

        The loop's thread dying is raised, so the host's ``stop_all_on_failure`` sees it. Shutdown is driven from
        here, outside the loop, with bounded waits (:meth:`_shutdown`).

        :raises RuntimeError: The runner was started before, or its loop stopped unexpectedly.
        """
        with self._lock:
            if self._started.is_set():
                raise RuntimeError("The background job runner was already started")
            self._started.set()
        try:
            if self._stop_requested.is_set():
                return
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._run_loop, name="ak-background-jobs", daemon=True)
            self._loop_thread.start()
            while not self._stop_requested.wait(self._SUPERVISE_SECONDS):
                if ThreadRunner.shutdown_event.is_set():
                    break
                if not self._loop_thread.is_alive():
                    raise RuntimeError("The background job loop stopped unexpectedly")
            self._shutdown()
        finally:
            with self._lock:
                self._accepting = False
            self._finished.set()

    def stop(self, timeout: Optional[float] = None) -> bool:
        """Shut the runner down; callable from any thread, before or after :meth:`start`.

        :param timeout: Seconds to wait for ``start()`` to return; ``None`` waits as long as it takes.
        :return: Whether ``start()`` has returned, or never ran.
        """
        self._stop_requested.set()
        with self._lock:
            if not self._started.is_set():
                self._accepting = False
                return True
        return self._finished.wait(timeout)

    @property
    def ready(self) -> bool:
        """Whether the job loop runs and takes jobs."""
        return self._ready.is_set() and self._accepting

    def wait_ready(self, timeout: float) -> bool:
        """Wait for the job loop to run, at most ``timeout`` seconds; for a host's startup, never on the pool's loop."""
        return self._ready.wait(timeout)

    def _run_loop(self) -> None:
        loop = self._loop
        asyncio.set_event_loop(loop)
        loop.call_soon(self._ready.set)
        try:
            loop.run_forever()
        finally:
            self._ready.clear()
            loop.close()

    # Jobs

    def submit(
        self,
        session_id: str,
        work: Callable[[BackgroundJob], Awaitable[Optional[str]]],
        *,
        name: str,
        job_id: Optional[str] = None,
        arguments: Optional[dict[str, Any]] = None,
        wait_until_heard: bool = False,
        on_error: Optional[Callable[[BackgroundJob, Exception], Optional[str]]] = None,
        refuse: Optional[Callable[[list[BackgroundJob]], Optional[str]]] = None,
    ) -> BackgroundJob:
        """Queue a job for the session's call; see ``RealtimeCall.run_in_background``. Never waits.

        :raises RuntimeError: The runner is not running yet, or is shutting down.
        :raises BackgroundJobRefused: ``refuse`` returned a reason.
        :raises ValueError: A queued or running job of the session has ``job_id``.
        """
        with self._lock:
            if not self._accepting:
                raise RuntimeError("The background job runner is shutting down: no new background job is started")
            if not self._ready.is_set():
                raise RuntimeError("The background job runner is not running yet: the realtime host starts it, with the pool")
            jobs = self._sessions.setdefault(session_id, _SessionJobs())
            pending = self._pending(jobs)
            # Under the lock, so the check and the submission it allows cannot race another submission.
            reason = refuse(pending) if refuse is not None else None
            if reason:
                raise BackgroundJobRefused(reason)
            job_id = job_id or f"{name}-{uuid.uuid4().hex[:8]}"
            if any(job.id == job_id for job in pending):
                raise ValueError(f"Session {session_id} already has a queued or running job {job_id}")
            job = BackgroundJob(self, session_id, job_id, name, dict(arguments or {}), work, on_error, wait_until_heard)
            jobs.queued.append(job)
            start = not jobs.serving
            jobs.serving = True
            if start:
                self._loop.call_soon_threadsafe(self._start_worker, session_id)
        _log.info(f"Queued background job {job.id} ({name}) for session {session_id}")
        return job

    def jobs(self, session_id: str) -> list[BackgroundJob]:
        """The session's queued and running jobs, the running one first."""
        with self._lock:
            jobs = self._sessions.get(session_id)
            return self._pending(jobs) if jobs is not None else []

    def post(self, job: BackgroundJob, text: str, *, key: Optional[str] = None, expires_after: float = 600.0) -> bool:
        """Say ``text`` into ``job``'s call, unless the call ended or the host is shutting down.

        The check and the queueing are one step under the runner's lock, which ``close()`` and shutdown take to
        suppress posts, so a post never reaches a closed call's outbox (``RealtimeCall.close()`` drops it after).
        """
        from .pool import RealtimeCall

        with self._lock:
            if job._closed:
                _log.info(f"Background job {job.id} of session {job.session_id} posted after its call ended; dropped: {text[:80]}")
                return False
            return RealtimeCall(job.session_id).post_when_idle(text, key=key, expires_after=expires_after)

    def close(self, session_id: str) -> None:
        """The session's call ended: the jobs whose work has not started are cancelled and never run (one still
        waiting to be heard included); the running one finishes, but what it posts is dropped. The next submission
        starts afresh."""
        with self._lock:
            jobs = self._sessions.get(session_id)
            if jobs is None:
                return
            cancelled = self._close_locked(jobs)
            if not jobs.serving:
                del self._sessions[session_id]
        for job in cancelled:
            _log.info(f"Background job {job.id} of session {session_id} cancelled: the call ended")
            self._report_error(job, BackgroundJobCancelled("The call ended before the job ran"), post=False)

    def _close_locked(self, jobs: _SessionJobs) -> list[BackgroundJob]:
        """Suppress the session's posts and end the jobs whose work has not started, with the lock held; returns them."""
        queued, jobs.queued = jobs.queued, []
        cancelled = []
        if jobs.running is not None:
            jobs.running._closed = True
            if jobs.running.status == "queued" and self._end_locked(jobs.running, "cancelled"):
                cancelled.append(jobs.running)  # its worker is still waiting for it to be heard
        for job in queued:
            job._closed = True
            if self._end_locked(job, "cancelled"):
                cancelled.append(job)
        return cancelled

    def _end(self, job: BackgroundJob, status: JobStatus, error: Optional[Exception] = None) -> bool:
        with self._lock:
            return self._end_locked(job, status, error)

    def _end_locked(self, job: BackgroundJob, status: JobStatus, error: Optional[Exception] = None) -> bool:
        """End ``job`` once, with the lock held: its terminal status, error and end time; ``False`` when it had ended."""
        if job.status in self._TERMINAL:
            return False
        job.status = status
        job.error = error
        job.finished_at = time.time()
        return True

    @staticmethod
    def _pending(jobs: _SessionJobs) -> list[BackgroundJob]:
        """The queued and running jobs, not one ``close()`` cancelled while its worker waited for it to be heard."""
        running = [jobs.running] if jobs.running is not None and jobs.running.status != "cancelled" else []
        return running + list(jobs.queued)

    def _start_worker(self, session_id: str) -> None:
        """Start the session's worker; on the job loop, so its coroutine is made only when it can run."""
        self._loop.create_task(self._serve(session_id))

    async def _serve(self, session_id: str) -> None:
        """Run the session's jobs one at a time until none is queued.

        Whatever ends it, the session's state is left as a later submission expects: a worker that is cancelled
        (shutdown) or fails clears the job it held and, while the runner takes jobs, hands the queue to a new worker.
        """
        jobs: Optional[_SessionJobs] = None
        try:
            while True:
                with self._lock:
                    jobs = self._sessions.get(session_id)
                    if jobs is None:
                        return
                    if not jobs.queued:
                        # Under the lock a submission checks: one made after this finds no session, and starts a worker.
                        del self._sessions[session_id]
                        jobs = None
                        return
                    job = jobs.queued.pop(0)
                    jobs.running = job
                await self._run(job)
                with self._lock:
                    jobs.running = None
        finally:
            if jobs is not None:
                self._recover_worker(session_id, jobs)

    def _recover_worker(self, session_id: str, jobs: _SessionJobs) -> None:
        """A worker ended with its session still served (it was cancelled, or failed): free the session for the next one."""
        with self._lock:
            if self._sessions.get(session_id) is not jobs:
                return
            jobs.running = None
            if not jobs.queued:
                del self._sessions[session_id]
                return
            restart = self._accepting and self._loop is not None and not self._loop.is_closed()
            jobs.serving = restart
        _log.warning(f"The background job worker of session {session_id} stopped early; {len(jobs.queued)} jobs still queued")
        if restart:
            self._loop.call_soon_threadsafe(self._start_worker, session_id)

    async def _run(self, job: BackgroundJob) -> None:
        if job._wait_until_heard:
            try:
                await self._until_heard(job)
            except asyncio.CancelledError:
                # The worker was cancelled before the job's work began: the job ends here, and _serve repairs the
                # session's state. Unless close() or shutdown ended it already, its on_error hears why.
                if self._end(job, "cancelled"):
                    self._report_error(job, BackgroundJobCancelled("The job was cancelled before its work began"), post=False)
                raise
        with self._lock:
            # The call may have ended, or the host begun shutting down, while the job waited: then it is cancelled.
            if job.status != "queued":
                return
            job.status = "running"
            job.started_at = time.time()
        try:
            text = await job._work(job)
        except asyncio.CancelledError:
            ended = self._end(job, "cancelled")
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                # The worker itself is being cancelled (shutdown): _serve repairs the session's state.
                if ended:
                    self._report_error(job, BackgroundJobCancelled("The job was cancelled while it ran"), post=False)
                raise
            # The work raised it itself (a cancelled call inside it, say): the job ends, and the queue goes on.
            _log.warning(f"Background job {job.id} ({job.name}) of session {job.session_id} was cancelled by its work")
            if ended:
                self._report_error(job, BackgroundJobCancelled("The job's work was cancelled"), post=True)
            return
        except Exception as e:
            _log.exception(f"Background job {job.id} ({job.name}) of session {job.session_id} failed")
            if self._end(job, "failed", e):
                self._report_error(job, e, post=True)
            return
        if self._end(job, "done"):
            _log.info(f"Background job {job.id} ({job.name}) of session {job.session_id} done")
            if text:
                self._post(job, text)

    @classmethod
    def _report_error(cls, job: BackgroundJob, error: Exception, post: bool) -> None:
        """Give ``on_error`` the job's end, outside every lock; post its text when asked and the call is still on."""
        if job._on_error is None:
            return
        try:
            text = job._on_error(job, error)
        except Exception:
            _log.exception(f"on_error of background job {job.id} ({job.name}) raised")
            return
        if text and post:
            cls._post(job, text)

    @staticmethod
    def _post(job: BackgroundJob, text: str) -> None:
        try:
            job.post(text)
        except Exception:
            _log.exception(f"Background job {job.id} ({job.name}) of session {job.session_id} could not post its result")

    async def _until_heard(self, job: BackgroundJob) -> None:
        """Wait until what the agent said before starting the job has been heard: the call has been quiet for a
        moment. At most ``HEARD_TIMEOUT_SECONDS``; at once without a live call, and no longer once the call ended."""
        call = self._call(job.session_id)
        deadline = time.monotonic() + self.HEARD_TIMEOUT_SECONDS
        while call.connected and not job._closed and time.monotonic() < deadline:
            quiet = call.quiet_for()
            if quiet is not None and quiet >= self.HEARD_SECONDS:
                return
            await asyncio.sleep(self.HEARD_POLL_SECONDS)

    @staticmethod
    def _call(session_id: str) -> "RealtimeCall":
        from .pool import RealtimeCall

        return RealtimeCall(session_id)

    # Shutdown

    def _shutdown(self) -> None:
        """Shut down, on the host's thread, with bounded waits; see the class's spec for the order."""
        with self._lock:
            self._accepting = False
            cancelled = [job for jobs in self._sessions.values() for job in self._close_locked(jobs)]
        for job in cancelled:
            _log.info(f"Background job {job.id} of session {job.session_id} cancelled: the host is shutting down")
            self._report_error(job, BackgroundJobCancelled("The host shut down before the job ran"), post=False)
        self._until_no_job_runs(self.SHUTDOWN_GRACE_SECONDS)
        # Always, not only when a job still runs: a cancelled job's worker may still be waiting, and finished work may
        # have left tasks of its own (an SDK's listener) on the loop. Job status alone does not prove the loop clean.
        unfinished = self._cancel_remaining_tasks()
        left = self._running()
        if left:
            names = ", ".join(f"{job.id} ({job.name}) of session {job.session_id}" for job in left)
            _log.error(f"Background jobs still running after the shutdown's cleanup budget, abandoned: {names}")
        if unfinished:
            _log.error(f"{unfinished} tasks on the background job loop did not finish their cleanup within {self.CANCEL_CLEANUP_SECONDS}s")
        self._call_on_loop(self._loop.stop if self._loop is not None else None)
        if self._loop_thread is not None:
            self._loop_thread.join(self._LOOP_JOIN_SECONDS)
            if self._loop_thread.is_alive():
                _log.error("The background job loop is blocked by application code; it stays until the process exits")

    def _running(self) -> list[BackgroundJob]:
        with self._lock:
            return [jobs.running for jobs in self._sessions.values() if jobs.running is not None and jobs.running.status == "running"]

    def _until_no_job_runs(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while self._running():
            if time.monotonic() >= deadline:
                return False
            time.sleep(self._SHUTDOWN_POLL_SECONDS)
        return True

    def _call_on_loop(self, callback: Optional[Callable[[], None]]) -> None:
        loop = self._loop
        if callback is None or loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(callback)
        except RuntimeError:  # closed meanwhile
            pass

    def _cancel_remaining_tasks(self) -> Optional[int]:
        """Cancel every task left on the loop and give them ``CANCEL_CLEANUP_SECONDS`` to end; from the host's thread.

        Returns how many did not end, or ``None`` when the loop never ran the cancellation (application code blocks
        it, which the join that follows reports): the host's wait is bounded either way.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return 0
        future: "concurrent.futures.Future[int]" = concurrent.futures.Future()

        def begin() -> None:
            # On the loop, only once it gets here: the coroutine is never made when the host stopped waiting first.
            if not future.set_running_or_notify_cancel():
                return
            loop.create_task(self._cancel_tasks()).add_done_callback(report)

        def report(done: "asyncio.Task[int]") -> None:
            if done.cancelled():
                future.set_exception(concurrent.futures.CancelledError())
            elif done.exception() is not None:
                future.set_exception(done.exception())
            else:
                future.set_result(done.result())

        try:
            loop.call_soon_threadsafe(begin)
        except RuntimeError:  # closed meanwhile
            return 0
        try:
            return future.result(timeout=self.CANCEL_CLEANUP_SECONDS + self._SHUTDOWN_POLL_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return None
        except (concurrent.futures.CancelledError, Exception):
            _log.exception("Cancelling the background job loop's tasks failed")
            return None

    async def _cancel_tasks(self) -> int:
        """On the loop: cancel every other task, and wait for them a bounded time; returns how many did not end."""
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current]
        for task in tasks:
            task.cancel()
        if not tasks:
            return 0
        _, pending = await asyncio.wait(tasks, timeout=self.CANCEL_CLEANUP_SECONDS)
        return len(pending)
