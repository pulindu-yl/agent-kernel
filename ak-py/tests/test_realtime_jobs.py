"""Background jobs for live calls (``RealtimeCall.run_in_background``): off the pool's loop, per session in order, in
sessions of their own, and ended with the call."""

import asyncio
import logging
import threading
import time

import pytest

from agentkernel import Agent, Runner
from agentkernel.core.model import AgentReplyText
from agentkernel.core.runtime import Runtime
from agentkernel.core.session.in_memory import InMemorySessionStore
from agentkernel.pipeline import BackgroundJob, BackgroundJobCancelled, BackgroundJobRefused, RealtimeCall
from agentkernel.pipeline.realtime.jobs import BackgroundJobRunner
from agentkernel.pipeline.realtime.pool import RealtimeConnectionPool
from agentkernel.pipeline.testing import BackgroundJobHost


@pytest.fixture(autouse=True)
def _runner_state(request, monkeypatch):
    """Every test runs in a job runner hosted as the realtime host would, but those about hosting itself."""
    monkeypatch.setattr(RealtimeConnectionPool, "_instance", None)
    if request.cls in (TestRealtimeCallJobs, TestBackgroundJobHosting):
        yield
        BackgroundJobRunner.reset()
        return
    with BackgroundJobHost():
        yield


@pytest.fixture
def posts(monkeypatch):
    """What jobs say into their calls, as (session, text, key)."""
    said = []

    def post_when_idle(self, text, *, key=None, expires_after=600.0, while_quiet=False):
        said.append((self.session_id, text, key))
        return True

    monkeypatch.setattr(RealtimeCall, "post_when_idle", post_when_idle)
    return said


def _wait(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.01)


class TestBackgroundJobRunner:
    def test_a_jobs_text_is_said_into_its_call(self, posts):
        async def work(job):
            return f"{job.id} finished: {job.arguments['question']}"

        job = RealtimeCall("s1").run_in_background(work, name="research", job_id="J1", arguments={"question": "GDPR?"})

        _wait(lambda: posts)
        assert posts == [("s1", "J1 finished: GDPR?", None)]
        assert (job.status, job.name) == ("done", "research")
        assert job.started_at is not None and job.finished_at >= job.started_at

    def test_a_sessions_jobs_run_one_at_a_time_in_order_and_other_sessions_beside_them(self, posts):
        order = []
        release = threading.Event()

        async def slow(job):
            order.append(("start", job.id))
            while not release.is_set():
                await asyncio.sleep(0.01)
            order.append(("end", job.id))

        async def quick(job):
            order.append(("run", job.id))

        call = RealtimeCall("s1")
        call.run_in_background(slow, name="a", job_id="A1")
        call.run_in_background(quick, name="b", job_id="A2")
        RealtimeCall("s2").run_in_background(quick, name="b", job_id="B1")

        _wait(lambda: ("run", "B1") in order)  # the other session did not wait for A1
        assert ("run", "A2") not in order
        assert [job.id for job in call.jobs] == ["A1", "A2"]
        release.set()
        _wait(lambda: ("run", "A2") in order)
        assert order.index(("end", "A1")) < order.index(("run", "A2"))
        _wait(lambda: call.jobs == [])

    def test_every_job_runs_on_one_loop_never_the_callers(self):
        seen = []

        async def work(job):
            seen.append((threading.current_thread().name, id(asyncio.get_running_loop())))

        for session in ("s1", "s2", "s3"):
            RealtimeCall(session).run_in_background(work, name="w")

        _wait(lambda: len(seen) == 3)
        assert {name for name, _ in seen} == {"ak-background-jobs"}
        assert len({loop for _, loop in seen}) == 1

    def test_a_failed_job_says_what_on_error_returns(self, posts, caplog):
        async def work(job):
            raise RuntimeError("search is down")

        errors = []

        def on_error(job, error):
            errors.append(error)
            return f"{job.id} could not run."

        with caplog.at_level(logging.ERROR, logger="ak.pipeline.realtime.jobs"):
            job = RealtimeCall("s1").run_in_background(work, name="research", job_id="J1", on_error=on_error)
            _wait(lambda: posts)

        assert posts == [("s1", "J1 could not run.", None)]
        assert job.status == "failed" and isinstance(job.error, RuntimeError) and errors == [job.error]
        assert "Background job J1 (research) of session s1 failed" in caplog.text

    def test_without_on_error_a_failure_is_only_logged_and_a_raising_on_error_is_logged(self, posts, caplog):
        async def work(job):
            raise RuntimeError("down")

        def on_error(job, error):
            raise ValueError("on_error broke")

        with caplog.at_level(logging.ERROR, logger="ak.pipeline.realtime.jobs"):
            quiet = RealtimeCall("s1").run_in_background(work, name="a")
            loud = RealtimeCall("s2").run_in_background(work, name="b", on_error=on_error)
            _wait(lambda: quiet.status == "failed" and loud.status == "failed")
            _wait(lambda: "on_error of background job" in caplog.text)

        assert posts == []

    def test_refuse_sees_the_pending_jobs_and_cannot_race_the_submission(self):
        release = threading.Event()

        async def work(job):
            while not release.is_set():
                await asyncio.sleep(0.01)

        def refuse(pending):
            time.sleep(0.05)  # widen the window two unguarded submissions would both pass through
            if any(job.arguments.get("account") == "A-1042" for job in pending):
                return "A booking for A-1042 is already under way."
            return None

        accepted, refused = [], []

        def submit():
            try:
                accepted.append(RealtimeCall("s1").run_in_background(work, name="booking", arguments={"account": "A-1042"}, refuse=refuse))
            except BackgroundJobRefused as e:
                refused.append(str(e))

        threads = [threading.Thread(target=submit) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        release.set()

        assert len(accepted) == 1 and refused == ["A booking for A-1042 is already under way."]

    def test_a_pending_job_id_cannot_be_reused(self):
        release = threading.Event()

        async def work(job):
            while not release.is_set():
                await asyncio.sleep(0.01)

        call = RealtimeCall("s1")
        call.run_in_background(work, name="w", job_id="J1")
        with pytest.raises(ValueError, match="already has a queued or running job J1"):
            call.run_in_background(work, name="w", job_id="J1")
        release.set()

    def test_ending_the_call_cancels_the_queued_jobs_and_silences_the_running_one(self, posts):
        started, release = threading.Event(), threading.Event()
        ran, cancelled = [], []

        async def running(job):
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            return "J1 finished."

        async def queued(job):
            ran.append(job.id)

        def on_error(job, error):
            cancelled.append((job.id, type(error)))
            return "never said"

        call = RealtimeCall("s1")
        first = call.run_in_background(running, name="w", job_id="J1")
        second = call.run_in_background(queued, name="w", job_id="J2", on_error=on_error)
        started.wait(2)

        call.close()
        release.set()
        _wait(lambda: first.status == "done")
        time.sleep(0.05)

        assert second.status == "cancelled" and ran == [] and cancelled == [("J2", BackgroundJobCancelled)]
        assert posts == []  # neither the running job's result nor the cancellation reached the call
        assert first.post("late") is False

        later = call.run_in_background(queued, name="w", job_id="J3")  # the session's next call starts afresh
        _wait(lambda: later.status == "done")
        assert ran == ["J3"]

    def test_ending_the_call_cancels_a_job_still_waiting_to_be_heard(self, monkeypatch):
        monkeypatch.setattr(RealtimeCall, "connected", property(lambda self: True))
        monkeypatch.setattr(RealtimeCall, "quiet_for", lambda self: None)  # never quiet: the job waits
        monkeypatch.setattr(BackgroundJobRunner, "HEARD_POLL_SECONDS", 0.01)
        ran, cancelled = [], []

        async def work(job):
            ran.append(job.id)

        call = RealtimeCall("s1")
        job = call.run_in_background(work, name="w", job_id="J1", wait_until_heard=True, on_error=lambda j, e: cancelled.append(type(e)))
        _wait(lambda: BackgroundJobRunner.current()._sessions["s1"].running is job)

        call.close()

        assert job.status == "cancelled" and cancelled == [BackgroundJobCancelled]
        assert call.jobs == []
        later = call.run_in_background(work, name="w", job_id="J2")  # the next call's job does not wait 20 s behind it
        _wait(lambda: later.status == "done", timeout=1.0)
        assert ran == ["J2"]

    def test_work_that_raises_cancelled_error_does_not_stall_the_calls_queue(self, posts):
        async def cancelled(job):
            raise asyncio.CancelledError()

        async def next_job(job):
            return "next done"

        reported = []
        call = RealtimeCall("s1")
        first = call.run_in_background(cancelled, name="w", on_error=lambda j, e: reported.append(type(e)) or "J1 was cancelled.")
        second = call.run_in_background(next_job, name="w")

        _wait(lambda: second.status == "done")
        assert first.status == "cancelled" and reported == [BackgroundJobCancelled]
        assert [text for _, text, _ in posts] == ["J1 was cancelled.", "next done"]
        assert call.jobs == []

    def test_a_cancelled_worker_hands_the_queue_to_a_new_one(self):
        started, ran = threading.Event(), []

        async def stuck(job):
            started.set()
            await asyncio.sleep(10)

        async def queued(job):
            ran.append(job.id)

        call = RealtimeCall("s1")
        first = call.run_in_background(stuck, name="w", job_id="J1")
        call.run_in_background(queued, name="w", job_id="J2")
        started.wait(2)
        runner = BackgroundJobRunner.current()
        for task in asyncio.run_coroutine_threadsafe(self._tasks(), runner._loop).result(2):
            runner._loop.call_soon_threadsafe(task.cancel)

        _wait(lambda: ran == ["J2"])
        assert first.status == "cancelled" and first.finished_at is not None and call.jobs == []

    def test_a_worker_cancelled_while_its_job_waits_to_be_heard_ends_that_job(self, monkeypatch, posts):
        monkeypatch.setattr(RealtimeCall, "connected", property(lambda self: True))
        monkeypatch.setattr(RealtimeCall, "quiet_for", lambda self: None)  # never quiet: the first job waits
        monkeypatch.setattr(BackgroundJobRunner, "HEARD_POLL_SECONDS", 0.01)
        ran, reported = [], []

        async def work(job):
            ran.append(job.id)

        call = RealtimeCall("s1")
        first = call.run_in_background(
            work, name="w", job_id="J1", wait_until_heard=True, on_error=lambda j, e: reported.append((j.id, type(e))) or "never said"
        )
        second = call.run_in_background(work, name="w", job_id="J2")
        runner = BackgroundJobRunner.current()
        _wait(lambda: runner._sessions["s1"].running is first)
        for task in asyncio.run_coroutine_threadsafe(self._tasks(), runner._loop).result(2):
            runner._loop.call_soon_threadsafe(task.cancel)

        _wait(lambda: second.status == "done")
        assert first.status == "cancelled" and first.finished_at is not None
        assert reported == [("J1", BackgroundJobCancelled)] and posts == []  # told, but nothing said
        assert ran == ["J2"] and call.jobs == []

    @staticmethod
    async def _tasks():
        return [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    def test_a_post_racing_the_calls_end_never_reaches_a_later_call(self, monkeypatch):
        """The check that the call is on and the queueing are one step: a close between them would let the post
        recreate the session's outbox for its next call."""
        pool = RealtimeConnectionPool()
        monkeypatch.setattr(RealtimeConnectionPool, "_instance", pool)
        entered, closed = threading.Event(), threading.Event()
        original = RealtimeCall.post_when_idle

        def slow_post(self, text, **kwargs):
            entered.set()
            time.sleep(0.1)  # a close() now must wait for the post to finish
            return original(self, text, **kwargs)

        monkeypatch.setattr(RealtimeCall, "post_when_idle", slow_post)

        async def work(job):
            job.post("late result")

        call = RealtimeCall("s1")
        job = call.run_in_background(work, name="w")
        entered.wait(2)
        call.close()
        closed.set()
        _wait(lambda: job.status == "done")

        assert "s1" not in pool._outboxes

    def test_a_job_that_waits_to_be_heard_starts_once_the_call_is_quiet(self, monkeypatch):
        quiet = [None, 0.4, 1.2]
        monkeypatch.setattr(RealtimeCall, "connected", property(lambda self: True))
        monkeypatch.setattr(RealtimeCall, "quiet_for", lambda self: quiet.pop(0) if len(quiet) > 1 else quiet[0])
        monkeypatch.setattr(BackgroundJobRunner, "HEARD_POLL_SECONDS", 0.01)
        started = []

        async def work(job):
            started.append(job.id)

        RealtimeCall("s1").run_in_background(work, name="w", job_id="J1", wait_until_heard=True)

        _wait(lambda: started)
        assert quiet == [1.2]

    def test_the_wait_to_be_heard_is_bounded(self, monkeypatch):
        monkeypatch.setattr(RealtimeCall, "connected", property(lambda self: True))
        monkeypatch.setattr(RealtimeCall, "quiet_for", lambda self: None)  # never quiet
        monkeypatch.setattr(BackgroundJobRunner, "HEARD_TIMEOUT_SECONDS", 0.1)
        monkeypatch.setattr(BackgroundJobRunner, "HEARD_POLL_SECONDS", 0.01)

        async def work(job):
            return None

        job = RealtimeCall("s1").run_in_background(work, name="w", wait_until_heard=True)

        _wait(lambda: job.status == "done")

    def test_without_a_live_call_a_job_does_not_wait_to_be_heard(self):
        async def work(job):
            return None

        job = RealtimeCall("s1").run_in_background(work, name="w", wait_until_heard=True)

        _wait(lambda: job.status == "done", timeout=1.0)


class _Runner(Runner):
    def __init__(self):
        super().__init__("test-runner")
        self.sessions = []

    async def run(self, agent, session, requests):
        self.sessions.append(session.id)
        return AgentReplyText(response=f"{agent.name} answered {requests[0].prompt}")

    async def stream(self, agent, session, requests):
        raise NotImplementedError()
        yield


class _Agent(Agent):
    def __init__(self, name, runner):
        super().__init__(name, runner)

    def get_description(self):
        return "fake"

    def get_a2a_card(self):
        return None

    def attach_tool(self, tool):
        pass

    def override_system_prompt(self, prompt):
        pass


class TestBackgroundJobAsk:
    @pytest.fixture
    def researcher(self):
        runner = _Runner()
        with Runtime(InMemorySessionStore()) as runtime:
            runtime.register(_Agent("researcher", runner))
            yield runner

    def test_a_jobs_asks_run_in_its_own_session_and_share_it(self, researcher, posts):
        async def work(job):
            first = await job.ask("researcher", "GDPR?")
            again = await job.ask("researcher", "and in the UK?")
            return f"{first} / {again}"

        RealtimeCall("call-1").run_in_background(work, name="research", job_id="J1")

        _wait(lambda: posts)
        assert posts == [("call-1", "researcher answered GDPR? / researcher answered and in the UK?", None)]
        assert researcher.sessions == ["call-1-job-J1", "call-1-job-J1"]

    def test_asking_an_agent_that_is_not_loaded_fails_the_job(self, researcher):
        async def work(job):
            return str(await job.ask("nobody", "hello"))

        job = RealtimeCall("call-1").run_in_background(work, name="research")

        _wait(lambda: job.status == "failed")
        assert isinstance(job.error, ValueError)

    def test_the_call_is_found_from_a_jobs_session(self):
        assert BackgroundJob.call_session_id("call-1-job-J1") == "call-1"
        assert BackgroundJob.call_session_id("call-1") == "call-1"


class TestRealtimeCallJobs:
    def test_a_call_without_jobs_has_none_and_closes_without_a_runner(self):
        call = RealtimeCall("s1")

        assert call.jobs == []
        assert call.close() is False
        assert BackgroundJobRunner.current() is None  # nothing started a job loop


class TestBackgroundJobHosting:
    """The realtime host runs the job runner: it starts it, readies it before any call, and shuts it down in order."""

    @staticmethod
    def _host():
        """A runner started as a host starts it, on a thread; returns the runner and that thread."""
        runner = BackgroundJobRunner.initialize()
        thread = threading.Thread(target=runner.start, daemon=True)
        thread.start()
        assert runner.wait_ready(5)
        return runner, thread

    def test_without_a_runner_a_submission_says_who_hosts_it(self):
        async def work(job):
            return None

        with pytest.raises(RuntimeError, match="realtime host.*BackgroundJobHost"):
            RealtimeCall("s1").run_in_background(work, name="w")

    def test_a_runner_not_yet_running_refuses_at_once(self):
        BackgroundJobRunner.initialize()

        async def work(job):
            return None

        started = time.monotonic()
        with pytest.raises(RuntimeError, match="not running yet"):
            RealtimeCall("s1").run_in_background(work, name="w")
        assert time.monotonic() - started < 0.5  # a submission never waits: it may be on the pool's loop

    def test_initializing_twice_is_one_runner_and_starts_no_thread(self):
        before = {t.name for t in threading.enumerate()}
        runner = BackgroundJobRunner.initialize()

        assert BackgroundJobRunner.initialize() is runner
        assert "ak-background-jobs" not in {t.name for t in threading.enumerate()} - before

    def test_both_realtime_hosts_run_the_job_runner_beside_the_pool(self, monkeypatch):
        from agentkernel.deployment.aws.containerized.akagentrunner import ECSRealtimeAgentRunner
        from agentkernel.pipeline.agent_runner import RealtimeAgentRunner

        monkeypatch.setattr(RealtimeConnectionPool, "_instance", None)
        pipeline = RealtimeAgentRunner._get_extra_tasks(object.__new__(RealtimeAgentRunner))
        ecs = ECSRealtimeAgentRunner._get_extra_tasks()

        for tasks in (pipeline, ecs):
            assert [task.thread_name for task in tasks] == ["realtime-pool", "background-jobs"]
            assert tasks[1].execution_function == BackgroundJobRunner.current().start
            assert tasks[1].graceful and tasks[1].stop_all_on_failure

    def test_a_runner_starts_once(self):
        runner, thread = self._host()

        with pytest.raises(RuntimeError, match="already started"):
            runner.start()
        runner.stop(5)
        thread.join(5)

    def test_shutdown_cancels_waiting_work_lets_running_work_finish_and_says_nothing(self, monkeypatch, posts):
        monkeypatch.setattr(RealtimeCall, "connected", property(lambda self: True))
        monkeypatch.setattr(RealtimeCall, "quiet_for", lambda self: None)  # never quiet: a job waiting to be heard waits
        monkeypatch.setattr(BackgroundJobRunner, "HEARD_POLL_SECONDS", 0.01)
        runner, thread = self._host()
        started, ended = threading.Event(), []

        async def running(job):
            started.set()
            await asyncio.sleep(0.2)
            return "J1 finished."  # within the grace: it finishes, but the call is closing

        async def never(job):
            ended.append(job.id)

        reported = []
        on_error = lambda job, error: reported.append((job.id, type(error))) or "never said"  # noqa: E731
        first = RealtimeCall("s1").run_in_background(running, name="w", job_id="J1", on_error=on_error)
        queued = RealtimeCall("s1").run_in_background(never, name="w", job_id="J2", on_error=on_error)
        waiting = RealtimeCall("s2").run_in_background(never, name="w", job_id="K1", wait_until_heard=True, on_error=on_error)
        started.wait(2)

        assert runner.stop(timeout=10)
        thread.join(5)

        assert first.status == "done" and posts == []
        assert (queued.status, waiting.status) == ("cancelled", "cancelled") and ended == []
        assert sorted(reported) == [("J2", BackgroundJobCancelled), ("K1", BackgroundJobCancelled)]
        assert not runner.ready
        with pytest.raises(RuntimeError, match="shutting down"):
            RealtimeCall("s1").run_in_background(never, name="w")
        assert not any(t.name == "ak-background-jobs" for t in threading.enumerate())

    def test_a_submission_racing_shutdown_is_cancelled_or_refused_never_left(self):
        runner, thread = self._host()

        async def work(job):
            return None

        outcomes = []

        def submit():
            for _ in range(50):
                try:
                    outcomes.append(RealtimeCall("s1").run_in_background(work, name="w"))
                except RuntimeError:
                    outcomes.append("refused")

        submitter = threading.Thread(target=submit)
        submitter.start()
        runner.stop(timeout=10)
        submitter.join(5)
        thread.join(5)

        assert "refused" in outcomes or outcomes  # some were refused, or all landed before
        assert all(job == "refused" or job.status in ("done", "cancelled") for job in outcomes)

    def test_shutdown_with_only_a_waiting_job_leaves_no_task_on_the_loop(self, monkeypatch):
        monkeypatch.setattr(RealtimeCall, "connected", property(lambda self: True))
        monkeypatch.setattr(RealtimeCall, "quiet_for", lambda self: None)  # never quiet: the job waits to be heard
        monkeypatch.setattr(BackgroundJobRunner, "HEARD_POLL_SECONDS", 0.01)
        runner, thread = self._host()

        async def work(job):
            return None

        job = RealtimeCall("s1").run_in_background(work, name="w", wait_until_heard=True)
        _wait(lambda: runner._sessions["s1"].running is job)

        assert runner.stop(timeout=10)
        thread.join(5)

        assert job.status == "cancelled"
        assert not asyncio.all_tasks(runner._loop)  # its worker was cancelled and ended, not destroyed with the loop

    def test_shutdown_cancels_what_finished_work_left_on_the_loop(self):
        """An SDK may leave a listener task behind its call; job status alone does not show it."""
        runner, thread = self._host()
        cleaned = threading.Event()

        async def listener():
            try:
                await asyncio.sleep(100)
            finally:
                cleaned.set()

        async def work(job):
            asyncio.get_running_loop().create_task(listener())
            await asyncio.sleep(0)

        job = RealtimeCall("s1").run_in_background(work, name="w")
        _wait(lambda: job.status == "done")

        assert runner.stop(timeout=10)
        thread.join(5)

        assert cleaned.is_set()  # its cancellation cleanup ran before the loop closed
        assert not asyncio.all_tasks(runner._loop)

    def test_no_worker_is_restarted_once_shutdown_began(self, monkeypatch):
        from agentkernel.pipeline.realtime.jobs import _SessionJobs

        runner, thread = self._host()
        restarted = []
        monkeypatch.setattr(runner, "_start_worker", lambda session_id: restarted.append(session_id))
        jobs = _SessionJobs()

        async def work(job):
            return None

        jobs.queued.append(BackgroundJob(runner, "s1", "J1", "w", {}, work, None, False))
        runner._sessions["s1"] = jobs
        with runner._lock:
            runner._accepting = False  # what shutdown does first

        runner._recover_worker("s1", jobs)
        time.sleep(0.05)

        assert restarted == []
        runner._sessions.clear()
        runner.stop(5)
        thread.join(5)

    def test_work_blocking_the_loop_is_reported_and_abandoned_within_the_budgets(self, monkeypatch, caplog):
        """The budgets bound the host's wait, not the work: blocked work stays until released (here) or the process exits."""
        monkeypatch.setattr(BackgroundJobRunner, "SHUTDOWN_GRACE_SECONDS", 0.2)
        monkeypatch.setattr(BackgroundJobRunner, "CANCEL_CLEANUP_SECONDS", 0.2)
        monkeypatch.setattr(BackgroundJobRunner, "_LOOP_JOIN_SECONDS", 0.2)
        runner, thread = self._host()
        blocking, release = threading.Event(), threading.Event()

        async def blocks(job):
            blocking.set()
            release.wait(10)  # blocks the job loop, as careless application code would

        RealtimeCall("s1").run_in_background(blocks, name="w", job_id="J1")
        blocking.wait(2)
        began = time.monotonic()
        with caplog.at_level(logging.ERROR, logger="ak.pipeline.realtime.jobs"):
            assert runner.stop(timeout=5)
        took = time.monotonic() - began

        assert took < 2.0
        assert "abandoned: J1 (w) of session s1" in caplog.text
        assert "blocked by application code" in caplog.text
        release.set()  # the double cooperates now: the loop ends, and no test thread is left behind
        loop_thread = runner._loop_thread
        loop_thread.join(5)
        thread.join(5)
        assert not loop_thread.is_alive() and not thread.is_alive()

    def test_a_readiness_timeout_is_not_remembered_as_ready(self, monkeypatch, caplog):
        monkeypatch.setattr(BackgroundJobRunner, "READY_TIMEOUT_SECONDS", 0.05)
        BackgroundJobRunner.initialize()  # made, never started
        pool = RealtimeConnectionPool()

        with caplog.at_level(logging.WARNING, logger="ak.pipeline.realtime.pool"):
            pool._wait_for_jobs()
            pool._wait_for_jobs()  # the next connection waits again

        assert caplog.text.count("The background job runner is not running") == 2

    def test_once_ready_no_connection_waits_again(self, monkeypatch):
        runner, thread = self._host()
        pool = RealtimeConnectionPool()
        pool._wait_for_jobs()
        monkeypatch.setattr(runner, "wait_ready", lambda timeout: pytest.fail("waited again"))

        pool._wait_for_jobs()

        runner.stop(5)
        thread.join(5)

    def test_concurrent_first_connections_both_wait_for_readiness(self, monkeypatch):
        runner = BackgroundJobRunner.initialize()
        pool = RealtimeConnectionPool()
        entered, release, second_done = threading.Event(), threading.Event(), threading.Event()

        def wait_ready(timeout):
            entered.set()
            return release.wait(5)

        monkeypatch.setattr(runner, "wait_ready", wait_ready)
        first = threading.Thread(target=pool._wait_for_jobs)
        second = threading.Thread(target=lambda: (pool._wait_for_jobs(), second_done.set()))
        first.start()
        assert entered.wait(2)
        second.start()

        assert not second_done.wait(0.1)  # the second connection waits for the first's outcome
        release.set()
        first.join(5)
        second.join(5)
        assert second_done.is_set() and pool._jobs_ready
