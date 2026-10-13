"""Unit tests for the team's coordination code; no live model, LiveKit room or Docker needed."""

import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from agentkernel.core.base import Session

import agents as prompts
from agents import BackgroundTeam, VoiceTeam
from project import AcceptanceResult, ProjectDocuments, ProjectStore
from team import (
    ActivityFeed,
    CallLink,
    DevJobRunner,
    HandoffGuard,
    JobRefused,
    ModelWaitWatcher,
    ProcessGates,
    TeamCall,
    TeamCalls,
)
from tools import TeamTools


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeConnection:
    """The turn state a ``RealtimeConnection`` exposes, set by hand."""

    def __init__(self):
        self.closed = False
        self.is_responding = False
        self.is_caller_speaking = False
        self.last_model_output_at = None
        self.last_model_generated_at = None
        self.last_agent_change_at = None
        self.last_tool_result_at = None
        self.last_caller_speech_at = None
        self.last_edge_text_at = None
        self.sent: list[str] = []

    def send_text(self, text: str) -> None:
        self.sent.append(text)


class TestProjectStore:
    def test_records_made_before_the_project_is_named_are_kept_and_moved(self, tmp_path):
        """Live: the Worker handed over before start_project, and the PM's requirements must not be lost."""
        store = ProjectStore(tmp_path, "room_01")
        assert list(tmp_path.iterdir()) == []

        store.add_requirement("Sign up", "Users sign up with email.", ["A user can sign up"], "pm")
        store.flush()
        assert "Sign up" in (tmp_path / "room-01" / "requirements.md").read_text()
        assert store.started is False

        folder = store.start("Attendance system", "Track arrivals and leaves.", "worker")
        store.flush()

        assert [p.name for p in tmp_path.iterdir()] == ["attendance-system"]
        assert "Sign up" in (folder / "requirements.md").read_text()
        assert store.started is True

    def test_every_document_is_rendered_from_the_records(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        folder = store.start("Login system", "Let customers log in to the web app.", "worker")
        store.add_requirement(
            "Sign up", "Users sign up with email.", ["A user can sign up", "An email is used once"], "pm"
        )
        store.add_assumption("Accounts lock for 15 minutes after 5 failed logins.", "pm")
        store.add_assumption("A reset link is valid for 30 minutes.", "coder", needs_settling=True)
        store.add_question("Does the reset email need branding?", "pm")
        store.add_decision(
            "Reset tokens",
            "Random single-use token, stored hashed",
            "Guessable links leak accounts",
            ["User id in the link: guessable"],
            "architect",
            challenge="A token in a log leaks an account: tokens are stored hashed and used once",
        )
        store.add_component("Auth service", "Sign-up, login and password reset", "architect")
        store.add_tasks(["Sign-up and login: R1", "Password reset"], "worker")
        store.flush()

        assert folder == tmp_path / "login-system"
        assert sorted(p.name for p in folder.iterdir()) == sorted(ProjectDocuments.FILES)
        requirements = (folder / "requirements.md").read_text()
        assert "**R1.1** A user can sign up" in requirements
        assert "**R1.2** An email is used once" in requirements
        assert "Accounts lock for 15 minutes" in requirements
        assert "A reset link is valid for 30 minutes. _(coder; not settled yet)_" in requirements
        design = (folder / "design.md").read_text()
        assert "User id in the link: guessable" in design
        assert "**Challenged:** A token in a log leaks an account" in design
        assert "**Auth service**" in design
        assert "| T2 Password reset | todo |" in (folder / "status.md").read_text()

    def test_starting_again_renames_the_folder(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        store.start("Login", "Log in.", "worker")
        store.flush()
        folder = store.start("Login system", "Log in.", "worker")
        store.flush()

        assert [p.name for p in tmp_path.iterdir()] == ["login-system"]
        assert folder.exists()
        assert store.start("Login system", "Log in with email.", "pm") == folder  # the same name keeps the folder

    def test_a_folder_an_earlier_call_left_is_never_reused(self, tmp_path):
        """Review: a new project with an earlier project's name got its old code, and a nested room-01 folder."""
        earlier = tmp_path / "login-system" / "code" / "app"
        earlier.mkdir(parents=True)
        (earlier / "old_main.py").write_text("# from an earlier call\n")
        store = ProjectStore(tmp_path, "room_01")
        store.add_requirement("Sign up", "Users sign up with email.", ["A user can sign up"], "pm")
        store.flush()

        folder = store.start("Login system", "Log in.", "pm")
        store.flush()

        assert folder == tmp_path / "login-system-2"
        assert sorted(p.name for p in folder.iterdir()) == sorted(ProjectDocuments.FILES)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["login-system", "login-system-2"]

    def test_a_change_returns_before_the_documents_are_written(self, tmp_path, monkeypatch):
        """A voice tool records on the realtime pool's loop, which paces the call's audio: it must not wait for files."""
        store = ProjectStore(tmp_path, "room_01")
        written = []

        def slow_write(state, folder):
            time.sleep(0.5)
            written.append(len(state.requirements))

        monkeypatch.setattr(ProjectDocuments, "write", staticmethod(slow_write))

        started = time.monotonic()
        store.add_requirement("Sign up", "Users sign up.", ["A user can sign up"], "pm")
        store.add_requirement("Log in", "Users log in.", ["A user can log in"], "pm")
        assert time.monotonic() - started < 0.2

        assert store.flush()
        assert written[-1] == 2  # the last write covers every change

    def test_a_repeated_task_is_not_added_twice(self, tmp_path):
        """Review: a voice turn that repeated create_tasks doubled every task, and with them the builds."""
        store = ProjectStore(tmp_path, "room_01")
        store.add_tasks(["Sign-up: R1", "Login: R1"], "worker")

        added = store.add_tasks(["sign-up: R1, again", "Password reset: R2"], "worker")

        assert [t.title for t in added] == ["Password reset"]
        assert [t.id for t in store.state.tasks] == ["T1", "T2", "T3"]

    def test_the_brief_carries_what_a_background_agent_needs(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        store.start("Login system", "Log in.", "worker")
        store.add_requirement("Reset", "Reset a forgotten password.", ["A reset link expires after 30 minutes"], "pm")
        store.add_findings(["app/auth.py: a reset token can be used twice", "app/auth.py: no rate limit"], "reviewer")
        store.set_finding_status("F1", "to fix", "architect", "A stolen link works twice")
        store.set_finding_status("F2", "waived", "architect", "Not in the requirements")
        store.add_build_note("Reset tokens are random 32-byte strings", "Cannot be guessed", "coder")

        brief = store.brief()

        assert "R1.1: A reset link expires after 30 minutes" in brief
        assert "Review findings to fix:\n- F1: app/auth.py: a reset token can be used twice" in brief
        assert "- F2: app/auth.py: no rate limit (waived: Not in the requirements)" in brief
        assert "- N1: Reset tokens are random 32-byte strings (Cannot be guessed)" in brief
        assert "The last test run failed" not in brief

    def test_failing_tests_reach_the_coder_and_the_voice_agents(self, tmp_path):
        """Review: only "2 failed, 5 passed" reached anyone, so a rebuild repeated the same mistake, and the
        Architect, told to read why the tests fail with get_progress, could not."""
        store = ProjectStore(tmp_path, "room_01")
        store.start("Booking", "Book slots.", "pm")
        output = (
            "....F.E\n"
            "E   assert 500 == 200\n"
            "FAILED tests/test_booking.py::test_r1_2_no_double_booking - assert 500 == 200\n"
            "ERROR tests/test_api.py::test_r2_1_list - fixture 'db' not found\n"
            "1 failed, 5 passed, 1 error in 0.42s\n"
        )
        store.set_test_output(output, "1 failed, 5 passed, 1 error in 0.42s")

        brief = store.brief()
        assert "The last test run failed (1 failed, 5 passed, 1 error in 0.42s)" in brief
        assert "E   assert 500 == 200" in brief
        for told in (store.overview(), store.progress_report()):
            assert "Failing tests: FAILED tests/test_booking.py::test_r1_2_no_double_booking" in told
            assert "ERROR tests/test_api.py::test_r2_1_list" in told

        store.set_test_output("......\n6 passed in 0.40s\n", "6 passed in 0.40s")
        assert "The last test run failed" not in store.brief() and "Failing tests" not in store.overview()

    def test_progress_reports_the_running_job(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        job = store.add_job("develop", "T1")
        store.update_job(job.id, "running")
        store.add_progress("Sign-up endpoint written", "coder")

        report = store.progress_report()

        assert "J1 (develop) has been running" in report
        assert "Sign-up endpoint written" in report


class TestHandoffGuard:
    def test_handoffs_are_refused_after_the_limit(self):
        guard = HandoffGuard(lambda: None, limit=3)

        assert [guard.allow() for _ in range(4)] == [True, True, True, False]
        assert "so you keep the call" in guard.refusal()
        # It does not make an agent that has just said where things stand say it again.
        assert "if you have, say nothing more" in guard.refusal()

    def test_a_finished_job_lets_the_team_hand_over_again(self, tmp_path):
        """Live: the guard counted the call bouncing while a review ran, then refused the one handover the
        review's result needed: to the Architect, to approve the code."""
        call = TeamCall("room_01", tmp_path)
        call.link.stop()
        call.guard = HandoffGuard(lambda: None, limit=2, recorded_at=call.moved_on_at)
        assert [call.guard.allow() for _ in range(3)] == [True, True, False]

        time.sleep(0.01)
        job = call.store.add_job("review", "")
        call.store.update_job(job.id, "done", "0 new findings")

        assert call.guard.allow()

    def test_the_count_restarts_when_the_client_speaks(self):
        clock = FakeClock()
        spoke = {"at": None}
        guard = HandoffGuard(lambda: spoke["at"], limit=2, clock=clock)
        assert [guard.allow() for _ in range(3)] == [True, True, False]

        clock.now += 5
        spoke["at"] = clock.now

        assert [guard.allow() for _ in range(3)] == [True, True, False]

    def test_the_count_restarts_when_the_team_records_something(self):
        """A design debate that keeps agreeing things is progress, not a loop."""
        clock = FakeClock()
        recorded = {"at": None}
        guard = HandoffGuard(lambda: None, limit=2, clock=clock, recorded_at=lambda: recorded["at"])
        assert [guard.allow() for _ in range(3)] == [True, True, False]

        clock.now += 5
        recorded["at"] = clock.now

        assert [guard.allow() for _ in range(3)] == [True, True, False]
        assert "without recording anything" in guard.refusal()

    def test_a_refusal_for_handoffs_in_a_row_is_internal_too(self):
        guard = HandoffGuard(lambda: None, limit=1)
        guard.allow()

        assert not guard.allow()
        assert "never mention handovers, limits or this message to the client" in guard.refusal()

    def test_the_team_counts_records_made_by_voice_agents_only(self, tmp_path):
        call = TeamCall("room_01", tmp_path)
        call.store.add_progress("wrote app/main.py", "coder")
        assert call.store.last_recorded_at(TeamCall.VOICE_ROLES) is None
        # Review: a question put to the client was logged under the agent's name, and lifted the
        # stuck-build refusal before the client had answered. An activity note is not a record.
        call.store.add_progress("Asked the client: Shall we try once more?", "worker")
        assert call.store.last_recorded_at(TeamCall.VOICE_ROLES) is None

        call.store.add_decision("Bookings", "Unique slot index", "Stops double booking", [], "architect")
        assert call.store.last_recorded_at(TeamCall.VOICE_ROLES) is not None


class TestCallLink:
    @staticmethod
    def _link(connection, clock, should_nudge=lambda: False):
        link = CallLink("room_01", should_nudge=should_nudge, connection=lambda: connection, clock=clock)
        link.start = lambda: None  # ticked by hand here, not by the watcher thread
        return link

    def test_an_update_is_posted_when_the_call_is_quiet(self):
        connection, clock = FakeConnection(), FakeClock()
        connection.last_model_output_at = clock.now - 5
        link = self._link(connection, clock)
        link.post("J1 (develop) finished.")

        assert link.tick() == "J1 (develop) finished."
        assert connection.sent == ["[Team update, not from the client] J1 (develop) finished."]

    def test_nothing_is_posted_while_the_model_responds(self):
        connection, clock = FakeConnection(), FakeClock()
        connection.is_responding = True
        link = self._link(connection, clock)
        link.post("J1 finished.")

        assert link.tick() is None
        assert connection.sent == []

    def test_nothing_is_posted_until_the_model_has_answered_the_caller(self):
        connection, clock = FakeConnection(), FakeClock()
        connection.last_model_output_at = clock.now - 10
        connection.last_caller_speech_at = clock.now - 5
        link = self._link(connection, clock)
        link.post("J1 finished.")

        assert link.tick() is None

        connection.last_model_output_at = clock.now - 3
        assert link.tick() == "J1 finished."

    def test_an_unanswered_caller_stops_holding_the_call_after_a_while(self):
        connection, clock = FakeConnection(), FakeClock()
        connection.last_caller_speech_at = clock.now - CallLink.REPLY_TIMEOUT_SECONDS - 1
        link = self._link(connection, clock)
        link.post("J1 finished.")

        assert link.tick() == "J1 finished."

    def test_nothing_is_posted_while_the_caller_is_speaking(self):
        """An adapter that reports the end of speech (OpenAI) says the caller is mid-answer, however long."""
        connection, clock = FakeConnection(), FakeClock()
        connection.is_caller_speaking = True
        connection.last_caller_speech_at = clock.now - 60
        link = self._link(connection, clock)
        link.post("J1 finished.")

        assert link.tick() is None

    def test_a_long_answer_is_not_taken_for_silence(self):
        """Review: the caller's turn start was the only timestamp, so a 20 s answer read as quiet after 15 s.

        Gemini reports more words as the caller keeps talking, which moves ``last_caller_speech_at``.
        """
        connection, clock = FakeConnection(), FakeClock()
        connection.last_model_output_at = clock.now - 30
        connection.last_caller_speech_at = clock.now - 0.5  # started 20 s ago, still heard
        link = self._link(connection, clock)
        link.post("J1 finished.")

        assert link.tick() is None

    def test_one_post_waits_for_the_model_to_answer_the_last(self):
        connection, clock = FakeConnection(), FakeClock()
        link = self._link(connection, clock)
        link.post("first")
        link.post("second")

        assert link.tick() == "first"
        clock.now += 5
        assert link.tick() is None

        connection.last_model_output_at = clock.now
        clock.now += CallLink.QUIET_SECONDS
        assert link.tick() == "second"

    def test_nothing_is_posted_without_a_live_call(self):
        clock = FakeClock()
        link = CallLink("room_01", connection=lambda: None, clock=clock)
        link.post("J1 finished.")

        assert link.tick() is None

    def test_a_stalled_team_is_nudged_at_most_twice_until_the_client_speaks(self):
        connection, clock = FakeConnection(), FakeClock()
        connection.last_model_output_at = clock.now
        link = self._link(connection, clock, should_nudge=lambda: True)

        clock.now += CallLink.NUDGE_AFTER_SECONDS - 1
        assert link.tick() is None
        for _ in range(2):
            clock.now += CallLink.NUDGE_AFTER_SECONDS
            connection.last_model_output_at = clock.now - CallLink.NUDGE_AFTER_SECONDS + 0.5
            clock.now += 1
            assert link.tick() == CallLink.NUDGE
            connection.last_model_output_at = clock.now
        clock.now += CallLink.NUDGE_AFTER_SECONDS
        assert link.tick() is None

        connection.last_caller_speech_at = clock.now
        connection.last_model_output_at = clock.now + 1
        clock.now += CallLink.NUDGE_AFTER_SECONDS + 1
        assert link.tick() == CallLink.NUDGE

    def test_words_count_as_heard_once_their_audio_has_played(self):
        connection, clock = FakeConnection(), FakeClock()
        link = self._link(connection, clock)

        connection.is_responding = True  # audio still waiting to be played
        assert not link.heard()
        connection.is_responding = False
        connection.last_model_output_at = clock.now
        assert not link.heard()
        clock.now += CallLink.HEARD_AFTER_SECONDS
        assert link.heard()
        assert CallLink("room_01", connection=lambda: None).heard()  # no live call: nothing to wait for

    def test_no_nudge_when_the_team_should_wait(self):
        connection, clock = FakeConnection(), FakeClock()
        connection.last_model_output_at = clock.now - 60
        link = self._link(connection, clock, should_nudge=lambda: False)

        assert link.tick() is None

    def test_the_model_answering_a_tool_result_is_not_taken_for_silence(self):
        """Review: a finished tool call counted as the model's output, so 1.5 s after its result, before the
        model had said a word, an update was posted, and on Gemini Live that cuts off the reply."""
        connection, clock = FakeConnection(), FakeClock()
        connection.last_caller_speech_at = clock.now - 10
        connection.last_model_output_at = clock.now - 8
        connection.last_tool_result_at = clock.now - 2  # e.g. the agent just handed the call over
        link = self._link(connection, clock)
        link.post("J1 (review) finished.")

        assert link.tick() is None

        connection.last_model_output_at = clock.now  # the new agent speaks
        clock.now += CallLink.QUIET_SECONDS
        assert link.tick() == "J1 (review) finished."

    def test_a_tool_result_the_model_never_answers_stops_holding_the_call(self):
        connection, clock = FakeConnection(), FakeClock()
        connection.last_model_output_at = clock.now - 30
        connection.last_tool_result_at = clock.now - CallLink.TOOL_REPLY_TIMEOUT_SECONDS - 1
        link = self._link(connection, clock)
        link.post("J1 (review) finished.")

        assert link.tick() == "J1 (review) finished."

    def test_text_the_client_typed_is_waited_on_and_counts_as_the_client_speaking(self):
        """Review: typed words were invisible: an update could cut into the reply, and the client 'never spoke'."""
        connection, clock = FakeConnection(), FakeClock()
        connection.last_model_output_at = clock.now - 30
        connection.last_edge_text_at = clock.now - 2
        link = self._link(connection, clock)
        link.post("J1 (review) finished.")

        assert link.caller_spoke_at() == clock.now - 2
        assert link.tick() is None
        connection.last_model_output_at = clock.now
        clock.now += CallLink.QUIET_SECONDS
        assert link.tick() == "J1 (review) finished."

    def test_an_update_that_no_longer_applies_is_dropped(self):
        connection, clock = FakeConnection(), FakeClock()
        link = self._link(connection, clock)
        still_open = {"question": True}
        link.post("The Coder asks: email or username?", valid=lambda: still_open["question"])
        link.post("J1 (develop) finished.")

        still_open["question"] = False  # timed out while the call was busy

        assert link.tick() == "J1 (develop) finished."
        assert connection.sent == ["[Team update, not from the client] J1 (develop) finished."]

    def test_a_full_queue_drops_its_oldest_update_with_a_warning(self, caplog):
        link = self._link(FakeConnection(), FakeClock())
        link.start = lambda: None
        for i in range(CallLink.MAX_PENDING + 1):
            link.post(f"update {i}")

        assert "dropped the oldest: update 0" in caplog.text
        assert link._next_pending() == "update 1"

    def test_a_new_model_connection_starts_the_link_over(self):
        """The pool replaces a failed connection with one that starts at the PM and remembers nothing."""
        first, second, clock = FakeConnection(), FakeConnection(), FakeClock()
        current = {"connection": first}
        reconnects = []
        link = CallLink(
            "room_01", connection=lambda: current["connection"], clock=clock, on_reconnect=lambda: reconnects.append(1)
        )
        link.start = lambda: None
        first.last_model_output_at = clock.now - 5
        link.post("first")
        assert link.tick() == "first"

        current["connection"] = second
        link.tick()

        assert reconnects == [1]
        assert link._last_post_at is None

    def test_words_said_after_a_tool_call_are_waited_for_too(self):
        """A job waits until its announcement has been heard, also when the agent speaks after the job tool."""
        connection, clock = FakeConnection(), FakeClock()
        link = self._link(connection, clock)
        connection.last_model_output_at = clock.now - 5
        connection.last_tool_result_at = clock.now - 1

        assert not link.heard()  # the reply to the tool result has not started yet
        clock.now += CallLink.HEARD_TOOL_REPLY_SECONDS
        assert link.heard()  # nothing came: there is nothing to wait for


class FakeService:
    """Stands in for ``AgentService``: runs a scripted reply, recording how many runs overlap.

    Given the call, a scripted Reviewer submits what its work asks for (a review, or every criterion
    passing), as the real one does with its tools.
    """

    running = 0
    most_at_once = 0
    lock = threading.Lock()

    def __init__(self, on_run=None, reply="Built it.", call=None):
        self.agent = None
        self.selected = None
        self.on_run = on_run
        self.reply = reply
        self.call = call

    def select(self, session_id, name):
        self.selected = (session_id, name)
        self.agent = SimpleNamespace(name=name)

    async def run(self, prompt):
        with FakeService.lock:
            FakeService.running += 1
            FakeService.most_at_once = max(FakeService.most_at_once, FakeService.running)
        try:
            await asyncio.sleep(0.05)
            if self.on_run:
                self.on_run(self.selected, prompt)
            if self.call is not None and self.selected[1] == "reviewer":
                self._submit(prompt)
            return self.reply
        finally:
            with FakeService.lock:
                FakeService.running -= 1

    def _submit(self, prompt):
        store = self.call.store
        if "submit_acceptance" in prompt:
            results = [
                AcceptanceResult(criterion_id=c, criterion=t, test="t", result="pass") for c, t in store.criteria()
            ]
            store.set_acceptance(results, "reviewer")
        elif "submit_review" in prompt:
            store.record_review("Looks fine.", "reviewer")


class TestDevJobRunner:
    @pytest.fixture
    def call(self, tmp_path, monkeypatch):
        call = TeamCall("room_01", tmp_path)
        call.store.start("Login system", "Log in.", "worker")
        call.store.add_tasks(["Sign-up", "Reset"], "worker")
        call.notices = []
        monkeypatch.setattr(call, "notify", call.notices.append)
        # These tests are about running jobs; the process checks have their own tests below.
        monkeypatch.setattr(call.gates, "blocker", lambda step, items=None: None)

        async def passing_tests(self):
            self._call.store.set_test_output("....\n4 passed in 0.10s\n", "4 passed in 0.10s")
            return True, "4 passed in 0.10s"

        monkeypatch.setattr(DevJobRunner, "run_tests", passing_tests)
        FakeService.running = FakeService.most_at_once = 0
        return call

    @staticmethod
    def _wait_until_idle(runner):
        deadline = time.time() + 5
        while runner.busy and time.time() < deadline:
            time.sleep(0.02)
        assert not runner.busy

    def test_a_develop_job_runs_the_coder_in_the_dev_session_and_records_the_real_test_run(self, call):
        seen = []

        def coder(selected, prompt):
            seen.append((selected, prompt))
            call.store.set_task_status("T1", "done", "coder")

        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(on_run=coder))

        job_id = call.jobs.submit("develop", ["T1"])
        self._wait_until_idle(call.jobs)

        [(selected, prompt)] = seen
        assert selected == ("room_01-dev", "coder")
        assert "T1 Sign-up" in prompt and "T2" not in prompt.split("Implement these tasks:")[1]
        state = call.store.state
        assert [t.status for t in state.tasks] == ["done", "todo"]
        assert state.jobs[0].status == "done"
        assert job_id == "J1"
        assert call.notices and "Tests: 4 passed in 0.10s" in call.notices[0]
        assert "request_review" in call.notices[0]

    def test_a_job_starts_once_the_agent_announcing_it_has_been_heard(self, call):
        """Live: the review was done on the board while the Architect was still saying it would ask for one."""
        connection = FakeConnection()
        connection.is_responding = True  # the Architect's words are still being played
        call.link = CallLink("room_01", connection=lambda: connection)
        reviewed = []
        call.jobs = DevJobRunner(
            call, service_factory=lambda: FakeService(on_run=lambda s, p: reviewed.append(s), call=call)
        )

        call.jobs.submit("review")
        time.sleep(0.6)
        assert reviewed == [] and call.store.state.jobs[0].status == "queued"

        connection.is_responding = False
        self._wait_until_idle(call.jobs)
        assert reviewed and call.store.state.jobs[0].status == "done"

    def test_a_coder_that_reports_no_task_is_asked_once_more(self, call):
        """Live: GPT-4.1 built it and passed its tests, but reported no task, so nothing counted as done."""
        prompts = []

        def coder(selected, prompt):
            prompts.append(prompt)
            if "You finished without reporting your tasks" in prompt:
                call.store.set_task_status("T1", "done", "coder")

        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(on_run=coder))

        call.jobs.submit("develop", ["T1"])
        self._wait_until_idle(call.jobs)

        assert len(prompts) == 2 and "for each of T1, call report_progress" in prompts[1]
        assert call.store.state.tasks[0].status == "done"

    def test_a_task_the_coder_did_not_finish_is_not_marked_done(self, call):
        """Live: the Coder finished T1 only, and passing tests used to mark every task done."""
        call.jobs = DevJobRunner(
            call,
            service_factory=lambda: FakeService(on_run=lambda s, p: call.store.set_task_status("T1", "done", "coder")),
        )

        call.jobs.submit("develop", ["T1", "T2"])
        self._wait_until_idle(call.jobs)

        assert [t.status for t in call.store.state.tasks] == ["done", "in progress"]
        assert "Tasks done: T1; not finished: T2" in call.notices[0]
        assert "starts development again for T2 with start_development" in call.notices[0]

    def test_a_task_reported_done_with_failing_tests_needs_work(self, call, monkeypatch):
        async def failing_tests(self):
            return False, "1 failed, 3 passed in 0.2s"

        monkeypatch.setattr(DevJobRunner, "run_tests", failing_tests)
        call.jobs = DevJobRunner(
            call,
            service_factory=lambda: FakeService(on_run=lambda s, p: call.store.set_task_status("T1", "done", "coder")),
        )

        call.jobs.submit("develop", ["T1"])
        self._wait_until_idle(call.jobs)

        assert call.store.state.tasks[0].status == "needs work"
        assert "fix the failing tests" in call.notices[0]

    def test_jobs_run_one_at_a_time_in_order(self, call):
        order = []

        def run(selected, prompt):
            order.append(selected[1])
            if selected[1] == "coder":  # a Coder that reports its tasks, so nothing is asked twice
                for task in ("T1", "T2"):
                    call.store.set_task_status(task, "done", "coder")

        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(on_run=run, call=call))

        call.jobs.submit("develop", ["T1"])
        call.jobs.submit("review")
        call.jobs.submit("develop", ["T2"])
        self._wait_until_idle(call.jobs)

        assert FakeService.most_at_once == 1
        assert order == ["coder", "reviewer", "coder"]
        assert [j.status for j in call.store.state.jobs] == ["done", "done", "done"]

    def test_a_failing_job_is_recorded_and_reported(self, call):
        def broken(selected, prompt):
            raise RuntimeError("model unavailable")

        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(on_run=broken))

        call.jobs.submit("develop")
        self._wait_until_idle(call.jobs)

        [job] = call.store.state.jobs
        assert job.status == "failed"
        assert "model unavailable" in call.notices[0]

    def test_a_job_finishing_after_the_call_ended_is_still_recorded(self, tmp_path, monkeypatch):
        call = TeamCall("room_02", tmp_path)
        call.store.start("Login system", "Log in.", "worker")
        call.link = CallLink("room_02", connection=lambda: None)
        monkeypatch.setattr(call.gates, "blocker", lambda step, items=None: None)
        monkeypatch.setattr(DevJobRunner, "run_tests", lambda self: asyncio.sleep(0, result=(True, "1 passed")))
        call.jobs = DevJobRunner(call, service_factory=FakeService)

        call.jobs.submit("develop")
        self._wait_until_idle(call.jobs)

        assert call.store.state.jobs[0].status == "done"
        assert call.link.tick() is None

    def test_an_error_reply_from_the_agent_fails_the_job(self, call):
        """A runner reports a failed run (a retired model, say) as an error reply, not an exception."""
        retired = "Error: Invalid model or resource not found. Please check your configuration."
        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(reply=retired))

        call.jobs.submit("develop", ["T1"])
        self._wait_until_idle(call.jobs)

        state = call.store.state
        assert state.jobs[0].status == "failed"
        assert "Invalid model" in state.jobs[0].summary
        assert state.tasks[0].status == "todo"
        assert "do not start the job again" in call.notices[0]

    def test_a_kind_that_keeps_failing_is_refused_until_the_client_speaks(self, call):
        """The live loop: the model was retired, and the Worker restarted development every few seconds."""
        connection = FakeConnection()
        call.link = CallLink("room_01", connection=lambda: connection)
        call.jobs = DevJobRunner(
            call, service_factory=lambda: FakeService(reply="Error: The service is temporarily unavailable.")
        )
        for _ in range(DevJobRunner.MAX_FAILURES_IN_A_ROW):
            call.jobs.submit("develop")
            self._wait_until_idle(call.jobs)

        with pytest.raises(JobRefused, match="Starting it again will not help"):
            call.jobs.submit("develop")
        assert call.jobs.refusal("review") is None

        connection.last_caller_speech_at = time.time() + 1
        assert call.jobs.refusal("develop") is None

    def test_delivery_with_failing_tests_is_not_delivered(self, call, monkeypatch):
        """Review: delivery ran the tests and ignored the result."""

        async def failing_tests(self):
            return False, "1 failed, 6 passed in 0.3s"

        async def some_files(self, destination):
            return 3

        monkeypatch.setattr(DevJobRunner, "run_tests", failing_tests)
        monkeypatch.setattr(DevJobRunner, "copy_workspace", some_files)
        call.jobs = DevJobRunner(call, service_factory=FakeService)

        call.jobs.submit("deliver")
        self._wait_until_idle(call.jobs)

        assert call.store.state.delivered is False
        assert "Not delivered: the tests fail (1 failed, 6 passed in 0.3s)" in call.notices[0]
        call.store.flush()
        assert "Delivered:** no" in (call.store.folder / "status.md").read_text()

    def test_delivery_that_copied_no_code_is_not_delivered(self, call, monkeypatch):
        """Review: delivery reported success even when nothing came out of the sandbox."""

        async def no_files(self, destination):
            return 0

        monkeypatch.setattr(DevJobRunner, "copy_workspace", no_files)
        call.jobs = DevJobRunner(call, service_factory=FakeService)

        call.jobs.submit("deliver")
        self._wait_until_idle(call.jobs)

        assert call.store.state.delivered is False
        assert "no code could be copied" in call.notices[0]

    def test_a_queued_job_is_checked_again_when_it_starts(self, tmp_path, monkeypatch):
        """Review: a delivery requested during a new build passed its check, waited, then shipped unchecked code."""
        call = TeamCall("room_03", tmp_path)
        call.link = CallLink("room_03", connection=lambda: None)
        notices = []
        monkeypatch.setattr(call, "notify", notices.append)
        store = call.store
        store.add_requirement("Booking", "Book a slot.", ["A slot can be booked"], "pm")
        store.add_component("Booking service", "Books slots", "architect")
        store.add_decision("Slots", "Unique index", "Stops double booking", [], "architect", challenge="Races: index")
        store.write_design("architect")
        store.confirm_requirements("pm")
        store.add_tasks(["Booking"], "worker")
        store.set_task_status("T1", "done", "coder")
        for kind in ("develop", "review"):
            job = store.add_job(kind, "")
            store.update_job(job.id, "done")
        time.sleep(0.01)
        store.set_acceptance(
            [AcceptanceResult(criterion_id="R1.1", criterion="c", test="t", result="pass")], "reviewer"
        )
        # A late review finding: fixing it is a build that leaves the (done) tasks as they are, so the
        # delivery check passes while it runs.
        store.add_findings(["app/booking.py: the error message names the database"], "reviewer")
        store.set_finding_status("F1", "to fix", "architect", "Leaks how the app is built")
        time.sleep(0.01)

        release = threading.Event()

        class BuildingService(FakeService):
            async def run(self, prompt):
                await asyncio.to_thread(release.wait, 5)
                return "Fixed it."

        async def passing_tests(self):
            return True, "1 passed"

        async def some_files(self, destination):
            return 3

        monkeypatch.setattr(DevJobRunner, "run_tests", passing_tests)
        monkeypatch.setattr(DevJobRunner, "copy_workspace", some_files)
        call.jobs = DevJobRunner(call, service_factory=BuildingService)

        call.jobs.submit("develop", ["F1"])  # the tool's check is skipped here; a new build is running
        assert call.gates.blocker("deliver") is None  # so the delivery tool would let it through
        call.jobs.submit("deliver")
        release.set()
        self._wait_until_idle(call.jobs)

        develop, deliver = store.state.jobs[-2:]
        assert develop.status == "done"
        assert deliver.kind == "deliver" and deliver.status == "skipped"
        assert "changed after the last acceptance check" in deliver.summary
        assert store.state.delivered is False
        assert any("did not start" in n for n in notices)

    def test_a_fix_the_reviewer_did_not_confirm_is_to_be_fixed_again(self, call):
        call.store.add_findings(["app/auth.py: passwords are logged"], "reviewer")
        call.store.set_finding_status("F1", "fixed", "coder")
        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(call=call))

        call.jobs.submit("review", ["F1"])
        self._wait_until_idle(call.jobs)

        assert call.store.state.findings[0].status == "to fix"  # the Architect already ruled on it
        assert "Next: the Worker fixes F1 again with start_development" in call.notices[-1]

    def test_builds_stuck_on_the_same_failing_tests_are_refused_until_something_changes(self, call):
        """Live: every build ended "2 failed, 5 passed", and the Worker started another every ten seconds, 13 times."""
        call.link = CallLink("room_01", connection=lambda: None)
        store = call.store
        for timing in ("0.89s", "0.92s", "0.90s"):  # the same failures, run times aside
            job = store.add_job("develop", "T3")
            store.update_job(
                job.id, "done", "The Coder says: blocked.", tests=f"2 failed, 5 passed, 2 warnings in {timing}"
            )

        refused = call.jobs.refusal("develop")

        assert "the last 3 builds ended with the same failing tests (2 failed, 5 passed, 2 warnings)" in refused
        assert "record_decision" in refused and "drop or simplify" in refused
        assert call.jobs.refusal("review") is None  # only building again is pointless
        with pytest.raises(JobRefused):
            call.jobs.submit("develop", ["T3"])

        time.sleep(0.01)
        store.add_decision(
            "Chat page",
            "Render with Jinja2 TemplateResponse(request, ...)",
            "Request first",
            [],
            "architect",
            challenge="Template error",
        )
        assert call.jobs.refusal("develop") is None  # the Architect changed the approach

    def test_builds_that_improve_or_pass_are_never_stuck(self, call):
        store = call.store
        for tests in ("4 failed, 1 passed in 0.2s", "2 failed, 3 passed in 0.2s", "1 failed, 4 passed in 0.2s"):
            job = store.add_job("develop", "T1")
            store.update_job(job.id, "done", "Built.", tests=tests)
        assert call.jobs.refusal("develop") is None  # different results: the builds are getting somewhere

        for _ in range(3):
            job = store.add_job("develop", "T1")
            store.update_job(job.id, "done", "Built.", tests="5 passed in 0.2s")
        assert call.jobs.refusal("develop") is None

    def test_one_job_thread_and_loop_serve_the_whole_call(self, call):
        """Live: a new event loop per batch of jobs left LiteLLM's logging task on a closed loop."""
        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(call=call))
        call.jobs.submit("review")
        self._wait_until_idle(call.jobs)
        first = call.jobs._thread

        time.sleep(1.2)  # idle for a while, as between jobs in a call
        call.jobs.submit("review")
        self._wait_until_idle(call.jobs)

        assert first.is_alive() and call.jobs._thread is first

    def test_a_review_the_reviewer_never_submitted_is_asked_for_once_more_then_fails(self, call):
        """Review: a Reviewer that skipped submit_review produced "0 new findings", read as a clean review."""
        prompts_seen = []
        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(on_run=lambda s, p: prompts_seen.append(p)))

        call.jobs.submit("review")
        self._wait_until_idle(call.jobs)

        assert len(prompts_seen) == 2 and "you finished without calling submit_review" in prompts_seen[1]
        [job] = call.store.state.jobs
        assert job.status == "failed" and "without calling submit_review" in job.summary
        assert call.store.state.code_approved_at is None

    def test_an_acceptance_check_never_submitted_does_not_report_results(self, call):
        """Review: with no submit_acceptance the job reported "0 of 2 pass" (or old results) and sent the Worker to fix them."""
        call.store.add_requirement("Booking", "Book a slot.", ["A slot can be booked", "No double booking"], "pm")
        submitted = {"second": False}

        def reviewer(selected, prompt):
            if "you finished without calling submit_acceptance" in prompt:
                submitted["second"] = True
                results = [
                    AcceptanceResult(criterion_id=c, criterion=t, test="t", result="pass")
                    for c, t in call.store.criteria()
                ]
                call.store.set_acceptance(results, "reviewer")

        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(on_run=reviewer))
        call.jobs.submit("acceptance")
        self._wait_until_idle(call.jobs)

        assert submitted["second"]
        assert "Acceptance: 2 of 2 criteria pass" in call.notices[-1]

    def test_a_failing_build_names_its_failures_and_gives_them_to_the_next_build(self, call, monkeypatch):
        """Live: 13 identical builds, ten seconds apart: the next build was never told what failed."""

        async def failing_tests(self):
            output = "FAILED tests/test_login.py::test_r1_1_login - assert 401 == 200\n1 failed, 3 passed in 0.2s\n"
            self._call.store.set_test_output(output, "1 failed, 3 passed in 0.2s")
            return False, "1 failed, 3 passed in 0.2s"

        monkeypatch.setattr(DevJobRunner, "run_tests", failing_tests)
        prompts_seen = []
        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(on_run=lambda s, p: prompts_seen.append(p)))

        call.jobs.submit("develop", ["T1"])
        self._wait_until_idle(call.jobs)
        assert "test_r1_1_login" in call.notices[-1] and "the Coder is given them" in call.notices[-1]

        call.jobs.submit("develop", ["T1"])
        self._wait_until_idle(call.jobs)
        assert "The last test run failed" in prompts_seen[-1] and "assert 401 == 200" in prompts_seen[-1]

    def test_pytest_summary_line(self):
        assert DevJobRunner.test_summary("....\n========= 12 passed in 0.41s =========\n") == "12 passed in 0.41s"
        assert (
            DevJobRunner.test_summary("FAILED tests/test_a.py::test_x\n1 failed, 3 passed in 0.2s")
            == "1 failed, 3 passed in 0.2s"
        )
        assert DevJobRunner.test_summary("") == "no test results"
        assert DevJobRunner.test_summary("ERROR tests/test_a.py\n1 error in 0.10s") == "1 error in 0.10s"

    def test_a_traceback_line_is_not_a_test_summary(self):
        """Live: "ran the tests: .../starlette/middleware/errors.py:164: in __call__" was told as the result."""
        traceback = "E   assert 500 == 200\n/usr/local/lib/python3.12/site-packages/starlette/middleware/errors.py:164: in __call__"

        assert DevJobRunner.test_summary(traceback) == "no test results"


class TestTeamTools:
    @pytest.fixture
    def calls(self, tmp_path):
        return TeamCalls(tmp_path)

    async def test_voice_tools_find_their_call_from_the_session(self, calls):
        pm = TeamTools(calls, "pm")

        async with Session("room_01"):
            pm.start_project("Login system", "Let customers log in.")
            pm.record_requirement("Sign up", "Users sign up with email.", ["A user can sign up"])
            state = pm.get_project_state()

        assert "R1 Sign up" in state
        assert calls.get("room_01").store.state.requirements[0].by == "pm"

    async def test_background_tools_reach_the_same_call_from_the_dev_session(self, calls):
        async with Session("room_01-dev"):
            TeamTools(calls, "coder").report_progress("Wrote the sign-up endpoint", "", "")

        assert calls.get("room_01").store.state.progress[0].what == "Wrote the sign-up endpoint"

    async def test_the_pm_answers_the_coders_question(self, calls):
        call = calls.get("room_01")
        call.notify = lambda text: None
        coder, pm = TeamTools(calls, "coder"), TeamTools(calls, "pm")

        async def answer_soon():
            while call.coder_question.open is None:
                await asyncio.sleep(0.01)
            async with Session("room_01"):
                return pm.answer_coder("Log in with email.")

        async with Session("room_01-dev"):
            reply, answered = await asyncio.gather(coder.ask_team("Login by email or username?"), answer_soon())

        assert reply == "The team says: Log in with email."
        assert answered == "Answer passed to the Coder."
        [question] = call.store.state.questions
        assert question.answer == "Log in with email."

    async def test_acceptance_results_cover_every_criterion(self, calls):
        reviewer = TeamTools(calls, "reviewer")
        store = calls.get("room_01").store
        store.add_requirement("Sign up", "Users sign up.", ["A user can sign up", "An email is used once"], "pm")

        async with Session("room_01-dev"):
            reviewer.submit_acceptance(["R1.1 | tests/test_auth.py::test_r1_1_signup | PASS", "not a result"])

        results = {r.criterion_id: (r.test, r.result) for r in store.state.acceptance}
        assert results == {"R1.1": ("tests/test_auth.py::test_r1_1_signup", "pass"), "R1.2": ("none", "not covered")}

    async def test_an_assumption_the_coder_makes_is_taken_to_the_call(self, calls):
        """Review: after an unanswered question the Coder decides alone; a business rule must reach the client."""
        call = calls.get("room_01")
        call.link.stop()
        call.store.start("salon booking", "Book appointments online", "pm")
        posted = []
        call.notify = posted.append

        async with Session("room_01-dev"):
            TeamTools(calls, "coder").record_assumption("A booking can be cancelled up to 24 hours before.")
        async with Session("room_01"):
            TeamTools(calls, "pm").record_assumption("Times are shown in the shop's time zone.")

        [notice] = posted
        assert "The Coder assumed A1" in notice and "asks them to confirm it" in notice
        assert "settle_assumption" in notice
        assumptions = call.store.state.assumptions
        assert [a.needs_settling for a in assumptions] == [True, False]

    async def test_nothing_is_recorded_before_the_project_is_started(self, calls):
        """Live: before saying hello the PM recorded its own greeting question, and the board showed it all call."""
        call = calls.get("room_01")
        pm = TeamTools(calls, "pm")

        async with Session("room_01"):
            question = pm.record_question("What would you like to build?")
            assumption = pm.record_assumption("The client wants to hear how the team works.")
            assert question == assumption == TeamTools.NOT_STARTED
            assert call.store.state.questions == [] and call.store.state.assumptions == []

            pm.start_project("salon booking", "Book appointments online")
            assert pm.record_question("Do bookings need a deposit?") == "Recorded open question Q1."

    async def test_settling_an_assumption(self, calls):
        call = calls.get("room_01")
        call.link.stop()
        call.store.add_assumption("A booking can be cancelled up to 24 hours before.", "coder", needs_settling=True)
        pm = TeamTools(calls, "pm")

        async with Session("room_01"):
            missing = pm.settle_assumption("A9", "The client confirmed it")
            settled = pm.settle_assumption("A1", "The client confirmed it")

        assert missing.startswith("Not settled: there is no assumption A9")
        assert settled == "Settled A1."
        assert call.store.state.assumptions[0].settled == "The client confirmed it"

    async def test_a_decision_records_its_challenge(self, calls):
        architect = TeamTools(calls, "architect")

        async with Session("room_01"):
            architect.record_decision(
                "Bookings",
                "A unique index on the slot",
                "The database refuses the second booking",
                "Two customers could book one slot at once: the index makes the second insert fail",
                ["Check, then insert: races"],
            )

        [decision] = calls.get("room_01").store.state.decisions
        assert decision.challenge.startswith("Two customers could book one slot at once")

    def test_each_role_gets_its_own_tools(self, calls):
        names = {
            role: [tool.__name__ for tool in TeamTools(calls, role).tools()]
            for role in ("worker", "pm", "architect", "coder", "reviewer")
        }

        assert "start_development" in names["worker"] and "start_development" not in names["pm"]
        assert "start_project" in names["pm"] and "start_project" not in names["worker"]
        assert "get_progress" in names["architect"]
        assert "answer_coder" in names["architect"] and "answer_coder" in names["pm"]
        assert "settle_assumption" in names["pm"] and "settle_assumption" in names["architect"]
        assert "settle_assumption" not in names["worker"]
        assert names["coder"] == ["report_progress", "ask_team", "record_assumption", "record_build_note"]
        assert "rule_on_finding" in names["architect"] and "approve_code" in names["architect"]
        assert "rule_on_finding" not in names["worker"]
        assert "submit_review" not in names["pm"]

    async def test_creating_the_same_tasks_again_adds_nothing(self, calls):
        call = calls.get("room_01")
        call.link.stop()
        monkey_ready(call)

        async with Session("room_01"):
            worker = TeamTools(calls, "worker")
            first = worker.create_tasks(["Sign-up: R1", "Login: R1"])
            again = worker.create_tasks(["Sign-up: R1", "Login: R1"])
            more = worker.create_tasks(["Login: R1", "Password reset: R2"])

        assert first == "Created T1 Sign-up; T2 Login."
        assert again.startswith("Not created: these tasks already exist (T1, T2)")
        assert more == "Created T3 Password reset. Already there, so not added again: T2 Login."

    async def test_the_coders_and_reviewers_words_for_done_and_pass_are_understood(self, calls):
        """Review: "completed" left a built task undone, and "passed" failed acceptance: both loops of rebuilding."""
        call = calls.get("room_01")
        call.link.stop()
        call.store.add_requirement("Booking", "Book a slot.", ["A slot can be booked", "No double booking"], "pm")
        call.store.add_tasks(["Booking"], "worker")

        async with Session("room_01-dev"):
            coder, reviewer = TeamTools(calls, "coder"), TeamTools(calls, "reviewer")
            coder.report_progress("Built booking", "T1", "Completed")
            odd = coder.report_progress("Built booking", "T1", "almost")
            reviewer.submit_acceptance(["R1.1 | tests/test_b.py::t1 | Passed", "R1.2 | tests/test_b.py::t2 | FAILED"])

        assert call.store.state.tasks[0].status == "done"
        assert 'is "in progress" or "done", not "almost"' in odd
        assert [r.result for r in call.store.state.acceptance] == ["pass", "fail"]

    async def test_a_review_resolves_only_findings_the_coder_fixed(self, calls):
        """Review: submit_review could resolve an open finding, skipping the Architect's ruling, or a waived one."""
        call = calls.get("room_01")
        call.link.stop()
        store = call.store
        store.add_findings(["a: open", "b: waived", "c: fixed"], "reviewer")
        store.set_finding_status("F2", "waived", "architect", "Not needed")
        store.set_finding_status("F3", "fixed", "coder")

        async with Session("room_01-dev"):
            reply = TeamTools(calls, "reviewer").submit_review("Mostly fine.", [], ["F1", "F2", "F3"])

        assert [f.status for f in store.state.findings] == ["open", "waived", "resolved"]
        assert (
            reply
            == "Recorded 0 new findings and 1 resolved. Not resolved: F1, F2, which the Coder has not fixed; they stay as they are."
        )
        assert store.state.review_submitted_at is not None

    async def test_the_job_tools_state_facts_and_leave_speaking_to_the_prompt(self, calls, monkeypatch):
        """Live: an agent said it would start a job, called the tool, and the reply asked it to say so again."""
        call = calls.get("room_01")
        call.link.stop()
        monkeypatch.setattr(call.gates, "blocker", lambda step, items=None: None)
        monkeypatch.setattr(call.jobs, "submit", lambda kind, items=None: "J1")

        async with Session("room_01"):
            replies = [
                TeamTools(calls, "worker").start_development(["T1"]),
                TeamTools(calls, "worker").deliver(),
                TeamTools(calls, "architect").request_review([]),
                TeamTools(calls, "pm").check_acceptance(),
            ]

        assert all(reply.startswith("Started ") and "J1" in reply for reply in replies)
        assert not any("Tell the client" in reply or "say" in reply.lower() for reply in replies)


def monkey_ready(call):
    """Bring the project to where tasks can be created: requirements approved, design written."""
    store = call.store
    store.add_requirement("Booking", "Book a slot.", ["A slot can be booked"], "pm")
    store.add_component("API", "Books slots", "architect")
    store.add_decision("Slots", "Unique index", "No double booking", [], "architect", challenge="Races")
    store.count_design_round("architect")
    store.count_design_round("architect")
    store.write_design("architect")
    store.confirm_requirements("pm")


class TestClientQuestionWatcher:
    """The team knows the client owes an answer from what the agent said, not from a tool it had to call."""

    @staticmethod
    def _watch(tmp_path):
        from team import ClientQuestionWatcher

        call = TeamCall("room_01", tmp_path)
        call.link.stop()
        return call, ClientQuestionWatcher(call)

    def test_a_question_to_the_client_leaves_the_team_waiting_for_the_answer(self, tmp_path):
        from server import BoardSnapshot

        call, watcher = self._watch(tmp_path)
        # The turn from the live log, as the gateway logged it.
        said = (
            "AI response: Understood, Pulindu. So customers can browse available juices and place preorders in the app. "
            "Should they also be able to pay online, or pay when they pick up their order?"
        )
        watcher.emit(logging.LogRecord(watcher.LOGGERS[0], logging.INFO, __file__, 0, said, None, None))
        watcher.emit(logging.LogRecord(watcher.LOGGERS[0], logging.INFO, __file__, 0, "Barge-in detected", None, None))

        assert call.awaiting_client()
        assert BoardSnapshot.of(call, 0.0).waiting_for_you == [
            "Should they also be able to pay online, or pay when they pick up their order?"
        ]
        assert call.store.state.progress[-1].who == "pm"

    def test_a_question_to_a_teammate_or_a_statement_is_not_one_for_the_client(self, tmp_path):
        call, watcher = self._watch(tmp_path)

        watcher.agent_said("Worker, how would you stop two people booking the same slot?")
        watcher.agent_said("The team has started building it.")
        assert not call.awaiting_client()

        call.handed_over("worker")  # a turn that ends by handing over asks the teammate, not the client
        watcher.agent_said("I have one concern. How would you stop two people booking the same slot?")
        assert not call.awaiting_client()

        watcher.agent_said("Pulindu, should a customer be able to cancel a preorder?")
        assert call.awaiting_client()
        assert call.store.state.progress[-1].who == "worker"

    def test_only_a_turn_that_ends_with_a_question_addressed_to_the_client_counts(self, tmp_path):
        call, watcher = self._watch(tmp_path)

        watcher.agent_said("Should it work offline? I'll note that we keep it simple either way.")
        watcher.agent_said("That's the plan. What do you think, Architect?")
        assert not call.awaiting_client()

        # Naming a role is not asking that teammate.
        watcher.agent_said("Pulindu, should the Architect plan for phones too?")
        assert call.awaiting_client()

    def test_a_typed_answer_is_an_answer(self, tmp_path):
        """Review: an answer typed in the page's chat box left the team waiting for the client forever."""
        call, watcher = self._watch(tmp_path)
        connection, clock = FakeConnection(), FakeClock()
        call.link = CallLink("room_01", connection=lambda: connection, clock=clock)
        watcher.agent_said("Pulindu, should customers pay online?")
        assert call.awaiting_client()

        connection.last_edge_text_at = time.time() + 1

        assert not call.awaiting_client()


class TestCodeReview:
    """The review is a conversation: the Architect asks the Worker why, rules on each finding, and approves the code."""

    @staticmethod
    def _reviewed(tmp_path, findings):
        calls = TeamCalls(tmp_path)
        call = calls.get("room_01")
        call.link.stop()
        store = call.store
        store.add_requirement("Booking", "Book a slot.", ["A slot cannot be booked twice"], "pm")
        store.add_component("Booking API", "Books slots", "architect")
        store.add_decision("Slots", "Unique index", "Stops double booking", [], "architect", challenge="Races")
        for _ in range(2):
            store.count_design_round("architect")
        store.write_design("architect")
        store.confirm_requirements("pm")
        store.add_tasks(["Booking"], "worker")
        store.set_task_status("T1", "done", "coder")
        for kind in ("develop", "review"):
            job = store.add_job(kind, "")
            store.update_job(job.id, "done")
        store.add_findings(findings, "reviewer")
        return calls, call

    async def test_the_architect_rules_only_after_asking_the_worker(self, tmp_path):
        calls, call = self._reviewed(
            tmp_path, ["app/booking.py: no check for a taken slot", "app/api.py: no rate limit"]
        )
        architect = TeamTools(calls, "architect")

        async with Session("room_01"):
            too_soon = architect.rule_on_finding("F1", "fix", "Two people could book one slot")
            call.store.count_review_round("architect")  # the Architect put the findings to the Worker
            fixed = architect.rule_on_finding("F1", "fix", "Two people could book one slot")
            waived = architect.rule_on_finding("F2", "waive", "Not in the requirements")
            again = architect.rule_on_finding("F2", "fix", "Changed my mind")
            unknown = architect.rule_on_finding("F9", "fix", "?")

        assert too_soon.startswith("Not ruled: put the findings to the Worker first")
        assert fixed == "Ruled on F1. Still to rule on: F2."
        assert waived == "Ruled on F2."
        assert again == "Not ruled: F2 is already waived." and unknown.startswith("Not ruled: there is no finding F9")
        assert [(f.status, f.reason) for f in call.store.state.findings] == [
            ("to fix", "Two people could book one slot"),
            ("waived", "Not in the requirements"),
        ]
        assert "F2 app/api.py: no rate limit (Not in the requirements)" in call.store.overview()

    async def test_the_architect_approves_once_the_review_is_settled_and_acceptance_waits_for_it(self, tmp_path):
        calls, call = self._reviewed(tmp_path, ["app/api.py: no rate limit"])
        architect = TeamTools(calls, "architect")

        async with Session("room_01"):
            early = architect.approve_code("Approved")
            call.store.count_review_round("architect")
            architect.rule_on_finding("F1", "waive", "Not in the requirements")
            blocked = TeamTools(calls, "pm").check_acceptance()
            approved = architect.approve_code("Approved; F1 waived: not in the requirements")

        assert "are still open, waiting for the Architect's ruling" in early
        assert "The Architect calls approve_code first" in blocked
        assert approved == "Code approved."
        assert call.gates.blocker("check_acceptance") is None
        assert "The Architect approved the code." in call.store.overview()
        assert "Approved the code: Approved; F1 waived" in call.store.state.log[-1].what

    async def test_the_review_conversation_counts_rounds_and_stops_after_two(self, tmp_path):
        calls, call = self._reviewed(tmp_path, ["app/booking.py: no check for a taken slot"])
        team = VoiceTeam(calls)
        transfer = SimpleNamespace(name="transfer_to_agent")

        def handover(sender, receiver):
            context = SimpleNamespace(agent_name=sender, actions=SimpleNamespace(transfer_to_agent=receiver))
            refused = team.limit_handoffs(transfer, {"agent_name": receiver}, context)
            if refused:
                return refused["result"]
            return team.brief_handover(transfer, {"agent_name": receiver}, context, None)["result"]

        call.holder = "architect"  # the review's findings came to the Architect
        async with Session("room_01"):
            first = handover("architect", "worker")
            back = handover("worker", "architect")
            handover("architect", "worker")
            handover("worker", "architect")
            third = handover("architect", "worker")
            TeamTools(calls, "architect").rule_on_finding("F1", "fix", "Two people could book one slot")
            to_fix = handover("architect", "worker")  # the fix, not another round

        assert call.store.state.review_rounds == 2
        # Only the Architect hears about rounds; the Worker hears what is waiting for it, as facts.
        assert "the Architect is reviewing the code with you and is waiting for your answer" in first
        assert "round" not in first
        assert "waiting for the Architect's ruling (code review round 1 of at most 2)" in back
        assert third.startswith("Not handed over: the review findings have had 2 rounds of discussion")
        assert "findings F1 are to be fixed. The Worker fixes them with start_development" in to_fix
        assert "Code review: round 2" in ProjectDocuments.log(call.store.state)
        from server import BoardSnapshot

        snapshot = BoardSnapshot.of(call, 0.0)
        assert snapshot.phase == "Code review" and not snapshot.code_approved
        assert [(f.status, f.reason) for f in snapshot.findings] == [("to fix", "Two people could book one slot")]
        status = ProjectDocuments.status(call.store.state)
        assert (
            "[to fix] app/booking.py: no check for a taken slot _(the Architect: Two people could book one slot)_"
            in status
        )
        assert "**Code approved by the Architect:** no" in status

    def test_the_review_job_starts_the_conversation(self, tmp_path, monkeypatch):
        call = TeamCall("room_01", tmp_path)
        call.link = CallLink("room_01", connection=lambda: None)
        notices = []
        monkeypatch.setattr(call, "notify", notices.append)
        monkeypatch.setattr(call.gates, "blocker", lambda step, items=None: None)

        def reviewer(selected, prompt):
            call.store.add_findings(["app/booking.py: no check for a taken slot"], "reviewer")

        call.jobs = DevJobRunner(call, service_factory=lambda: FakeService(on_run=reviewer, call=call))
        call.jobs.submit("review")
        deadline = time.time() + 5
        while call.jobs.busy and time.time() < deadline:
            time.sleep(0.02)

        assert "Next: the code review. The Architect puts F1 to the Worker as questions about their code" in notices[-1]
        assert "rule_on_finding" in notices[-1]

    async def test_the_coder_records_build_notes_the_worker_answers_from(self, tmp_path):
        calls = TeamCalls(tmp_path)
        call = calls.get("room_01")
        call.link.stop()
        call.store.start("Shop", "Sell juice.", "pm")

        async with Session("room_01-dev"):
            reply = TeamTools(calls, "coder").record_build_note("Prices are stored in cents", "No rounding errors")

        assert reply == "Noted N1."
        assert "N1 Prices are stored in cents (No rounding errors)" in call.store.overview()
        assert "- N1: Prices are stored in cents (No rounding errors)" in call.store.brief()
        assert "## Build notes" in ProjectDocuments.design(call.store.state)


class TestProcessGates:
    """The order holds even when a model skips ahead: each step says what is missing and who should act."""

    def test_each_step_waits_for_the_one_before_it(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        gates = ProcessGates(store)

        assert "no requirements are recorded" in gates.blocker("confirm_requirements")
        store.add_requirement("Sign up", "Users sign up with email.", ["A user can sign up"], "pm")
        # The client approves the requirements and the design together, once the design is written.
        assert "the design is not written yet" in gates.blocker("confirm_requirements")

        blocked = gates.blocker("create_tasks")
        assert "the client has not approved the requirements" in blocked and "write_design" in blocked
        assert "record_decision" in gates.blocker("write_design")
        store.add_component("Auth service", "Sign-up and login", "architect")
        store.add_decision("Passwords", "bcrypt", "Slow hashes resist brute force", ["SHA-256: too fast"], "architect")
        assert "the challenge behind decisions D1" in gates.blocker("write_design")
        store.add_decision(
            "Passwords",
            "bcrypt",
            "Slow hashes resist brute force",
            ["SHA-256: too fast"],
            "architect",
            challenge="A leaked database exposes passwords: bcrypt makes cracking them slow",
        )
        assert [d.id for d in store.state.decisions] == ["D1"]  # the same title revises D1
        store.write_design("architect")
        assert gates.blocker("confirm_requirements") is None
        assert "the client has not approved the requirements" in gates.blocker("start_development")
        store.confirm_requirements("pm")
        assert gates.blocker("create_tasks") is None
        assert "no tasks yet" in gates.blocker("start_development")
        store.add_tasks(["Sign-up"], "worker")
        assert gates.blocker("start_development") is None

        assert "nothing has been built" in gates.blocker("request_review")
        develop = store.add_job("develop", "T1")
        store.update_job(develop.id, "done")
        assert gates.blocker("request_review") is None

        assert "not been reviewed" in gates.blocker("check_acceptance")
        review = store.add_job("review", "")
        store.update_job(review.id, "done")
        store.add_findings(["app/auth.py: passwords are logged"], "reviewer")
        assert "F1 are still open" in gates.blocker("check_acceptance")
        assert "not ruled to be fixed" in gates.blocker("start_development", ["F1"])
        store.set_finding_status("F1", "to fix", "architect", "Passwords end up in the logs")
        assert gates.blocker("start_development", ["F1"]) is None
        store.set_finding_status("F1", "resolved", "reviewer")
        assert "approve_code first" in gates.blocker("check_acceptance")
        assert gates.blocker("approve_code") is None
        store.approve_code("Approved after one fix", "architect")
        assert gates.blocker("check_acceptance") is None

        # Live: delivered with every task "in progress", because the Coder never reported one done.
        assert "tasks T1 are not done" in gates.blocker("deliver")
        store.set_task_status("T1", "done", "coder")
        assert "check_acceptance first" in gates.blocker("deliver")

    def test_the_next_step_names_what_the_project_waits_for_and_who_does_it(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        gates = ProcessGates(store)

        assert "The PM gathers them" in gates.next_step()
        store.add_requirement("Sign up", "Users sign up with email.", ["A user can sign up"], "pm")
        assert "The Architect proposes it to the Worker and they debate it, 2 to 4 rounds" in gates.next_step()
        store.write_design("architect")
        assert "The PM asks them to look them over on their page" in gates.next_step()
        store.confirm_requirements("pm")
        assert "The Worker creates the tasks" in gates.next_step()
        store.add_tasks(["Sign-up"], "worker")
        develop = store.add_job("develop", "T1")
        assert gates.next_step().startswith("J1 (develop) is running, and nothing is needed from anyone")
        store.update_job(develop.id, "done")
        assert "tasks T1 are not done. The Worker starts development" in gates.next_step()
        store.set_task_status("T1", "done", "coder")
        assert "The Architect requests a review first" in gates.next_step()
        review = store.add_job("review", "")
        store.update_job(review.id, "done")
        assert "The Architect calls approve_code first" in gates.next_step()
        store.approve_code("Clean review", "architect")
        assert "The PM runs check_acceptance first" in gates.next_step()
        store.set_acceptance(
            [AcceptanceResult(criterion_id="R1.1", criterion="sign up", test="t", result="pass")], "reviewer"
        )
        assert "The Worker delivers the project" in gates.next_step()
        store.mark_delivered()
        assert "the project is delivered" in gates.next_step()

    def test_the_design_is_debated_for_two_to_four_rounds(self, tmp_path):
        """The client wants the Architect and the Worker to argue the design out, not agree in one polite exchange."""
        store = ProjectStore(tmp_path, "room_01")
        gates = ProcessGates(store)
        store.add_requirement("Booking", "Book a slot.", ["A slot cannot be booked twice"], "pm")
        store.add_component("Booking API", "Books slots", "architect")
        store.add_decision("Slots", "Unique index", "Stops double booking", ["Check then insert"], "architect", "Races")

        assert "at least 2 rounds and so far 0" in gates.blocker("write_design")
        assert store.count_design_round("architect") == 1
        assert "so far 1" in gates.blocker("write_design")
        store.count_design_round("architect")
        assert gates.blocker("write_design") is None

        assert gates.handover_blocker("architect", "worker") is None
        store.count_design_round("architect")
        store.count_design_round("architect")
        assert "the design has had 4 rounds of debate" in gates.handover_blocker("architect", "worker")
        assert gates.handover_blocker("worker", "architect") is None  # the Worker still answers the last round
        assert "Design debate: round 4" in ProjectDocuments.log(store.state)

        store.write_design("architect")
        assert "**Debated:** 4 rounds between the Architect and the Worker" in ProjectDocuments.design(store.state)
        assert gates.handover_blocker("architect", "worker") is None  # a fix after the build, not the debate
        store.add_decision("Cancel", "Soft delete", "Keeps history", [], "architect", "Lost bookings")
        assert store.state.design_rounds == 0  # a design changed after it was written is debated again

    @staticmethod
    def _built_and_reviewed(tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        store.add_requirement(
            "Booking", "Book a slot.", ["A slot can be booked", "A slot cannot be booked twice"], "pm"
        )
        for kind in ("develop", "review"):
            job = store.add_job(kind, "")
            store.update_job(job.id, "done")
        store.approve_code("Clean review", "architect")
        return store, ProcessGates(store)

    @staticmethod
    def _accept(store, results):
        store.set_acceptance(
            [AcceptanceResult(criterion_id=c, criterion=c, test="t", result=r) for c, r in results.items()], "reviewer"
        )

    def test_a_fix_blocks_acceptance_until_it_is_re_reviewed(self, tmp_path):
        """Review: a fixed finding is only the Coder's word; the Reviewer must confirm it."""
        store, gates = self._built_and_reviewed(tmp_path)
        store.add_findings(["app/booking.py: two people can book one slot"], "reviewer")
        store.set_finding_status("F1", "fixed", "coder")

        blocked = gates.blocker("check_acceptance")

        assert "F1 have not been re-reviewed" in blocked and "request_review" in blocked
        store.set_finding_status("F1", "resolved", "reviewer")
        assert gates.blocker("check_acceptance") is None

    def test_delivery_needs_every_criterion_to_pass(self, tmp_path):
        """Review: delivery used to need acceptance results to exist, not to pass."""
        store, gates = self._built_and_reviewed(tmp_path)
        self._accept(store, {"R1.1": "pass", "R1.2": "fail"})

        assert "R1.2 do not pass" in gates.blocker("deliver")

        self._accept(store, {"R1.1": "pass", "R1.2": "pass"})
        assert gates.blocker("deliver") is None

    def test_a_new_requirement_needs_a_new_acceptance_check(self, tmp_path):
        """Review: password reset added after login passed acceptance must not ride on the old results."""
        store, gates = self._built_and_reviewed(tmp_path)
        self._accept(store, {"R1.1": "pass", "R1.2": "pass"})

        store.add_requirement("Reset", "Reset a password.", ["A reset link expires after 30 minutes"], "pm")

        assert "R2.1 were added after the last acceptance check" in gates.blocker("deliver")

    def test_acceptance_needs_a_review_of_the_latest_build(self, tmp_path):
        """Review: a build after the last review went to acceptance with its changes unreviewed."""
        store, gates = self._built_and_reviewed(tmp_path)
        assert gates.blocker("check_acceptance") is None
        time.sleep(0.01)

        job = store.add_job("develop", "T2")
        store.update_job(job.id, "done")

        assert "the latest build has not been reviewed" in gates.blocker("check_acceptance")

    def test_an_assumption_the_coder_made_alone_blocks_delivery_until_settled(self, tmp_path):
        """Review: an unanswered business question became the Coder's decision, and nobody had to confirm it."""
        store, gates = self._built_and_reviewed(tmp_path)
        self._accept(store, {"R1.1": "pass", "R1.2": "pass"})
        store.add_assumption("Times are shown in the shop's time zone.", "pm")
        store.add_assumption("A booking can be cancelled up to 24 hours before.", "coder", needs_settling=True)

        blocked = gates.blocker("deliver")

        assert "the Coder's assumptions A2 are not settled" in blocked and "settle_assumption" in blocked
        store.settle_assumption("A2", "The client confirmed it", "pm")
        assert gates.blocker("deliver") is None

    def test_code_built_after_the_acceptance_check_needs_another(self, tmp_path):
        store, gates = self._built_and_reviewed(tmp_path)
        self._accept(store, {"R1.1": "pass", "R1.2": "pass"})
        time.sleep(0.01)

        job = store.add_job("develop", "F1")
        store.update_job(job.id, "done")

        assert "changed after the last acceptance check" in gates.blocker("deliver")

    async def test_a_blocked_step_says_why_and_starts_nothing(self, tmp_path):
        calls = TeamCalls(tmp_path)
        calls.get("room_01").link.stop()

        async with Session("room_01"):
            worker = TeamTools(calls, "worker")
            reply = worker.start_development([])
            tasks = worker.create_tasks(["Sign-up"])

        assert reply.startswith("Not started: the client has not approved the requirements")
        assert tasks.startswith("Not created")
        assert calls.get("room_01").store.state.jobs == []
        assert calls.get("room_01").store.state.tasks == []


class TestRefusedJobs:
    async def test_a_refused_job_answers_the_voice_agent_instead_of_failing_the_tool(self, tmp_path, monkeypatch):
        calls = TeamCalls(tmp_path)
        call = calls.get("room_01")
        call.link.stop()

        def refuse(kind, items=None):
            raise JobRefused("Not started: the last 2 develop jobs failed.")

        monkeypatch.setattr(call.jobs, "submit", refuse)
        monkeypatch.setattr(call.gates, "blocker", lambda step, items=None: None)
        async with Session("room_01"):
            reply = TeamTools(calls, "worker").start_development([])

        assert reply == "Not started: the last 2 develop jobs failed."


class TestStallRecovery:
    """A voice model may name a teammate ("PM?") and stop without handing over; the call must not stay silent."""

    def test_every_voice_agent_gets_the_meeting_rules_and_its_role(self, tmp_path):
        pm = VoiceTeam(TeamCalls(tmp_path)).build()
        worker, architect = pm.sub_agents

        assert pm.instruction == f"{prompts.MEETING_RULES}\n\n{prompts.PM}"
        assert worker.instruction.endswith(prompts.WORKER) and architect.instruction.endswith(prompts.ARCHITECT)

    def test_a_call_is_watched_from_the_moment_it_is_seen(self, tmp_path):
        call = TeamCalls(tmp_path).get("room_01")
        try:
            assert call.link._thread is not None and call.link._thread.is_alive()
        finally:
            call.link.stop()

    def test_an_announced_but_not_called_handover_is_nudged(self, tmp_path):
        """Live: no project was started, the PM said "I'll bring in the Architect" and stopped, nothing was posted."""
        connection, clock = FakeConnection(), FakeClock()
        call = TeamCall("room_01", tmp_path)
        call.link = CallLink("room_01", should_nudge=call.should_nudge, connection=lambda: connection, clock=clock)
        connection.last_caller_speech_at = clock.now - 20
        connection.last_model_output_at = clock.now

        clock.now += CallLink.NUDGE_AFTER_SECONDS
        posted = call.link.tick()

        assert posted == CallLink.NUDGE
        assert "call the transfer tool now" in connection.sent[0]


class TestNudgeTiming:
    def test_no_nudge_before_the_client_has_spoken(self, tmp_path):
        """While the team greets and waits for the client's first words, silence is expected."""
        connection, clock = FakeConnection(), FakeClock()
        call = TeamCall("room_01", tmp_path)
        call.link = CallLink("room_01", should_nudge=call.should_nudge, connection=lambda: connection, clock=clock)
        connection.last_model_output_at = clock.now

        clock.now += 60

        assert call.link.tick() is None


class TestTeams:
    def test_the_pm_opens_the_meeting_with_the_worker_and_the_architect_as_sub_agents(self, tmp_path):
        """The PM is the entry agent, so the call does not start with a handover from the Worker to the PM."""
        team = VoiceTeam(TeamCalls(tmp_path))
        pm = team.build()

        assert pm.name == "pm"
        assert [a.name for a in pm.sub_agents] == ["worker", "architect"]
        assert all(a.model == VoiceTeam.MODEL for a in [pm, *pm.sub_agents])
        assert all(a.before_tool_callback == team.limit_handoffs for a in [pm, *pm.sub_agents])
        assert all(a.after_tool_callback == team.brief_handover for a in [pm, *pm.sub_agents])

    def test_the_background_team_is_adk_agents_on_a_litellm_model(self, tmp_path):
        """One framework for everyone: the Coder and the Reviewer are ADK agents on a text model through LiteLLM."""
        from google.adk.models.lite_llm import LiteLlm

        coder, reviewer = BackgroundTeam(TeamCalls(tmp_path)).build()

        assert (coder.name, reviewer.name) == ("coder", "reviewer")
        # Each run sees its brief and its own tool calls, not every earlier job's transcript.
        assert coder.include_contents == reviewer.include_contents == "none"
        assert isinstance(coder.model, LiteLlm) and coder.model.model == BackgroundTeam.MODEL == "openai/gpt-4.1"
        assert BackgroundTeam(TeamCalls(tmp_path), "anthropic/claude-x").build()[1].model.model == "anthropic/claude-x"

    async def test_a_refused_transfer_answers_in_the_tools_place(self, tmp_path):
        calls = TeamCalls(tmp_path)
        team = VoiceTeam(calls)
        calls.get("room_01").guard = HandoffGuard(lambda: None, limit=1)
        transfer = SimpleNamespace(name="transfer_to_agent")

        async with Session("room_01"):
            assert team.limit_handoffs(transfer, {"agent_name": "pm"}, None) is None
            assert "Handoff refused" in team.limit_handoffs(transfer, {"agent_name": "pm"}, None)["result"]
            assert team.limit_handoffs(SimpleNamespace(name="get_progress"), {}, None) is None

    async def test_a_transfer_tells_the_agent_taking_the_call_where_the_project_stands(self, tmp_path):
        """Live: the Architect handed the Worker the call mid-design without a question; the Worker said it could not
        build yet and handed it straight back. The transfer's result now says what the project is waiting for."""
        calls = TeamCalls(tmp_path)
        team = VoiceTeam(calls)
        calls.get("room_01").store.add_requirement("Sum", "Sum the primes to 100.", ["Prints 1060"], "pm")
        transfer = SimpleNamespace(name="transfer_to_agent")

        def context(target):
            return SimpleNamespace(agent_name="architect", actions=SimpleNamespace(transfer_to_agent=target))

        async with Session("room_01"):
            brief = team.brief_handover(transfer, {"agent_name": "worker"}, context("worker"), None)["result"]
            refused = team.brief_handover(
                transfer, {"agent_name": "worker"}, context(None), {"result": "Handoff refused"}
            )
            other = team.brief_handover(SimpleNamespace(name="get_progress"), {}, context(None), "ok")

        assert brief.startswith("The Worker has the call: you are the Worker now, so speak as yourself")
        assert "Where the project stands: the Architect is debating the design" in brief
        assert "round" not in brief  # told rounds, the Worker acted out every round alone
        assert refused is None and other is None

    async def test_handovers_in_the_design_debate_count_rounds_and_stop_after_the_last(self, tmp_path):
        calls = TeamCalls(tmp_path)
        team = VoiceTeam(calls)
        call = calls.get("room_01")
        call.link.stop()
        call.store.add_requirement("Sum", "Sum the primes to 100.", ["Prints 1060"], "pm")
        transfer = SimpleNamespace(name="transfer_to_agent")

        def handover(sender, receiver):
            context = SimpleNamespace(agent_name=sender, actions=SimpleNamespace(transfer_to_agent=receiver))
            refused = team.limit_handoffs(transfer, {"agent_name": receiver}, context)
            if refused:
                return refused["result"]
            return team.brief_handover(transfer, {"agent_name": receiver}, context, None)["result"]

        call.holder = "architect"
        async with Session("room_01"):
            briefs = [handover(*pair) for pair in [("architect", "worker"), ("worker", "architect")] * 4]
            last = handover("architect", "worker")

        assert call.store.state.design_rounds == 4
        # Briefs alternate: to the Worker (facts, no rounds), then back to the Architect (the round).
        assert all("debating the design with you" in brief and "round" not in brief for brief in briefs[0::2])
        assert "round 1 of 2 to 4, with at least one more round" in briefs[1]
        assert "round 2 of 2 to 4, with more rounds only if a point is still open" in briefs[3]
        assert "round 4, the last" in briefs[7]
        assert last.startswith("Not handed over: the design has had 4 rounds of debate")


def test_output_folder_defaults_to_output(monkeypatch):
    from server import SoftwareTeamServer

    monkeypatch.delenv("TEAM_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("TEAM_TEXT_MODEL", raising=False)

    server = SoftwareTeamServer.from_env()

    assert server.calls.output_root == Path("./output")
    assert server.text_model is None


class TestServer:
    """Every agent is a Google ADK agent; only the entry agent of the voice team and the background agents are registered."""

    @staticmethod
    def _loaded(server):
        from agentkernel.core.runtime import Runtime
        from agentkernel.core.session.in_memory import InMemorySessionStore

        with Runtime(InMemorySessionStore()) as runtime:
            server.load()
            return runtime.agents()

    def test_every_agent_is_a_google_adk_agent(self, tmp_path):
        from server import SoftwareTeamServer

        agents = self._loaded(SoftwareTeamServer(tmp_path))

        assert {name: type(agent).__name__ for name, agent in agents.items()} == {
            "pm": "GoogleADKAgent",
            "coder": "GoogleADKAgent",
            "reviewer": "GoogleADKAgent",
        }

    def test_the_text_model_comes_from_the_environment(self, tmp_path, monkeypatch):
        from server import SoftwareTeamServer

        monkeypatch.setenv("TEAM_TEXT_MODEL", "anthropic/claude-sonnet-5-5")
        server = SoftwareTeamServer.from_env()

        assert server.text_model == "anthropic/claude-sonnet-5-5"
        assert self._loaded(server)["coder"].agent.model.model == "anthropic/claude-sonnet-5-5"


class TestActivityFeed:
    """Every background tool call becomes progress in plain words, without relying on the agent to report it."""

    def test_tool_calls_are_described_in_plain_words(self):
        describe = ActivityFeed.describe
        tests_ran = json.dumps({"stdout": "....\n4 passed in 0.10s\n", "exit_code": 0})

        assert describe("write_sandbox_file", {"path": "app/main.py"}, "{}") == "wrote app/main.py"
        assert (
            describe("run_command", {"command": "python -m pytest -q"}, tests_ran) == "ran the tests: 4 passed in 0.10s"
        )
        assert (
            describe("run_command", {"command": "pip install -q -r requirements.txt"}, "{}")
            == "installed the dependencies"
        )
        assert describe("report_progress", {"message": "x"}, "Noted.") is None

    def test_a_recorded_call_is_project_progress(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        feed = ActivityFeed(store, clock=FakeClock())

        feed.record_tool("coder", "write_sandbox_file", {"path": "app/main.py"}, "{}")

        assert feed.last.text == "wrote app/main.py"
        assert store.state.progress[-1].what == "wrote app/main.py"

    async def test_model_waits_become_activity(self, tmp_path):
        """Live: the SDKs retried rate-limited requests silently, and the call could not tell waiting from stuck."""
        calls = TeamCalls(tmp_path)
        call = calls.get("room_01")
        call.link.stop()
        watcher = ModelWaitWatcher(calls)

        def line(logger, message):
            return logging.LogRecord(logger, logging.INFO, __file__, 0, message, None, None)

        watcher.emit(
            line("openai._base_client", "Retrying request to /responses in 2.0 seconds")
        )  # no session: ignored
        async with Session("room_01-dev"):
            # The wording of the openai client's own retry log line, which LiteLLM's OpenAI calls go through.
            watcher.emit(line("openai._base_client", "Retrying request to /chat/completions in 15.900000 seconds"))
            watcher.emit(line("openai._base_client", "Retrying request to /chat/completions in 0.432 seconds"))
            watcher.emit(line("openai._base_client", "HTTP Request: POST https://api.openai.com/v1/chat/completions"))

        entries = call.activity.since(0)
        assert [entry.text for entry in entries] == [
            "is waiting 16 s for the model's rate limit",
            "is waiting 1 s for the model's rate limit",
        ]
        assert ModelWaitWatcher.LOGGERS == ("openai._base_client",)


class TestNoNarration:
    """Running work is followed on the board, not narrated: every post into a live call makes the agent speak."""

    def test_a_running_job_is_not_narrated_into_the_call(self, tmp_path):
        connection, clock = FakeConnection(), FakeClock()
        call = TeamCall("room_01", tmp_path)
        call.link.stop()
        call.link = CallLink("room_01", should_nudge=call.should_nudge, connection=lambda: connection, clock=clock)
        job = call.store.add_job("develop", "T1")
        call.store.update_job(job.id, "running")
        call.jobs._active = job.id  # the job runner is busy with it
        call.activity.record_tool("coder", "write_sandbox_file", {"path": "app/main.py"}, "{}")
        connection.last_caller_speech_at = clock.now - 600
        connection.last_model_output_at = clock.now - 600

        assert call.link.tick() is None and connection.sent == []
        assert "wrote app/main.py" in call.store.progress_report()  # what an agent asked "how is it going?" tells


class TestActivityCallback:
    """The background agents feed the activity feed from ADK's after-tool callback."""

    async def test_background_agents_record_through_after_tool_callback(self, tmp_path):
        calls = TeamCalls(tmp_path)
        calls.get("room_01").link.stop()
        team = BackgroundTeam(calls)
        assert all(agent.after_tool_callback == team.record_activity for agent in team.build())

        async with Session("room_01-dev"):
            kept = team.record_activity(
                SimpleNamespace(name="write_sandbox_file"),
                {"path": "app/x.py"},
                SimpleNamespace(agent_name="coder"),
                "{}",
            )

        assert kept is None
        assert calls.get("room_01").activity.last.text == "wrote app/x.py"

    async def test_starting_development_returns_at_once(self, tmp_path, monkeypatch):
        calls = TeamCalls(tmp_path)
        call = calls.get("room_01")
        call.link.stop()
        monkeypatch.setattr(call.gates, "blocker", lambda step, items=None: None)
        monkeypatch.setattr(call.jobs, "submit", lambda kind, items=None: "J1")

        async with Session("room_01"):
            reply = TeamTools(calls, "worker").start_development(["T1"])

        assert reply.startswith("Started development job J1")
        assert "minutes" not in reply


class TestProjectBoard:
    """The board shows the client what the team recorded, and keeps every page in the room up to date."""

    @pytest.fixture
    def call(self, tmp_path):
        call = TeamCall("room_01", tmp_path)
        call.link = CallLink("room_01", connection=lambda: None)
        return call

    def test_the_snapshot_shows_what_was_recorded(self, call):
        from server import BoardSnapshot

        store = call.store
        store.start("Salon booking", "Customers book a stylist.", "pm")
        store.add_requirement(
            "Booking", "Book a slot.", ["A slot can be booked", "A slot cannot be booked twice"], "pm"
        )
        store.add_decision("Slots", "Unique index", "Stops double booking", [], "architect", challenge="Races: index")
        store.set_acceptance(
            [AcceptanceResult(criterion_id="R1.1", criterion="c", test="t", result="pass")], "reviewer"
        )

        snapshot = BoardSnapshot.of(call, 100.0)

        assert snapshot.project == "Salon booking" and snapshot.phase == "Agreeing what to build"
        [requirement] = snapshot.requirements
        assert [(c.id, c.result) for c in requirement.criteria] == [("R1.1", "pass"), ("R1.2", "not checked")]
        assert snapshot.decisions[0].challenge == "Races: index"

    def test_the_snapshot_says_what_the_team_is_waiting_for(self, call):
        from server import BoardSnapshot

        store = call.store
        store.add_progress("Asked the client: Can a customer cancel?", "pm")
        call.ask_client()
        store.add_question("Do stylists have breaks?", "pm")
        store.add_assumption("A booking lasts one hour.", "coder", needs_settling=True)
        store.add_assumption("Prices are in rupees.", "pm")

        assert BoardSnapshot.of(call, 0.0).waiting_for_you == [
            "Can a customer cancel?",
            "Do stylists have breaks?",
            "Confirm: A booking lasts one hour.",
        ]

    def test_a_running_job_shows_its_latest_activity(self, call):
        from server import BoardSnapshot

        job = call.store.add_job("develop", "T1")
        call.store.update_job(job.id, "running")
        started = call.store.state.jobs[0].started_at
        call.activity = ActivityFeed(call.store, clock=lambda: started + 1)
        for path in ("app/a.py", "app/b.py"):
            call.activity.record_tool("coder", "write_sandbox_file", {"path": path}, "{}")
        call.jobs.running = lambda: call.store.state.jobs[0]

        snapshot = BoardSnapshot.of(call, started + 30)

        assert snapshot.phase == "Building"
        assert snapshot.job.seconds == 30
        assert snapshot.job.activity == ["coder wrote app/a.py", "coder wrote app/b.py"]

    def test_the_steps_follow_the_process_and_the_recent_log_is_shown(self, call):
        from server import BoardSnapshot

        def steps():
            return {s.name: s.state for s in BoardSnapshot.of(call, 0.0).steps}

        store = call.store
        assert steps() == {
            "Requirements": "current", "Design": "todo", "Approval": "todo", "Build": "todo",
            "Review": "todo", "Acceptance": "todo", "Delivered": "todo",
        }  # fmt: skip

        store.add_requirement("Booking", "Book a slot.", ["A slot can be booked"], "pm")
        assert steps()["Requirements"] == "current"  # still being gathered until the design is written
        store.add_component("Booking service", "Books slots", "architect")
        store.add_decision("Slots", "Unique index", "Stops double booking", [], "architect", challenge="Races")
        store.write_design("architect")
        assert steps()["Design"] == "done" and steps()["Approval"] == "current"
        assert BoardSnapshot.of(call, 0.0).phase == "Waiting for your approval"
        store.confirm_requirements("pm")
        store.add_tasks(["Booking"], "worker")
        assert steps()["Approval"] == "done" and steps()["Build"] == "current"

        store.set_task_status("T1", "done", "coder")
        job = store.add_job("develop", "T1")
        store.update_job(job.id, "done")
        assert steps()["Build"] == "done" and steps()["Review"] == "current"

        recent = BoardSnapshot.of(call, 0.0).recent
        assert recent[-1].who == "jobs" and "J1 (develop) is done" in recent[-1].what
        assert len(recent) <= BoardSnapshot.MAX_RECENT

    def test_listeners_hear_a_change_once_it_is_written_and_a_failing_one_does_not_fail_it(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        heard = []
        store.on_change(lambda: heard.append((store.folder / "requirements.md").read_text()))
        store.on_change(lambda: 1 / 0)

        store.add_requirement("Booking", "Book a slot.", ["A slot can be booked"], "pm")
        store.flush()

        assert heard and "Booking" in heard[-1]  # the documents already show the change
        assert store.state.requirements[0].title == "Booking"

    async def test_changes_are_sent_at_most_once_an_interval(self, call):
        from server import ProjectBoard, RoomAccess

        sent = []

        class FakeParticipant:
            async def send_text(self, text, topic, destination_identities=None):
                sent.append((json.loads(text)["project"], topic, destination_identities))

        # The agents' participants are in the room too; only the client's page is sent the board.
        room = SimpleNamespace(
            local_participant=FakeParticipant(), remote_participants={"agent-pm": object(), "client": object()}
        )
        board = ProjectBoard(call, RoomAccess("room_01", "", "", ""))
        board.MIN_INTERVAL_SECONDS = 0.2
        serving = asyncio.ensure_future(board.serve(room))
        await asyncio.sleep(0.05)
        assert sent == [("New project", "ak.project", ["client"])]  # the current state, at once

        for _ in range(3):
            call.store.add_progress("wrote app/main.py", "coder")
        await asyncio.to_thread(call.store.flush)
        await asyncio.sleep(0.05)
        call.store.start("Salon booking", "Customers book a stylist.", "pm")
        await asyncio.to_thread(call.store.flush)
        await asyncio.sleep(0.4)
        serving.cancel()

        assert len(sent) == 3  # the first burst in one send, the change during the wait in the next
        assert sent[-1][0] == "Salon booking"

        await board.send(room, ["late-viewer"])
        assert sent[-1][2] == ["late-viewer"]

        before = len(sent)
        await board.send(SimpleNamespace(local_participant=FakeParticipant(), remote_participants={"agent-pm": 1}))
        assert len(sent) == before  # nobody watching: nothing sent

    async def test_the_clients_words_from_the_log_reach_their_page(self, call):
        """Agent Kernel logs each caller turn ("Caller said: ..."); the board passes it on, no core change needed."""
        from server import ClientWordsWatcher, ProjectBoard, RoomAccess

        sent = []

        class FakeParticipant:
            async def send_text(self, text, topic, destination_identities=None):
                sent.append((json.loads(text), topic, destination_identities))

        room = SimpleNamespace(local_participant=FakeParticipant(), remote_participants={"agent-pm": 1, "Pulindu": 2})
        board = ProjectBoard(call, RoomAccess("room_01", "", "", ""))
        board._loop, board._room = asyncio.get_running_loop(), room
        watcher = ClientWordsWatcher(board)

        def line(message):
            return logging.LogRecord("ak.adk.realtime", logging.INFO, __file__, 0, message, None, None)

        watcher.emit(line("Caller said: I need a booking system."))
        watcher.emit(line("Connecting to Gemini Live (agent=pm)"))  # not the caller's words
        await asyncio.sleep(0.05)

        assert sent == [({"text": "I need a booking system."}, "ak.client", ["Pulindu"])]

    def test_no_board_without_livekit_settings(self, call):
        from server import ProjectBoard, RoomAccess

        board = ProjectBoard(call, RoomAccess("room_01", "", "", ""))
        board.start()

        assert board._thread is None

    def test_the_livekit_sdk_is_ready_before_the_board_thread_starts(self, call, monkeypatch):
        """Live: the board's and the gateway's threads first used the SDK at once, and the gateway's room never connected."""
        from server import ProjectBoard, RoomAccess

        order = []
        monkeypatch.setattr(ProjectBoard, "prepare_livekit", staticmethod(lambda: order.append("sdk ready")))
        monkeypatch.setattr(ProjectBoard, "_run", lambda self: order.append("board thread"))
        board = ProjectBoard(call, RoomAccess("room_01", "wss://x", "key", "secret"))

        board.start()
        board._thread.join(timeout=5)

        assert order == ["sdk ready", "board thread"]


class TestClientPage:
    """The page joins as the client: it talks and listens, and sends no data the agents would take as the client's words."""

    ACCESS = ("room_01", "wss://example.livekit.cloud", "key", "secret-of-at-least-32-characters-long")

    def test_the_tokens_grant_what_each_participant_needs(self):
        from livekit import api

        from server import RoomAccess

        access = RoomAccess(*self.ACCESS)
        verify = api.TokenVerifier("key", "secret-of-at-least-32-characters-long").verify

        board = verify(access.board_token())
        client = verify(access.client_token())

        assert board.identity == "project-board" and board.video.room == "room_01"
        assert board.video.can_publish is False and board.video.can_publish_data is True
        assert client.identity == "Client" and client.name == "Client"  # no name given
        # The client may type: links and names are easier written than said.
        assert client.video.can_publish is True and client.video.can_publish_data is True

    def test_the_client_joins_under_their_name(self):
        from livekit import api

        from server import RoomAccess

        access = RoomAccess(*self.ACCESS)
        verify = api.TokenVerifier("key", "secret-of-at-least-32-characters-long").verify

        pulindu = verify(access.client_token("  Pulindu   Vidmal "))
        # Capitals kept: the gateway greets the caller by their identity ("A user named Pulindu-Vidmal").
        assert (pulindu.identity, pulindu.name) == ("Pulindu-Vidmal", "Pulindu Vidmal")
        assert RoomAccess.client_identity("Pulindu") == "Pulindu"
        # A name that would look like an agent or the board is kept apart from them.
        assert RoomAccess.client_identity("Agent Smith") == "client-Agent-Smith"
        assert RoomAccess.client_identity("Project board") == "client-Project-board"
        assert RoomAccess.client_identity("!!!") == "client"

    def test_the_token_route(self, tmp_path):
        from fastapi import HTTPException

        from server import ClientPage, RoomAccess

        reply = ClientPage(RoomAccess(*self.ACCESS), TeamCalls(tmp_path), tmp_path).token("Pulindu")
        assert reply["url"] == "wss://example.livekit.cloud" and reply["room"] == "room_01" and reply["token"]
        assert reply["name"] == "Pulindu"

        with pytest.raises(HTTPException) as unset:
            ClientPage(RoomAccess("room_01", "", "", ""), TeamCalls(tmp_path), tmp_path).token()
        assert unset.value.status_code == 503

    def test_an_unbuilt_page_says_how_to_build_it(self, tmp_path):
        from fastapi import HTTPException

        from server import ClientPage, RoomAccess

        page = ClientPage(RoomAccess(*self.ACCESS), TeamCalls(tmp_path), tmp_path)

        assert page.index().status_code == 503 and b"npm run dev" in page.index().body
        with pytest.raises(HTTPException):
            page.asset("index.js")
        (tmp_path / "index.html").write_text("<div id=root></div>")
        assert page.index().status_code == 200


class TestClientCorrections:
    """Live: "cosmetics" was heard as "cost metrics"; the client corrected it, and nothing could change the record."""

    def test_a_revised_requirement_keeps_its_id_and_needs_confirming_and_checking_again(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        store.add_requirement("Display cost metrics", "Show costs.", ["Costs are shown"], "pm")
        store.add_requirement("Manage inventory", "Track stock.", ["Stock is tracked"], "pm")
        store.confirm_requirements("pm")
        store.set_acceptance(
            [
                AcceptanceResult(criterion_id="R1.1", criterion="c", test="t", result="pass"),
                AcceptanceResult(criterion_id="R2.1", criterion="c", test="t", result="pass"),
            ],
            "reviewer",
        )

        store.revise_requirement(
            "R1", "Display cosmetic items", "Show the shop's items.", ["At least three items are shown"], "pm"
        )

        state = store.state
        assert [(r.id, r.title) for r in state.requirements] == [
            ("R1", "Display cosmetic items"),
            ("R2", "Manage inventory"),
        ]
        assert state.requirements_confirmed is False
        assert [a.criterion_id for a in state.acceptance] == ["R2.1"]  # R1's old result no longer counts

    def test_a_dropped_requirements_id_is_never_reused(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        for title in ("A", "B", "C"):
            store.add_requirement(title, title, [f"{title} works"], "pm")

        store.drop_requirement("R3", "Not wanted", "pm")
        store.drop_requirement("R1", "Not wanted", "pm")
        added = store.add_requirement("D", "D", ["D works"], "pm")

        assert [r.id for r in store.state.requirements] == ["R2", "R4"]
        assert added.id == "R4"
        store.flush()
        assert "Dropped requirement R1 (A): Not wanted" in (store.folder / "log.md").read_text()

    async def test_the_pm_corrects_the_record_with_tools(self, tmp_path):
        calls = TeamCalls(tmp_path)
        calls.get("room_01").link.stop()
        pm = TeamTools(calls, "pm")

        async with Session("room_01"):
            pm.record_requirement("Display cost metrics", "Show costs.", ["Costs are shown"])
            revised = pm.revise_requirement("R1", "Display cosmetic items", "Show items.", ["Three items are shown"])
            missing = pm.drop_requirement("R9", "No such thing")

        assert revised.startswith("Revised R1 (Display cosmetic items) with criteria R1.1")
        assert missing.startswith("Not dropped: there is no requirement R9")
        assert calls.get("room_01").store.state.requirements[0].acceptance_criteria == ["Three items are shown"]

    async def test_a_confirmed_project_is_not_started_over(self, tmp_path):
        """Live: the model connection dropped, the new one remembered nothing, and the PM renamed the project "project ideas"."""
        calls = TeamCalls(tmp_path)
        call = calls.get("room_01")
        call.link.stop()
        pm = TeamTools(calls, "pm")

        async with Session("room_01"):
            pm.start_project("Metrics dashboard", "Track metrics.")
            renamed = pm.start_project(
                "Cosmetics shop", "Track cosmetics stock and sales."
            )  # allowed: not confirmed yet
            pm.record_requirement("Manage inventory", "Track stock.", ["Stock is tracked"])
            call.store.confirm_requirements("pm")
            refused = pm.start_project("Project ideas", "Brainstorm.")

        assert renamed.startswith("Project 'Cosmetics shop' started")
        assert refused.startswith("Not started: the project 'Cosmetics shop' is already under way")
        assert "call get_project_state and continue it" in refused
        assert call.store.state.name == "Cosmetics shop"

    def test_the_pm_has_the_tools_and_is_told_to_use_them(self, tmp_path):
        names = [tool.__name__ for tool in TeamTools(TeamCalls(tmp_path), "pm").tools()]

        assert "revise_requirement" in names and "drop_requirement" in names


class TestProjectFiles:
    """The client views and downloads the project's files, and shares files with the team, from the page."""

    @pytest.fixture
    def client(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from server import ClientPage, RoomAccess

        calls = TeamCalls(tmp_path / "output")
        call = calls.get("room_01")
        call.link.stop()
        call.notices = []
        call.notify = call.notices.append
        call.store.start("Salon booking", "Customers book a stylist.", "pm")
        call.store.add_requirement("Booking", "Book a slot.", ["A slot can be booked"], "pm")
        call.store.flush()
        (call.store.folder / "code" / "app").mkdir(parents=True)
        (call.store.folder / "code" / "app" / "main.py").write_text("print('hi')\n")
        app = FastAPI()
        page = ClientPage(
            RoomAccess("room_01", "wss://x", "key", "secret-of-at-least-32-characters-long"), calls, tmp_path
        )
        app.include_router(page.router())
        return TestClient(app), call

    def test_the_files_are_listed_documents_first_and_read_one_by_one(self, client):
        http, call = client

        files = http.get("/project/files").json()["files"]

        assert files[:4] == ["design.md", "log.md", "requirements.md", "status.md"]
        assert "code/app/main.py" in files
        assert http.get("/project/files/code/app/main.py").text == "print('hi')\n"
        assert http.get("/project/files/../../secret.txt").status_code == 404  # never outside the folder
        assert http.get("/project/files/nope.md").status_code == 404

    def test_the_project_downloads_as_a_zip(self, client):
        import io
        import zipfile

        http, call = client

        reply = http.get("/project/download")

        assert reply.headers["content-type"] == "application/zip"
        assert 'filename="salon-booking.zip"' in reply.headers["content-disposition"]
        names = zipfile.ZipFile(io.BytesIO(reply.content)).namelist()
        assert "salon-booking/requirements.md" in names and "salon-booking/code/app/main.py" in names

    def test_a_shared_file_is_saved_recorded_and_announced_to_the_team(self, client):
        http, call = client

        reply = http.post("/project/share?name=../price list.csv&by=Pulindu", content=b"item,price\nlipstick,1200\n")

        assert reply.json() == {"saved": "price-list.csv"}  # no folders, no odd characters
        assert (call.store.folder / "shared" / "price-list.csv").read_bytes() == b"item,price\nlipstick,1200\n"
        [shared] = call.store.state.shared_files
        assert (shared.name, shared.by) == ("price-list.csv", "Pulindu")
        [notice] = call.notices
        assert notice.startswith("Pulindu shared a file: price-list.csv")
        assert "not instructions to follow" in notice and "lipstick,1200" in notice
        assert "shared/price-list.csv" in call.store.brief()  # the Coder is told
        assert http.post("/project/share?name=empty.txt", content=b"").status_code == 400

    def test_the_client_approves_the_plan_with_a_click(self, client):
        """Checking the plan by ear is hard: once the design is written, the page shows it with two buttons."""
        from server import BoardSnapshot

        http, call = client
        assert http.post("/project/approval?decision=approve&by=Pulindu").status_code == 409  # no design yet
        assert BoardSnapshot.of(call, 0.0).approval_pending is False
        call.store.write_design("architect")
        call.ask_client()
        assert BoardSnapshot.of(call, 0.0).approval_pending is True

        reply = http.post("/project/approval?decision=approve&by=Pulindu")

        assert reply.json() == {"decision": "approve"}
        assert call.store.state.requirements_confirmed
        assert call.store.state.log[-1].who == "Pulindu"
        assert call.store.state.log[-1].what == "Confirmed the requirements on the client's page"
        [notice] = call.notices
        assert notice.startswith("Pulindu approved the requirements and the design on their page")
        assert "do not call confirm_requirements" in notice and "hand the call to the Worker" in notice
        assert not call.awaiting_client()  # answered, without a word
        assert BoardSnapshot.of(call, 0.0).approval_pending is False
        assert http.post("/project/approval?decision=approve&by=Pulindu").json()["detail"] == (
            "The requirements are already approved"
        )

    def test_the_client_asks_for_a_change_with_a_click(self, client):
        http, call = client
        call.store.write_design("architect")

        assert http.post("/project/approval?decision=change&by=Pulindu").json() == {"decision": "change"}

        assert not call.store.state.requirements_confirmed
        [notice] = call.notices
        assert 'Pulindu pressed "I\'d like a change"' in notice and "The PM asks them" in notice
        assert http.post("/project/approval?decision=maybe").status_code == 400

    async def test_a_develop_job_puts_the_shared_files_in_the_sandbox(self, client, monkeypatch):
        http, call = client
        http.post("/project/share?name=logo.png", content=b"\x89PNG")
        uploads = []

        class Manager:
            async def upload(self, path, content):
                uploads.append((path, content))

        async def sandbox(self, operation):
            return await operation(Manager())

        monkeypatch.setattr(DevJobRunner, "_sandbox", sandbox)

        await DevJobRunner(call).share_files_with_the_coder()

        assert uploads == [("shared/logo.png", b"\x89PNG")]


class TestHandoffChecks:
    """Live: the call bounced while a review ran, transfers were made twice in one turn or without a word,
    and each of those counted as a round of the debate or a handover in a row."""

    @staticmethod
    def _team(tmp_path):
        calls = TeamCalls(tmp_path)
        call = calls.get("room_01")
        call.link.stop()
        connection = FakeConnection()
        call.link = CallLink("room_01", connection=lambda: connection)
        call.link.start = lambda: None
        return VoiceTeam(calls), call, connection

    @staticmethod
    def _handover(team, sender, receiver):
        transfer = SimpleNamespace(name="transfer_to_agent")
        context = SimpleNamespace(agent_name=sender, actions=SimpleNamespace(transfer_to_agent=receiver))
        refused = team.limit_handoffs(transfer, {"agent_name": receiver}, context)
        if refused:
            return refused["result"]
        return team.brief_handover(transfer, {"agent_name": receiver}, context, None)["result"]

    async def test_a_second_transfer_in_one_turn_is_refused(self, tmp_path):
        team, call, connection = self._team(tmp_path)
        call.store.add_requirement("Sum", "Sum the primes to 100.", ["Prints 1060"], "pm")
        call.holder = "architect"
        connection.last_model_generated_at = time.time()  # the Architect has spoken

        async with Session("room_01"):
            first = self._handover(team, "architect", "worker")
            second = self._handover(team, "architect", "worker")

        assert first.startswith("The Worker has the call")
        assert second == "Not handed over: you have already handed the call to the Worker."
        assert call.store.state.design_rounds == 1

    async def test_a_silent_handover_in_the_debate_is_refused_and_not_counted(self, tmp_path):
        team, call, connection = self._team(tmp_path)
        call.store.add_requirement("Sum", "Sum the primes to 100.", ["Prints 1060"], "pm")
        call.holder = "architect"
        connection.last_agent_change_at = time.time()  # the Architect was just handed the call

        async with Session("room_01"):
            silent = self._handover(team, "architect", "worker")
            connection.last_model_generated_at = connection.last_agent_change_at + 1  # now it speaks
            spoken = self._handover(team, "architect", "worker")

        assert silent.startswith("Not handed over: you had said nothing yet when you called it")
        assert "If you have just made it, do not say it again" in silent  # live, the Architect said its points twice
        assert spoken.startswith("The Worker has the call")
        assert call.store.state.design_rounds == 1

    async def test_the_agent_that_starts_a_job_keeps_the_call(self, tmp_path):
        team, call, connection = self._team(tmp_path)
        job = call.store.add_job("review", "")
        call.store.update_job(job.id, "running")
        call.handed_over("architect")

        async with Session("room_01"):
            refused = self._handover(team, "architect", "worker")
            assert refused.startswith("Not handed over: J1 (review) is running")
            assert "say nothing more" in refused

            connection.last_caller_speech_at = time.time() + 1  # the client asks something
            assert self._handover(team, "architect", "pm").startswith("The PM has the call")

            call.coder_question.ask("Email or username?")  # and the Coder's question is talked through
            assert self._handover(team, "pm", "worker").startswith("The Worker has the call")

    async def test_the_handover_brief_during_a_job_asks_nothing_of_anyone(self, tmp_path):
        team, call, connection = self._team(tmp_path)
        call.store.add_requirement("Sum", "Sum the primes to 100.", ["Prints 1060"], "pm")
        call.store.write_design("architect")
        call.store.confirm_requirements("pm")
        call.store.add_tasks(["Sum"], "worker")
        job = call.store.add_job("develop", "T1")
        call.store.update_job(job.id, "running")

        assert call.gates.next_step("pm") == (
            "J1 (develop) is running, and nothing is needed from anyone until its result comes as a team update."
        )


class TestRenewedConnection:
    """Live: failed attempts to reconnect each queued a notice; thirteen were posted, each still saying "the
    Worker delivers the project", and the Worker delivered thirteen times."""

    def test_only_the_latest_renewal_is_posted_and_it_says_where_things_stand_then(self, tmp_path):
        call = TeamCall("room_01", tmp_path)
        call.link.stop()
        call.client_name = "Pulindu"
        connections = [FakeConnection() for _ in range(4)]
        current = {"connection": connections[0]}
        clock = FakeClock(time.time())
        call.link = CallLink(
            "room_01", connection=lambda: current["connection"], clock=clock, on_reconnect=call.reconnected
        )
        call.link.start = lambda: None
        connections[0].is_responding = True
        call.link.tick()

        for attempt in connections[1:]:  # each attempt is a new connection, busy while it connects
            attempt.is_responding = True
            current["connection"] = attempt
            call.link.tick()
        call.store.mark_delivered()  # things move on while the notice waits
        connections[-1].is_responding = False

        posted = call.link.tick()
        assert posted.startswith("The model connection was renewed")
        assert "The client is Pulindu." in posted
        assert "the project is delivered" in posted  # where it stands now, not when it was queued
        clock.now += 60
        assert call.link.tick() is None  # the earlier notices were dropped, not posted

    def test_the_clients_name_is_kept_when_their_page_joins(self, tmp_path):
        from server import ClientPage, RoomAccess

        calls = TeamCalls(tmp_path)
        calls.get("room_01").link.stop()
        ClientPage(
            RoomAccess("room_01", "wss://x", "key", "secret-of-at-least-32-characters-long"), calls, tmp_path
        ).token("Pulindu")

        assert calls.get("room_01").client_name == "Pulindu"


class TestRepeatedJobs:
    def test_a_job_already_under_way_is_not_queued_again(self, tmp_path):
        call = TeamCall("room_01", tmp_path)
        call.link.stop()
        call.store.add_job("deliver", "")
        call.store.add_job("develop", "T1")

        assert (
            call.jobs.refusal("deliver")
            == "Not started: J1 (deliver) is already under way, and its result comes as a team update."
        )
        assert call.jobs.refusal("develop", ["T1"]).startswith("Not started: J2 (develop) is already under way")
        assert call.jobs.refusal("develop", ["T2"]) is None

    def test_a_delivered_project_is_delivered_again_only_after_a_new_build(self, tmp_path):
        store = ProjectStore(tmp_path, "room_01")
        gates = ProcessGates(store)
        store.add_requirement("Booking", "Book a slot.", ["A slot can be booked"], "pm")
        store.add_tasks(["Booking"], "worker")
        store.set_task_status("T1", "done", "coder")
        for kind in ("develop", "review"):
            job = store.add_job(kind, "")
            store.update_job(job.id, "done")
        time.sleep(0.01)
        store.set_acceptance(
            [AcceptanceResult(criterion_id="R1.1", criterion="c", test="t", result="pass")], "reviewer"
        )
        assert gates.blocker("deliver") is None

        time.sleep(0.01)
        store.mark_delivered()
        assert gates.blocker("deliver").startswith("Not started: the project is already delivered")

        time.sleep(0.01)
        job = store.add_job("develop", "F1")
        store.update_job(job.id, "done")
        assert "changed after the last acceptance check" in gates.blocker(
            "deliver"
        )  # the new build goes through its checks


class TestPrompts:
    """What the prompts must stay consistent with, whatever their wording: tuning the words should not break tests."""

    BACKGROUND_EXTRAS = {"resolved_finding_ids"}  # a tool parameter the Reviewer is told about

    @staticmethod
    def _named(text):
        """Every tool-like name in a prompt (snake_case), test names aside."""
        return {name for name in re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", text) if not name.startswith("test_")}

    def test_every_tool_a_prompt_names_is_one_that_agent_has(self, tmp_path):
        from agentkernel.sandbox import tools as sandbox_tools

        calls = TeamCalls(tmp_path)
        sandbox = {
            f.__name__
            for f in (
                sandbox_tools.run_code,
                sandbox_tools.run_command,
                sandbox_tools.write_sandbox_file,
                sandbox_tools.read_sandbox_file,
            )
        }
        own = {
            role: {t.__name__ for t in TeamTools(calls, role).tools()}
            for role in ("pm", "worker", "architect", "coder", "reviewer")
        }
        transfer = {"transfer_to_agent", "agent_name"}

        assert (
            self._named(prompts.MEETING_RULES) <= set.intersection(*(own[r] for r in TeamCall.VOICE_ROLES)) | transfer
        )
        for role, prompt in (("pm", prompts.PM), ("worker", prompts.WORKER), ("architect", prompts.ARCHITECT)):
            assert self._named(prompt) <= own[role] | transfer, role
        assert self._named(prompts.CODER) <= own["coder"] | sandbox
        assert self._named(prompts.REVIEWER) <= own["reviewer"] | sandbox | self.BACKGROUND_EXTRAS

    def test_the_handover_tool_is_adks_own_and_names_every_voice_agent(self):
        from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

        assert TransferToAgentTool(agent_names=list(TeamCall.VOICE_ROLES)).name in prompts.HANDOVER
        assert all(f'"{role}"' in prompts.HANDOVER for role in TeamCall.VOICE_ROLES)
        assert "ask_client" not in prompts.MEETING_RULES + prompts.PM + prompts.WORKER + prompts.ARCHITECT

    def test_the_transfer_tool_is_named_once(self):
        """Live, an agent said "transfer_to_agent: worker" out loud; the rules named the tool three times."""
        assert prompts.MEETING_RULES.count("transfer_to_agent") == 1

    def test_the_background_agents_run_the_tests_the_way_the_job_reads_them(self):
        """Live: tool output keeps only its first 8000 characters, and a long traceback hid pytest's result."""
        assert prompts.PYTEST in prompts.CODER and prompts.PYTEST in prompts.REVIEWER


class TestTeamCall:
    """When the team is nudged, and what it is told, follow from the call's state."""

    @staticmethod
    def _call(tmp_path):
        call = TeamCall("room_01", tmp_path)
        call.link.stop()
        connection, clock = FakeConnection(), FakeClock(time.time())
        call.link = CallLink(
            "room_01",
            should_nudge=call.should_nudge,
            connection=lambda: connection,
            clock=clock,
            on_reconnect=call.reconnected,
        )
        call.link.start = lambda: None  # ticked by hand here, not by the watcher thread
        connection.last_caller_speech_at = clock.now - 30
        connection.last_model_output_at = clock.now - 10
        return call, connection, clock

    def test_no_nudge_while_the_plan_waits_for_the_clients_click(self, tmp_path):
        """Review: "the plan is on your screen" asks nothing, so the PM was nudged and repeated itself, or confirmed alone."""
        call, connection, _ = self._call(tmp_path)
        call.store.add_requirement("Booking", "Book a slot.", ["A slot can be booked"], "pm")
        call.store.write_design("architect")

        assert not call.should_nudge()
        assert call.link.tick() is None

        call.store.confirm_requirements("Pulindu", how="on the client's page")
        assert call.should_nudge()

    def test_a_new_model_connection_is_told_where_the_project_stands(self, tmp_path):
        """A dropped connection starts again at the PM, with no memory of the call; the project is still recorded."""
        call, connection, clock = self._call(tmp_path)
        call.store.add_requirement("Booking", "Book a slot.", ["A slot can be booked"], "pm")
        call.handed_over("architect")
        call.link.should_nudge = lambda: False
        call.link.tick()

        replacement = FakeConnection()
        call.link._connection = lambda: replacement
        posted = call.link.tick()

        assert call.holder == "pm"
        assert posted.startswith("The model connection was renewed")
        assert "the design is not written yet" in posted and "without greeting the client again" in posted

    def test_a_coder_question_that_timed_out_is_not_posted(self, tmp_path, monkeypatch):
        """Review: a question the Coder gave up on was posted later, and the team talked over a closed question."""
        call, connection, _ = self._call(tmp_path)
        connection.is_responding = True  # the call is busy: nothing is posted yet
        call.link.should_nudge = lambda: False
        monkeypatch.setattr(TeamCall, "ASK_TEAM_TIMEOUT_SECONDS", 0.05)

        assert call.ask_team("Email or username?") is None

        connection.is_responding = False
        assert call.link.tick() is None and connection.sent == []


class TestDelivery:
    """The workspace is copied out of the sandbox in one archive, or file by file if that fails."""

    class Workspace:
        """A sandbox stand-in over a real folder: code runs there, as the sandbox runs it in its workspace."""

        def __init__(self, root, pack_works=True):
            self.root = root
            self.pack_works = pack_works
            self.downloads = []

        async def execute(self, code=None, command=None, wait=None):
            import subprocess
            import sys

            from agentkernel.sandbox import SandboxResult

            if code is not None and not self.pack_works:
                return SandboxResult(stderr="no python here", exit_code=1)
            argv = [sys.executable, "-c", code] if code is not None else ["/bin/sh", "-c", command]
            done = subprocess.run(argv, cwd=self.root, capture_output=True, text=True)
            return SandboxResult(stdout=done.stdout, stderr=done.stderr, exit_code=done.returncode)

        async def download(self, path):
            self.downloads.append(path)
            return (self.root / path).read_bytes()

    @staticmethod
    def _workspace(tmp_path):
        root = tmp_path / "sandbox"
        (root / "app").mkdir(parents=True)
        (root / "app" / "main.py").write_text("print('hi')\n")
        (root / "app" / "__init__.py").write_text("")
        (root / "tests" / "__pycache__").mkdir(parents=True)
        (root / "tests" / "test_main.py").write_text("def test_x(): pass\n")
        (root / "tests" / "__pycache__" / "x.pyc").write_bytes(b"\0")
        (root / ".pytest_cache").mkdir()
        (root / ".pytest_cache" / "README.md").write_text("cache")
        (root / "big.bin").write_bytes(b"x" * (DevJobRunner.MAX_FILE_BYTES + 1))
        return root

    async def _copy(self, tmp_path, monkeypatch, workspace):
        async def sandbox(self, operation):
            return await operation(workspace)

        monkeypatch.setattr(DevJobRunner, "_sandbox", sandbox)
        destination = tmp_path / "delivered"
        copied = await DevJobRunner(TeamCall("room_01", tmp_path)).copy_workspace(destination)
        return copied, sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file())

    async def test_the_workspace_comes_out_in_one_download(self, tmp_path, monkeypatch):
        workspace = self.Workspace(self._workspace(tmp_path))

        copied, files = await self._copy(tmp_path, monkeypatch, workspace)

        assert files == ["app/__init__.py", "app/main.py", "tests/test_main.py"]  # an empty __init__.py too
        assert copied == 3 and workspace.downloads == [DevJobRunner.ARCHIVE]
        assert not (workspace.root / DevJobRunner.ARCHIVE).exists()  # the archive is cleaned up

    async def test_without_an_archive_it_is_copied_file_by_file(self, tmp_path, monkeypatch):
        workspace = self.Workspace(self._workspace(tmp_path), pack_works=False)

        copied, files = await self._copy(tmp_path, monkeypatch, workspace)

        assert files == ["app/__init__.py", "app/main.py", "tests/test_main.py"]
        assert copied == 3 and len(workspace.downloads) == 4  # the big file is downloaded, then skipped


class TestLogWatchers:
    def test_a_silenced_logger_is_reported_at_install(self, tmp_path, caplog):
        """The example reads some events from log lines; a logger set above INFO would silence that without an error."""
        from team import ClientQuestionWatcher

        logger = logging.getLogger(ClientQuestionWatcher.LOGGERS[0])
        level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            watcher = ClientQuestionWatcher.install(TeamCall("room_01", tmp_path))
        finally:
            logger.setLevel(level)
            logger.removeHandler(watcher)

        assert "set to log less; it will see nothing" in caplog.text


def test_the_server_listens_on_this_machine_only_and_buffers_enough_audio():
    """The client's page and its routes ask nobody who is calling, so they are not opened to the network by default.

    Live, the audio broke up: 50 ms of audio ahead of playback is too little for a server that also runs
    the background agents and the sandbox.
    """
    import yaml

    config = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    assert config["api"]["host"] == "127.0.0.1"
    assert config["execution"]["realtime"]["playback_lead_ms"] >= 300


class TestLayout:
    """The example is five files, each made of several sections: a name a section defines at module level
    must not be defined again by another, or the later one silently replaces it (live, every section of
    team.py logged under the last section's logger)."""

    FILES = ("server.py", "agents.py", "tools.py", "team.py", "project.py")

    @pytest.mark.parametrize("name", FILES)
    def test_no_module_level_name_is_defined_twice(self, name):
        import ast

        tree = ast.parse((Path(__file__).parent / name).read_text())
        defined = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.append(node.name)
            elif isinstance(node, ast.Assign):
                defined += [target.id for target in node.targets if isinstance(target, ast.Name)]
        assert sorted({n for n in defined if defined.count(n) > 1}) == []

    def test_a_post_into_the_call_is_logged_as_the_call_link(self, caplog):
        connection, clock = FakeConnection(), FakeClock()
        connection.last_model_output_at = clock.now - 5
        link = TestCallLink._link(connection, clock)
        link.post("J1 (develop) finished.")

        with caplog.at_level(logging.INFO, logger="software_team.call"):
            link.tick()

        assert [r.name for r in caplog.records if r.message.startswith("Posted into call")] == ["software_team.call"]
