"""Making background work observable: what the Coder and the Reviewer are doing, on the board and in the log.

- ``ActivityFeed``: every tool call they make, in plain words, from the frameworks' own hooks.
- ``ModelWaitWatcher``: the model making them wait (a rate limit).

Neither speaks into the call. Every post into a live call makes the voice agent answer it, so
running work was narrated over and over in near-identical words; the client follows it on the
board instead, and an agent asked "how is it going?" answers from ``get_progress``.
"""

import json
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import BaseModel

from team.jobs import DevJobRunner
from team.log_watch import LogLineWatcher
from team.project import ProjectStore

if TYPE_CHECKING:
    from team.state import TeamCalls

_log = logging.getLogger("software_team.activity")


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
        _log.info(f"{entry.who} {entry.text}")
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
