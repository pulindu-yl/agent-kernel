"""The two agents that work behind the call: the Coder and the Reviewer.

They are Google ADK agents, like the voice agents, but on a text model reached through LiteLLM
(OpenAI's ``gpt-4.1`` by default), as in the ``examples/api/adk`` example. A build makes many model
calls in a row, which a realtime voice model is not made for and the Gemini API free tier cannot
cover. ``TEAM_TEXT_MODEL`` picks another model in LiteLLM's ``provider/model`` form, such as
``openai/gpt-4.1-mini`` or ``anthropic/claude-sonnet-5-5``.

They are run through ``AgentService`` by ``DevJobRunner`` (team/jobs.py), and they get the sandbox
tools because ``config.yaml`` names them (``sandbox.agents``). They never join the call: what they do
reaches the board through the activity feed (team/progress.py), and the call through the job's result.

Each run sees only its brief and its own tool calls (ADK's ``include_contents="none"``). Both agents
share one AK session, and so one ADK session, so otherwise every model call of every job would
resend every earlier job, whole file writes and test output included. The brief
(``ProjectStore.brief``) already carries everything a job needs.
"""

import logging
from typing import Any, Optional

from agentkernel.framework.adk import GoogleADKModule, GoogleADKToolBuilder
from google.adk.agents import Agent as GoogleAgent
from google.adk.models.lite_llm import LiteLlm

from members import prompts
from team.state import TeamCalls
from team.tools import TeamTools

_log = logging.getLogger("software_team.agents")


class BackgroundTeam:
    """Builds the Coder and the Reviewer, and registers them with Agent Kernel."""

    MODEL = "openai/gpt-4.1"
    # A build makes many model calls in a row; a rate-limited request is retried rather than failing the job.
    # The OpenAI client logs each retry, which the call hears about (team/progress.py, ModelWaitWatcher).
    MAX_RETRIES = 6

    def __init__(self, calls: TeamCalls, model: Optional[str] = None):
        self._calls = calls
        self._model = model or self.MODEL

    def build(self) -> list[GoogleAgent]:
        coder = GoogleAgent(
            name="coder",
            model=LiteLlm(model=self._model, max_retries=self.MAX_RETRIES),
            description="Background engineer: writes the code and tests in the sandbox.",
            instruction=prompts.CODER,
            tools=GoogleADKToolBuilder.bind(TeamTools(self._calls, "coder").tools()),
            after_tool_callback=self.record_activity,
            include_contents="none",
        )
        reviewer = GoogleAgent(
            name="reviewer",
            model=LiteLlm(model=self._model, max_retries=self.MAX_RETRIES),
            description="Background reviewer: reads the real code and runs the tests.",
            instruction=prompts.REVIEWER,
            tools=GoogleADKToolBuilder.bind(TeamTools(self._calls, "reviewer").tools()),
            after_tool_callback=self.record_activity,
            include_contents="none",
        )
        return [coder, reviewer]

    def load(self) -> None:
        GoogleADKModule(self.build())

    def record_activity(self, tool: Any, args: dict, tool_context: Any, tool_response: Any) -> None:
        """An after-tool callback: every tool call goes to the call's activity feed.

        Returns ``None``, so the tool's result is kept; a failure here never fails the tool call.
        """
        try:
            self._calls.current().activity.record_tool(tool_context.agent_name, tool.name, args, tool_response)
        except Exception:
            _log.exception(f"Could not record {tool.name} in the activity feed")
        return None
