import logging
from collections.abc import AsyncGenerator
from typing import Any

from ...core import Session
from ...core.event import StreamEvent
from ...core.model import AgentReply, AgentRequest, ResumeDecision
from ...core.paused_run import PausedRun
from ...framework.claude.claude import ClaudeRunner
from .openllmetry import TraceloopContext


class OpenLLMetryClaudeRunner(ClaudeRunner):

    def __init__(self) -> None:
        """
        Initializes an OpenLLMetryClaudeRunner instance.

        OpenLLMetry has no Claude Agent SDK instrumentation, and its Anthropic instrumentation cannot see the
        model calls, which the Claude Code CLI subprocess makes. Each run gets the outer Agent Kernel span only.
        """
        super().__init__()
        self._log = logging.getLogger("ak.trace.openllmetry.claude")

    async def run(self, agent: Any, session: Session, requests: list[AgentRequest]) -> AgentReply:
        """
        Runs the Claude agent with provided multi modal inputs.
        :param agent: The Claude agent to run.
        :param session: The session to use for the agent.
        :param requests: The requests to the agent.
        :return: The result of the agent's execution.
        """

        with TraceloopContext(app_name="AgentKernel Claude", association_properties={"session_id": session.id}):
            result = await super().run(agent, session, requests)
        return result

    async def resume(
        self, agent: Any, session: Session, requests: list[AgentRequest], decisions: list[ResumeDecision], record: PausedRun
    ) -> AgentReply:
        with TraceloopContext(app_name="AgentKernel Claude", association_properties={"session_id": session.id}):
            return await super().resume(agent, session, requests, decisions, record)

    async def stream(self, agent: Any, session: Session, requests: list[AgentRequest]) -> AsyncGenerator[StreamEvent, None]:
        with TraceloopContext(app_name="AgentKernel Claude", association_properties={"session_id": session.id}):
            async for event in super().stream(agent, session, requests):
                yield event

    async def resume_stream(
        self, agent: Any, session: Session, requests: list[AgentRequest], decisions: list[ResumeDecision], record: PausedRun
    ) -> AsyncGenerator[StreamEvent, None]:
        with TraceloopContext(app_name="AgentKernel Claude", association_properties={"session_id": session.id}):
            async for event in super().resume_stream(agent, session, requests, decisions, record):
                yield event
