import logging
from collections.abc import AsyncGenerator
from typing import Any

from langfuse import Langfuse, propagate_attributes
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor

from ...core import Session
from ...core.event import StreamEvent
from ...core.model import AgentReply, AgentRequest, ResumeDecision
from ...core.paused_run import PausedRun
from ...framework.claude.claude import ClaudeRunner


class LangFuseClaudeRunner(ClaudeRunner):

    def __init__(self, client: Langfuse):
        """
        Initializes a LangFuseClaudeRunner instance.

        The model calls run inside the Claude Code CLI subprocess, out of reach of Python-side LLM
        instrumentation, so the SDK's own OpenInference instrumentor supplies the agent and tool spans.
        :param client: The Langfuse client instance.
        """
        super().__init__()
        self._client = client
        self._log = logging.getLogger("ak.trace.langfuse.claude")

        ClaudeAgentSDKInstrumentor().instrument()

    async def run(self, agent: Any, session: Session, requests: list[AgentRequest]) -> AgentReply:
        """
        Runs the Claude agent with provided multi modal inputs.
        :param agent: The Claude agent to run.
        :param session: The session to use for the agent.
        :param requests: The requests to the agent.
        :return: The result of the agent's execution.
        """

        with propagate_attributes(session_id=session.id, tags=["agentkernel"]):

            with self._client.start_as_current_observation(name="Agent Kernel Claude", as_type="span") as span:

                result = await super().run(agent, session, requests)
                span.update(input=result.prompt, output=str(result))

        return result

    async def resume(
        self, agent: Any, session: Session, requests: list[AgentRequest], decisions: list[ResumeDecision], record: PausedRun
    ) -> AgentReply:
        with propagate_attributes(session_id=session.id, tags=["agentkernel"]):
            with self._client.start_as_current_observation(name="Agent Kernel Claude resume", as_type="span") as span:
                result = await super().resume(agent, session, requests, decisions, record)
                span.update(output=str(result))
        return result

    async def stream(self, agent: Any, session: Session, requests: list[AgentRequest]) -> AsyncGenerator[StreamEvent, None]:
        with propagate_attributes(session_id=session.id, tags=["agentkernel"]):
            with self._client.start_as_current_observation(name="Agent Kernel Claude stream", as_type="span"):
                async for event in super().stream(agent, session, requests):
                    yield event

    async def resume_stream(
        self, agent: Any, session: Session, requests: list[AgentRequest], decisions: list[ResumeDecision], record: PausedRun
    ) -> AsyncGenerator[StreamEvent, None]:
        with propagate_attributes(session_id=session.id, tags=["agentkernel"]):
            with self._client.start_as_current_observation(name="Agent Kernel Claude resume", as_type="span"):
                async for event in super().resume_stream(agent, session, requests, decisions, record):
                    yield event
