"""The three agents the client talks to: Worker, PM and Architect, on Gemini Live.

They are ordinary Google ADK agents. The PM is the entry agent: it opens the meeting, as a project
manager does with a client. The Worker and the Architect are its sub-agents, so, by ADK's own rules,
any of the three can hand the call to the other two with ``transfer_to_agent`` (the PM to its
sub-agents, each sub-agent to its parent and to its peer). Agent Kernel's realtime runner plays each agent in its own voice, and the
LiveKit gateway shows each one as its own participant in the room.
"""

import logging
from typing import Any, Optional

from agentkernel.framework.adk import GoogleADKModule, GoogleADKToolBuilder
from google.adk.agents import Agent as GoogleAgent
from google.genai import types

from members import prompts
from team.state import TeamCall, TeamCalls
from team.tools import TeamTools

_log = logging.getLogger("software_team.agents")


class VoiceTeam:
    """Builds the PM, the Worker and the Architect, and registers them with Agent Kernel."""

    MODEL = "gemini-3.1-flash-live-preview"
    NAMES = TeamCall.NAMES

    def __init__(self, calls: TeamCalls):
        self._calls = calls

    def build(self) -> GoogleAgent:
        """The PM, with the Worker and the Architect as its sub-agents."""
        worker = GoogleAgent(
            name="worker",
            model=self.MODEL,
            description="Software engineer: plans the tasks, drives development and delivery.",
            instruction=f"{prompts.MEETING_RULES}\n\n{prompts.WORKER}",
            tools=self._tools("worker"),
            generate_content_config=self._voice("Puck"),
            before_tool_callback=self.limit_handoffs,
            after_tool_callback=self.brief_handover,
        )
        architect = GoogleAgent(
            name="architect",
            model=self.MODEL,
            description="Software architect: owns the technical design, challenges technical choices, reviews the code.",
            instruction=f"{prompts.MEETING_RULES}\n\n{prompts.ARCHITECT}",
            tools=self._tools("architect"),
            generate_content_config=self._voice("Charon"),
            before_tool_callback=self.limit_handoffs,
            after_tool_callback=self.brief_handover,
        )
        pm = GoogleAgent(
            name="pm",
            model=self.MODEL,
            description="Project manager: opens the meeting, owns the business requirements, asks the client business questions, checks acceptance.",
            instruction=f"{prompts.MEETING_RULES}\n\n{prompts.PM}",
            tools=self._tools("pm"),
            generate_content_config=self._voice("Aoede"),
            before_tool_callback=self.limit_handoffs,
            after_tool_callback=self.brief_handover,
            sub_agents=[worker, architect],
        )
        return pm

    def load(self) -> None:
        """Register the team with Agent Kernel; the LiveKit gateway serves its entry agent, the PM."""
        GoogleADKModule([self.build()])

    def limit_handoffs(self, tool: Any, args: dict, tool_context: Any) -> Optional[dict]:
        """A before-tool callback: refuses a transfer that should not happen.

        In turn: a second transfer in one turn (the agent no longer holds the call), a debate past
        its last round, a handover while a job runs with nothing new, a silent handover in the design
        debate or the code review, and too many handovers in a row without the client.

        Returning a result answers in the tool's place, so the transfer does not happen (Agent
        Kernel's realtime runner runs ADK's tool callbacks, as an ADK run does).
        """
        if tool.name != "transfer_to_agent":
            return None
        call = self._calls.current()
        sender, receiver = getattr(tool_context, "agent_name", None), args.get("agent_name")
        blocked = (
            self._already_handed_over(call, sender)
            or call.gates.handover_blocker(sender, receiver)
            or call.job_blocker()
            or self._silent_in_debate(call, sender, receiver)
        )
        if blocked:
            return {"result": blocked}
        return None if call.guard.allow() else {"result": call.guard.refusal()}

    def _already_handed_over(self, call: TeamCall, sender: Optional[str]) -> Optional[str]:
        """Live, the Architect called the transfer twice in one turn, and each counted as a round of the debate."""
        if sender is None or sender == call.holder:
            return None
        return f"Not handed over: you have already handed the call to the {self.NAMES.get(call.holder, call.holder)}."

    def _silent_in_debate(self, call: TeamCall, sender: Optional[str], receiver: Optional[str]) -> Optional[str]:
        """Live, the Architect and the Worker bounced the call without a word, and each bounce counted as a round."""
        if call.gates.debate() is None or {sender, receiver} != {"architect", "worker"}:
            return None
        if call.link.agent_spoke_since_handover():
            return None
        teammate = self.NAMES.get(receiver, receiver)
        return f"Not handed over: you have said nothing since you got the call. Say your part to the {teammate} by name first, then hand over."

    def brief_handover(self, tool: Any, args: dict, tool_context: Any, tool_response: Any) -> Optional[dict]:
        """An after-tool callback: a transfer's result tells the agent taking the call where the project stands.

        The agent taking the call reads the conversation so far, the transfer and its result
        included. A teammate's words do not always say why it was handed the call (a Worker handed
        the call mid-design took it to be asked to build, and handed it straight back), but the
        project's next step, and whose it is, does.
        """
        if tool.name != "transfer_to_agent" or not tool_context.actions.transfer_to_agent:
            return None  # another tool, or a transfer limit_handoffs refused
        call = self._calls.current()
        target = tool_context.actions.transfer_to_agent
        call.handed_over(target)
        if (tool_context.agent_name, target) == ("architect", "worker"):
            # A round of the design debate or of the code review conversation, whichever is on.
            debate = call.gates.debate()
            if debate == "design":
                call.store.count_design_round("architect")
            elif debate == "review":
                call.store.count_review_round("architect")
        return {
            "result": f"The {self.NAMES.get(target, target)} has the call. Where the project stands: {call.gates.next_step(target)}"
        }

    def _tools(self, role: str) -> list:
        return GoogleADKToolBuilder.bind(TeamTools(self._calls, role).tools())

    @staticmethod
    def _voice(name: str) -> types.GenerateContentConfig:
        """A Gemini prebuilt voice, so each agent sounds different."""
        return types.GenerateContentConfig(
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=name))
            )
        )
