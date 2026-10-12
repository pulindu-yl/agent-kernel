import logging

from agentkernel.claude import ClaudeAgentSpec, ClaudeModule, ClaudeToolBuilder
from agentkernel.cli import CLI
from agentkernel.core import ToolContext
from claude_agent_sdk import ClaudeAgentOptions

MODEL = "sonnet"


def get_weather(city: str) -> str:
    """Returns the weather for a given city (example stub)."""
    logger = logging.getLogger(__name__)
    logger.debug("Session ID: %s", ToolContext.get().session.id)

    if city == "Tokyo":
        return "The weather in Tokyo is sunny."
    else:
        return f"Cannot find weather for {city}."


# tools=[] and setting_sources=[] turn off Claude Code's built-in tools (Bash, Read, Write, ...) and ~/.claude
# settings; permission_mode="dontAsk" denies any tool call that is not explicitly approved. Agent Kernel approves
# the tools bound with ClaudeToolBuilder.
math_agent = ClaudeAgentSpec(
    name="math",
    description="Specialist agent for math questions",
    options=ClaudeAgentOptions(
        model=MODEL,
        tools=[],
        setting_sources=[],
        permission_mode="dontAsk",
        system_prompt="You provide help with math problems. Give short and direct answers exactly to the question. "
        "Don't provide any explanations nor additional details.",
    ),
)

general_agent = ClaudeAgentSpec(
    name="general",
    description="Agent for general questions",
    options=ClaudeAgentOptions(
        model=MODEL,
        tools=[],
        setting_sources=[],
        permission_mode="dontAsk",
        system_prompt="You provide assistance with general queries. Give short and direct answers exactly to the question. "
        "Don't provide any explanations nor additional details",
    ),
)

weather_agent = ClaudeAgentSpec(
    name="weather",
    description="Agent that provides weather information",
    options=ClaudeAgentOptions(
        model=MODEL,
        tools=[],
        setting_sources=[],
        permission_mode="dontAsk",
        system_prompt="You provide weather information upon request. Use the get_weather tool for all weather-related questions.",
    ),
    tools=ClaudeToolBuilder.bind([get_weather]),
)

triage_agent = ClaudeAgentSpec(
    name="triage",
    description="Front-line agent that routes each question to the right specialist",
    options=ClaudeAgentOptions(
        model=MODEL,
        tools=[],
        setting_sources=[],
        permission_mode="dontAsk",
        system_prompt="You determine which agent to use based on the user's question. Give short and direct answers "
        "exactly to the question. Don't provide any explanations nor additional details",
    ),
    subagents=[general_agent, math_agent, weather_agent],
)

ClaudeModule([triage_agent, math_agent, general_agent, weather_agent])

if __name__ == "__main__":
    CLI.main()
