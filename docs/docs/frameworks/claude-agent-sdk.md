---
sidebar_position: 8
---

# Claude Agent SDK

Integrate the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview) with Agent Kernel. The
Claude Agent SDK is the agent harness behind Claude Code, packaged as a library. Choose it when you want that
harness: built-in file, shell and web tools, subagents, skills, hooks, a permission engine and automatic
context compaction. For a plain tool-calling agent on Claude models, [Pydantic AI](./pydantic-ai) with its
Anthropic provider is lighter.

:::info How it runs
Each run starts a Claude Code CLI subprocess and talks to it over stdin/stdout. The CLI is bundled in the
`claude-agent-sdk` wheel (a native binary of about 240 MB; Node.js is not needed). Expect roughly half a
second to start a run and a few hundred MB of memory per concurrent run; Anthropic suggests planning for
1 GiB RAM and 1 CPU per concurrent agent. See [Deployment](#deployment).
:::

## Installation

```bash
pip install agentkernel[claude]
```

Authenticate with an Anthropic API key or a cloud provider. Anthropic does not allow third-party products to
offer a claude.ai login, so use one of:

```bash
export ANTHROPIC_API_KEY=...          # Claude API
export CLAUDE_CODE_USE_BEDROCK=1      # Amazon Bedrock (with AWS credentials)
export CLAUDE_CODE_USE_VERTEX=1       # Google Vertex AI (with GCP credentials)
export CLAUDE_CODE_USE_FOUNDRY=1      # Microsoft Foundry
```

## Basic Usage

The SDK has no named agent object: an agent is configured by `ClaudeAgentOptions`. Agent Kernel adds a
`ClaudeAgentSpec` that gives it a name, a description and its tools, and passes the options through unchanged.

```python
from claude_agent_sdk import ClaudeAgentOptions

from agentkernel.claude import ClaudeAgentSpec, ClaudeModule, ClaudeToolBuilder
from agentkernel.cli import CLI


def get_weather(city: str) -> str:
    """
    Returns the weather for a city.
    :param city: The city to report the weather for.
    """
    return f"The weather in {city} is sunny."


general = ClaudeAgentSpec(
    name="general",
    description="Agent for general and weather questions",
    options=ClaudeAgentOptions(
        model="sonnet",
        system_prompt="You are a helpful assistant. Use get_weather for weather questions.",
        tools=[],                   # no built-in Claude Code tools
        setting_sources=[],         # ignore ~/.claude settings and CLAUDE.md
        permission_mode="dontAsk",  # deny anything not explicitly approved
        verbatim_prompts=True,      # don't expand @path mentions in user text
    ),
    tools=ClaudeToolBuilder.bind([get_weather]),
)

ClaudeModule([general])

if __name__ == "__main__":
    CLI.main()
```

## Server Safety

Agent Kernel keeps the SDK's own defaults, and on a server those defaults are wide open. Unless you set
otherwise, an agent:

- has the **full Claude Code toolset** (Bash, Read, Write, Edit, WebFetch, …) acting on the host the agent runs
  on, outside the [sandbox](../advanced/sandbox) capability;
- runs in the CLI's default permission mode, which approves many calls on its own;
- loads user and project settings, CLAUDE.md and auto-memory from disk;
- expands `@path` file mentions in prompts from the local disk.

For agents that serve untrusted users, start from this lockdown and open up only what an agent needs:

| Option | Lockdown value | Effect |
|--------|----------------|--------|
| `tools` | `[]`, or an explicit list such as `["Agent"]` | No built-in tools beyond the ones listed |
| `permission_mode` | `"dontAsk"` | Calls that are not explicitly approved are denied |
| `setting_sources` | `[]` | No settings, CLAUDE.md or hooks from disk |
| `strict_mcp_config` | `True` | Ignore `.mcp.json` and user MCP servers |
| `verbatim_prompts` | `True` | User text is sent as written |
| `env` | `{"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CONFIG_DIR": <per tenant>}` | No auto-memory; per-tenant CLI state |
| `cwd` | a dedicated directory | Where built-in file tools act |
| `max_turns` / `max_budget_usd` | a limit | Bounds a runaway run |

Tools you bind with `ClaudeToolBuilder` are always approved, the same as a bound tool simply runs in every other
framework. To gate one, add a native `PreToolUse` hook matched on `mcp__ak__<tool name>`; hooks are evaluated
before approvals.

## Tool Binding

`ClaudeToolBuilder.bind` turns plain functions into the SDK's in-process tools. The input schema comes from the
signature (parameters without a default are required; Pydantic models are accepted), and the description is
the docstring. Sync functions run on a worker thread so they don't block the
SDK's message loop. Tools reach the run through `ToolContext.get()`.

```python
from agentkernel.core import ToolContext

def lookup_order(order_id: str) -> dict:
    """
    Looks an order up.
    :param order_id: The order to look up.
    """
    session = ToolContext.get().session
    return {"order_id": order_id, "status": "shipped"}

tools = ClaudeToolBuilder.bind([lookup_order], readOnlyHint=True)
```

Keyword options become the tools' `ToolAnnotations` (`readOnlyHint`, `maxResultSizeChars`, …) and apply to
every function in the call. Tools made with the SDK's own `@tool` decorator can go in `ClaudeAgentSpec.tools`
too.

Agent Kernel serves an agent's tools from an in-process MCP server named `ak`, so the model calls them as
`mcp__ak__<tool name>`; stream events report the plain tool name. The `ak` server name is reserved, and
`mcp_servers` must be a dict (not a config-file path) on an agent that has tools.

## Multi-Agent System

Claude's native multi-agent primitive is the **subagent**: the parent delegates to it through its `Agent` tool.
List other `ClaudeAgentSpec`s in `subagents=`, as you would pass agents to OpenAI's `handoffs=` or ADK's
`sub_agents=`:

```python
weather = ClaudeAgentSpec(
    name="weather",
    description="Agent that provides weather information",
    options=ClaudeAgentOptions(system_prompt="Use the get_weather tool for weather questions.", tools=[]),
    tools=ClaudeToolBuilder.bind([get_weather]),
)

triage = ClaudeAgentSpec(
    name="triage",
    description="Routes each question to the right specialist",
    options=ClaudeAgentOptions(system_prompt="Delegate each question to the right agent.", tools=[], permission_mode="dontAsk"),
    subagents=[weather],
)

ClaudeModule([triage, weather])
```

For each subagent, Agent Kernel builds the SDK's `AgentDefinition` from its description, system prompt (a string
or custom prompt), model and tools, and adds it to the parent's `agents=`. A subagent runs inside the parent's
session, so the parent serves the subagent's Agent Kernel tools, and the subagent may use only those plus the
built-in tools its own `tools=` lists (the session's tools when left unset). The parent's `Agent` tool is enabled and
approved. Native `agents=` definitions still work alongside; a name used in both is rejected.

Subagent traffic is not streamed: the subagent's answer arrives as the result of the parent's `Agent` tool call. A
subagent's own `subagents` are not included.

## Conversations and Sessions

Each agent is its own Claude conversation. The CLI writes its transcript to local disk, which a stateless
replica does not have on the next turn, so Agent Kernel keeps the transcript in the **Agent Kernel session**
through the SDK's `SessionStore` interface:

- the SDK mirrors every transcript line to the store as the CLI runs;
- once the turn's result arrives, the lines become part of the session, which your configured session store
  (in-memory, Redis, Valkey, DynamoDB, …) persists as usual;
- on the next turn, on any replica, the SDK restores the transcript and the CLI resumes the same Claude session.

No configuration is needed. A turn that fails before its result, or a stream that is abandoned, leaves the
previously stored transcript untouched. Two agents in one session do not share a transcript.

:::caution Transcript size
The transcript grows with every turn, including tool output and base64 images, and the whole session value is
rewritten after each turn. DynamoDB caps an item at 400 KB and Firestore a document at 1 MiB; long-running
conversations on those backends should be bounded (for example with `max_turns` per run and new sessions for
new tasks).
:::

Working-directory files and CLAUDE.md memory are **not** part of the transcript and do not follow a
conversation to another replica.

## Structured Output

Set `output_format` to a JSON schema; the result comes back as `AgentReplyAny`:

```python
from pydantic import BaseModel

class Weather(BaseModel):
    city: str
    summary: str

options = ClaudeAgentOptions(output_format={"type": "json_schema", "schema": Weather.model_json_schema()})
```

## Streaming

Token streaming is supported (`execution.mode: stream`). The runner enables the SDK's partial messages and maps
them onto Agent Kernel stream events: text and thinking are keyed by `<message id>:<block index>`, and tool calls
keep their native `tool_use` id, which their results carry too. To receive reasoning text from current models,
set `thinking={"type": "adaptive", "display": "summarized"}`.

## Multimodal Input

Images (base64 or an `https` URL) and documents (PDF as base64 or URL, and text files) are sent as content
blocks. `s3://` references and other file types are rejected; voice input is not supported.

## Per-run context/state

The SDK has no per-run caller-state slot: tool handlers receive only their arguments. A `framework_context` set
on the session is **not injected** and is left untouched, with one warning logged per runner. Tools that need
session data read it through `ToolContext.get().session`.

## Native run options

`Module.run_options(spec, **options)` takes any `ClaudeAgentOptions` field, applied over the agent's options
for each run (a name that is not a field is rejected):

```python
module = ClaudeModule([general])
module.run_options(general, max_turns=5, max_budget_usd=0.50)
```

The runner owns these fields, so they are rejected in run options (and overwritten if set on the declared
options):
`resume`, `session_id`, `continue_conversation`, `fork_session`, `resume_session_at`, `resume_drops_turn`
(the conversation), `session_store`, `session_store_flush` (the transcript), `enable_file_checkpointing` (not
allowed with a session store) and `include_partial_messages` (set in stream mode).

## Human in the loop

Not supported yet: `supports_pause` is `False`. The SDK's `can_use_tool` callback waits inside the live CLI
process, so it cannot survive a restart or move to another replica. The SDK's durable `defer` hook decision is
the candidate mechanism for a later release.

## Tracing

| Backend | What you get |
|---------|--------------|
| Langfuse | An Agent Kernel span per run, plus agent and tool spans from the OpenInference Claude Agent SDK instrumentor |
| Logfire | An Agent Kernel span per run, plus `logfire.instrument_claude_agent_sdk()` |
| CloudWatch | An Agent Kernel span per run, plus agent and tool spans from the OpenInference instrumentor |
| OpenLLMetry | An Agent Kernel span per run only |

The model calls happen inside the CLI subprocess, so Python-side LLM instrumentation does not see them. The CLI
can export its own OpenTelemetry spans (`CLAUDE_CODE_ENABLE_TELEMETRY=1` and the `OTEL_*` variables in `env`);
see Anthropic's [observability guide](https://code.claude.com/docs/en/agent-sdk/observability).

## Deployment

- **Containers (ECS, Kubernetes)** are the natural fit. Size each replica for its concurrent runs (each pipeline
  consumer thread may run one CLI at a time) and set resource requests accordingly.
- **AWS Lambda** needs a **container image**: the SDK is too large for a zip package. Point `HOME` at `/tmp`, and
  raise the memory and timeout well above Agent Kernel's agent-runner defaults (512 MB, 45 s).
- The wheel is published for glibc Linux (x86_64, aarch64), macOS and Windows; Alpine (musl) images need a
  separately installed Claude Code CLI and `cli_path`.

## Terms

The Claude Agent SDK is MIT-licensed; the bundled Claude Code CLI is proprietary and used under Anthropic's
Commercial Terms of Service. Products built on it must authenticate with an API key or a supported cloud
provider, and may not be branded "Claude Code". See Anthropic's
[Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview).

## Example

See [`examples/cli/claude`](https://github.com/yaalalabs/agent-kernel/tree/develop/examples/cli/claude).
