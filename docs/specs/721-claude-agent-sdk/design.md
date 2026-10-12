# #721: Integrate the Claude Agent SDK as a Framework Adapter

Adds the Claude Agent SDK as an AK framework adapter (`framework/claude/`), following the
`Agent`/`Runner`/`Module`/`ToolBuilder` pattern of the existing adapters. What it adds over every
existing adapter is the Claude Code agent harness itself: built-in file, shell and web tools, subagents,
hooks, a permission engine and automatic context compaction. The SDK drives a Claude Code CLI subprocess
and keeps its transcript on local disk, so the design keeps that transcript on the AK session through
the SDK's `SessionStore` interface, which lets a conversation continue on any replica. Written alongside
the implementation on `feature/721-claude-agent-sdk`; the requirements describe the implemented
behaviour, and departures from the house patterns are called out as such.

## Motivation

- AK can already call Claude models, but only as a model behind another framework's loop:
  - Pydantic AI natively (`pydantic-ai-slim[anthropic]`), LangGraph via `langchain-anthropic`, and the
    OpenAI Agents SDK and Google ADK via LiteLLM.
  - None of them gives an agent Claude Code's harness: tools that work a filesystem and shell,
    subagents, skills, hooks, permission modes and auto-compaction.
- The Claude Agent SDK is that harness packaged as a library. Others wrap it the same way:
  Microsoft Agent Framework ships an `agent-framework-claude` package with its own `ClaudeAgent` class;
  AWS AgentCore Runtime hosts SDK agents.
- The SDK has no named agent object, unlike every framework AK wraps today:
  - Its unit of configuration is `ClaudeAgentOptions` (`claude_agent_sdk/types.py:1971`, SDK 0.2.165),
    with no `name` or `description`.
  - `AgentDefinition` (`types.py:108`) is named only as a dict key and describes **subagents**; its
    `tools` are names, not functions.
- It is a subprocess, not an in-process library:
  - Every `ClaudeSDKClient` spawns the bundled native CLI (252,755,128 bytes in the 0.2.165 Linux wheel;
    no Node.js) and speaks NDJSON with it.
  - Measured locally: about 0.5 s to connect and about 200 MB resident per idle process. Anthropic's
    hosting guide suggests 1 GiB RAM and 1 CPU per concurrent agent.
- Its conversation state is a local JSONL transcript, but AK serves any session on any replica:
  - The Helm chart runs `replicaCount: 2` agent runners with no shared volume
    (`ak-deployment/ak-k8s/chart/values.yaml:143-146`), and Lambda keeps nothing between invocations.
  - The SDK ships a `SessionStore` protocol (`types.py:1609`, added in 0.1.64) that mirrors transcript
    entries out while the CLI runs and restores them on `resume` before spawning the CLI
    (`claude_agent_sdk/_internal/session_resume.py:130`).
  - AK already persists each framework's session value with pickle after every run
    (`core/session/serde.py:17-27`, `core/session/dynamodb.py:107`), so the transcript can live there
    with no new configuration.

| Dimension | Pydantic AI + Anthropic (`framework/pydanticai/`) | Claude Agent SDK (proposed) |
|---|---|---|
| Version (2026-10-09) | `pydantic-ai-slim~=2.13.0` | 0.2.165, classified "Alpha", about 3–4 releases a week; bundles Claude Code CLI 2.1.294 |
| Runtime | In-process library | One Claude Code CLI subprocess per run |
| Models | Many providers, Claude included | Claude only: Anthropic API, Bedrock, Vertex, Foundry |
| Built-in tools | None | Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch, Agent, … (on by default) |
| Custom tools | Native `Tool` | In-process MCP server (`create_sdk_mcp_server`), called as `mcp__<server>__<tool>` |
| Multi-agent | Delegation via tool calls | Native subagents (`agents=` + the `Agent` tool) |
| Conversation state | Message list AK stores | Local JSONL transcript; `SessionStore` for external persistence |
| Durable human-in-the-loop | `DeferredToolRequests` | `PreToolUse` hook returning `defer` (`types.py:441`, `types.py:1356`), one tool call per turn only |
| Tracing | Full OpenInference / Logfire instrumentation | Model calls run inside the CLI; agent and tool spans via the SDK's OpenInference or Logfire instrumentor |
| Licence and terms | MIT | SDK MIT; bundled CLI proprietary under Anthropic's Commercial Terms; API-key or cloud auth only; "Claude Code" may not be used as a product name |

Sources: code.claude.com/docs/en/agent-sdk/overview, /hosting, /session-storage; pypi.org/project/claude-agent-sdk;
the installed 0.2.165 package — retrieved 2026-10-09.

**Recommendation: integrate as an adapter, positioned for agents that need the harness.** For a plain
tool-calling agent on Claude, Pydantic AI with its Anthropic provider stays the lighter choice; the docs
say so.

## Requirements

The Pydantic AI adapter (`framework/pydanticai/pydanticai.py`) is the structural template, per
`.agents/skills/ak-dev-new-framework-integration`.

### Package layout and naming

- Adapter at `ak-py/src/agentkernel/framework/claude/` (`__init__.py` + `claude.py`); public alias
  `ak-py/src/agentkernel/claude.py`.
- Classes in the order the other adapters use: `ClaudeSession`, `ClaudeRunner(Runner)`,
  `ClaudeAgentSpec`, `ClaudeAgent(Agent)`, `ClaudeModule(Module)`, `ClaudeToolBuilder(ToolBuilder)`
  (`claude.py:65-776`).
  - Helpers are private (`_ClaudeSessionStore`, `_ClaudeStreamMapper`), as ADK keeps `_AdkStreamMapper`.
  - Public names: `ClaudeAgentSpec`, `ClaudeModule`, `ClaudeToolBuilder`.
- `FRAMEWORK = "claude"` (`claude.py:60`) is the session key and the runner name.
- Naming avoids "Claude Code", which Anthropic's terms do not allow in product branding.

### Agent declaration — departure from "wrap the native object"

- `ClaudeAgentSpec(name, description, options, tools, subagents)` (`claude.py:577`) declares an agent.
  - It adds only what the SDK lacks (a name, a description, the agent's tools) and passes
    `ClaudeAgentOptions` through unchanged.
  - Departure: `AGENTS.md:66` asks adapters to wrap the native object. There is no native named agent to
    wrap; MAF's Claude integration defines its own class for the same reason.
- `ClaudeAgent` copies the declaration's options and tool list at init (`claude.py:625-626`), so options
  or tools shared between declarations are not altered when AK appends system instructions and tools.
- `ClaudeAgent` implements all four abstract `Agent` methods (`core/base.py:659-686`):
  - `get_description()` falls back to a plain-string system prompt.
  - `override_system_prompt()` appends in the declared form: string, custom prompt, or the
    `claude_code` preset's `append`; a file prompt is left alone with a warning. The SDK's default (None)
    sends an empty system prompt (`subprocess_cli.py:577`).
  - `attach_tool()` binds the callable and skips a name already present.
  - `get_a2a_card()` lists one skill per tool.

### Execution model

- Each run creates its own `ClaudeSDKClient`, after `ToolContext.set()`, so the in-process tool handlers
  (which run in tasks the client spawns) inherit the context.
- The client lives inside `run()`/`stream()`; closing the stream early disconnects it and stops the CLI,
  which matters because `Runtime.stream()` always `aclose()`s the runner's generator
  (`core/runtime.py:534`).
- Long-lived or pooled clients are out of scope: they need sticky routing, and the SDK offers no process
  pool.

### Conversation state

- `ClaudeSession` (`claude.py:65`) holds, per agent, the Claude session id and the transcript entries by
  subpath (`""` main, `subagents/agent-<id>` subagents).
  - One conversation per agent, because the CLI records an agent's system prompt in its transcript.
  - Entries are the CLI's transcript lines as plain JSON objects: the pickled session carries no SDK
    objects.
- `_ClaudeSessionStore` (`claude.py:108`) implements the SDK `SessionStore` protocol for one run:
  - `append()` buffers entries, deduplicated by `uuid`; `load()` returns stored then buffered entries;
    `list_subkeys()` restores subagent transcripts.
  - `commit()` saves the buffer only once the turn's `ResultMessage` arrives (`claude.py:514`,
    `claude.py:568`), so a failed or abandoned turn leaves the stored conversation as it was.
  - `project_key` is ignored: the store is already scoped to one agent's conversation, so the CLI's
    working directory need not match across replicas.
- First turn passes `session_id=<new uuid>`; later turns pass `resume=<stored id>`.
- No new configuration: the transcript is persisted by whichever session backend is configured.
- Departure from "pluggable by default": `session_store` is reserved, so a user cannot substitute their
  own store (see Open questions).

### `ClaudeRunner.run()`

- Follows the Pydantic AI lifecycle: `ToolContext` set in `try`/`finally`; run options resolved once,
  after the no-content early return.
- `_process_requests()` (`claude.py:329`) maps requests to Anthropic content blocks:
  - text; images as base64 or `https` URL; PDF documents as base64 or URL; text files decoded.
  - `s3://` references and other file types raise `ValueError`; voice is skipped.
  - Text-only content goes as a string; anything else as one streaming-input user message
    (`claude.py:379`).
- Reply: `ResultMessage.structured_output` through `AgentReplyAny.from_output()`, else
  `ResultMessage.result`, else the joined assistant text.
- An error result (`is_error`) is committed (the transcript is whole) and then reported; an API failure
  that arrives as subtype `success` names the HTTP status instead (`claude.py:465`).
- The catch-all error contract is preserved: any exception becomes
  `AgentReplyText(response=user_facing_error_message(e))`.

### `ClaudeRunner.stream()`

- Real token streaming: `include_partial_messages=True` (`claude.py:442`) and the raw Anthropic stream
  events mapped by `_ClaudeStreamMapper` (`claude.py:186`):
  - text and thinking keyed `<message id>:<block index>`; tool calls keyed by the native `tool_use` id,
    which `ToolCallResult` carries too.
  - Tool names served by the AK server are reported without the `mcp__ak__` prefix.
  - Subagent traffic (`parent_tool_use_id` set) is not emitted.
- The transcript is committed only after a full drain; an error result raises after its events.

### Tool binding

- `ClaudeToolBuilder.bind(funcs, **options)` (`claude.py:786`) returns the SDK's `SdkMcpTool`s:
  - The input schema is built from the signature with Pydantic (parameters without a default are
    required); the description is the docstring. The SDK takes a schema, not a function.
  - Sync functions run on a worker thread (`claude.py:841`), which keeps the SDK's message loop serviced
    and inherits the `ToolContext`.
  - Keyword options become the tools' `ToolAnnotations`.
- Per run, the agent's tools are served from an in-process MCP server named
  `ClaudeToolBuilder.MCP_SERVER = "ak"` (`claude.py:782`) and added to `allowed_tools`, the same as a
  bound tool simply runs in every other adapter.
  - `mcp_servers` must be a dict on an agent with tools, and the `ak` name is reserved
    (`claude.py:640`).

### Subagents — an AK convenience over the native `agents=`

- `ClaudeAgentSpec.subagents=[...]` lets a parent list other specs, as OpenAI's `handoffs=` and ADK's
  `sub_agents=` take agents.
  - Each becomes an `AgentDefinition` built from its description, system prompt, model and tools
    (`claude.py:447`); the parent serves the subagents' AK tools (`claude.py:427`) and enables and
    approves its `Agent` tool (`claude.py:438`).
  - A subagent may use only the built-in tools its own `tools=` lists plus its AK tools; left at the SDK
    default (None) it may use the session's tools.
  - Native `agents=` keeps working alongside; a name in both, a nameless subagent, or a preset or file
    subagent prompt is rejected at load (`claude.py:652`).
- Departure: this is AK translating its own objects into the SDK's, which `AGENTS.md:66` leans against.
  Kept because a subagent's tools must be served by the parent session, which otherwise has to be wired
  by hand (see Open questions).

### Run options (#754, #758)

- Any `ClaudeAgentOptions` field may be declared; a name that is not a field is rejected at declaration
  (`claude.py:667`), unlike the kwargs adapters, because the options form one dataclass.
- Options are merged with `dataclasses.replace`, and AK-owned fields are written last through
  `_native_kwargs` (`core/base.py:367`).
- `RESERVED_RUN_OPTIONS` (`claude.py:603`): `resume`, `session_id`, `continue_conversation`,
  `fork_session`, `resume_session_at`, `resume_drops_turn` (the conversation); `session_store`,
  `session_store_flush` (the transcript); `enable_file_checkpointing` (not allowed with a store);
  `include_partial_messages` (stream mode); plus the `ak` entry of `mcp_servers`.

### Framework context (#526)

- Unsupported: the SDK has no per-run caller-state slot (tool handlers receive only their arguments).
  A set `framework_context` is not injected, is left untouched, and is warned about once per runner
  (`claude.py:394`), as CrewAI does.

### Human in the loop (#606)

- `supports_pause` stays `False` (`core/base.py:403`).
  - `can_use_tool` waits inside the live CLI process, so it cannot survive a restart or move to another
    replica — the reason CrewAI and smolagents were declined
    (`docs/specs/606-human-in-the-loop/research/adapter-strategies.md:311-346`).
  - The SDK's durable `defer` decision is the candidate mechanism but works only for a single tool call
    per turn and is unverified against a live model.

### Security defaults

- The SDK's defaults are kept, per `AGENTS.md:75` ("make it the same default the native tool itself
  would use"). On a server they give the model the full Claude Code toolset on the host and load
  `~/.claude` settings (`types.py:2297`), and prompts expand `@path` mentions unless
  `verbatim_prompts=True` (`types.py:2219`).
- The docs page, quick start and example lead with the lockdown: `tools=[]`, `setting_sources=[]`, an
  explicit `permission_mode` (`dontAsk` in examples).

### Tracing

- `BaseTrace.claude()` is **non-abstract** and raises `NotImplementedError` naming the tracer
  (`trace/base.py:56`), so bring-your-own tracers written before this adapter still load; `Trace.claude()`
  dispatches (`trace/trace.py:120`).
- Traced runners for all four built-in backends wrap `run()` in the outer AK span:
  - Langfuse and CloudWatch also activate the OpenInference `ClaudeAgentSDKInstrumentor` (agent and tool
    spans).
  - Logfire calls `logfire.instrument_claude_agent_sdk()`.
  - OpenLLMetry has no SDK instrumentor, so it gets the outer span only.
- The SDK forwards `TRACEPARENT` to the CLI when a span is active (`subprocess_cli.py:843`), so the CLI's
  own OpenTelemetry export nests under AK's span.

### Packaging (`ak-py/pyproject.toml`)

- A `claude` extra (`pyproject.toml:62-65`): `claude-agent-sdk>=0.2.165,<0.3` and
  `openinference-instrumentation-claude-agent-sdk>=0.1.20`.
  - It co-resolves with every other extra in the shared `uv.lock`; no `[tool.uv].conflicts` entry is
    needed. The lock update also moves the shared `openinference-instrumentation` to 0.1.71.
  - The upper bound reflects the release cadence.

### Deployment

- No deployment code changes. The docs state the constraints:
  - The SDK is too large for a zip Lambda; AK's default agent runner (`LocalZip`, 45 s, 512 MB,
    `ak-deployment/ak-aws/serverless/variables.tf:381-385`) needs a container image and raised limits.
  - Size container replicas for one CLI process per concurrent run; no musllinux wheel, so Alpine
    images need a separately installed CLI.

### Tests

- `ak-py/tests/test_claude_runner.py`, driven by a fake `ClaudeSDKClient` replaying real SDK message
  objects, covers:
  - reply mapping, structured output, error results;
  - conversation continuity (`session_id` then `resume`, commit only after a result, a `BinarySerde`
    round-trip, per-agent conversations) and the session store;
  - native options, tool server and approvals, `ToolContext` in tool handlers;
  - reserved, unknown and colliding run options; framework-context warning; multimodal blocks;
  - stream mapping, early close, subagents, and agent prompt handling.
- `ak-py/tests/test_tool_claude.py` covers schema, handlers (sync on a worker thread keeping the
  `ToolContext`, async, Pydantic parameters) and annotations.
- Trace tests: CloudWatch and Logfire runners activate their instrumentation; a pre-existing BYO tracer
  still loads and raises clearly from `claude()`.

### Examples and docs

- `examples/cli/claude/`: the triage/math/general/weather shape of the other CLI demos, with triage
  declaring `subagents=[...]` and every agent locked down.
- `docs/docs/frameworks/claude-agent-sdk.md` and a sidebar entry; the capability tables (frameworks
  overview, runner, AG-UI, tracing, human in the loop); installation and quick start; landing, features
  and persona pages; READMEs; the bundled user skills and their evals; developer skills.

## Non-goals

- Human-in-the-loop pause and resume (follow-up once `defer` is validated against a live model).
- A per-run framework-context mapping.
- Long-lived, pooled or sticky CLI processes.
- Persisting working-directory files or CLAUDE.md memory across replicas (the transcript only).
- AK-side safety defaults that override the SDK's.
- Claude Managed Agents or an Anthropic Messages API tool-runner adapter (different products; separate
  issues if wanted).
- Changes to AK core or to any existing adapter beyond the tracing hook.

## Open questions

- **Tool context robustness:** tool handlers inherit the `ToolContext` because the SDK starts its tool
  tasks after `connect()`, verified on 0.2.165. Should each run wrap its handlers to set the context
  explicitly, so a future SDK change cannot break it silently?
- **Pluggable transcript storage:** should users be able to supply their own SDK `SessionStore` (S3,
  Redis), for long transcripts that would exceed DynamoDB's 400 KB item limit? That un-reserves
  `session_store`.
- **`subagents=`:** keep the convenience, or keep the adapter strictly thin and use native `agents=`?
- **CI:** add an Anthropic key (or Bedrock access) so `examples/cli/claude` runs in
  `.github/test-config.yaml`, and generate its `uv.lock` once the extra is published.
- **Terms:** Anthropic's Commercial Terms have an unclear clause on paying for usage on end users'
  behalf; confirm with legal for hosted deployments.
- **Live validation:** the adapter is verified against the real CLI only up to authentication; streaming,
  tools and subagents need one end-to-end run with a key.
