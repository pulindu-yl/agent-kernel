from __future__ import annotations

import asyncio
import base64
import dataclasses
import inspect
import json
import logging
import typing
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, List, Mapping
from uuid import uuid4

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SdkMcpTool,
    ServerToolResultBlock,
)
from claude_agent_sdk import StreamEvent as ClaudeStreamEvent
from claude_agent_sdk import TextBlock, ToolAnnotations, ToolResultBlock, UserMessage, create_sdk_mcp_server
from pydantic import Field, create_model

from ...core import Agent as BaseAgent
from ...core import Module
from ...core import Runner as BaseRunner
from ...core import Runtime, Session, ToolBuilder, ToolContext
from ...core.builder import A2ACardBuilder
from ...core.config import AKConfig
from ...core.event import (
    MessageEnd,
    MessageStart,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningStart,
    StreamEvent,
    TextDelta,
    ToolCallArgs,
    ToolCallEnd,
    ToolCallResult,
    ToolCallStart,
)
from ...core.model import (
    AgentReply,
    AgentReplyAny,
    AgentReplyText,
    AgentRequest,
    AgentRequestAny,
    AgentRequestFile,
    AgentRequestImage,
    AgentRequestText,
)
from ...core.util.error_util import user_facing_error_message
from ...trace import Trace

FRAMEWORK = "claude"

_log = logging.getLogger("ak.claude.runner")


class ClaudeSession:
    """
    ClaudeSession stores each agent's Claude conversation: the Claude session id and the transcript the CLI
    mirrored out of it.

    The CLI keeps its transcript on local disk, which another replica does not have, so the transcript is kept
    here and persisted with the Agent Kernel session. Entries are the CLI's own transcript lines as plain JSON
    objects, so a pickled session carries no SDK objects. Each agent has its own conversation, because the CLI
    records an agent's system prompt in its transcript.
    """

    def __init__(self) -> None:
        """
        Initializes a ClaudeSession instance.
        """
        self._session_ids: dict[str, str] = {}
        self._transcripts: dict[str, dict[str, list[dict]]] = {}

    def session_id(self, agent_name: str) -> str | None:
        """
        :param agent_name: The agent whose conversation to read.
        :return: The Claude session id the agent's last turn ended on, or None before its first turn.
        """
        return self._session_ids.get(agent_name)

    def transcript(self, agent_name: str) -> dict[str, list[dict]]:
        """
        :param agent_name: The agent whose conversation to read.
        :return: The transcript entries by subpath: "" is the main transcript, "subagents/agent-<id>" a subagent's.
        """
        return self._transcripts.get(agent_name, {})

    def save(self, agent_name: str, session_id: str, transcript: dict[str, list[dict]]) -> None:
        """
        Records the conversation an agent's turn ended on.
        :param agent_name: The agent whose conversation to record.
        :param session_id: The Claude session id from the turn's result.
        :param transcript: The transcript entries by subpath.
        """
        self._session_ids[agent_name] = session_id
        self._transcripts[agent_name] = transcript


class _ClaudeSessionStore:
    """
    The SDK `SessionStore` for one run, backed by the agent's conversation in the ClaudeSession.

    The SDK mirrors transcript entries to `append()` while the CLI runs, and on `resume` restores the transcript
    from `load()` before spawning the CLI. Appends are buffered and saved only by `commit()`, once the turn's
    result arrives, so a failed or abandoned turn leaves the stored conversation as it was. `project_key` is
    ignored: the store is already scoped to one agent's conversation, so the CLI's working directory does not
    need to match across replicas.
    """

    def __init__(self, session: ClaudeSession, agent_name: str) -> None:
        """
        :param session: The ClaudeSession holding the conversation.
        :param agent_name: The agent whose conversation this run continues.
        """
        self._session = session
        self._agent_name = agent_name
        self._pending: dict[tuple[str, str], list[dict]] = {}
        self._seen: dict[tuple[str, str], set[str]] = {}

    def _stored(self, session_id: str, subpath: str) -> list[dict]:
        """
        :return: The saved entries of a transcript, or none for any Claude session other than the saved one.
        """
        if session_id != self._session.session_id(self._agent_name):
            return []
        return self._session.transcript(self._agent_name).get(subpath, [])

    async def append(self, key: Mapping[str, Any], entries: list[dict]) -> None:
        """
        Buffers mirrored transcript entries, skipping any whose `uuid` is already stored.
        :param key: The SDK SessionKey the entries belong to.
        :param entries: The transcript entries.
        """
        slot = (key["session_id"], key.get("subpath") or "")
        seen = self._seen.setdefault(slot, {entry["uuid"] for entry in self._stored(*slot) if entry.get("uuid")})
        buffer = self._pending.setdefault(slot, [])
        for entry in entries:
            uid = entry.get("uuid")
            if uid and uid in seen:
                continue
            if uid:
                seen.add(uid)
            buffer.append(entry)

    async def load(self, key: Mapping[str, Any]) -> list[dict] | None:
        """
        :param key: The SDK SessionKey to load.
        :return: The saved entries followed by this run's buffered ones, or None when there are none.
        """
        slot = (key["session_id"], key.get("subpath") or "")
        return [*self._stored(*slot), *self._pending.get(slot, [])] or None

    async def list_subkeys(self, key: Mapping[str, Any]) -> list[str]:
        """
        :param key: The SDK key naming the Claude session.
        :return: The subagent transcripts held for it, so a resume restores them too.
        """
        session_id = key["session_id"]
        stored = self._session.transcript(self._agent_name) if session_id == self._session.session_id(self._agent_name) else {}
        subpaths = {subpath for subpath in stored if subpath}
        subpaths |= {subpath for (pending_id, subpath) in self._pending if pending_id == session_id and subpath}
        return sorted(subpaths)

    def commit(self, session_id: str) -> None:
        """
        Saves this run's buffered entries onto the conversation the turn ended on.
        :param session_id: The session id from the turn's ResultMessage.
        """
        same = session_id == self._session.session_id(self._agent_name)
        transcript = {subpath: list(entries) for subpath, entries in self._session.transcript(self._agent_name).items()} if same else {}
        for (pending_id, subpath), entries in self._pending.items():
            if pending_id == session_id:
                transcript.setdefault(subpath, []).extend(entries)
        self._session.save(self._agent_name, session_id, transcript)


class _ClaudeStreamMapper:
    """
    Maps one run's SDK messages onto Agent Kernel stream events.

    Boundaries come from the raw Anthropic stream events the SDK forwards with `include_partial_messages`; tool
    results from the `ToolResultBlock`s in the following user message. Ids are the framework's own: a text or
    thinking stream is `<message id>:<block index>`, a tool call keeps its `tool_use` id. Traffic inside a
    subagent (`parent_tool_use_id` set) is not emitted. One instance per run: it holds the run's open blocks.
    """

    def __init__(self) -> None:
        self._message_id: str | None = None
        self._open: dict[int, tuple[str, str]] = {}  # block index -> (kind, stream id)

    def events(self, message: Any) -> list[StreamEvent]:
        """
        :param message: Any message the SDK yields.
        :return: The Agent Kernel events it produces, possibly none.
        """
        if isinstance(message, ClaudeStreamEvent):
            return self._raw_events(message.event) if message.parent_tool_use_id is None else []
        if isinstance(message, (UserMessage, AssistantMessage)) and message.parent_tool_use_id is None and isinstance(message.content, list):
            return [
                ToolCallResult(tool_call_id=block.tool_use_id, content=self._as_text(block.content))
                for block in message.content
                if isinstance(block, (ToolResultBlock, ServerToolResultBlock))
            ]
        return []

    def close(self) -> list[StreamEvent]:
        """
        :return: The closing events for any block left open by a stream that ended mid-block.
        """
        return [event for index in sorted(self._open) for event in self._close(index)]

    def _raw_events(self, event: dict[str, Any]) -> list[StreamEvent]:
        """
        :param event: A raw Anthropic stream event.
        :return: The Agent Kernel events it produces.
        """
        kind = event.get("type")
        if kind == "message_start":
            self._message_id = (event.get("message") or {}).get("id") or uuid4().hex
            return []
        index = event.get("index")
        if not isinstance(index, int):
            return []
        if kind == "content_block_start":
            return self._open_block(index, event.get("content_block") or {})
        if kind == "content_block_delta":
            return self._delta(index, event.get("delta") or {})
        if kind == "content_block_stop":
            return self._close(index)
        return []

    def _open_block(self, index: int, block: dict[str, Any]) -> list[StreamEvent]:
        events = self._close(index)
        block_type = block.get("type")
        if block_type in ("text", "thinking"):
            stream_id = f"{self._message_id or uuid4().hex}:{index}"
            self._open[index] = (block_type, stream_id)
            if block_type == "text":
                events.append(MessageStart(message_id=stream_id))
                if block.get("text"):
                    events.append(TextDelta(message_id=stream_id, content=block["text"]))
            else:
                events.append(ReasoningStart(message_id=stream_id))
                if block.get("thinking"):
                    events.append(ReasoningDelta(message_id=stream_id, content=block["thinking"]))
        elif block_type in ("tool_use", "server_tool_use") and block.get("id"):
            self._open[index] = ("tool", block["id"])
            name = (block.get("name") or "").removeprefix(f"mcp__{ClaudeToolBuilder.MCP_SERVER}__")
            events.append(ToolCallStart(tool_call_id=block["id"], name=name))
        return events

    def _delta(self, index: int, delta: dict[str, Any]) -> list[StreamEvent]:
        opened = self._open.get(index)
        if opened is None:
            return []
        kind, stream_id = opened
        delta_type = delta.get("type")
        if kind == "text" and delta_type == "text_delta" and delta.get("text"):
            return [TextDelta(message_id=stream_id, content=delta["text"])]
        if kind == "thinking" and delta_type == "thinking_delta" and delta.get("thinking"):
            return [ReasoningDelta(message_id=stream_id, content=delta["thinking"])]
        if kind == "tool" and delta_type == "input_json_delta" and delta.get("partial_json"):
            return [ToolCallArgs(tool_call_id=stream_id, delta=delta["partial_json"])]
        return []

    def _close(self, index: int) -> list[StreamEvent]:
        opened = self._open.pop(index, None)
        if opened is None:
            return []
        kind, stream_id = opened
        if kind == "text":
            return [MessageEnd(message_id=stream_id)]
        if kind == "thinking":
            return [ReasoningEnd(message_id=stream_id)]
        return [ToolCallEnd(tool_call_id=stream_id)]

    @staticmethod
    def _as_text(value: Any) -> str:
        """
        :param value: A tool result: a string, a list of content blocks, or any JSON value.
        :return: The text content for `ToolCallResult`.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value and all(isinstance(block, dict) and block.get("type") == "text" for block in value):
            return "".join(str(block.get("text", "")) for block in value)
        return json.dumps(value, default=str)


class ClaudeRunner(BaseRunner):
    """
    ClaudeRunner class provides a runner for Claude Agent SDK agents.

    Each run spawns its own `ClaudeSDKClient` (one Claude Code CLI subprocess), created after the ToolContext is
    set so the in-process tool handlers, which run in tasks the client spawns, inherit it.
    """

    def __init__(self) -> None:
        """
        Initializes a ClaudeRunner instance.
        """
        super().__init__(FRAMEWORK)
        self._context_warned = False

    @staticmethod
    def _session(session: Session) -> ClaudeSession:
        """
        Returns the Claude session associated with the provided session.
        :param session: The session to retrieve the Claude session for.
        :return: ClaudeSession instance.
        """
        if session is None:
            return ClaudeSession()
        claude_session: ClaudeSession = session.get(FRAMEWORK) or session.set(FRAMEWORK, ClaudeSession())
        return claude_session

    @staticmethod
    def _process_requests(requests: list[AgentRequest]) -> tuple[str, list[dict[str, Any]]]:
        """
        Process requests and extract prompt text and Anthropic content blocks.
        :param requests: The requests to process.
        :return: Tuple of (prompt, content).
        :raises ValueError: For an image or file the Claude Agent SDK cannot take.
        """
        prompt = ""
        content: list[dict[str, Any]] = []

        for req in requests:
            if isinstance(req, AgentRequestAny):
                continue

            if isinstance(req, AgentRequestText):
                prompt = prompt + "\n" + req.prompt if prompt else req.prompt
                content.append({"type": "text", "text": req.prompt})

            elif isinstance(req, AgentRequestImage):
                if not req.image_data:
                    raise ValueError("no image input provided")
                if req.image_data.startswith(("http://", "https://")):
                    content.append({"type": "image", "source": {"type": "url", "url": req.image_data}})
                elif req.image_data.startswith("s3://"):
                    raise ValueError("s3:// image references are not supported by the Claude Agent SDK; provide base64 data or an https URL")
                elif not req.mime_type:
                    raise ValueError("mime_type is missing for image input, either in the base64 or explicitly")
                else:
                    content.append({"type": "image", "source": {"type": "base64", "media_type": req.mime_type, "data": req.image_data}})

            elif isinstance(req, AgentRequestFile):
                if not req.file_data:
                    raise ValueError("no file input provided")
                if req.file_data.startswith(("http://", "https://")):
                    content.append({"type": "document", "source": {"type": "url", "url": req.file_data}})
                elif req.file_data.startswith("s3://"):
                    raise ValueError("s3:// file references are not supported by the Claude Agent SDK; provide base64 data or an https URL")
                elif not req.mime_type:
                    raise ValueError("mime_type is missing for file input, either in the base64 or explicitly")
                elif req.mime_type == "application/pdf":
                    content.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": req.file_data}})
                elif req.mime_type.startswith("text/"):
                    text = base64.b64decode(req.file_data).decode("utf-8", errors="replace")
                    content.append({"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": text}})
                else:
                    raise ValueError(f"file type '{req.mime_type}' is not supported by the Claude Agent SDK; send a PDF or a text file")

        return prompt, content

    @staticmethod
    def _prompt(content: list[dict[str, Any]]) -> str | AsyncIterator[dict[str, Any]]:
        """
        Text-only content goes as a string; content with an image or document goes as one streaming-input user
        message, the only form the SDK accepts it in.
        :param content: The content blocks for this turn.
        :return: The prompt for `ClaudeSDKClient.query`.
        """
        if all(block["type"] == "text" for block in content):
            return "\n".join(block["text"] for block in content)

        async def message() -> AsyncIterator[dict[str, Any]]:
            yield {"type": "user", "message": {"role": "user", "content": content}, "parent_tool_use_id": None}

        return message()

    def _warn_framework_context(self, session: Session) -> None:
        """
        The Claude Agent SDK has no per-run caller-state slot, so a framework_context is not injected. Warn once,
        and leave the stored context untouched.
        :param session: The session to check.
        """
        if not self._context_warned and session is not None and session.get_framework_context():
            _log.warning("framework_context is set but the Claude Agent SDK has no per-run caller context/state slot; ignoring it.")
            self._context_warned = True

    def _native_options(
        self, agent: Any, options: Mapping[str, Any], claude_session: ClaudeSession, store: _ClaudeSessionStore, *, stream: bool
    ) -> ClaudeAgentOptions:
        """
        Builds the options for one run: the agent's options, the resolved run options over them, and the fields
        Agent Kernel owns written last. The agent's tools, and its subagents' tools, are served by an in-process MCP
        server and approved by name, the same as a bound tool simply runs in every other framework. Subagents become
        the SDK's `agents=` definitions, delegated to through the `Agent` tool, which is enabled and approved.
        :param agent: The ClaudeAgent being run.
        :param options: The run options resolved for this run.
        :param claude_session: The ClaudeSession holding the agent's conversation.
        :param store: The store mirroring this run's transcript.
        :param stream: Whether this is a streamed run.
        :return: A new ClaudeAgentOptions; the agent's own is untouched.
        """
        spec: ClaudeAgentSpec = agent.agent
        declared = dataclasses.replace(spec.options, **options)
        ak_owned: dict[str, Any] = {"session_store": store, "session_store_flush": "batched"}
        if claude_session.transcript(agent.name).get(""):
            ak_owned["resume"] = claude_session.session_id(agent.name)
        else:
            ak_owned["session_id"] = str(uuid4())
        # A subagent runs inside this session and can only use tools that exist here, so its tools are served too.
        served = {tool.name: tool for agent_spec in (spec, *spec.subagents) for tool in agent_spec.tools}
        approved = [ClaudeToolBuilder.mcp_tool_name(name) for name in served]
        if served:
            server = create_sdk_mcp_server(ClaudeToolBuilder.MCP_SERVER, tools=list(served.values()))
            # A dict whenever the agent has tools: ClaudeAgent rejects a config-file path at declaration.
            servers = declared.mcp_servers if isinstance(declared.mcp_servers, dict) else {}
            ak_owned["mcp_servers"] = {**servers, ClaudeToolBuilder.MCP_SERVER: server}
        if spec.subagents:
            ak_owned["agents"] = {**(declared.agents or {}), **{s.name: self._subagent(s) for s in spec.subagents}}
            if isinstance(declared.tools, list) and "Agent" not in declared.tools:
                ak_owned["tools"] = [*declared.tools, "Agent"]
            approved.append("Agent")
        if approved:
            ak_owned["allowed_tools"] = declared.allowed_tools + [name for name in approved if name not in declared.allowed_tools]
        if stream:
            ak_owned["include_partial_messages"] = True
        native: ClaudeAgentOptions = dataclasses.replace(spec.options, **self._native_kwargs(options, **ak_owned))
        return native

    @staticmethod
    def _subagent(spec: ClaudeAgentSpec) -> AgentDefinition:
        """
        Converts a subagent declaration into the SDK's `AgentDefinition`: its description, system prompt and model,
        and the tools it may use. With `tools` left at the SDK default (None) the subagent may use every tool of the
        session, as the SDK defines; otherwise it may use the listed built-in tools plus its own Agent Kernel tools.
        :param spec: The subagent declaration.
        :return: The AgentDefinition for the parent's `agents=`.
        """
        declared = spec.options.system_prompt
        # A custom prompt carries its text under "prompt"; ClaudeAgent rejects the preset and file forms at declaration.
        prompt = str(declared.get("prompt", "")) if isinstance(declared, Mapping) else declared or ""
        tools = spec.options.tools
        allowed = (
            None if tools is None else [*(tools if isinstance(tools, list) else []), *(ClaudeToolBuilder.mcp_tool_name(t.name) for t in spec.tools)]
        )
        return AgentDefinition(description=spec.description or spec.name, prompt=prompt, tools=allowed, model=spec.options.model)

    @staticmethod
    def _error(result: ResultMessage) -> RuntimeError:
        """
        :param result: A ResultMessage with `is_error` set.
        :return: An error naming why the turn failed. An API failure arrives as subtype "success", so the HTTP
            status is named instead.
        """
        detail = "; ".join(result.errors or []) or result.result or ""
        if result.subtype != "success":
            reason = result.subtype
        else:
            reason = f"API error {result.api_error_status}" if result.api_error_status else "error result"
        return RuntimeError(f"Claude Agent SDK run failed ({reason}){': ' + detail if detail else ''}")

    async def run(self, agent: Any, session: Session, requests: list[AgentRequest]) -> AgentReply:
        """
        Runs the Claude agent with the provided multi modal inputs.
        :param agent: The Claude agent to run.
        :param session: The session to use for the agent.
        :param requests: The requests to the agent.
        :return: The result of the agent's execution.
        """
        context: ToolContext | None = None
        prompt = ""
        try:
            context = ToolContext(Runtime.current(), agent, session, requests).set()
            prompt, content = self._process_requests(requests)

            if not content:
                return AgentReplyText(response="Sorry. No valid content found in the requests")

            # Resolved once per run: the static options with a declared factory's result merged over them.
            options = await agent.resolve_run_options(session, requests)
            self._warn_framework_context(session)
            claude_session = self._session(session)
            store = _ClaudeSessionStore(claude_session, agent.name)

            texts: list[str] = []
            result: ResultMessage | None = None
            async with ClaudeSDKClient(options=self._native_options(agent, options, claude_session, store, stream=False)) as client:
                await client.query(self._prompt(content))
                async for message in client.receive_response():
                    if isinstance(message, ResultMessage):
                        result = message
                    elif isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
                        texts.extend(block.text for block in message.content if isinstance(block, TextBlock))

            if result is None:
                raise RuntimeError("Claude Agent SDK run ended without a result")
            # The transcript is whole once the result arrives, even for an error result such as max turns.
            store.commit(result.session_id)
            if result.is_error:
                raise self._error(result)

            structured = AgentReplyAny.from_output(result.structured_output, prompt)
            if structured is not None:
                return structured
            return AgentReplyText(response=result.result if result.result is not None else "\n".join(texts), prompt=prompt)
        except Exception as e:
            return AgentReplyText(response=user_facing_error_message(e), prompt=prompt)
        finally:
            if context is not None:
                context.reset()

    async def stream(self, agent: Any, session: Session, requests: list[AgentRequest]) -> AsyncGenerator[StreamEvent, None]:
        """
        Streams the Claude agent response as Agent Kernel stream events. The client lives inside the generator, so
        closing the generator early disconnects it and stops the CLI.
        :param agent: The Claude agent to run.
        :param session: The session to use for the agent.
        :param requests: The requests to the agent.
        :return: An async generator yielding StreamEvent objects.
        """
        context: ToolContext | None = None
        try:
            context = ToolContext(Runtime.current(), agent, session, requests).set()
            _, content = self._process_requests(requests)

            if not content:
                return

            # Resolved once per run: the static options with a declared factory's result merged over them.
            options = await agent.resolve_run_options(session, requests)
            self._warn_framework_context(session)
            claude_session = self._session(session)
            store = _ClaudeSessionStore(claude_session, agent.name)
            mapper = _ClaudeStreamMapper()  # local — the runner is shared across sessions

            result: ResultMessage | None = None
            async with ClaudeSDKClient(options=self._native_options(agent, options, claude_session, store, stream=True)) as client:
                await client.query(self._prompt(content))
                async for message in client.receive_response():
                    if isinstance(message, ResultMessage):
                        result = message
                        continue
                    for event in mapper.events(message):
                        yield event

            for event in mapper.close():
                yield event

            if result is None:
                raise RuntimeError("Claude Agent SDK stream ended without a result")
            # After a normal drain only — a disconnect or error leaves the stored transcript intact.
            store.commit(result.session_id)
            if result.is_error:
                raise self._error(result)
        finally:
            if context is not None:
                context.reset()


@dataclass
class ClaudeAgentSpec:
    """
    Declares a Claude Agent SDK agent. The SDK configures an agent with `ClaudeAgentOptions`, which carries no
    name or description; this adds those two, the agent's tools and its subagents, and passes the options through
    unchanged.

    name: str : the name Agent Kernel registers the agent under
    description: str : what the agent does, used for routing, the A2A card and, for a subagent, delegation
    options: ClaudeAgentOptions : the SDK's options, passed to every run
    tools: list[SdkMcpTool] : in-process tools, from `ClaudeToolBuilder.bind` or the SDK's `@tool`
    subagents: list[ClaudeAgentSpec] : agents this one delegates to, as Claude subagents through its `Agent` tool;
        each becomes an SDK `AgentDefinition` built from its description, system prompt, model and tools
    """

    name: str
    description: str = ""
    options: ClaudeAgentOptions = field(default_factory=ClaudeAgentOptions)
    tools: list[SdkMcpTool] = field(default_factory=list)
    subagents: list[ClaudeAgentSpec] = field(default_factory=list)


class ClaudeAgent(BaseAgent):
    """
    ClaudeAgent class provides an agent wrapping for Claude Agent SDK agents.
    """

    RESERVED_RUN_OPTIONS: ClassVar[Mapping[str, str]] = {
        "resume": "the runner resumes the Claude session stored on the Agent Kernel session",
        "session_id": "the runner assigns the Claude session id on a conversation's first turn",
        "continue_conversation": "the runner resumes the Claude session stored on the Agent Kernel session",
        "fork_session": "the runner keeps one transcript per agent conversation",
        "resume_session_at": "the runner keeps one transcript per agent conversation",
        "resume_drops_turn": "the runner keeps one transcript per agent conversation",
        "session_store": "the runner keeps the transcript on the Agent Kernel session through its own SessionStore",
        "session_store_flush": "the runner saves the transcript when the turn's result arrives",
        "enable_file_checkpointing": "the SDK does not allow file checkpointing together with a SessionStore",
        "include_partial_messages": "set by the runner in stream mode, where its token events are mapped",
    }

    def __init__(self, name: str, runner: ClaudeRunner, spec: ClaudeAgentSpec) -> None:
        """
        Initializes a ClaudeAgent instance.
        :param name: Name of the agent.
        :param runner: Runner associated with the agent.
        :param spec: The agent declaration. Its options and tool list are copied, so ones shared between
            declarations are not altered when this agent's system instructions and tools are added.
        """
        super().__init__(name, runner)
        spec.options = dataclasses.replace(spec.options)
        spec.tools = list(spec.tools)
        self._agent = spec
        self._attach_system_tools()
        self._setup_system_prompt()
        self._check_mcp_servers(spec.options.mcp_servers)
        self._check_subagents()

    @property
    def agent(self) -> ClaudeAgentSpec:
        """
        Returns the Claude agent declaration.
        """
        return self._agent

    def _check_mcp_servers(self, servers: Any) -> None:
        """
        :param servers: An `mcp_servers` value for this agent.
        :raises ValueError: If it names the server the runner serves the agent's tools from, or, for an agent with
            tools, is a config-file path the runner cannot add that server to.
        """
        if isinstance(servers, Mapping):
            if ClaudeToolBuilder.MCP_SERVER in servers:
                raise ValueError(f"MCP server name '{ClaudeToolBuilder.MCP_SERVER}' is reserved by the '{FRAMEWORK}' adapter for agent tools")
        elif self.agent.tools:
            raise ValueError(f"mcp_servers for agent '{self.name}' must be a dict, so the '{FRAMEWORK}' adapter can add its tool server")

    def _check_subagents(self) -> None:
        """
        :raises ValueError: If a subagent has no name, shares a name with an `agents=` entry, or has a system prompt an
            `AgentDefinition` cannot carry (the `claude_code` preset or a file).
        """
        native = set(self.agent.options.agents or {})
        for subagent in self.agent.subagents:
            if not subagent.name:
                raise ValueError(f"A subagent of agent '{self.name}' has no name")
            if subagent.name in native:
                raise ValueError(f"Subagent '{subagent.name}' of agent '{self.name}' is also defined in its options' agents=")
            prompt = subagent.options.system_prompt
            if isinstance(prompt, Mapping) and prompt.get("type") != "custom":
                raise ValueError(f"Subagent '{subagent.name}' of agent '{self.name}' needs a string system prompt, not a {prompt.get('type')} prompt")

    def validate_run_options(self, options: Mapping[str, Any]) -> None:
        """
        Rejects reserved keys, keys that are not ClaudeAgentOptions fields, and an `mcp_servers` value the runner
        cannot add its tool server to.
        :param options: The run options about to be declared for this agent.
        :raises ValueError: Naming the offending keys.
        """
        super().validate_run_options(options)
        unknown = sorted(set(options) - {f.name for f in dataclasses.fields(ClaudeAgentOptions)})
        if unknown:
            raise ValueError(f"Run option(s) {unknown} for agent '{self.name}' are not ClaudeAgentOptions fields")
        if "mcp_servers" in options:
            self._check_mcp_servers(options["mcp_servers"])

    def get_description(self) -> str:
        """
        Returns the description of the agent, falling back to a plain-string system prompt.
        """
        if self.agent.description:
            return self.agent.description
        prompt = self.agent.options.system_prompt
        return prompt if isinstance(prompt, str) else ""

    def override_system_prompt(self, prompt: str) -> None:
        """
        Appends additional instructions to the system prompt in the form it was declared: a string, a custom
        prompt, or the `claude_code` preset's `append`. The SDK's default (None) sends an empty system prompt, so
        the instructions become the prompt. A prompt loaded from a file cannot be extended.
        :param prompt: The instruction text to append.
        """
        if not prompt:
            return
        current = self.agent.options.system_prompt
        declared: dict[str, Any] = dict(current) if isinstance(current, Mapping) else {}
        if current is None:
            updated: Any = prompt
        elif isinstance(current, str):
            updated = f"{current}\n\n{prompt}"
        elif declared.get("type") == "preset":
            updated = {**declared, "append": f"{declared['append']}\n\n{prompt}" if declared.get("append") else prompt}
        elif declared.get("type") == "custom":
            updated = {**declared, "prompt": f"{declared.get('prompt', '')}\n\n{prompt}"}
        else:
            _log.warning(f"Agent '{self.name}' loads its system prompt from a file, which cannot be extended; system instructions were not added")
            return
        self.agent.options.system_prompt = updated

    def attach_tool(self, tool: Any) -> None:
        """
        Accepts a raw Callable, wraps it with ClaudeToolBuilder, and adds it to the agent's tools unless a tool of
        that name is already present. Called by the base Agent._attach_system_tools() at init.
        :param tool: Raw Python callable to attach.
        """
        names = {t.name for t in self.agent.tools}
        self.agent.tools.extend(t for t in ClaudeToolBuilder.bind([tool]) if t.name not in names)

    def get_a2a_card(self) -> Any:
        """
        Returns the A2A AgentCard associated with the agent.
        """
        from a2a.types import AgentSkill

        skills = [AgentSkill(id=t.name, name=t.name, description=t.description or "", tags=[]) for t in self.agent.tools]
        return A2ACardBuilder.build(name=self.name, description=self.get_description(), skills=skills)


class ClaudeModule(Module):
    """
    ClaudeModule class provides a module for Claude Agent SDK agents.
    """

    def __init__(self, agents: list[ClaudeAgentSpec], runner: ClaudeRunner | None = None) -> None:
        """
        Initializes a ClaudeModule instance.
        :param agents: List of agents in the module.
        :param runner: Custom runner associated with the module.
        """
        super().__init__()
        self.runner: ClaudeRunner
        if runner is not None:
            self.runner = runner
        elif AKConfig.get().trace.enabled:
            self.runner = typing.cast(ClaudeRunner, Trace.get().claude())
        else:
            self.runner = ClaudeRunner()
        self.load(agents)

    def _wrap(self, agent: ClaudeAgentSpec, agents: List[ClaudeAgentSpec]) -> BaseAgent:
        """
        Wraps the provided agent in a ClaudeAgent instance.
        :param agent: Agent to wrap.
        :param agents: List of agents in the module.
        :return: ClaudeAgent instance.
        :raises ValueError: If the agent has no name.
        """
        if not agent.name:
            raise ValueError("ClaudeAgentSpec passed to ClaudeModule must have a name")
        return ClaudeAgent(agent.name, self.runner, agent)

    def load(self, agents: list[ClaudeAgentSpec]) -> ClaudeModule:
        """
        Loads the specified agents into the module. By replacing the current agents.
        :param agents: List of agents to load.
        :return: ClaudeModule instance.
        """
        super().load(agents)
        return self


class ClaudeToolBuilder(ToolBuilder):
    """
    Tool builder for the Claude Agent SDK. Wraps generic tool functions into the SDK's in-process `SdkMcpTool`s;
    AK tools reach execution context via `ToolContext.get()`.
    """

    MCP_SERVER: ClassVar[str] = "ak"
    """The in-process MCP server the runner serves an agent's tools from; the model calls them mcp__ak__<tool>."""

    @classmethod
    def bind(cls, funcs: list[Callable], **options: Any) -> list[SdkMcpTool]:
        """
        Bind generic tool functions to Claude Agent SDK tools.

        Keyword options become the tools' `ToolAnnotations` (e.g. `readOnlyHint=True`, `maxResultSizeChars=N`) and
        apply to every function in the call, so tools wanting different options are bound in separate calls.

        :param funcs: List of generic tool functions to bind.
        :param options: Keyword arguments for `ToolAnnotations`.
        :return: List of `SdkMcpTool` definitions.
        :raises TypeError: If any item in funcs is not callable.
        """
        annotations = ToolAnnotations(**options) if options else None
        return [cls._wrap(func, annotations) for func in funcs]

    @classmethod
    def mcp_tool_name(cls, name: str) -> str:
        """
        :param name: A tool name.
        :return: The name the model calls the tool by, for example in an `AgentDefinition`'s `tools`.
        """
        return f"mcp__{cls.MCP_SERVER}__{name}"

    @classmethod
    def _wrap(cls, func: Callable, annotations: ToolAnnotations | None) -> SdkMcpTool:
        """
        Wraps a generic tool function as an `SdkMcpTool`. The SDK takes a JSON schema rather than a function, so the
        schema is built from the signature (parameters without a default are required) and the description is the
        docstring. Sync functions run on a worker thread, which keeps the SDK's message loop serviced.

        :param func: The generic tool function to wrap.
        :param annotations: The annotations to attach, if any.
        :return: The SdkMcpTool serving the function.
        :raises TypeError: If the function is not callable.
        """
        if not callable(func):
            raise TypeError(f"Expected a callable, got {type(func).__name__}")

        name = getattr(func, "__name__", type(func).__name__)
        try:
            hints = typing.get_type_hints(func)
        except Exception:
            hints = {}
        fields: dict[str, Any] = {}
        for param in inspect.signature(func).parameters.values():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            annotation = hints.get(param.name, Any if param.annotation is inspect.Parameter.empty else param.annotation)
            fields[param.name] = (annotation, Field(... if param.default is inspect.Parameter.empty else param.default))
        arguments = create_model(f"{name}_arguments", **fields)
        is_async = inspect.iscoroutinefunction(func)

        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            values = arguments.model_validate(args)
            kwargs = {key: getattr(values, key) for key in arguments.model_fields}
            result = await func(**kwargs) if is_async else await asyncio.to_thread(func, **kwargs)
            text = result if isinstance(result, str) else json.dumps(result, default=str)
            return {"content": [{"type": "text", "text": text}]}

        description = inspect.getdoc(func) or name
        return SdkMcpTool(name=name, description=description, input_schema=arguments.model_json_schema(), handler=handler, annotations=annotations)
