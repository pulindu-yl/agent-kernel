"""
Tests for the Claude Agent SDK adapter (framework/claude).

`ClaudeSDKClient` is replaced by `FakeClient`, which records the options a run built and replays a script of
real SDK message objects. A script step may also be a coroutine function, run with the client in the middle of
the stream, whose return value (if any) is yielded: that is how the tests mirror transcript entries into the run's
SessionStore (as the SDK does before yielding a ResultMessage), stamp the result with the session id the run chose,
and call tool handlers (as the CLI does through the in-process MCP server).
"""

import base64
import logging
from unittest.mock import AsyncMock

import pytest
from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    UserMessage,
)

import agentkernel.framework.claude.claude as claude_mod
from agentkernel.core import Session
from agentkernel.core.builder import SessionStoreBuilder
from agentkernel.core.model import AgentReplyAny, AgentReplyText, AgentRequestFile, AgentRequestImage, AgentRequestText
from agentkernel.core.runtime import Runtime
from agentkernel.core.session.serde import BinarySerde
from agentkernel.core.tool import ToolContext
from agentkernel.framework.claude.claude import (
    FRAMEWORK,
    ClaudeAgent,
    ClaudeAgentSpec,
    ClaudeModule,
    ClaudeRunner,
    ClaudeSession,
    ClaudeToolBuilder,
    _ClaudeSessionStore,
    _ClaudeStreamMapper,
)

AK_MCP_SERVER = ClaudeToolBuilder.MCP_SERVER


class FakeClient:
    """Stands in for ClaudeSDKClient: records what a run built, replays a scripted message list."""

    instances: list["FakeClient"] = []
    script: list = []

    def __init__(self, options: ClaudeAgentOptions):
        self.options = options
        self.prompts: list = []
        self.entered = False
        self.exited = False
        FakeClient.instances.append(self)

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False

    async def query(self, prompt):
        if isinstance(prompt, str):
            self.prompts.append(prompt)
        else:
            self.prompts.append([message async for message in prompt])

    @property
    def session_id(self) -> str:
        return self.options.resume or self.options.session_id

    async def receive_response(self):
        for step in self.script:
            message = await step(self) if callable(step) else step
            if message is not None:
                yield message


@pytest.fixture
def fake_client(monkeypatch):
    FakeClient.instances = []
    FakeClient.script = []
    monkeypatch.setattr(claude_mod, "ClaudeSDKClient", FakeClient)
    return FakeClient


@pytest.fixture
def runtime():
    with Runtime(SessionStoreBuilder.build()) as rt:
        yield rt


def _result(session_id="", result="done", is_error=False, subtype="success", structured_output=None, errors=None):
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id=session_id,
        result=result,
        structured_output=structured_output,
        errors=errors,
    )


def _mirror(*entries):
    """A script step that mirrors entries into the run's store, as the SDK does before yielding the result."""

    async def step(client):
        await client.options.session_store.append({"project_key": "p", "session_id": client.session_id}, list(entries))

    return step


def _finish(*entries, **result):
    """A script step that mirrors entries, then yields a ResultMessage carrying the session id the run chose."""

    async def step(client):
        await client.options.session_store.append({"project_key": "p", "session_id": client.session_id}, list(entries))
        return _result(session_id=client.session_id, **result)

    return step


async def _turn(fake_client, agent, session, prompt="hi"):
    """Runs one turn that mirrors a user and an assistant entry and ends in a successful result."""
    fake_client.script = [_finish({"type": "user", "uuid": f"u-{prompt}"}, {"type": "assistant", "uuid": f"a-{prompt}"})]
    return await agent.runner.run(agent, session, [AgentRequestText(prompt=prompt)])


def _module(*specs):
    return ClaudeModule(list(specs))


def _spec(name="general", **options):
    return ClaudeAgentSpec(name=name, description="A helpful agent", options=ClaudeAgentOptions(**options))


def double(x: int) -> int:
    """
    Double a number.
    :param x: The number to double.
    """
    return 2 * x


def _echo(text: str) -> str:
    """Echo the text back."""
    return text


# --- run ------------------------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_reply_is_the_result_text(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        fake_client.script = [_result(result="hello")]

        reply = await agent.runner.run(agent, Session("s"), [AgentRequestText(prompt="hi")])

        assert isinstance(reply, AgentReplyText)
        assert reply.response == "hello"
        assert reply.prompt == "hi"
        assert fake_client.instances[0].prompts == ["hi"]
        assert fake_client.instances[0].exited

    @pytest.mark.asyncio
    async def test_falls_back_to_assistant_text_when_the_result_has_none(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        fake_client.script = [AssistantMessage(content=[TextBlock(text="partial")], model="m"), _result(result=None)]

        reply = await agent.runner.run(agent, Session("s"), [AgentRequestText(prompt="hi")])

        assert reply.response == "partial"

    @pytest.mark.asyncio
    async def test_structured_output_becomes_agent_reply_any(self, fake_client, runtime):
        agent = _module(_spec(output_format={"type": "json_schema", "schema": {"type": "object"}})).get_agent("general")
        fake_client.script = [_result(result="{}", structured_output={"city": "Colombo"})]

        reply = await agent.runner.run(agent, Session("s"), [AgentRequestText(prompt="hi")])

        assert isinstance(reply, AgentReplyAny)
        assert reply.content == {"city": "Colombo"}

    @pytest.mark.asyncio
    async def test_error_result_is_reported_and_the_transcript_still_committed(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        session = Session("s")
        fake_client.script = [
            _finish({"type": "user", "uuid": "u1"}, is_error=True, subtype="error_max_turns", result=None, errors=["max turns reached"])
        ]

        reply = await agent.runner.run(agent, session, [AgentRequestText(prompt="hi")])

        assert "error_max_turns" in reply.response and "max turns reached" in reply.response
        assert session.get(FRAMEWORK).transcript("general")[""] == [{"type": "user", "uuid": "u1"}]

    def test_api_error_names_the_status_not_the_success_subtype(self):
        failed = _result(is_error=True, result="Overloaded")
        failed.api_error_status = 529

        message = str(ClaudeRunner._error(failed))

        assert message == "Claude Agent SDK run failed (API error 529): Overloaded"

    @pytest.mark.asyncio
    async def test_a_run_without_a_result_commits_nothing(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        session = Session("s")
        fake_client.script = [_mirror({"type": "user", "uuid": "u1"})]

        reply = await agent.runner.run(agent, session, [AgentRequestText(prompt="hi")])

        assert "without a result" in reply.response
        assert not session.get(FRAMEWORK).transcript("general")

    @pytest.mark.asyncio
    async def test_an_exception_mid_run_leaves_the_stored_transcript_intact(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        session = Session("s")
        await _turn(fake_client, agent, session, prompt="first")
        before = session.get(FRAMEWORK).transcript("general")[""][:]

        async def boom(client):
            raise RuntimeError("CLI died")

        fake_client.script = [_mirror({"type": "user", "uuid": "u-second"}), boom]
        reply = await agent.runner.run(agent, session, [AgentRequestText(prompt="second")])

        assert "CLI died" in reply.response
        assert session.get(FRAMEWORK).transcript("general")[""] == before

    @pytest.mark.asyncio
    async def test_no_content_returns_before_resolving_options(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        agent.resolve_run_options = AsyncMock(return_value={})

        reply = await agent.runner.run(agent, Session("s"), [])
        events = [e async for e in agent.runner.stream(agent, Session("s"), [])]

        assert "No valid content" in reply.response
        assert events == []
        agent.resolve_run_options.assert_not_awaited()
        assert fake_client.instances == []


# --- conversation continuity -------------------------------------------------------------------------


class TestConversation:
    @pytest.mark.asyncio
    async def test_first_turn_assigns_a_session_id_and_later_turns_resume_it(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        session = Session("s")

        await _turn(fake_client, agent, session, prompt="one")
        first = fake_client.instances[0].options
        await _turn(fake_client, agent, session, prompt="two")
        second = fake_client.instances[1].options

        assert first.resume is None and first.session_id
        assert second.resume == first.session_id and second.session_id is None
        claude_session = session.get(FRAMEWORK)
        assert claude_session.session_id("general") == first.session_id
        assert [e["uuid"] for e in claude_session.transcript("general")[""]] == ["u-one", "a-one", "u-two", "a-two"]

    @pytest.mark.asyncio
    async def test_the_session_survives_the_session_store_serde(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        session = Session("s")
        await _turn(fake_client, agent, session, prompt="one")

        serde = BinarySerde()
        restored = serde.loads(serde.dumps(session.get(FRAMEWORK)))

        assert restored.session_id("general") == session.get(FRAMEWORK).session_id("general")
        assert restored.transcript("general") == session.get(FRAMEWORK).transcript("general")

    @pytest.mark.asyncio
    async def test_each_agent_keeps_its_own_conversation(self, fake_client, runtime):
        module = _module(_spec("a"), _spec("b"))
        session = Session("s")

        await _turn(fake_client, module.get_agent("a"), session, prompt="for-a")
        await _turn(fake_client, module.get_agent("b"), session, prompt="for-b")

        data = session.get(FRAMEWORK)
        assert data.session_id("a") != data.session_id("b")
        assert fake_client.instances[1].options.resume is None  # b starts fresh, not on a's transcript


# --- session store --------------------------------------------------------------------------------------


class TestSessionStore:
    @staticmethod
    def _saved(session_id, transcript):
        claude_session = ClaudeSession()
        claude_session.save("a", session_id, transcript)
        return claude_session

    @pytest.mark.asyncio
    async def test_load_returns_none_for_an_unknown_transcript(self):
        store = _ClaudeSessionStore(ClaudeSession(), "a")
        assert await store.load({"project_key": "p", "session_id": "x"}) is None

    @pytest.mark.asyncio
    async def test_appends_are_buffered_until_commit_and_deduplicated_by_uuid(self):
        claude_session = ClaudeSession()
        store = _ClaudeSessionStore(claude_session, "a")
        key = {"project_key": "p", "session_id": "sid"}

        await store.append(key, [{"type": "user", "uuid": "u1"}, {"type": "title"}])
        await store.append(key, [{"type": "user", "uuid": "u1"}, {"type": "assistant", "uuid": "a1"}])

        assert claude_session.transcript("a") == {}
        assert await store.load(key) == [{"type": "user", "uuid": "u1"}, {"type": "title"}, {"type": "assistant", "uuid": "a1"}]
        store.commit("sid")
        assert claude_session.session_id("a") == "sid"
        assert len(claude_session.transcript("a")[""]) == 3

    @pytest.mark.asyncio
    async def test_project_key_is_ignored(self):
        store = _ClaudeSessionStore(self._saved("sid", {"": [{"type": "user", "uuid": "u1"}]}), "a")

        assert await store.load({"project_key": "another-cwd", "session_id": "sid"}) == [{"type": "user", "uuid": "u1"}]

    @pytest.mark.asyncio
    async def test_subagent_transcripts_are_listed_and_restored(self):
        store = _ClaudeSessionStore(self._saved("sid", {"": [{"type": "user"}], "subagents/agent-1": [{"type": "user"}]}), "a")
        await store.append({"project_key": "p", "session_id": "sid", "subpath": "subagents/agent-2"}, [{"type": "user"}])

        assert await store.list_subkeys({"project_key": "p", "session_id": "sid"}) == ["subagents/agent-1", "subagents/agent-2"]
        assert await store.list_subkeys({"project_key": "p", "session_id": "other"}) == []

    @pytest.mark.asyncio
    async def test_commit_keeps_only_the_session_the_turn_ended_on(self):
        claude_session = self._saved("old", {"": [{"type": "user", "uuid": "o1"}]})
        store = _ClaudeSessionStore(claude_session, "a")
        await store.append({"project_key": "p", "session_id": "new"}, [{"type": "user", "uuid": "n1"}])

        store.commit("new")

        assert claude_session.session_id("a") == "new"
        assert claude_session.transcript("a") == {"": [{"type": "user", "uuid": "n1"}]}

    def test_other_agents_conversations_are_untouched(self):
        claude_session = self._saved("sid-a", {"": [{"type": "user"}]})
        store = _ClaudeSessionStore(claude_session, "b")

        store.commit("sid-b")

        assert claude_session.session_id("a") == "sid-a"
        assert claude_session.session_id("b") == "sid-b"


# --- native options ----------------------------------------------------------------------------------


class TestNativeOptions:
    @pytest.mark.asyncio
    async def test_tools_are_served_by_the_ak_server_and_auto_approved(self, fake_client, runtime):
        spec = _spec(allowed_tools=["Read"])
        spec.tools = ClaudeToolBuilder.bind([double])
        agent = _module(spec).get_agent("general")
        fake_client.script = [_result()]

        await agent.runner.run(agent, Session("s"), [AgentRequestText(prompt="hi")])

        options = fake_client.instances[0].options
        assert AK_MCP_SERVER in options.mcp_servers
        assert options.allowed_tools == ["Read", "mcp__ak__double"]
        assert agent.agent.options.allowed_tools == ["Read"]  # the declaration is never mutated
        assert agent.agent.options.mcp_servers == {}

    @pytest.mark.asyncio
    async def test_no_tools_means_no_ak_server(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        fake_client.script = [_result()]

        await agent.runner.run(agent, Session("s"), [AgentRequestText(prompt="hi")])

        options = fake_client.instances[0].options
        assert AK_MCP_SERVER not in options.mcp_servers
        assert options.session_store is not None
        assert options.include_partial_messages is False

    @pytest.mark.asyncio
    async def test_tool_handlers_see_the_tool_context_of_the_run(self, fake_client, runtime):
        seen = {}

        def whoami() -> str:
            """Report the agent the tool runs for."""
            seen["agent"] = ToolContext.get().agent.name
            seen["session"] = ToolContext.get().session.id
            return "ok"

        spec = _spec()
        spec.tools = ClaudeToolBuilder.bind([whoami])
        agent = _module(spec).get_agent("general")

        async def call_tool(client):
            tool = next(t for t in agent.agent.tools if t.name == "whoami")
            seen["result"] = await tool.handler({})

        fake_client.script = [call_tool, _result()]
        await agent.runner.run(agent, Session("tool-session"), [AgentRequestText(prompt="hi")])

        assert seen["agent"] == "general"
        assert seen["session"] == "tool-session"
        assert seen["result"] == {"content": [{"type": "text", "text": "ok"}]}


# --- run options ---------------------------------------------------------------------------------------


class TestRunOptions:
    def test_reserved_keys(self):
        assert set(ClaudeAgent.RESERVED_RUN_OPTIONS) == {
            "resume",
            "session_id",
            "continue_conversation",
            "fork_session",
            "resume_session_at",
            "resume_drops_turn",
            "session_store",
            "session_store_flush",
            "enable_file_checkpointing",
            "include_partial_messages",
        }

    @pytest.mark.asyncio
    async def test_declared_options_reach_the_client_and_the_declared_dict_is_untouched(self, fake_client, runtime):
        module = _module(_spec())
        spec = module.get_agent("general").agent
        module.run_options(spec, max_turns=3, model="sonnet")
        agent = module.get_agent("general")
        fake_client.script = [_result(), _result()]

        await agent.runner.run(agent, Session("s"), [AgentRequestText(prompt="hi")])
        await agent.runner.run(agent, Session("t"), [AgentRequestText(prompt="hi")])

        for client in fake_client.instances:
            assert client.options.max_turns == 3
            assert client.options.model == "sonnet"
        assert agent.run_options == {"max_turns": 3, "model": "sonnet"}
        assert agent.agent.options.max_turns is None

    @pytest.mark.asyncio
    async def test_a_factory_merged_value_wins_and_resolution_happens_once_per_run_and_stream(self, fake_client, runtime):
        agent = _module(_spec(max_turns=1)).get_agent("general")
        agent.run_options.update({"max_turns": 2})
        agent.resolve_run_options = AsyncMock(return_value={"max_turns": 9})
        session = Session("s")
        requests = [AgentRequestText(prompt="hi")]
        fake_client.script = [_result()]

        await agent.runner.run(agent, session, requests)
        _ = [e async for e in agent.runner.stream(agent, session, requests)]

        assert agent.resolve_run_options.await_count == 2
        agent.resolve_run_options.assert_awaited_with(session, requests)
        assert [c.options.max_turns for c in fake_client.instances] == [9, 9]

    @pytest.mark.asyncio
    async def test_an_ak_owned_field_wins_over_a_bypassed_declaration(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        bypassed = {"session_store": object(), "session_store_flush": "eager", "session_id": "declared"}
        agent.run_options.update(bypassed)  # mutating the live dict skips validation
        fake_client.script = [_result()]

        await agent.runner.run(agent, Session("s"), [AgentRequestText(prompt="hi")])

        options = fake_client.instances[0].options
        assert isinstance(options.session_store, _ClaudeSessionStore)
        assert options.session_store_flush == "batched"
        assert options.session_id != "declared"

    def test_reserved_unknown_and_colliding_keys_are_rejected_at_declaration(self, runtime):
        spec = _spec()
        spec.tools = ClaudeToolBuilder.bind([double])
        module = _module(spec)

        for key in ClaudeAgent.RESERVED_RUN_OPTIONS:
            with pytest.raises(ValueError) as exc:
                module.run_options(spec, **{key: object()})
            assert f"'{key}'" in str(exc.value) and "'claude'" in str(exc.value)
        with pytest.raises(ValueError, match="not ClaudeAgentOptions fields"):
            module.run_options(spec, temperature=0.2)
        with pytest.raises(ValueError, match="reserved"):
            module.run_options(spec, mcp_servers={AK_MCP_SERVER: {}})
        with pytest.raises(ValueError, match="must be a dict"):
            module.run_options(spec, mcp_servers="/etc/mcp.json")

        module.run_options(spec, max_turns=2)
        assert module.get_agent("general").run_options == {"max_turns": 2}

    def test_declared_mcp_servers_are_checked_at_load(self, runtime):
        with pytest.raises(ValueError, match="reserved"):
            _module(_spec(mcp_servers={AK_MCP_SERVER: {}}))
        with_tools = _spec(mcp_servers="/etc/mcp.json")
        with_tools.tools = ClaudeToolBuilder.bind([double])
        with pytest.raises(ValueError, match="must be a dict"):
            _module(with_tools)

        _module(_spec("no-tools", mcp_servers="/etc/mcp.json"))  # a config-file path is fine without tools


# --- framework context -----------------------------------------------------------------------------


class TestFrameworkContext:
    @pytest.mark.asyncio
    async def test_context_is_not_injected_and_warns_once(self, fake_client, runtime, caplog):
        agent = _module(_spec()).get_agent("general")
        session = Session("s")
        session.set_framework_context({"user": "u1"})
        fake_client.script = [_result()]

        with caplog.at_level(logging.WARNING, logger="ak.claude.runner"):
            await agent.runner.run(agent, session, [AgentRequestText(prompt="hi")])
            await agent.runner.run(agent, session, [AgentRequestText(prompt="hi")])

        assert len([r for r in caplog.records if "framework_context" in r.getMessage()]) == 1
        assert session.get_framework_context() == {"user": "u1"}


# --- multimodal ----------------------------------------------------------------------------------------


class TestMultimodal:
    @pytest.mark.asyncio
    async def test_an_image_goes_as_one_streaming_input_message(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        fake_client.script = [_result()]
        requests = [AgentRequestText(prompt="what is this?"), AgentRequestImage(image_data="aGk=", name="a.png", mime_type="image/png")]

        await agent.runner.run(agent, Session("s"), requests)

        [messages] = fake_client.instances[0].prompts
        assert messages == [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
                    ],
                },
                "parent_tool_use_id": None,
            }
        ]

    def test_content_blocks(self):
        text_file = base64.b64encode(b"hello").decode()
        _, blocks = ClaudeRunner._process_requests(
            [
                AgentRequestImage(image_data="https://x/a.png", name="a"),
                AgentRequestFile(file_data="JVBE", name="a.pdf", mime_type="application/pdf"),
                AgentRequestFile(file_data=text_file, name="a.txt", mime_type="text/plain"),
                AgentRequestFile(file_data="https://x/a.pdf", name="b.pdf"),
            ]
        )
        assert blocks == [
            {"type": "image", "source": {"type": "url", "url": "https://x/a.png"}},
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBE"}},
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "hello"}},
            {"type": "document", "source": {"type": "url", "url": "https://x/a.pdf"}},
        ]

    @pytest.mark.parametrize(
        "request_, message",
        [
            (AgentRequestImage(image_data="s3://b/k", name="a"), "s3://"),
            (AgentRequestImage(image_data="aGk=", name="a"), "mime_type is missing"),
            (AgentRequestFile(file_data="aGk=", name="a.zip", mime_type="application/zip"), "not supported"),
        ],
    )
    def test_unsupported_content_is_rejected(self, request_, message):
        with pytest.raises(ValueError, match=message):
            ClaudeRunner._process_requests([request_])


# --- stream --------------------------------------------------------------------------------------------


def _raw(event, parent=None):
    return StreamEvent(uuid="u", session_id="s", event=event, parent_tool_use_id=parent)


def _text_message(message_id="msg_1", text=("Hel", "lo")):
    return [
        _raw({"type": "message_start", "message": {"id": message_id}}),
        _raw({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        *[_raw({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}}) for t in text],
        _raw({"type": "content_block_stop", "index": 0}),
        _raw({"type": "message_stop"}),
    ]


class TestStream:
    @pytest.mark.asyncio
    async def test_text_deltas_are_keyed_by_the_native_message_id(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        fake_client.script = [*_text_message(), _result()]

        events = [e async for e in agent.runner.stream(agent, Session("s"), [AgentRequestText(prompt="hi")])]

        assert [e.type for e in events] == ["message_start", "text_delta", "text_delta", "message_end"]
        assert {e.message_id for e in events} == {"msg_1:0"}
        assert "".join(e.content for e in events if e.type == "text_delta") == "Hello"
        assert fake_client.instances[0].options.include_partial_messages is True

    @pytest.mark.asyncio
    async def test_thinking_and_tool_calls_map_with_native_ids_and_ak_tool_names(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        fake_client.script = [
            _raw({"type": "message_start", "message": {"id": "msg_1"}}),
            _raw({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            _raw({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
            _raw({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}}),
            _raw({"type": "content_block_stop", "index": 0}),
            _raw({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "mcp__ak__double"}}),
            _raw({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"x": 2}'}}),
            _raw({"type": "content_block_stop", "index": 1}),
            UserMessage(content=[ToolResultBlock(tool_use_id="toolu_1", content=[{"type": "text", "text": "4"}])]),
            _result(),
        ]

        events = [e async for e in agent.runner.stream(agent, Session("s"), [AgentRequestText(prompt="hi")])]

        assert [e.type for e in events] == [
            "reasoning_start",
            "reasoning_delta",
            "reasoning_end",
            "tool_call_start",
            "tool_call_args",
            "tool_call_end",
            "tool_call_result",
        ]
        start = events[3]
        assert (start.tool_call_id, start.name) == ("toolu_1", "double")
        assert events[4].delta == '{"x": 2}'
        assert (events[6].tool_call_id, events[6].content) == ("toolu_1", "4")

    @pytest.mark.asyncio
    async def test_subagent_traffic_is_not_emitted(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        inner = [_raw(e.event, parent="toolu_task") for e in _text_message("msg_sub")]
        fake_client.script = [
            *inner,
            UserMessage(content=[ToolResultBlock(tool_use_id="x", content="r")], parent_tool_use_id="toolu_task"),
            _result(),
        ]

        events = [e async for e in agent.runner.stream(agent, Session("s"), [AgentRequestText(prompt="hi")])]

        assert events == []

    @pytest.mark.asyncio
    async def test_the_transcript_is_committed_after_a_full_drain(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        session = Session("s")
        fake_client.script = [*_text_message(), _finish({"type": "user", "uuid": "u1"})]

        _ = [e async for e in agent.runner.stream(agent, session, [AgentRequestText(prompt="hi")])]

        assert session.get(FRAMEWORK).transcript("general")[""]

    @pytest.mark.asyncio
    async def test_an_error_result_raises_after_the_events(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        fake_client.script = [*_text_message(), _result(is_error=True, subtype="error_during_execution", result=None)]

        events = []
        with pytest.raises(RuntimeError, match="error_during_execution"):
            async for event in agent.runner.stream(agent, Session("s"), [AgentRequestText(prompt="hi")]):
                events.append(event)
        assert [e.type for e in events][-1] == "message_end"

    @pytest.mark.asyncio
    async def test_closing_the_stream_early_disconnects_and_commits_nothing(self, fake_client, runtime):
        agent = _module(_spec()).get_agent("general")
        session = Session("s")
        fake_client.script = [*_text_message(), _mirror({"type": "user", "uuid": "u1"}), _result()]

        stream = agent.runner.stream(agent, session, [AgentRequestText(prompt="hi")])
        first = await stream.__anext__()
        await stream.aclose()

        assert first.type == "message_start"
        assert fake_client.instances[0].exited
        assert not session.get(FRAMEWORK).transcript("general")

    def test_mapper_closes_blocks_left_open(self):
        mapper = _ClaudeStreamMapper()
        for message in _text_message()[:3]:
            mapper.events(message)

        assert [e.type for e in mapper.close()] == ["message_end"]


# --- agent ---------------------------------------------------------------------------------------------


class TestAgent:
    @pytest.mark.parametrize(
        "declared, expected",
        [
            (None, "EXTRA"),
            ("Be brief.", "Be brief.\n\nEXTRA"),
            ({"type": "preset", "preset": "claude_code"}, {"type": "preset", "preset": "claude_code", "append": "EXTRA"}),
            ({"type": "preset", "preset": "claude_code", "append": "A"}, {"type": "preset", "preset": "claude_code", "append": "A\n\nEXTRA"}),
            ({"type": "custom", "prompt": "P"}, {"type": "custom", "prompt": "P\n\nEXTRA"}),
        ],
    )
    def test_override_system_prompt_appends_in_the_declared_form(self, runtime, declared, expected):
        agent = _module(_spec(system_prompt=declared)).get_agent("general")
        agent.override_system_prompt("EXTRA")
        assert agent.agent.options.system_prompt == expected

    def test_a_file_system_prompt_is_left_alone_with_a_warning(self, runtime, caplog):
        agent = _module(_spec(system_prompt={"type": "file", "path": "/p.md"})).get_agent("general")
        with caplog.at_level(logging.WARNING, logger="ak.claude.runner"):
            agent.override_system_prompt("EXTRA")
        assert agent.agent.options.system_prompt == {"type": "file", "path": "/p.md"}
        assert any("cannot be extended" in r.getMessage() for r in caplog.records)

    def test_shared_options_are_not_altered_through_one_declaration(self, runtime):
        shared = ClaudeAgentOptions(system_prompt="Base")
        module = _module(ClaudeAgentSpec(name="a", options=shared), ClaudeAgentSpec(name="b", options=shared))
        module.get_agent("a").override_system_prompt("ONLY-A")

        assert shared.system_prompt == "Base"
        assert module.get_agent("b").agent.options.system_prompt == "Base"

    def test_a_shared_tool_list_is_not_altered_through_one_declaration(self, runtime):
        shared = ClaudeToolBuilder.bind([double])
        module = _module(ClaudeAgentSpec(name="a", tools=shared), ClaudeAgentSpec(name="b", tools=shared))
        module.get_agent("a").attach_tool(_echo)

        assert [t.name for t in shared] == ["double"]
        assert [t.name for t in module.get_agent("b").agent.tools] == ["double"]

    def test_description_falls_back_to_a_string_system_prompt(self, runtime):
        module = _module(ClaudeAgentSpec(name="a", options=ClaudeAgentOptions(system_prompt="You plan trips.")))
        assert module.get_agent("a").get_description() == "You plan trips."

    def test_attach_tool_skips_a_name_already_present(self, runtime):
        spec = _spec()
        spec.tools = ClaudeToolBuilder.bind([double])
        agent = _module(spec).get_agent("general")

        agent.attach_tool(double)

        assert [t.name for t in agent.agent.tools] == ["double"]

    def test_module_requires_a_name(self, runtime):
        with pytest.raises(ValueError, match="must have a name"):
            _module(ClaudeAgentSpec(name=""))

    def test_does_not_pause(self):
        assert ClaudeRunner().supports_pause is False
        assert ClaudeRunner().supports_streaming is True


# --- subagents -----------------------------------------------------------------------------------------


class TestSubagents:
    @staticmethod
    def _options(agent):
        claude_session = ClaudeSession()
        return agent.runner._native_options(agent, {}, claude_session, _ClaudeSessionStore(claude_session, agent.name), stream=False)

    def test_subagents_become_definitions_whose_tools_the_parent_serves(self, runtime):
        weather = ClaudeAgentSpec(
            name="weather",
            description="Weather questions",
            options=ClaudeAgentOptions(system_prompt="Use get_weather.", tools=[], model="haiku"),
            tools=ClaudeToolBuilder.bind([double]),
        )
        triage = ClaudeAgentSpec(name="triage", options=ClaudeAgentOptions(tools=[], allowed_tools=["Read"]), subagents=[weather])
        agent = _module(triage, weather).get_agent("triage")

        options = self._options(agent)

        assert options.agents["weather"].description == "Weather questions"
        assert options.agents["weather"].prompt == "Use get_weather."
        assert options.agents["weather"].tools == ["mcp__ak__double"]
        assert options.agents["weather"].model == "haiku"
        assert AK_MCP_SERVER in options.mcp_servers
        assert options.tools == ["Agent"]
        assert options.allowed_tools == ["Read", "mcp__ak__double", "Agent"]
        assert agent.agent.options.tools == [] and agent.agent.options.agents is None  # the declaration is untouched

    def test_a_subagent_with_default_tools_inherits_the_sessions_tools(self, runtime):
        triage = ClaudeAgentSpec(name="triage", subagents=[ClaudeAgentSpec(name="helper", options=ClaudeAgentOptions(system_prompt="Help."))])

        options = self._options(_module(triage).get_agent("triage"))

        assert options.agents["helper"].tools is None
        assert options.agents["helper"].description == "helper"
        assert options.tools is None  # the full built-in toolset, which already includes Agent

    def test_a_custom_prompt_is_unwrapped_and_native_agents_are_kept(self, runtime):
        native = AgentDefinition(description="Native", prompt="P")
        sub = ClaudeAgentSpec(name="sub", options=ClaudeAgentOptions(system_prompt={"type": "custom", "prompt": "Custom."}))
        triage = ClaudeAgentSpec(name="triage", options=ClaudeAgentOptions(agents={"native": native}), subagents=[sub])

        options = self._options(_module(triage).get_agent("triage"))

        assert options.agents["native"] is native
        assert options.agents["sub"].prompt == "Custom."

    def test_invalid_subagents_are_rejected_at_load(self, runtime):
        clash = ClaudeAgentSpec(
            name="triage",
            options=ClaudeAgentOptions(agents={"sub": AgentDefinition(description="d", prompt="p")}),
            subagents=[ClaudeAgentSpec(name="sub")],
        )
        with pytest.raises(ValueError, match="also defined"):
            _module(clash)

        preset = ClaudeAgentSpec(name="sub", options=ClaudeAgentOptions(system_prompt={"type": "preset", "preset": "claude_code"}))
        with pytest.raises(ValueError, match="string system prompt"):
            _module(ClaudeAgentSpec(name="triage2", subagents=[preset]))
