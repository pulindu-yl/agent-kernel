# #810: Realtime multi-agent collaboration — Implementation Spec

Implements the requirements in [`design.md`](design.md). The realtime pool keeps each call's turn state, says
application posts at quiet moments through `RealtimeOutbox` / `RealtimeCall`, and feeds `RealtimeObserver`s. The Gemini
adapter runs ADK's tool callbacks, seeds transfers with authored history, applies the call's `RunConfig`, and recovers
lost sockets by replaying the conversation. A connection owns its tool calls, so a tool that awaits long work (another
agent's run) is cancelled with its call. The LiveKit gateway signs participant tokens. The idea: voice is the meeting
room, not the engine.

## Verification performed for this spec

Each item was checked against `develop` at `4c78c482`, the branch at `58015003`, the installed `google-adk==2.8.0` and
`google-genai==2.21.0`, the `google-adk==1.24.0` wheel (the floor, `ak-py/pyproject.toml:128`), or the Live API docs.

- ADK 1.24.0 and 2.8.0: `RunConfig` has `realtime_input_config`, `input_audio_transcription` and
  `output_audio_transcription`; `google.adk.flows.llm_flows.contents._present_other_agent_message` exists in both
- ADK's tool-callback order (2.8.0, `google/adk/flows/llm_flows/functions.py:621-682`): before-callbacks run until one
  returns a truthy result; the tool runs when the result is still `None`; error callbacks run until one returns non-`None`,
  else the error is raised; after-callbacks run until one returns a truthy result, which replaces the response when non-`None`
- google-genai 2.21.0: `LiveConnectConfig` has `realtime_input_config`
- Live API ([Session management](https://ai.google.dev/gemini-api/docs/live-session), 2026-10-15): a connection lasts about
  10 minutes, then the server closes it; each new connection is a new session
- `develop` behaviour this change replaces
  - A Gemini socket error is reported as `error` (`adk.py:1219-1234`); `RealtimeConnection._fail` closes the connection
    (`realtime_pool.py:159-171`); the next input opens one at the entry agent (`:95-98`)
  - Audio and text are dropped while no socket is open (`adk.py:1298-1308`, `:1360-1361`)
  - `ToolContext.set()` keeps the reset token on the instance (`core/tool.py:102-116`)
  - The pool's sweep skips closed connections (`realtime_pool.py:401`); a closed one is closed only when its session
    sends more input (`get_connection`, `:441-453`)
- `agentkernel.pipeline` exports `RealtimeCall` and no other `realtime_pool` name (`pipeline/__init__.py`)
- The `RealtimeRunner` callback contract (`core/realtime/runner.py`) documents `speech_heard`, `speech_stopped` and
  `caller_transcript`, not `reconnected` or the `done` status `reconnecting`
- `LiveKitEdgeGateway.__init__` raises without a token or an API key and secret, and contacts no server
  (`integration/livekit/adapter.py`, `_single_participant`); the example builds its gateway only under `__main__`,
  so its tests import it without LiveKit settings
- CI runs each example listed in `.github/test-config.yaml` with `./build.sh local`, the branch's own build, then
  `uv run pytest -s` (`.github/scripts/run_single_test.py:87-112`); no `examples/api/livekit-voice/*` example was listed
- AG-UI cannot carry a voice call: `ag-ui-protocol` 0.1.22 has no audio events, AK's AG-UI handler answers one HTTP
  request per run (`integration/agui/handler.py:221`) on the stream path, and rejects audio input (`run_input.py:216`)

## Design

### Package layout

```
ak-py/src/agentkernel/
├── livekit.py                 # NEW: public alias of integration.livekit
├── core/
│   ├── __init__.py            # exports RealtimeEvent, RealtimeObserver
│   └── realtime/
│       ├── observer.py        # NEW: RealtimeEvent, RealtimeObserver
│       └── runner.py          # reports_speech_stopped; the new adapter events
├── framework/
│   ├── adk/adk.py             # GoogleADKRealtimeRunner; the ADK tool wrapper
│   └── openai/openai.py       # speech_stopped
├── integration/
│   └── livekit/adapter.py     # participant_token
└── pipeline/
    ├── __init__.py            # exports RealtimeCall
    └── realtime_pool.py       # turn state, RealtimeOutbox, RealtimeCall, observers, recovery readiness, sweep
```

### Core contracts (`core/`)

```python
class RealtimeEvent(BaseModel):
    type: Literal["connected", "caller_said", "edge_text", "agent_said", "agent_changed", "reconnected", "disconnected"]
    session_id: str
    agent: Optional[str] = None
    text: str = ""
    interrupted: bool = False
    at: float = Field(default_factory=time.time)

class RealtimeObserver(ABC):
    async def on_event(self, session: Session, agent: Agent, event: RealtimeEvent) -> None: ...
    def name(self) -> str: ...

class RealtimeCall:   # pipeline/realtime_pool.py
    @staticmethod
    def observe(observer: RealtimeObserver) -> None: ...     # process-wide; each new connection takes the list
    @staticmethod
    def unobserve(observer: RealtimeObserver) -> None: ...

class RealtimeRunner:
    reports_speech_stopped: bool = False   # True only where the model signals the end of the caller's speech
```

Rules:

1. `core/realtime/observer.py` imports nothing outside `core/` and pydantic; `core/hooks.py`, `Agent` and `Module` are unchanged.
2. The adapter callback contract (`RealtimeRunner` docstring) gains, none reaching the edge:
   - `speech_heard {}`: the caller is still speaking;
   - `speech_stopped {}`: the caller stopped; only from an adapter with `reports_speech_stopped`;
   - `caller_transcript {"text"}`: optional, the caller's words in one turn;
   - `reconnected {"agent", "seconds", "seeded"}`: the socket was replaced, with `seeded` true when the conversation was replayed;
   - `done` with status `reconnecting`: the turn under way ended because the socket was lost.
3. Observers live on the pool class (`RealtimeConnectionPool._observers`), not on `Agent`: a call is observed, not an agent, and the list can be filled before the pool starts. `get_or_create` passes it to each new `RealtimeConnection(..., observers=)`.
4. `set()` / `reset()` are unchanged for their callers, which activate once per run.

### ADK realtime runner (`GoogleADKRealtimeRunner`)

#### Tool callbacks

`execute_tool` runs `_run_tool(caller, tool, args, tool_context)`, `caller` being the agent that made the call:

1. Before-callbacks (`caller.canonical_before_tool_callbacks`) in order, each awaited when it returns an awaitable; stop at the first truthy result.
2. If the result is still `None`, run `tool.run_async`. If it raises, error callbacks in order until one returns non-`None`, which becomes the result; with none, re-raise.
3. After-callbacks in order with `tool_response`; stop at the first truthy result; a non-`None` one replaces the result.
4. Then read `tool_context.actions.transfer_to_agent`, as today.

The order matches ADK's (Verification); plugins do not run.

#### Caller turn events

- `_caller_turn_reported` is set when a caller turn is reported (`speech_started` or `interrupt`) and cleared when the model speaks or the turn completes
- A transcription of the caller (interim or final) while the model is silent reports `speech_started` once per turn, then `speech_heard` for each later one; while the model speaks, the existing barge-in rule applies
- The caller's words are reported as `caller_transcript` and logged (`Caller said: ...`) when the turn ends: `turn_complete`, a tool call, text sent in, a transfer or a lost socket

#### Authored history

- `_history` holds `_Said(author, content)`: `"user"` for the caller, else the agent's name
- `_presented_history()` builds a seed: caller entries as they are; agent entries through `_present_other_agent_message(Event(author=..., content=...))`
- A seed ends on the user's side, so it is sent with `turn_complete=True`, followed by `"."` on Gemini 3.x Live, as ADK does

#### Live settings

- `connect` reads `agent.run_options.get("run_config")` once; `_connect_config` sets, on every socket:
  - `realtime_input_config` from the `RunConfig`;
  - `input_audio_transcription` / `output_audio_transcription` from the `RunConfig`, else `AudioTranscriptionConfig()`.

#### Recovery

| Constant | Value | Meaning |
|---|---|---|
| `RECONNECT_ATTEMPTS` | 3 | Attempts per loss; also the quick drops in a row that give up |
| `RECONNECT_DELAYS_SECONDS` | (1, 2, 4) | Wait before each attempt |
| `RECONNECT_TIMEOUT_SECONDS` | 10 | Bound on one attempt's connect |
| `QUICK_DROP_SECONDS` | 20 | A socket lost this soon after a reconnect is a quick drop |
| `STALL_SECONDS` / `STALL_CHECK_SECONDS` | 15 / 1 | No model progress this long is a stall; checked this often |
| `HOLD_AUDIO_SECONDS` | 5 | Caller audio held while no socket is open |
| `SPEECH_LEVEL` | 500 | RMS of 16-bit samples above which held audio is speech |
| `RECONNECTED_NOTE` / `RECONNECTED_QUIET_NOTE` | text | The replay note when the caller spoke / did not; `""` adds none |

Triggers: `_listen` raising (a socket error, or the server closing the connection at its ~10 minute limit) and the
stall watchdog. Each calls
`_recover_soon(reason)`, which starts `_recover` unless one is running or the runner is closing.

`_recover(reason)`:

1. Set `_connection = None`, stop listening, close the socket.
2. Move the open socket's batches to `_lost_batches`; clear `_callers`, `_transfers` and the stall wait.
3. End the turn (report the caller's words), reset the turn flags, report `done {"status": "reconnecting"}`.
4. Count a quick drop when the last reconnect was under `QUICK_DROP_SECONDS` ago, else reset the count; at `RECONNECT_ATTEMPTS`, report `error` and stop.
5. For each attempt, after its delay (stop if closing):
   1. **Replay**: the note (`RECONNECTED_NOTE` when the caller was speaking at the loss or held audio has speech, else `RECONNECTED_QUIET_NOTE`), passed to `_open_socket(..., extra=[note])`, which adds it to that seed only. `_history` never holds a note.
   2. On success: listen, report `reconnected {"agent", "seconds", "seeded"}`, then send the held input.
6. When every attempt fails, report `error`.

Rules:

1. **Late tool results.** `send_tool_result` for a call in `_lost_batches` records the result; when its batch is complete, the calls and responses are appended to `_history` under the agent that made them, without ending the current turn, and nothing is sent to the socket. A transfer asked by a lost call is dropped.
2. **Stall wait.** `_owed_since` is set to now after a tool response, after text is sent, after a seed that completes a turn, and on each model audio or output transcription while a response is open; it is cleared on `turn_complete`, `interrupted` and a tool call. No other message touches it.
3. **Held input.** While `_connection` is `None` and the runner is not closing, `append_audio` keeps the latest `HOLD_AUDIO_SECONDS` and `send_text` queues the text; both go to the next socket after a reconnect or a transfer.
5. **Text.** `send_text` sends as ADK's live connection does, by the model the open socket runs (`_socket_model`): `send_realtime_input(text=...)` on Gemini 3.x Live, a completed `send_client_content` turn otherwise. A seed stays client content (`history_config.initial_history_in_client_content`) followed by ADK's placeholder realtime text.
4. **Shutdown.** `disconnect()` cancels and awaits the watchdog and any recovery before closing the socket.

### OpenAI realtime runner (`OpenAIRealtimeRunner`)

`reports_speech_stopped = True`; `input_audio_buffer.speech_stopped` reports `speech_stopped`. Nothing else changes.

### Realtime pool (`pipeline/realtime_pool.py`)

#### Turn state (`RealtimeConnection`)

| Adapter event or call | Updates |
|---|---|
| `audio_delta`, `transcript_delta` | `last_model_generated_at`; response open; `last_model_output_at` when paced out to the edge |
| `agent_changed` | `last_agent_change_at` |
| `done`, `interrupt`, `speech_started` | response closed |
| `interrupt`, `speech_started`, `speech_heard`, `speech_stopped` | `last_caller_speech_at`; `is_caller_speaking` set by the first two when `reports_speech_stopped`, cleared by `speech_stopped` or model audio |
| `done` with status `reconnecting` | recovering |
| `reconnected` | not recovering; when `seeded`, `last_reconnect_at` |
| a tool call / its result sent | `_tools_in_flight` up / down; `last_tool_result_at` |
| `send_text(text, from_edge=True)` | `last_edge_text_at`; an `edge_text` observer event |

- `is_responding`: queued output, a tool in flight, or a response open with model output in the last `_UNFINISHED_RESPONSE_SECONDS` (15 s)
- `is_ready`: connected, not closed, not recovering
- `agent_has_spoken`: `last_model_generated_at` is later than `last_agent_change_at`

#### Outbox (`RealtimeOutbox`)

`QUIET_SECONDS = 1.5`, `REPLY_TIMEOUT_SECONDS = 15.0`, `MAX_PENDING = 20`; the pool's loop calls `_deliver_posts` every
`_POST_POLL_SECONDS = 0.25`.

`take(connection, now)`:

1. Drop expired posts; with none waiting, return `None`.
2. A new `connection` (not the last one posted into) owes nothing to earlier posts.
3. Return `None` while the connection is not ready, is responding, the caller is speaking, or a reply is owed: one of
   `last_caller_speech_at`, `last_edge_text_at`, `last_tool_result_at`, `last_reconnect_at` or the last post is later
   than `last_model_output_at` and younger than `REPLY_TIMEOUT_SECONDS`.
4. Return `None` until the newest of those times, `connected_at` and `last_model_output_at` is `QUIET_SECONDS` old.
5. Pop the first post, record the post time, return its text.

Rule: every write to the outbox's state (`_pending`, `_connection`, `_last_post_at`) is made under its lock, from
`quiet_for` as from `take`.

#### `RealtimeCall` and the pool's API

| `RealtimeCall` | Pool | Without a pool |
|---|---|---|
| `post_when_idle(text, key=None, expires_after=600.0) -> bool` | queues in the session's outbox; returns whether a call is connected | warning, `False` |
| `withdraw(key) -> bool` | drops a waiting post | `False` |
| `connected` | a ready connection exists | `False` |
| `quiet_for() -> float \| None` | seconds quiet, `None` while busy | `None` |
| `agent_has_spoken -> bool \| None` | as on the connection | `None` |
| `close() -> bool` | pops the connection and its outbox, closes it on the loop | `False` |

#### Observers

- A connection with observers gets an `asyncio.Queue` and one task; `_observe` puts on the loop, `_observe_soon` from other threads
- Sources: `connect()` (`connected`); the pacing loop (`agent_changed`, `agent_said` at `done`, `interrupt` or a change); `caller_transcript` (`caller_said`); `send_text(from_edge=True)` (`edge_text`); `reconnected`; `close()` or a failure (`disconnected`, once)
- On close, the observers get `_OBSERVER_DRAIN_SECONDS` (2 s), then the task is cancelled

#### Sweep and loop lag

- Every 5 s the loop pops and closes connections idle for `_IDLE_TIMEOUT_SECONDS` (300 s) **and** connections already closed by a failure
- The loop measures its own sleep; lag beyond `_LOOP_LAG_WARNING_SECONDS` (1 s) logs a warning

### LiveKit gateway (`LiveKitEdgeGateway`)

- `AGENT_IDENTITY_PREFIX = "agent-"`; agent identities are built from it
- `participant_token(identity, name=None, **grants)`: `ValueError` for an identity starting with the prefix, for `room` or `room_join` in `grants`, and with no key and secret; else `_sign(identity, name, **grants)`
- `_sign` builds `api.AccessToken(...).with_identity(...)`, adds the name when given, and grants `VideoGrants(room_join=True, room=self.session_id, **grants)`; `_mint_token` uses it with `can_update_own_metadata=True`

### Packaging

- `agentkernel/livekit.py` re-exports `integration.livekit`, as `slack.py` does
- No new extra; `ak-py/pyproject.toml` and `ak-py/uv.lock` unchanged

### Example (`examples/api/livekit-voice/google/handoffs/`)

| File | Change |
|---|---|
| `server.py` | `verify_account(account_id, tool_context)`, a supervisor tool, keeps a verified account in the ADK tool state, which lasts for the call across transfers; `require_account`, the supervisor's `before_tool_callback`, answers in place of `transfer_to_agent` to `billing` or `tech_support` until an account is verified; `book_technician`, an `async` tool, waits for the dispatch system and answers with the confirmation (Application building blocks, Example); a `CallLog(RealtimeObserver)` logging `caller_said`, `agent_said`, `agent_changed` and `reconnected`, registered with `RealtimeCall.observe` under `__main__`, where the gateway is built |
| `server_test.py` (new) | Offline tests (Testing) |
| `README.md` | Refusing a handoff, a tool that waits, the call log, the walkthrough, tests; the transfer gap now holds the caller's words |

`.github/test-config.yaml`: `- type: api` / `path: examples/api/livekit-voice/google/handoffs`; the tests need no
model, LiveKit server or key.

### Application building blocks

Verified for this part against the branch at `d1a464bd` and the installed `openai-agents==0.20.0` and
`google-adk==2.8.0`: `GoogleADKToolBuilder._wrap` keeps a coroutine function a coroutine function (`adk.py`, the
`asyncio.iscoroutinefunction` branch), and ADK's `FunctionTool.run_async` awaits it, so an `async` tool runs on the
pool's loop; `OpenAIRunner.run` catches `Exception` (a framework error becomes error text) but not
`asyncio.CancelledError`, and resets its `ToolContext` in `finally`; `Session.__aexit__` resets the current session
from its own token and releases the lock on any exit, cancellation included (`core/base.py`), as `Agent._activate`
does for the current agent. `Runtime._current` is class-level (`core/runtime.py:155`). No change to `core/model.py`,
`core/runtime.py` or the framework adapters' unary paths (design.md Non-goals).

```
ak-py/src/agentkernel/
├── core/
├── integration/adapter/gateway.py  # GatewayRunner.stop
└── pipeline/
    ├── realtime_pool.py         # RealtimeCall facts; tool calls the connection owns; while_quiet
    └── __init__.py              # RealtimeCall's lazy export points at realtime_pool
```

#### Module layout (`pipeline/realtime_pool.py`)

- The pool stays in `pipeline/realtime_pool.py`, where agentkernel 0.9.5 shipped `RealtimeConnectionPool` and
  `RealtimeConnection`: their imports, the module settings its tests patch (`_SHUTDOWN_GRACE_SECONDS`, ...) and the
  logger (`ak.pipeline.realtime_pool`) are unchanged. Line citations to `realtime_pool.py` earlier in this spec refer to
  the file before this change
- The public path is `agentkernel.pipeline` (`RealtimeCall`), whose lazy export points at `realtime_pool`
- `core/realtime/` keeps the adapter contracts and audio primitives; the gateways stay in `integration/adapter/`

#### Tool calls the connection owns (`pipeline/realtime_pool.py`)

```python
_TOOL_CANCEL_SECONDS = 2.0

class RealtimeConnection:
    _tool_tasks: set[asyncio.Task]                  # the tool calls still running
    async def close(self) -> None: ...               # also cancels them, before the adapter disconnects
```

Rules:

1. **Owned.** `handle_framework_event("tool_call", ...)` creates `_execute_tool_and_reply` as a task of the
   connection, adds it to `_tool_tasks`, and removes it when it is done (`add_done_callback(discard)`).
2. **Busy.** A running tool call counts toward `_tools_in_flight`, so `is_responding` is true: the outbox says no post
   and `quiet_for()` is `None`. Unchanged; a tool that awaits long work makes it last.
3. **Closing.** `close()` sets `closed` and, the first time, stops pacing, then `_cancel_tools()`: cancel every
   running tool task except the current task (a tool that closes its own call), `asyncio.wait` them up to
   `_TOOL_CANCEL_SECONDS` (a warning for those still running), then `adapter.disconnect()`. The cancellation reaches
   the tool's awaits, so a nested `AgentService.run_multi` unwinds its own session and context; the tool's
   `async with session` and `Agent._activate()` then unwind the call's.
4. **No late result.** `_execute_tool_and_reply` sends the result only while `closed` is false: a tool that finished
   once the connection was closed or its socket failed (`_fail`), even one that ignored its cancellation, is logged
   and its result dropped. `last_tool_result_at` moves only for a result that was sent. A cancelled tool sends
   nothing and its `CancelledError` propagates out of its task.
5. **Recovered sockets.** A socket the Gemini adapter replaces itself leaves the connection open (`closed` false): a
   result goes to `send_tool_result`, whose lost-call rule (above, Recovery rule 1) keeps it in the history and sends
   it to no socket; `_wait_for_lost_calls` waits up to `LOST_TOOL_WAIT_SECONDS` for it first. AK never runs a tool
   again.
6. **Nothing else.** No time limit, no retry, no queue: the tool's own code bounds its wait.

#### Call facts and `while_quiet` posts (`pipeline/realtime_pool.py`)

```python
class RealtimeCall:
    last_caller_input_at: Optional[float]           # max(last_caller_speech_at, last_edge_text_at)
    last_agent_change_at: Optional[float]
    last_tool_result_at: Optional[float]
    def post_when_idle(self, text, *, key=None, expires_after=600.0, while_quiet=False) -> bool: ...
    def close(self) -> bool: ...                     # the connection's close() cancels its running tool calls
```

Rules:

1. The facts read the session's open connection (`closed` false), ready or recovering; `None` without one.
2. `RealtimeOutbox._quiet_since(connection, now)` (under the outbox lock): the newest of the times `_quiet_for` takes,
   or `None` when busy; `quiet_for` is `now - _quiet_since`.
3. No quiet-call event: an application polls `RealtimeCall.quiet_for()` (from a task on the realtime loop that an
   observer starts on `connected` and cancels on `disconnected`).
4. `_quiet_since` takes `last_agent_change_at` with the other times: a handover ends a quiet stretch.
5. `while_quiet`: `add` takes the session's ready connection and records `stretch = _quiet_since`; with no connection or
   a busy call it queues nothing and returns `False`. `take` drops (info log) each `while_quiet` post whose `stretch`
   is not the current `_quiet_since`, which a new connection also changes (its `connected_at`).
6. `GatewayRunner` unregisters with `StatefulEdgeRegistry.unregister(session_id, adapter=...)`, which removes the
   entry only while it is that adapter.

#### Gateways (`integration/adapter/gateway.py`)

```python
class GatewayRunner:
    def stop(self, timeout: Optional[float] = None) -> bool: ...   # whether start() has returned (or never ran)
```

Rules:

1. `stop()` sets a `threading.Event`; `_run_until_shutdown` ends on it as on the shutdown event, stopping the adapter on
   its own loop. `start()` after `stop()` returns at once, registering nothing. `stop(timeout)` waits for `start()` to
   return when it has begun.
2. A gateway per session is the application's: it keeps a `GatewayRunner` per session under its own lock, hosts each
   on a daemon thread running `start()`, forgets the entry when `start()` returns or raises (only if it is still its
   own), and calls `stop(timeout)` outside the lock.

#### Example (`examples/api/livekit-voice/google/handoffs/server.py`)

- `book_technician(account_id, day)` is `async`: it awaits `Dispatch.book`, which sleeps `DISPATCH_SECONDS` (the
  dispatch system's time) and returns the confirmation as the tool's result; a second booking for an account already
  being booked is refused (two calls in one model turn). No thread, no post

### Consumer changes

- `RealtimeConnectionPool.dispatch`: text requests call `send_text(prompt, from_edge=True)`; the ECS realtime agent runner goes through `dispatch` (its test assertion changes)
- `RealtimeConnection.send_text` gains a keyword-only `from_edge=False`; other callers are unchanged
- LiveKit examples: public aliases; the `sqs` token handlers call `participant_token`

### Config changes

- No new fields, models, extras or env vars; existing YAML and `AK_*` variables are unaffected

### Behavioural changes

1. ADK realtime tool calls run the agent's tool callbacks. Intentional, the feature.
2. A lost or stalled Gemini socket reconnects the same agent instead of failing the call and restarting at the entry agent. Intentional.
3. Audio and text sent while no Gemini socket is open are held and sent to the next one, instead of dropped. Intentional.
4. A transfer's seed presents every agent's turn as user-side `[agent] ...` context, instead of model turns. Intentional, ADK's own presentation.
5. The ADK adapter reports `speech_started`, `speech_heard` and `caller_transcript`; the OpenAI adapter reports `speech_stopped`. None reaches the edge; `speech_started` is a barge-in only while paced audio is queued, as before.
6. The Live config takes turn detection and transcription from the call's `RunConfig`. Without a `run_config` option it is as before.
7. The pool's loop wakes every 0.25 s instead of 0.5 s; the idle sweep still runs every 5 s and now also closes failed connections.
8. LiveKit agent tokens are signed through `_sign`, with the same claims.
9. Closing a realtime connection cancels its running tool calls, and a result that comes once it is closed or its socket failed is not sent. Before, such a tool ran on with the session lock, and its result went to the disconnected adapter. Intentional.
10. No `Module.realtime_hook` / `Agent.realtime_hooks`: observers are registered with `RealtimeCall.observe`. Nothing of the hook API was released.

**Non-changes:** the reply models and every adapter's error reply; the existing `RealtimeCall` members; `GatewayRunner.start()` for a runner never stopped; `IOHandler.run(gateways=...)`; the stream event union and `AgentChanged`'s fields; queue message formats; OpenAI handoffs and the
LiveKit gateway's participants; `ToolContext.set()` / `reset()`; `AKConfig`; the extras; other examples' behaviour.

## Error handling

| Condition | Where | Surfaces as |
|---|---|---|
| Socket drops, server closes, stall | `GoogleADKRealtimeRunner._recover` | `done {"status": "reconnecting"}`, then `reconnected`; observers get `reconnected` |
| Every reconnect attempt fails | `_recover` | `error` → the pool fails the connection; the edge gets an error chunk |
| Quick drops `RECONNECT_ATTEMPTS` times in a row | `_recover` | `error` ("keeps dropping") |
| A transfer's socket does not open in 15 s | `_transfer` | `error`, as before |
| A tool raises with no error callback answering | `_run_tool` → the pool | `Error: ...` as the tool result, as before |
| A tool result arrives for a lost socket | `send_tool_result` | recorded in the history; info log |
| A realtime observer raises | `_serve_observers` | logged; the next observer runs |
| An observer is still busy at close | `_finish_observers` | warning after 2 s; its task cancelled |
| `post_when_idle` with no pool in the process | `RealtimeCall` | warning; the post is dropped |
| A full outbox / an expired post | `RealtimeOutbox` | the oldest dropped with a warning / dropped with an info log |
| `participant_token` misuse | `LiveKitEdgeGateway` | `ValueError` |
| A tool still running when its connection closes | `RealtimeConnection.close` | cancelled (info log); no result sent |
| A cancelled tool still running after 2 s | `_cancel_tools` | warning; the adapter disconnects; its result is never sent |
| A tool finishes after its connection closed or its socket failed | `_execute_tool_and_reply` | info log; result not sent |

## Testing

### Changed: `ak-py/tests/test_realtime_pool.py`

- `TestRealtimeTurnState`: each row of the turn-state table; a pause mid-answer; a lost `done`; readable from another thread; `is_ready` false between `reconnecting` and `reconnected`
- `TestADKRealtimeToolCallbacks`: before-callbacks stop a transfer, run in order, can ask for one; error and after-callbacks; async callbacks; the calling agent's callbacks
- `TestADKRealtimeSpeechStarted`, `TestADKCallerTranscript`, `TestOpenAIBargeIn`: the caller events of both adapters
- `TestADKRealtimeTransfers`: seeds as ADK presents them
- `TestADKConnectConfig`: the `RunConfig`'s turn detection and transcription on every socket
- `TestADKRealtimeRecovery`: replay with the right note; the note absent from the next seed; held audio and text; quick drops give up; every attempt failing fails; a late tool result in the next seed; stalls (a caller transcription keeps the wait; a response that stops mid-way recovers; an answering socket does not)
- `TestRealtimeOutbox`: quiet, busy, unready and recovering calls; owed replies, a seeded reconnect included; keys, withdrawal, expiry, overflow; a new connection
- `TestRealtimeCall`: posts with and without a call; `quiet_for`, `agent_has_spoken`; no pool; `close()` with and without a pool
- `TestRealtimeObservers`: events from connect to close, `reconnected` included; a failing observer; edge text from another thread; no observers, no queue; observers registered with `RealtimeCall.observe` reach the connections made after
- The pool: the sweep closes a failed connection without further input; the loop-lag warning
- `TestADKFirstAudioLog`: the wait logged once per turn

### New in `ak-py/tests/test_realtime_pool.py`

- `TestRealtimeAwaitedTools`: a tool awaiting a nested run in a session of its own keeps the call busy (no post said,
  `quiet_for()` is `None`), its result is sent once and the call's session and agent are current again after it; closing the
  connection, directly or through `RealtimeCall.close()`, cancels it, releases both sessions' locks, leaves no task and
  sends nothing; a tool that ignores its cancellation still sends nothing; a result after a failed socket is not sent
- `TestADKRealtimeRecovery`: a tool slower than `STALL_SECONDS` is not taken for a stall, and its result is answered
  on the same socket

### Changed: other files

- `test_livekit_gateway.py`: participant tokens (room, grants, refusals, no key); the alias
- `test_ecs_akagentrunner_realtime.py`: `send_text(..., from_edge=True)`
- `test_realtime_pool.py`: `TestRealtimeCallFacts` (each fact; `None` without a connection; read while recovering); `TestRealtimeOutbox` (`while_quiet` dropped by activity, said when quiet); `TestOpenAIRealtimeToolGuards` unchanged (approval and any tool guardrail excluded)
- `test_integration_gateway_runner.py`: `stop()` from another thread, before `start()`, and its wait; a failed start leaving nothing registered, a slowly leaving gateway not unregistering its replacement
- Lifecycle cases (`test_realtime_pool.py`): a `while_quiet` post refused when busy and dropped on a new connection or a handover

### New: `examples/api/livekit-voice/google/handoffs/server_test.py`

- `require_account` answers in place of a transfer to a specialist until the account is verified, then lets it through; other tools and other transfers pass
- `verify_account` records a verified account in the tool state and refuses a malformed one
- `book_technician` answers with the confirmation once the dispatch system's wait ends, and refuses a second booking for the account made at the same time
- `require_account` is the supervisor's callback; `CallLog`, a realtime observer, logs both sides of the call

### Live check

On `gemini-3.1-flash-live-preview`, with the application's researcher and reviewer on `gpt-4.1`: a handoff refused by
a callback, research and a review awaited by a voice tool (how long the caller waits, and what they hear), a forced
socket drop, and a call past 10 minutes.

### Commands

```bash
cd ak-py && ./build.sh && uv run pytest tests/test_realtime_pool.py tests/test_module.py tests/test_livekit_gateway.py tests/test_ecs_akagentrunner_realtime.py -v
cd ak-py && uv run pytest
make lint-check-all
cd examples/api/livekit-voice/google/handoffs && ./build.sh local && uv run pytest -s
```

## Migration surface (docs and skills)

- `docs/docs/integrations/livekit.md`: tool callbacks and refusing a handoff, "Tokens for your users", "Talking into a live call", "Watching a live call"; new sections on the call's Live settings (`run_config`) and on recovery; the limitation at `:285` (pre- and post-hooks do not run on realtime turns; realtime observers see the call) and at `:288` (a Gemini socket failure no longer restarts the call)
- `docs/docs/integrations/hooks.md`: realtime mode runs no hooks; a pointer to realtime observers
- `livekit.md`: "Tools that wait for long work" (a separate session, cancellation, detecting a failed run natively), "Rooms per session" (a `GatewayRunner` per room), the call facts and `while_quiet`
- `realtime-internals.md`: tool calls that wait, `while_quiet`
- `docs/docs/architecture/realtime-internals.md`: the adapter events, turn state as internal to the pool, `RealtimeOutbox`, realtime observers, tool callbacks, a connection recovery section, the stall limit
- `core/realtime/runner.py`: the callback contract gains `reconnected` and the `reconnecting` status (Rule 2)
- `.agents/skills/ak-dev-architecture/SKILL.md`: a realtime section (it has none); `.agents/skills/ak-dev-testing-conventions/SKILL.md`: the `test_realtime_pool.py` row
- No new integration, so the docs-site landing and features pages are unchanged
- Bundled skills: no change (the one match, `ak-py/src/agentkernel/skills/ak-cloud-deploy/SKILL.md:561`, is about WebSocket async mode)
- `docs/versioned_docs/`: not edited

## Open items for review

- `RECONNECTED_NOTE` is prompt text inside an adapter; it is a class constant a subclass can change or empty
- The replayed history is unbounded; a long call sends every turn on a transfer or a replayed reconnect
- The outbox, reconnect and stall thresholds come from one application's live runs
- Neither adapter acts on Gemini's `toolCallCancellation`; what Gemini does with the caller's speech while a blocking tool call is open is not verified live
- `RealtimeCall.connected` and `quiet_for()` lost their only consumer with the job runner, and `withdraw` has none in the driving application or the examples; kept as the public reading of the quiet rule, for review
