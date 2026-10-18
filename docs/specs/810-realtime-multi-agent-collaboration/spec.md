# #810: Realtime multi-agent collaboration with background work — Implementation Spec

Implements the requirements in [`design.md`](design.md). The realtime pool keeps each call's turn state, says
application posts at quiet moments through `RealtimeOutbox` / `RealtimeCall`, and feeds `RealtimeHook`s. The Gemini
adapter runs ADK's tool callbacks, seeds transfers with authored history, applies the call's `RunConfig`, and recovers
lost sockets by replaying the conversation. The
LiveKit gateway signs participant tokens. The idea: voice is the meeting room, not the engine.

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
│   ├── hooks.py               # RealtimeEvent, RealtimeHook
│   ├── base.py                # Agent.realtime_hooks
│   ├── module.py              # Module.realtime_hook
│   └── realtime/runner.py     # reports_speech_stopped; the new adapter events
├── framework/
│   ├── adk/adk.py             # GoogleADKRealtimeRunner; the ADK tool wrapper
│   └── openai/openai.py       # speech_stopped
├── integration/
│   └── livekit/adapter.py     # participant_token
└── pipeline/
    ├── __init__.py            # exports RealtimeCall
    └── realtime_pool.py       # turn state, RealtimeOutbox, RealtimeCall, hooks, recovery readiness, sweep
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

class RealtimeHook(ABC):
    async def on_event(self, session: Session, agent: Agent, event: RealtimeEvent) -> None: ...
    def name(self) -> str: ...

class Module:
    def realtime_hook(self, agent: Any, hooks: list[RealtimeHook]) -> Self: ...   # chains like pre_hook / post_hook

class RealtimeRunner:
    reports_speech_stopped: bool = False   # True only where the model signals the end of the caller's speech
```

Rules:

1. `core/hooks.py` imports nothing outside `core/` and pydantic.
2. The adapter callback contract (`RealtimeRunner` docstring) gains, none reaching the edge:
   - `speech_heard {}`: the caller is still speaking;
   - `speech_stopped {}`: the caller stopped; only from an adapter with `reports_speech_stopped`;
   - `caller_transcript {"text"}`: optional, the caller's words in one turn;
   - `reconnected {"agent", "seconds", "seeded"}`: the socket was replaced, with `seeded` true when the conversation was replayed;
   - `done` with status `reconnecting`: the turn under way ended because the socket was lost.
3. `Agent.realtime_hooks` is a list on the wrapped agent, like `pre_hooks`.
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
| `send_text(text, from_edge=True)` | `last_edge_text_at`; an `edge_text` hook event |

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

#### Hooks

- An agent with `realtime_hooks` gets an `asyncio.Queue` and one task per connection; `_observe` puts on the loop, `_observe_soon` from other threads
- Sources: `connect()` (`connected`); the pacing loop (`agent_changed`, `agent_said` at `done`, `interrupt` or a change); `caller_transcript` (`caller_said`); `send_text(from_edge=True)` (`edge_text`); `reconnected`; `close()` or a failure (`disconnected`, once)
- On close, the hooks get `_HOOK_DRAIN_SECONDS` (2 s), then the task is cancelled

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
| `server.py` | `verify_account(account_id, tool_context)`, a supervisor tool, keeps a verified account in the ADK tool state, which lasts for the call across transfers; `require_account`, the supervisor's `before_tool_callback`, answers in place of `transfer_to_agent` to `billing` or `tech_support` until an account is verified; `book_technician` starts a short background job (a thread) that confirms the slot with `RealtimeCall(session_id).post_when_idle("[Dispatch] ...", key="dispatch")`, the session id read with `ToolContext.get()` inside the tool; a `CallLog(RealtimeHook)` logging `caller_said`, `agent_said`, `agent_changed` and `reconnected`, attached with `Module.realtime_hook(supervisor, ...)`; the gateway is built under `__main__` |
| `server_test.py` (new) | Offline tests (Testing) |
| `README.md` | Refusing a handoff, background updates, the call log, the walkthrough, tests; the transfer gap now holds the caller's words |

`.github/test-config.yaml`: `- type: api` / `path: examples/api/livekit-voice/google/handoffs`; the tests need no
model, LiveKit server or key.

### Application building blocks

Verified for this part against the branch at `640bd057` and the installed `openai-agents==0.20.0` (the floor,
`ak-py/pyproject.toml:118`): `ToolInputGuardrailData` and `ToolInputGuardrailTripwireTriggered` are exported by
`agents`; the SDK's realtime session runs a tool's input guardrails in order, returns the first `reject_content`
message as the tool's output and raises on `raise_exception` (`agents/realtime/session.py:807-815`); `Runtime._current`
is class-level, so a thread other than the caller's sees the same runtime (`core/runtime.py:155`). No change to
`core/model.py`, `core/runtime.py` or the framework adapters' unary paths (design.md Non-goals).

```
ak-py/src/agentkernel/
├── core/
│   └── hooks.py                 # RealtimeEvent "idle" + quiet_for; RealtimeHook.idle_after_seconds
├── framework/openai/openai.py   # tool input guardrails in realtime
├── integration/adapter/gateway.py  # GatewayRunner.stop, GatewayPool
└── pipeline/
    ├── realtime/                # NEW package: the pool and the jobs together (Package layout, below)
    │   ├── __init__.py          # RealtimeCall, BackgroundJob, BackgroundJobRefused, BackgroundJobCancelled
    │   ├── pool.py              # was realtime_pool.py: RealtimeCall facts, run_in_background, jobs; idle; while_quiet
    │   └── jobs.py              # NEW: BackgroundJob, BackgroundJobRunner, the job errors
    ├── testing.py               # BackgroundJobHost
    └── __init__.py              # exports BackgroundJob, BackgroundJobRefused, BackgroundJobCancelled
```

#### Package layout (`pipeline/realtime/`)

- `pipeline/realtime_pool.py` moved to `pipeline/realtime/pool.py`; the jobs live in `pipeline/realtime/jobs.py`. Line
  citations to `realtime_pool.py` earlier in this spec refer to the file before the move
- No compatibility module: `pipeline/realtime_pool.py` was never in a release, and the public path is
  `agentkernel.pipeline` (`RealtimeCall`, `BackgroundJob`, `BackgroundJobRefused`, `BackgroundJobCancelled`), whose lazy
  exports now point at `realtime.pool` / `realtime.jobs`. Loggers follow the modules: `ak.pipeline.realtime.pool`,
  `ak.pipeline.realtime.jobs`
- `pipeline/realtime/__init__.py` imports the four public names, as `core/realtime/__init__.py` does its own; importing
  it starts no thread and loads no framework SDK
- `core/realtime/` keeps the adapter contracts and audio primitives; the gateways stay in `integration/adapter/`

#### Background jobs (`pipeline/realtime/jobs.py`)

```python
class BackgroundJobRefused(Exception): ...          # str(e) is the refusal, said to the model
class BackgroundJobCancelled(Exception): ...        # given to on_error for a job the call's close() dropped

class BackgroundJob:
    SESSION_INFIX = "-job-"
    id: str; name: str; session_id: str; arguments: dict[str, Any]
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    created_at: float; started_at: float | None; finished_at: float | None
    @property
    def job_session_id(self) -> str: ...             # f"{session_id}{SESSION_INFIX}{id}"
    async def ask(self, agent: str, prompt: str | Sequence[AgentRequest]) -> AgentReply: ...
    def post(self, text: str, *, key: str | None = None, expires_after: float = 600.0) -> bool: ...
    @staticmethod
    def call_session_id(session_id: str) -> str: ... # the text before SESSION_INFIX, else session_id

class BackgroundJobRunner:                           # internal; process-wide like the pool, hosted like it
    HEARD_SECONDS = 1.0; HEARD_TIMEOUT_SECONDS = 20.0; HEARD_POLL_SECONDS = 0.2
    SHUTDOWN_GRACE_SECONDS = 5.0; CANCEL_CLEANUP_SECONDS = 5.0; READY_TIMEOUT_SECONDS = 10.0
    @classmethod
    def initialize(cls) -> "BackgroundJobRunner": ...  # the process's runner, created once; starts no thread
    @classmethod
    def current(cls) -> "BackgroundJobRunner | None": ...
    @classmethod
    def reset(cls) -> None: ...                      # tests: stop() the runner, forget it
    def get_task(self) -> ThreadRunner.Task: ...      # "background-jobs", graceful, stop_all_on_failure
    def start(self) -> None: ...                      # blocking: run and supervise the loop until shutdown or stop()
    def stop(self, timeout: Optional[float] = None) -> bool: ...  # any thread; whether start() returned (or never ran)
    def wait_ready(self, timeout: float) -> bool: ...
    def submit(self, session_id, work, *, name, job_id=None, arguments=None, wait_until_heard=False,
               on_error=None, refuse=None) -> BackgroundJob: ...
    def jobs(self, session_id) -> list[BackgroundJob]: ...
    def close(self, session_id) -> None: ...
    def post(self, job, text, *, key=None, expires_after=600.0) -> bool: ...  # BackgroundJob.post delegates here
```

Rules:

1. **Hosting.** `initialize()` creates the runner (once; `RLock`-free class lock like the pool's) and starts nothing.
   `RealtimeAgentRunner._get_extra_tasks()` and `ECSRealtimeAgentRunner._get_extra_tasks()` return the pool's task
   and `get_task()`. `start()` creates the loop and a daemon thread (`ak-background-jobs`) running it with
   `run_forever`; the loop sets `_ready` from its first callback. `start()` then waits on the runner's stop event or
   `ThreadRunner.shutdown_event` (0.5 s slices), raising `RuntimeError` if the loop's thread has died (the host's
   `stop_all_on_failure`), and on either signal runs the shutdown (Rule 9) and returns. `start()` a second time
   raises. `stop(timeout)` sets the stop event and waits for `start()` to return, as `GatewayRunner.stop` does.
1a. **Admission.** `submit` raises `RuntimeError` at once, under the lock, while `_ready` is not set ("not ready") or
   admission is closed ("shutting down"); `RealtimeCall.run_in_background` raises `RuntimeError` when there is no
   runner, naming the realtime host and `BackgroundJobHost`. `RealtimeConnectionPool.get_or_create`, after its own
   loop's readiness wait, runs `_wait_for_jobs()`: with `_jobs_ready` unset it takes `_jobs_ready_lock` (no connection
   or job lock is held), checks again, and calls `wait_ready(READY_TIMEOUT_SECONDS)` on `BackgroundJobRunner.current()`;
   `_jobs_ready` is set only when there is no runner or the wait succeeded. A connection opening while another waits
   blocks on the lock and sees that outcome; after a timeout (a warning) the next connection waits again.
2. **Per session, in order.** Each session has a list of queued jobs, the job running and whether a worker is scheduled,
   all under the runner's lock. `submit` appends and, when no worker is scheduled, marks one and schedules it with
   `call_soon_threadsafe`; the worker pops the next job under the lock and, finding none, unmarks itself under the same
   lock. So a submission never waits behind a worker that is exiting.
3. **Admission.** Under the lock, before the job exists: `refuse(pending)` (queued and running jobs) returning text
   raises `BackgroundJobRefused`; a `job_id` equal to a pending job's raises `ValueError`. The default id is
   `f"{name}-{uuid4().hex[:8]}"`, so a default id never reuses an earlier job's AK session.
4. **Running a job.** `wait_until_heard`: poll every `HEARD_POLL_SECONDS` while `RealtimeCall.connected` and
   `quiet_for()` is below `HEARD_SECONDS`, at most `HEARD_TIMEOUT_SECONDS`. Then `status = "running"`,
   `text = await work(job)`; `done` and `job.post(text)` when it returned text. When `work` raises: `failed`, the
   exception kept as `job.error`, logged with its traceback, and `on_error(job, error)`'s text posted; `on_error`
   raising is logged.
5. **`ask`.** `AgentService.ensure_agent_available(agent)` (`ValueError` for an agent not loaded), a fresh
   `AgentService`, `select(job.job_session_id, agent)`, `run_multi([AgentRequestText(prompt=prompt)])` (a sequence is
   passed as it is). The reply is returned as it is: an adapter reports a framework error as error text, so `work`
   detects failure through its framework's own signal during the run and raises (the docs show it with the OpenAI
   Agents SDK's `RunHooks.on_agent_end`).
6. **Close.** `close(session_id)`: under the lock, the queued jobs become `cancelled` and are dropped, and the running
   job is marked closed; if its status is still `queued` (its worker waits for it to be heard) it is `cancelled` too.
   Outside the lock, `on_error(job, BackgroundJobCancelled())` is called for each cancelled job, its text ignored. The
   worker moves a job to `running` under the lock, after the wait, only if it was not cancelled; the wait itself ends
   once the job is closed. `post` checks `_closed` and queues the text under the runner's lock, which `close` takes
   before `RealtimeCall.close()` drops the call's posts, so a post never recreates a closed call's outbox; a closed
   job's `post` drops the text (info log) and returns `False`. Later submissions start afresh.
8. **Cancellation.** `work` raising `asyncio.CancelledError` while its task is not being cancelled: the job is
   `cancelled`, `on_error(job, BackgroundJobCancelled())`'s text is posted, the worker goes on. The worker task being
   cancelled (`Task.cancelling()`), while the job runs or while it waits to be heard: the job is `cancelled` with its
   `finished_at`, `on_error(job, BackgroundJobCancelled())` is called (its text not posted), and the error propagates;
   `_serve`'s `finally` frees the session (`running` cleared) and, with jobs still queued, schedules a new worker,
   unless `reset()` is stopping the runner. A worker's coroutine is made on the job loop, when it can run.
7. **Callbacks are sync**, called on the job loop (`on_error` for a cancelled job: on the closing thread), and must be
   short; `work` is the async part. Every callback runs outside the lock except `refuse`, which must not call back
   into the runner.
7a. **One end per job.** `_end(job, status, error=None) -> bool`, under the lock, sets the terminal status, `error`
   and `finished_at` only for a job not yet ended, and says whether it did; `on_error` is called only by the caller
   that ended the job, so at most once. `post` (runner-owned, `BackgroundJob.post` delegates) checks `_closed` and
   queues under the lock.

9. **Shutdown.** Under the lock: admission closed, every job's `_closed` set, queued jobs and a running job still
   `queued` (waiting to be heard) ended `cancelled` with `_end`. Outside it: their `on_error(job,
   BackgroundJobCancelled())`, text ignored. Then, on the host's thread: poll (50 ms) until no job is `running`, at
   most `SHUTDOWN_GRACE_SECONDS`; then always, whatever the jobs' status (a cancelled job's worker may still wait, and
   finished work may leave tasks of its own), `_cancel_remaining_tasks()`: `call_soon_threadsafe` a callback that, on
   the loop, creates a task cancelling every other task and `asyncio.wait`ing them up to `CANCEL_CLEANUP_SECONDS`, its
   count of unfinished tasks returned through a `concurrent.futures.Future` the host waits on for
   `CANCEL_CLEANUP_SECONDS` plus 50 ms (on a timeout the future is cancelled, so the task is never created); log an
   error naming each job still running, and the tasks that did not finish; `call_soon_threadsafe(loop.stop)` and join
   the loop's thread for 1 s, logging that the loop is blocked by application code and stays
   until the process exits if it has not ended. The loop's thread closes the loop after `run_forever` returns. A
   worker cancelled during shutdown is not replaced. Never: `loop.close()` from another thread, a wait without a bound.
10. **Test support.** `BackgroundJobHost` (`pipeline/testing.py`): `__enter__` refuses (`RuntimeError`) when
   `BackgroundJobRunner.current()` exists, else `initialize()`s, runs `start()` on a daemon thread and waits for
   `wait_ready`; `__exit__` calls `stop()`, joins its thread and forgets the runner only if it is still the one it
   created. It never sets `ThreadRunner.shutdown_event`.

#### Call facts, the idle event and `while_quiet` posts (`pipeline/realtime/pool.py`, `core/hooks.py`)

```python
class RealtimeEvent(BaseModel):
    type: Literal[..., "idle"]
    quiet_for: Optional[float] = None               # idle only

class RealtimeHook(ABC):
    idle_after_seconds: Optional[float] = None      # set to receive "idle"

class RealtimeCall:
    last_caller_input_at: Optional[float]           # max(last_caller_speech_at, last_edge_text_at)
    last_agent_change_at: Optional[float]
    last_tool_result_at: Optional[float]
    jobs: list[BackgroundJob]
    def post_when_idle(self, text, *, key=None, expires_after=600.0, while_quiet=False) -> bool: ...
    def run_in_background(self, work, *, name, job_id=None, arguments=None, wait_until_heard=False, on_error=None, refuse=None) -> BackgroundJob: ...
    def close(self) -> bool: ...                     # also BackgroundJobRunner.close, when a runner exists
```

Rules:

1. The facts read the session's open connection (`closed` false), ready or recovering; `None` without one.
2. `RealtimeOutbox.quiet_since(connection, now)`: the newest of the times `_quiet_for` takes, or `None` when busy;
   `quiet_for` is `now - quiet_since`.
3. Every `_POST_POLL_SECONDS`, after the posts, `_report_idle(now)`: for each ready connection whose agent has a hook with
   `idle_after_seconds` (`n`), `since = quiet_since`; for each such hook, keep `(since, told)`, `told` reset to 0 for a
   new `since`; when `now - since >= n * (told + 1)`, count one more and queue `idle` for that hook alone. So a hook is
   told at `n`, `2n`, `3n` seconds of one stretch, never twice for the same `n`. A threshold of 0 or less asks for nothing.
4. The hook queue holds `(event, only, stretch)`; `only` and `stretch` are `None` for every other event. `_serve_hooks`
   drops an `idle` whose `stretch` is no longer the outbox's `quiet_since`, and gives an event to `only` when set, else
   to every hook.
5. `quiet_since` takes `last_agent_change_at` with the other times: a handover ends a quiet stretch.
6. `while_quiet`: `add` takes the session's ready connection and records `stretch = quiet_since`; with no connection or
   a busy call it queues nothing and returns `False`. `take` drops (info log) each `while_quiet` post whose `stretch`
   is not the current `quiet_since`, which a new connection also changes (its `connected_at`).
7. `GatewayRunner` unregisters with `StatefulEdgeRegistry.unregister(session_id, adapter=...)`, which removes the
   entry only while it is that adapter.

#### Gateways (`integration/adapter/gateway.py`)

```python
class GatewayRunner:
    def stop(self, timeout: Optional[float] = None) -> bool: ...   # whether start() has returned (or never ran)

class GatewayPool:
    STOP_TIMEOUT_SECONDS = 10.0
    def __init__(self, factory: Callable[[str], StatefulEdgeAdapter]): ...
    def open(self, session_id: str) -> StatefulEdgeAdapter: ...
    def get(self, session_id: str) -> Optional[StatefulEdgeAdapter]: ...
    def is_open(self, session_id: str) -> bool: ...
    def close(self, session_id: str, timeout: float = STOP_TIMEOUT_SECONDS) -> bool: ...
    def close_all(self) -> None: ...
```

Rules:

1. `stop()` sets a `threading.Event`; `_run_until_shutdown` ends on it as on the shutdown event, stopping the adapter on
   its own loop. `start()` after `stop()` returns at once, registering nothing. `stop(timeout)` waits for `start()` to
   return when it has begun.
2. `open`: under the pool's lock, the open gateway, else `factory(session_id)`, a `ValueError` when its `session_id`
   differs, a `GatewayRunner`, and a daemon thread `gateway-<session_id>` running `start()`. The thread forgets its entry
   when `start()` returns or raises (logged), only if the entry is still its own.
3. `close`: pops the entry under the lock and calls `stop(timeout)` outside it.

#### OpenAI realtime tool input guardrails (`OpenAIRealtimeRunner`)

- `_offered_tools` skips only tools with `needs_approval` set or `tool_output_guardrails`; the warning names those two
- `execute_tool`, with the AK context set and before `invoke_function_tool`: `_input_guardrail_rejection(tool, sdk_context, caller)`
  runs each guardrail with `ToolInputGuardrailData(context=sdk_context, agent=caller)`; `raise_exception` raises
  `ToolInputGuardrailTripwireTriggered`, `reject_content` returns its message, which `execute_tool` returns without
  running the tool

#### Example (`examples/api/livekit-voice/google/handoffs/server.py`)

- `book_technician(account_id, day)`: `RealtimeCall(session_id).run_in_background(confirm_visit, name="dispatch",
  arguments={"account_id": ..., "day": ...}, refuse=...)`, refusing while a dispatch job for the account is pending;
  `confirm_visit(job)` sleeps `DISPATCH_SECONDS` and posts the confirmation with `key="dispatch"`. No thread

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
9. OpenAI realtime offers tools that carry input guardrails, and runs the guardrails; before, such a tool was not offered. Tools needing approval or carrying output guardrails are still not offered.
10. `RealtimeCall.close()` also cancels the session's queued background jobs. New: before, there were no AK jobs.
11. Realtime hooks receive the queue as `(event, only)` internally; a hook without `idle_after_seconds` sees exactly the events it saw before.

**Non-changes:** the reply models and every adapter's error reply; the existing `RealtimeCall` members; `GatewayRunner.start()` for a runner never stopped; `IOHandler.run(gateways=...)`; the stream event union and `AgentChanged`'s fields; queue message formats; OpenAI handoffs and the
LiveKit gateway's participants; `ToolContext.set()` / `reset()`; `AKConfig`; the extras; other examples' behaviour.

## Error handling

| Condition | Where | Surfaces as |
|---|---|---|
| Socket drops, server closes, stall | `GoogleADKRealtimeRunner._recover` | `done {"status": "reconnecting"}`, then `reconnected`; hooks get `reconnected` |
| Every reconnect attempt fails | `_recover` | `error` → the pool fails the connection; the edge gets an error chunk |
| Quick drops `RECONNECT_ATTEMPTS` times in a row | `_recover` | `error` ("keeps dropping") |
| A transfer's socket does not open in 15 s | `_transfer` | `error`, as before |
| A tool raises with no error callback answering | `_run_tool` → the pool | `Error: ...` as the tool result, as before |
| A tool result arrives for a lost socket | `send_tool_result` | recorded in the history; info log |
| A realtime hook raises | `_serve_hooks` | logged; the next hook runs |
| A hook is still busy at close | `_finish_hooks` | warning after 2 s; its task cancelled |
| `post_when_idle` with no pool in the process | `RealtimeCall` | warning; the post is dropped |
| A full outbox / an expired post | `RealtimeOutbox` | the oldest dropped with a warning / dropped with an info log |
| `participant_token` misuse | `LiveKitEdgeGateway` | `ValueError` |
| A job's `work` raises | `BackgroundJobRunner` | `failed`, traceback logged, `on_error`'s text posted |
| A job asks an agent that is not loaded | `BackgroundJob.ask` | `ValueError` (into the rule above) |
| `refuse` returns text / a pending job has the id | `BackgroundJobRunner.submit` | `BackgroundJobRefused` / `ValueError` |
| A closed call's job posts | `BackgroundJob.post` | dropped, info log, `False` |
| `on_error` raises | `BackgroundJobRunner` | logged |
| A tool input guardrail rejects / trips | `OpenAIRealtimeRunner.execute_tool` | its message is the result / `Error: ...` from the pool, logged |
| A pooled gateway's `start()` fails | `GatewayPool` | logged; the entry forgotten |
| A factory raises, or returns an adapter for another session | `GatewayPool.open` | raised / `ValueError` |

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
- `TestRealtimeHooks`: events from connect to close, `reconnected` included; a failing hook; edge text from another thread; no hooks, no queue
- The pool: the sweep closes a failed connection without further input; the loop-lag warning
- `TestADKFirstAudioLog`: the wait logged once per turn

### New: `ak-py/tests/test_realtime_jobs.py`

- `TestBackgroundJobRunner`: a session's jobs in order, other sessions beside them; text returned is posted; `failed` and `on_error`; `refuse` under the lock (two threads submitting at once, one refused); a duplicate id; `close` cancels the queued, lets the running finish and drops its post; `wait_until_heard` waits for a quiet call and gives up after the timeout; one loop for every job
- `TestBackgroundJobAsk`: the job's own session, shared by its asks; the reply returned as it is; an unknown agent; `call_session_id`
- `TestRealtimeCallJobs`: `run_in_background`, `jobs`, `close()` ending the jobs

### Changed: other files

- `test_module.py`: `realtime_hook` attaches to the wrapped agent and chains
- `test_livekit_gateway.py`: participant tokens (room, grants, refusals, no key); the alias
- `test_ecs_akagentrunner_realtime.py`: `send_text(..., from_edge=True)`
- `test_realtime_pool.py`: `TestRealtimeCallFacts` (each fact; `None` without a connection; read while recovering); `TestRealtimeIdle` (at each multiple of the threshold while quiet, started over by activity, a hook's own threshold, only to hooks that ask, in order with other events); `TestRealtimeOutbox` (`while_quiet` dropped by activity, said when quiet); `TestOpenAIRealtimeToolGuardrails` (offered; a rejection skips the tool; a tripwire raises; approval and output guardrails still excluded)
- `test_integration_gateway_runner.py`: `stop()` from another thread, before `start()`, and its wait; `GatewayPool` open, reuse, close, a failed start forgotten, a factory error, a mismatched session, a slowly leaving gateway not unregistering its replacement
- Lifecycle cases (`test_realtime_jobs.py`, `test_realtime_pool.py`): close during the heard wait, work raising `CancelledError`, a cancelled worker, a post racing close; a stale `idle` dropped, a `while_quiet` post refused when busy and dropped on a new connection or a handover

### New: `examples/api/livekit-voice/google/handoffs/server_test.py`

- `require_account` answers in place of a transfer to a specialist until the account is verified, then lets it through; other tools and other transfers pass
- `verify_account` records a verified account in the tool state and refuses a malformed one
- `book_technician` posts the confirmation once its job ends (a fake `RealtimeCall`), keyed `dispatch`, and refuses a second booking for the account while one is pending
- `require_account` is the supervisor's callback, and `CallLog` its realtime hook; `CallLog` logs both sides of the call

### Live check

On `gemini-3.1-flash-live-preview`, with background agents on `openai/gpt-4.1` through LiteLLM: a handoff refused by a
callback, a background post said at a quiet moment, a forced socket drop, and a call past 10 minutes.

### Commands

```bash
cd ak-py && ./build.sh && uv run pytest tests/test_realtime_pool.py tests/test_module.py tests/test_livekit_gateway.py tests/test_ecs_akagentrunner_realtime.py -v
cd ak-py && uv run pytest
make lint-check-all
cd examples/api/livekit-voice/google/handoffs && ./build.sh local && uv run pytest -s
```

## Migration surface (docs and skills)

- `docs/docs/integrations/livekit.md`: tool callbacks and refusing a handoff, "Tokens for your users", "Talking into a live call", "Watching a live call"; new sections on the call's Live settings (`run_config`) and on recovery; the limitation at `:285` (pre- and post-hooks do not run on realtime turns, realtime hooks do) and at `:288` (a Gemini socket failure no longer restarts the call)
- `docs/docs/integrations/hooks.md`: `RealtimeHook`, its `idle` event
- `livekit.md`: "Background work" (`run_in_background`, detecting a failed run natively), "Rooms per session" (`GatewayPool`), the call facts and `while_quiet`, OpenAI tool input guardrails
- `realtime-internals.md`: the job loop, the idle event, `while_quiet`
- `docs/docs/architecture/realtime-internals.md`: the adapter events, turn state as internal to the pool, `RealtimeOutbox`, realtime hooks, tool callbacks, a connection recovery section, the stall limit
- `core/realtime/runner.py`: the callback contract gains `reconnected` and the `reconnecting` status (Rule 2)
- `.agents/skills/ak-dev-architecture/SKILL.md`: a realtime section (it has none); `.agents/skills/ak-dev-testing-conventions/SKILL.md`: the `test_realtime_pool.py` row
- No new integration, so the docs-site landing and features pages are unchanged
- Bundled skills: no change (the one match, `ak-py/src/agentkernel/skills/ak-cloud-deploy/SKILL.md:561`, is about WebSocket async mode)
- `docs/versioned_docs/`: not edited

## Open items for review

- `RECONNECTED_NOTE` is prompt text inside an adapter; it is a class constant a subclass can change or empty
- The replayed history is unbounded; a long call sends every turn on a transfer or a replayed reconnect
- The outbox, reconnect and stall thresholds come from one application's live runs
