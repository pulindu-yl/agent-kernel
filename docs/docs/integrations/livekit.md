---
sidebar_position: 10
---

# LiveKit Voice and Vision Integration

Agent Kernel integrates with [LiveKit Agents](https://docs.livekit.io/agents/) to expose an
existing Agent Kernel agent through a real-time WebRTC voice session. LiveKit handles voice
activity detection, speech-to-text (STT), interruptions, and text-to-speech (TTS). Agent Kernel
handles the agent, tools, hooks, and session.

LiveKit is an integration, not an Agent Kernel framework adapter. You can use it with any
supported Agent Kernel framework.

## Architecture

For each finalized transcript, the integration's LiveKit
[custom `Agent.llm_node` pipeline node](https://docs.livekit.io/agents/logic/nodes/) creates Agent
Kernel requests. In `streaming` mode it calls `AgentService.stream_multi()` and yields text deltas
to LiveKit as soon as they are available, allowing TTS to begin before the full response is
complete. In `buffered` mode it calls `AgentService.run_multi()` and sends the complete response
to LiveKit as one segment.

```text
LiveKit room -> STT -> Agent.llm_node -> AgentService -> Agent Kernel runner
LiveKit room <- TTS <- text deltas  <-                 <-
```

Session IDs use the LiveKit room SID and, when available, the dispatched participant identity.
Additional participants in the same room-scoped job share its Agent Kernel conversation. A new
room SID starts a new session even when the room name is reused. If no SID is available, the room
name is used.

## Installation

Install the API, LiveKit, and selected framework extras. For example, with the OpenAI Agents SDK:

```bash
pip install "agentkernel[api,livekit,openai]"
```

Replace `openai` with the extra for your framework. Webcam vision additionally requires the
multimodal dependencies:

```bash
pip install "agentkernel[api,livekit,openai,multimodal]"
```

You also need:

1. A [LiveKit Cloud](https://cloud.livekit.io/) project or self-hosted LiveKit server.
2. `AK_LIVEKIT__URL`, `AK_LIVEKIT__API_KEY`, and `AK_LIVEKIT__API_SECRET`.
3. Credentials for your STT and TTS providers, such as `DEEPGRAM_API_KEY` and
   `OPENAI_API_KEY`.

The LiveKit extra targets the LiveKit Agents 1.6 API.

## Configuration

```yaml
livekit:
  agent: "my-voice-agent"
  worker_name: "agent-kernel-worker"
  stt_provider: "deepgram"    # deepgram or openai
  tts_provider: "openai"      # openai, elevenlabs, or google
  streaming_mode: "streaming" # streaming or buffered
  vision_enabled: false
  # Prefer environment variables for these secrets:
  # url: "wss://your-project.livekit.cloud"
  # api_key: "your_api_key"
  # api_secret: "your_api_secret"
```

Every setting can be supplied as an environment variable. Nested keys use `__`:

```bash
export AK_LIVEKIT__AGENT="my-voice-agent"
export AK_LIVEKIT__WORKER_NAME="agent-kernel-worker"
export AK_LIVEKIT__URL="wss://your-project.livekit.cloud"
export AK_LIVEKIT__API_KEY="your_api_key"
export AK_LIVEKIT__API_SECRET="your_api_secret"
export AK_LIVEKIT__STREAMING_MODE="streaming"
```

### Streaming modes

| Mode | Behavior |
| --- | --- |
| `streaming` | Calls `AgentService.stream_multi()` and forwards each Agent Kernel text delta immediately. |
| `buffered` | Calls `AgentService.run_multi()` and waits for the complete response before sending it to LiveKit TTS. |

Choose the mode explicitly for the selected Agent Kernel runner. OpenAI Agents SDK, LangGraph,
and Google ADK support streaming. Use `buffered` with CrewAI and Smolagents because their current
Agent Kernel adapters do not implement streaming.

Also use `buffered` when output guardrails must validate the complete response, when a post-hook
only implements complete-response `PostHook.on_run`, or when the agent uses structured output.
The streaming path invokes `PostHook.on_stream_chunk` for deltas and does not invoke the
complete-response `on_run` path.

Failures before output return a generic message. Failures after the first delta are logged and
end the stream.

For structured output, ensure the complete returned representation is suitable for speech,
because LiveKit sends that representation to TTS.

### Conversation history

LiveKit uses Agent Kernel session memory but does not create Conversation Thread Manager threads,
so its conversations do not appear in thread-listing APIs.

### Webcam vision

Vision requires both LiveKit capture and Agent Kernel multimodal processing:

```yaml
livekit:
  vision_enabled: true

multimodal:
  enabled: true
  description_model: "gpt-4o"
```

Install the `multimodal` extra as shown above. The latest captured video frame is attached to the
next voice turn.

## Usage

Register an Agent Kernel agent and include `AgentLiveKitRequestHandler` when starting the REST API:

```python
from agentkernel.api import RESTAPI
from agentkernel.livekit import AgentLiveKitRequestHandler
from agentkernel.openai import OpenAIModule
from agents import Agent

voice_agent = Agent(
    name="my-voice-agent",
    instructions="You are a concise voice assistant. Do not use markdown or emojis.",
)
OpenAIModule([voice_agent])

if __name__ == "__main__":
    RESTAPI.run([AgentLiveKitRequestHandler()])
```

The handler starts the worker named by `livekit.worker_name` and exposes
`GET /livekit/token?room=<room>&identity=<identity>`. Generated tokens include an
[agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch/) for that worker and the
participant identity used for session scoping.

LiveKit applies token-embedded room configuration only when that token creates the room. If the
room already exists, use a fresh room name or dispatch the named agent through LiveKit's dispatch
API before participants join.

The endpoint has no authentication unless you provide an `auth_dependency`. Protect it in every
production deployment and authorize both the requested room and identity:

```python
handler = AgentLiveKitRequestHandler(auth_dependency=require_authenticated_user)
```

Run the bundled example under `examples/api/livekit`, then connect with the
[LiveKit Agents Playground](https://cloud.livekit.io/projects/p_/agents) or your own frontend.
