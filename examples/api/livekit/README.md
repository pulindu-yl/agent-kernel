# LiveKit Voice and Vision Integration Example

This example connects an OpenAI Agents SDK agent registered with Agent Kernel to a LiveKit
real-time voice room. LiveKit handles VAD, STT, interruption, and TTS. The integration forwards
Agent Kernel response deltas from `AgentService.stream_multi()` into the LiveKit TTS pipeline so
speech can begin before the complete response is available.

The optional vision path attaches the latest webcam frame to the next voice turn through Agent
Kernel's multimodal pipeline.

## Prerequisites

1. Python 3.12 or 3.13.
2. A [LiveKit Cloud](https://cloud.livekit.io/) project or self-hosted LiveKit server.
3. The LiveKit WebSocket URL, API key, and API secret.
4. API keys for the configured providers. This example uses OpenAI for the agent and TTS, and
   Deepgram for STT.

## Setup

### 1. Set credentials

Create a `.env` file in this directory or export the variables:

```bash
export AK_LIVEKIT__URL="wss://your-project.livekit.cloud"
export AK_LIVEKIT__API_KEY="your_api_key"
export AK_LIVEKIT__API_SECRET="your_api_secret"
export OPENAI_API_KEY="your_openai_key"
export DEEPGRAM_API_KEY="your_deepgram_key"
```

Keep the LiveKit API secret on the server. Never send it to a browser or mobile client.

### 2. Review the configuration

The included `config.yaml` enables webcam vision and Agent Kernel streaming:

```yaml
livekit:
  agent: "my-voice-agent"
  worker_name: "agent-kernel-worker"
  stt_provider: "deepgram"
  tts_provider: "openai"
  streaming_mode: "streaming"
  vision_enabled: true

multimodal:
  enabled: true
  description_model: "gpt-4o"
```

`streaming_mode` accepts:

| Mode | Behavior |
| --- | --- |
| `streaming` | Calls `AgentService.stream_multi()` and forwards every text delta immediately. |
| `buffered` | Calls `AgentService.run_multi()` and waits for the complete response before sending it to TTS. |

Choose the mode explicitly. OpenAI, LangGraph, and Google ADK support streaming. Use `buffered`
for CrewAI and Smolagents because their current Agent Kernel adapters do not implement streaming.
Also use it for full-response guardrails, post-hooks that only implement `on_run`, and structured
output. Streaming post-hooks must implement `on_stream_chunk`.

To run voice without webcam input, set both `livekit.vision_enabled` and `multimodal.enabled` to
`false`. Vision requires the `multimodal` package extra; the example build already installs it.

### 3. Install

The build script installs this repository's Agent Kernel source in editable mode with:

```text
agentkernel[api,livekit,multimodal,openai]
```

Run:

```bash
./build.sh
```

For a separate application installed from PyPI, use:

```bash
pip install "agentkernel[api,livekit,openai,multimodal]"
```

Remove `multimodal` when vision is disabled, and replace `openai` with the Agent Kernel framework
extra used by your agent.

## Run

```bash
uv run server.py
```

The REST API starts on `http://localhost:8000` and starts the named LiveKit worker configured by
`livekit.worker_name` in the same process.

## Connect a client

For local development, request a participant token:

```text
http://localhost:8000/livekit/token?room=demo-room&identity=user1
```

The generated token explicitly dispatches the configured worker to the room and carries the
participant identity in dispatch metadata. Paste it into the
[LiveKit Agents Playground](https://cloud.livekit.io/projects/p_/agents), or use it in your own
LiveKit frontend. Start speaking; with vision enabled, turn on the camera and ask a question about
what it shows.

The token's agent dispatch is applied only when that token creates the room. For repeat testing,
use a fresh room name; when joining an existing room, create the named dispatch through LiveKit's
dispatch API first.

The `/livekit/token` endpoint in this example is intentionally unauthenticated for local testing.
In production, construct `AgentLiveKitRequestHandler` with an `auth_dependency`, authorize the
requested `room` and `identity`, and apply normal rate limits.

Sessions are scoped by room SID and dispatched participant identity. Participants in the same
room-scoped job share the selected conversation. Recreating a room produces a new session; if no
SID is available, the room name is used.

## Troubleshooting

### The participant connects but the agent does not join

- Confirm the worker registered successfully in the server logs.
- Verify the token came from this server's `/livekit/token` endpoint so it includes named-agent
  dispatch.
- Confirm `AK_LIVEKIT__URL`, `AK_LIVEKIT__API_KEY`, and `AK_LIVEKIT__API_SECRET` belong to the same
  LiveKit project.

### No voice response

- Confirm `livekit.agent` matches the agent registered in `server.py`.
- Check the server logs for provider authentication or quota errors.
- Confirm microphone permission is granted and the microphone is not muted.
- When using `streaming_mode: streaming`, verify that the selected framework supports Agent Kernel
  streaming. Otherwise select `buffered` explicitly.
- If speech stops partway through a response, inspect the server logs.

### Vision does not work

- Confirm `livekit.vision_enabled: true` and `multimodal.enabled: true`.
- Confirm the application installed the `multimodal` extra.
- Use a vision-capable `multimodal.description_model` and provide its API credentials.
- Confirm the camera track is published before asking about the image.

## Resources

- [LiveKit Agents documentation](https://docs.livekit.io/agents/)
- [Agent Kernel LiveKit integration guide](../../../docs/docs/integrations/livekit.md)
