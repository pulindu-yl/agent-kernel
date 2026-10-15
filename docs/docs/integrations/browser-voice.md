# Browser Voice Integration

The browser voice edge lets a web page talk to OpenAI Agents SDK and Google ADK realtime agents directly over one WebSocket: the microphone goes up, the agent's voice comes down. It needs no media server only Agent Kernel's server and a browser. It runs on Agent Kernel's realtime execution pipeline.

## When to use it

| | Browser voice | LiveKit |
|---|---|---|
| Infrastructure | Your server only | A LiveKit server or LiveKit Cloud |
| Network | Plain HTTPS/WSS, through any proxy or firewall that passes WebSockets | WebRTC (UDP), with TURN behind strict firewalls |
| Echo cancellation | The browser's own, on the page's audio (test your target browsers) | Built into WebRTC |
| Several people in one call | No: one browser per call | Yes |
| Calls started and ended at runtime | Yes: one per connection | One gateway per room, started with the process |

## Quick start

```bash
pip install "agentkernel[browser,adk,api]"
```

```python
from fastapi import WebSocket
from google.adk.agents import Agent

from agentkernel.adk import GoogleADKModule
from agentkernel.api import RESTAPI
from agentkernel.browser import BrowserVoiceEndpoint
from agentkernel.pipeline import IOHandler

agent = Agent(name="general", model="gemini-3.1-flash-live-preview", instruction="Give short answers.")
GoogleADKModule([agent])


def session_for(websocket: WebSocket) -> str | None:
    """The session a call joins, or None to refuse the connection."""
    return websocket.query_params.get("session") or None


if __name__ == "__main__":
    RESTAPI.add(BrowserVoiceEndpoint(session_for, agent_name="general").router("/voice"))
    IOHandler.run()
```

With `execution.mode: realtime` in `config.yaml`. The [browser-voice example](https://github.com/yaalalabs/agent-kernel/tree/develop/examples/api/browser-voice/google) adds the page: about a hundred lines of plain JavaScript.

## How it works

`BrowserVoiceEndpoint` serves a WebSocket route. For each connection it:

1. Asks `session_for(websocket)` which session the call joins. This is where the application checks the origin, a login, or that a meeting exists. `None` refuses the connection (close code `4403`).
2. Creates a `BrowserVoiceEdge` for that connection and registers it in `StatefulEdgeRegistry` under the session, so the Response Handler routes the session's output back to this socket.
3. Unregisters it when the socket closes.

A new connection for a session that already has one (a reload, a second tab) replaces it; the old one is closed with code `4409`. Sessions come and go with their connections, so one process serves many calls with nothing started in advance; no `GatewayRunner` is involved.

The edge sends the microphone and typed text to the input queue like any edge. The realtime pool runs the model connection per session and paces the agent's audio, and the edge forwards each streamed chunk to the browser. One socket carries everything in order, so a change of agent reaches the page exactly where the new agent's audio starts.

## Wire protocol

**Binary frames**, both ways: 16-bit little-endian mono PCM at 24 kHz (`EDGE_SAMPLE_RATE`). Up, the microphone, which the edge batches per `execution.realtime.input_batch_ms`. Down, the agent's voice, paced by the pool `execution.realtime.playback_lead_ms` ahead of real time.

**Text frames**, JSON objects:

| Direction | Message | Meaning |
|---|---|---|
| Up | `{"type": "hello", "name": "Ann"}` | The caller joined; the agent greets them by this name (`BrowserVoiceEdge.GREETING`, trimmed to 40 characters) |
| Up | `{"type": "text", "text": "..."}` | Typed words, taken as the caller's |
| Down | `{"type": "agent", "agent": "pm", "agents": ["pm", "ba"]}` | The agent speaking from here on; `agents`, the whole team, on the first only |
| Down | `{"type": "said", "agent": "pm", "text": "...", "at": 1760500000.1}` | What an agent said, once its turn ended or the call moved to another agent; `at` is the server's time (epoch seconds), to order it among words the page gets elsewhere, such as the caller's from a `RealtimeHook` |
| Down | `{"type": "interrupt"}` | The caller barged in: drop the audio still queued for playback |
| Down | `{"type": "error", "message": "..."}` | The turn failed; the message is a generic one, never the raw error |

A page that stops reading is disconnected (close code `1013`) once `BrowserVoiceEdge.MAX_OUTBOX` messages wait for it, rather than buffered without bound.

## The page

The page needs the microphone, a playback queue it can clear, and the socket:

- Capture with `getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true}})` and an `AudioWorklet`; resample to 24 kHz and send `Int16Array` buffers.
- Play the binary frames through a second `AudioWorklet` that queues them, and clear the queue on `interrupt`.
- Create the `AudioContext` from a click: browsers start audio only after the user acts.

The browser's echo cancellation must keep the agent's voice, played by the page, out of the microphone; otherwise the agent hears itself and stops. Test the browsers and devices you target, and suggest headphones when in doubt.

## Ending a call

`BrowserVoiceEndpoint.disconnect(session_id)` closes a session's browser connection, and `RealtimeCall(session_id).close()` closes its model connection at once (otherwise it is evicted after five idle minutes):

```python
await endpoint.disconnect(meeting_id)
RealtimeCall(meeting_id).close()
```

## Security

Any web page can open a WebSocket to a server its visitor can reach, and the browser sends the cookies of your site along. Check the `Origin` header in `session_for`, and authenticate the user there, before accepting a call.
