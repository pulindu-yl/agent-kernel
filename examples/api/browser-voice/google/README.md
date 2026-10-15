# Browser voice (Google Gemini Live)

Talk to a Gemini Live agent from a web page. Agent Kernel's browser voice edge (`agentkernel.browser`) carries the
call over one WebSocket between the page and the server: the microphone up, the agent's voice down, as 16-bit PCM
at 24 kHz. No media server, no LiveKit account: one Python process and a browser.

`index.html` is the whole client, in plain JavaScript: audio worklets for the microphone and playback, and the
wire protocol described in `docs/docs/integrations/browser-voice.md`. Each browser tab is a call of its own.

## Quickstart

```bash
./build.sh            # or ./build.sh local, for Agent Kernel built from this repository
export GEMINI_API_KEY="..."
uv run server.py
```

Open **http://localhost:8000**, press **Start talking**, allow the microphone, and speak.

The page relies on the browser's echo cancellation to keep the agent's voice out of the microphone; with
laptop speakers and no echo cancellation the agent hears itself and stops. Headphones always work.

## Tests

```bash
uv run pytest server_test.py
```
