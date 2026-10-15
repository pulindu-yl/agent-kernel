import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from agentkernel.adk import GoogleADKModule, GoogleADKToolBuilder
from agentkernel.api import RESTAPI
from agentkernel.browser import BrowserVoiceEndpoint
from agentkernel.pipeline import IOHandler
from fastapi import APIRouter, WebSocket
from fastapi.responses import HTMLResponse

from google.adk.agents import Agent as GoogleAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
)
# The edge enqueues the microphone ten times a second, and the producer logs every message.
logging.getLogger("ak.integration.producer").setLevel(logging.WARNING)


def get_weather(location: str) -> str:
    """Get the current weather in a given location."""
    return f"The weather in {location} is sunny and 75 degrees."


general_agent = GoogleAgent(
    name="general",
    model="gemini-3.1-flash-live-preview",
    description="Agent for general questions",
    instruction="You provide assistance with general queries. Give short and direct answers.",
    tools=GoogleADKToolBuilder.bind([get_weather]),
)

GoogleADKModule([general_agent])


def session_for(websocket: WebSocket) -> Optional[str]:
    """The session a browser's call joins: the one its page made up. Only pages of this site may call:
    any web page can open a WebSocket to a server its visitor can reach."""
    origin = websocket.headers.get("origin")
    if origin and urlsplit(origin).hostname != urlsplit(f"//{websocket.headers.get('host', '')}").hostname:
        return None
    return websocket.query_params.get("session") or None


page = APIRouter()


@page.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    RESTAPI.add(page)
    RESTAPI.add(BrowserVoiceEndpoint(session_for, agent_name="general").router("/voice"))
    IOHandler.run()
