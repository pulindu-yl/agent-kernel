"""Start here: builds the software team and serves it in a LiveKit room.

    uv run server.py

Every agent is a Google ADK agent. The voice agents (members/voice.py) run on Gemini Live; the Coder
and the Reviewer (members/background.py) run on a text model through LiteLLM, OpenAI's ``gpt-4.1``
unless ``TEAM_TEXT_MODEL`` says otherwise.

The client joins from the page at http://localhost:8000 (panel/, built with ``npm run build``), which
shows the call beside the project board (team/board.py).
"""

import logging
import os
from pathlib import Path
from typing import Optional

from agentkernel.api import RESTAPI
from agentkernel.integration.adapter import GatewayRunner
from agentkernel.integration.livekit import LiveKitEdgeGateway
from agentkernel.pipeline import IOHandler

from members.background import BackgroundTeam
from members.voice import VoiceTeam
from team.board import ClientWordsWatcher, ProjectBoard
from team.client_page import ClientPage
from team.client_questions import ClientQuestionWatcher
from team.progress import ModelWaitWatcher
from team.room_access import RoomAccess
from team.state import TeamCalls

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
)
# The gateway enqueues the caller's microphone audio ten times a second, and the producer logs every chunk,
# which buries the lines worth reading (transfers, tool calls, the team's posts, the agents' replies).
logging.getLogger("ak.integration.producer").setLevel(logging.WARNING)
# LiteLLM logs every model call of the Coder and the Reviewer, twice; their tool calls are logged as activity instead.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class SoftwareTeamServer:
    """Loads the team and serves it in a LiveKit room.

    ``TEAM_TEXT_MODEL`` overrides the background agents' model (any LiteLLM ``provider/model``);
    ``TEAM_OUTPUT_DIR`` is where each project's folder is written (default ``./output``).
    """

    ROOM = "room_01"

    def __init__(self, output_root: Path, text_model: Optional[str] = None):
        self.text_model = text_model
        self.calls = TeamCalls(output_root)

    @classmethod
    def from_env(cls) -> "SoftwareTeamServer":
        return cls(Path(os.getenv("TEAM_OUTPUT_DIR", "./output")), os.getenv("TEAM_TEXT_MODEL") or None)

    def load(self) -> None:
        """Register the voice team and the background agents with Agent Kernel."""
        ModelWaitWatcher.install(self.calls)
        VoiceTeam(self.calls).load()
        BackgroundTeam(self.calls, self.text_model).load()

    def run(self) -> None:
        self.load()
        access = RoomAccess.from_config(self.ROOM)
        RESTAPI.add(ClientPage(access, self.calls).router())
        board = ProjectBoard(self.calls.get(self.ROOM), access)
        board.start()
        ClientWordsWatcher.install(board)
        ClientQuestionWatcher.install(self.calls.get(self.ROOM))
        gateway = LiveKitEdgeGateway(session_id=self.ROOM, agent_name="pm")
        IOHandler.run(gateways=[GatewayRunner(gateway)])


if __name__ == "__main__":
    SoftwareTeamServer.from_env().run()
