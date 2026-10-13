"""Tokens for joining the call's LiveKit room, made from the settings the gateway already uses.

Two participants join besides the agents: the project board (team/board.py), which only sends
data, and the client's page (panel/), which talks and listens.
"""

import re

from agentkernel.core.config import AKConfig
from livekit import api


class RoomAccess:
    """Makes LiveKit tokens for one room, from ``AK_LIVEKIT__URL``, ``AK_LIVEKIT__API_KEY`` and ``AK_LIVEKIT__API_SECRET``."""

    BOARD_IDENTITY = "project-board"
    CLIENT_IDENTITY = "client"
    # The gateway names its agent participants agent-<name>; a client must never look like one.
    AGENT_PREFIX = "agent-"
    MAX_NAME = 40

    def __init__(self, room: str, url: str, api_key: str, api_secret: str):
        self.room = room
        self.url = url
        self._api_key = api_key
        self._api_secret = api_secret

    @classmethod
    def from_config(cls, room: str) -> "RoomAccess":
        settings = AKConfig.get().livekit
        return cls(room, settings.url, settings.api_key, settings.api_secret)

    @property
    def configured(self) -> bool:
        """Whether tokens can be made: the URL, key and secret are all set."""
        return bool(self.url and self._api_key and self._api_secret)

    def board_token(self) -> str:
        """For the project board: it sends data only, with no microphone, and subscribes to nothing."""
        grants = api.VideoGrants(
            room_join=True, room=self.room, can_publish=False, can_subscribe=False, can_publish_data=True
        )
        return self._token(self.BOARD_IDENTITY, "Project board", grants)

    def client_token(self, name: str = "") -> str:
        """For the client's page: it publishes the microphone, hears the agents, and can type.

        Joins under the client's name (``client_identity``), which the gateway greets them by. Typed
        messages go to the room chat, which the gateway passes to the agent with the call as the
        client's words: for links, names and anything easier written than said.
        """
        grants = api.VideoGrants(
            room_join=True, room=self.room, can_publish=True, can_subscribe=True, can_publish_data=True
        )
        display = self.display_name(name)
        return self._token(self.client_identity(display), display, grants)

    @classmethod
    def display_name(cls, name: str) -> str:
        return " ".join(name.split())[: cls.MAX_NAME] or "Client"

    @classmethod
    def client_identity(cls, display_name: str) -> str:
        """A room identity from the client's name, never one an agent or the board uses.

        It keeps the name's capitals: the gateway greets the caller by their identity ("A user named
        Pulindu just connected"), so the team calls them by it.
        """
        identity = re.sub(r"[^A-Za-z0-9]+", "-", display_name).strip("-") or cls.CLIENT_IDENTITY
        if identity.lower().startswith(cls.AGENT_PREFIX) or identity.lower() == cls.BOARD_IDENTITY:
            identity = f"{cls.CLIENT_IDENTITY}-{identity}"
        return identity

    def _token(self, identity: str, name: str, grants: api.VideoGrants) -> str:
        return (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(identity)
            .with_name(name)
            .with_grants(grants)
            .to_jwt()
        )
