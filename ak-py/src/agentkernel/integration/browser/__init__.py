"""
Agent Kernel Integration with web browsers

This package contains the Agent Kernel edge for live voice calls from a browser over a WebSocket.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("agentkernel")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

from .edge import BrowserVoiceEdge, BrowserVoiceEndpoint
