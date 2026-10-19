from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Callable

from ..base import Agent, Runner, Session
from ..event import StreamEvent
from ..model import AgentReply, AgentRequest
from .pcm import EDGE_SAMPLE_RATE

if TYPE_CHECKING:  # typing only; ..tool imports ..base, so this avoids a cycle
    from ..tool import ToolContext


class RealtimeRunner(Runner):
    """Base class for framework adapters that handle realtime streaming over a persistent model socket (e.g. the OpenAI Realtime API)."""

    input_sample_rate: int = EDGE_SAMPLE_RATE
    output_sample_rate: int = EDGE_SAMPLE_RATE
    # True when the adapter reports ``speech_stopped`` after every ``speech_started`` and
    # ``interrupt`` (a model with server voice activity detection that signals the end of speech).
    # The pool then counts the user as speaking until it arrives.
    reports_speech_stopped: bool = False

    async def run(self, agent: Any, session: Session, requests: list[AgentRequest]) -> AgentReply:
        """Not supported for realtime runners."""
        raise NotImplementedError("Realtime runners do not support unary run().")

    @staticmethod
    async def _cancel_and_wait(task: asyncio.Task | None) -> None:
        """Cancel a background task of the adapter and wait for it to end; a no-op for None and for the calling task itself."""
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.wait({task})

    def team(self, agent: "Agent") -> list[str]:
        """The names of every agent a conversation with ``agent`` can hand off to, ``agent`` first; it must not raise."""
        return [agent.name]

    async def stream(self, agent: Any, session: Session, requests: list[AgentRequest]) -> AsyncGenerator[StreamEvent, None]:
        """Not supported for realtime runners."""
        raise NotImplementedError("Realtime runners do not support unary stream().")
        yield

    @abstractmethod
    async def connect(self, session: Session, agent: "Agent", callback: Callable) -> None:
        """Establishes the persistent model socket and reports model events through ``callback`` in the order the model sent them."""
        raise NotImplementedError()

    @abstractmethod
    async def append_audio(self, base64_audio: str) -> None:
        """Appends audio to the model's input buffer, at :attr:`input_sample_rate`."""
        raise NotImplementedError()

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Sends a text message and requests a response."""
        raise NotImplementedError()

    @abstractmethod
    async def send_tool_result(self, call_id: str, result: str) -> None:
        """Sends a tool execution result back to the model."""
        raise NotImplementedError()

    @abstractmethod
    async def execute_tool(self, name: str, arguments: str, context: "ToolContext", call_id: str) -> str:
        """Executes the named tool with JSON ``arguments`` in the framework's own tool context and returns its string result."""
        raise NotImplementedError()

    @abstractmethod
    async def disconnect(self) -> None:
        """Closes the active model socket; safe to call more than once and after a :meth:`connect` that failed or never ran."""
        raise NotImplementedError()
