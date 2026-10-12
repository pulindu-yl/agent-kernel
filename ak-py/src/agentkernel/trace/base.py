from abc import ABC, abstractmethod

from ..core import Runner


class BaseTrace(ABC):
    @abstractmethod
    def init(self):
        """
        Initialize trace instrumentation
        """
        raise NotImplementedError

    @abstractmethod
    def openai(self) -> Runner:
        """
        Initialize OpenAI Agents SDK instrumentation
        """
        raise NotImplementedError

    @abstractmethod
    def langgraph(self) -> Runner:
        """
        Initialize LangGraph instrumentation
        """
        raise NotImplementedError

    @abstractmethod
    def crewai(self) -> Runner:
        """
        Initialize CrewAI instrumentation
        """
        raise NotImplementedError

    @abstractmethod
    def adk(self) -> Runner:
        """
        Initialize Google ADK instrumentation
        """
        raise NotImplementedError

    @abstractmethod
    def smolagents(self) -> Runner:
        """
        Initialize Smolagents instrumentation
        """
        raise NotImplementedError

    @abstractmethod
    def pydanticai(self) -> Runner:
        """
        Initialize Pydantic AI instrumentation
        """
        raise NotImplementedError

    def claude(self) -> Runner:
        """
        Initialize Claude Agent SDK instrumentation.

        Not abstract, so a bring-your-own tracer written before this adapter existed still loads; such a tracer
        raises here only when a Claude module is built with tracing enabled.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support Claude Agent SDK tracing")
