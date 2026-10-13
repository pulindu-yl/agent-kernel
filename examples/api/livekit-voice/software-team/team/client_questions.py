"""Knowing when the team is waiting for the client's answer, from what the agents say."""

import re
from typing import TYPE_CHECKING, Optional

from team.log_watch import LogLineWatcher

if TYPE_CHECKING:
    from team.state import TeamCall


class ClientQuestionWatcher(LogLineWatcher):
    """Notices an agent's turn that ends by asking the client something, and marks the team as waiting for the answer.

    The team must know when the client owes an answer: no "nobody is speaking" nudge then, and the
    board shows the question under "Waiting for you". The agents used to call an ``ask_client``
    tool around each question, but a tool answering in the middle of a spoken turn makes Gemini
    Live start talking again: live, the PM asked the same question three times. So an agent just
    asks, and this reads the question from its words.

    The LiveKit gateway logs each agent turn once its audio has played ("AI response: ..."; an
    interrupted turn is not logged). Reading that line is the example's shortcut, as
    ``ClientWordsWatcher`` reads the client's. A question put to a teammate is not one for the client:
    it addresses them by name ("Worker, how would you ...?", "... what do you think, Architect?"),
    and the turn ends with a handover. A question that merely mentions a role ("should the Architect
    plan for phones too?") is still the client's.
    """

    LOGGERS = ("ak.integration.livekit",)
    PREFIX = "AI response: "
    SENTENCE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
    TEAMMATE = re.compile(r"^(worker|architect|pm)\b|\b(worker|architect|pm)\s*\?$", re.IGNORECASE)

    def __init__(self, call: "TeamCall"):
        super().__init__()
        self._call = call
        self._handovers_seen = 0

    def line(self, message: str) -> None:
        if message.startswith(self.PREFIX):
            self.agent_said(message.removeprefix(self.PREFIX))

    @classmethod
    def closing_question(cls, text: str) -> Optional[str]:
        """The turn's last sentence, if it is a question; ``None`` otherwise."""
        sentences = [s.strip() for s in cls.SENTENCE.findall(text) if s.strip()]
        return sentences[-1] if sentences and sentences[-1].endswith("?") else None

    def agent_said(self, text: str) -> None:
        """One agent turn, as it was heard: if it ended by asking the client something, the team now waits for the answer."""
        handovers = self._call.handovers
        handed_over, self._handovers_seen = handovers > self._handovers_seen, handovers
        question = self.closing_question(text)
        if handed_over or question is None or self.TEAMMATE.search(question):
            return
        self._call.ask_client()
        self._call.store.add_progress(f"Asked the client: {question}", self._call.holder)
