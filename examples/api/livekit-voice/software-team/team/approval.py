"""The client's approval of what the team will build, given with a button on their page."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from team.state import TeamCall


class ClientApproval:
    """Is this what we build? The client answers on their page, with the requirements and the design in front of them.

    Once the Architect has written the design, the page shows the requirements and the design with
    two buttons. Checking documents by ear is hard, and a click is a clear answer: "Looks good" records
    the approval at once, as the PM's ``confirm_requirements`` does after a spoken yes, and tells the
    team; "I'd like a change" tells the team to ask what. Either way the voice agents hear it as a team
    update, so the meeting carries on from it.
    """

    def __init__(self, call: "TeamCall"):
        self._call = call

    def pending(self) -> bool:
        """Whether the requirements and the design are ready and waiting for the client's approval."""
        state = self._call.store.state
        return bool(state.requirements) and state.design_written and not state.requirements_confirmed

    def approve(self, by: str) -> None:
        """Record the client's approval and tell the team.

        :raises ValueError: Nothing is waiting for approval.
        """
        self._check_pending()
        self._call.store.confirm_requirements(by, how="on the client's page")
        self._call.client_answered()
        self._call.notify(
            f"{by} approved the requirements and the design on their page, so the requirements are confirmed: do not "
            "call confirm_requirements. Thank them in one sentence, and hand the call to the Worker to start the build."
        )

    def ask_for_change(self, by: str) -> None:
        """Tell the team the client wants something changed first.

        :raises ValueError: Nothing is waiting for approval.
        """
        self._check_pending()
        self._call.client_answered()
        self._call.notify(
            f'{by} pressed "I\'d like a change" on their page: they want to change something before approving. The PM '
            "asks them, in one easy sentence, what they would like to change, and records it."
        )

    def _check_pending(self) -> None:
        if not self.pending():
            state = self._call.store.state
            if state.requirements_confirmed:
                raise ValueError("The requirements are already approved")
            raise ValueError(
                "There is nothing to approve yet: the team is still working out the requirements and the design"
            )
