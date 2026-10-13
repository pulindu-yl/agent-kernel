"""The records the team keeps: requirements, decisions, tasks, jobs and so on.

Plain data classes (pydantic), with no behaviour; ``ProjectStore`` (team/project.py) changes them.
"""

from typing import Optional

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    id: str
    title: str
    description: str
    acceptance_criteria: list[str]
    by: str

    def criterion_id(self, index: int) -> str:
        return f"{self.id}.{index + 1}"


class Note(BaseModel):
    """An assumption, an open question, or one of the Coder's build notes."""

    id: str
    text: str
    by: str
    answer: Optional[str] = None
    # An assumption the Coder made on its own must be settled in the call (confirmed, or replaced
    # by a requirement) before delivery; ``settled`` says how.
    needs_settling: bool = False
    settled: Optional[str] = None


class Decision(BaseModel):
    id: str
    title: str
    choice: str
    reason: str
    rejected: list[str]
    by: str
    # The objection raised in the debate and how it was answered: agreement has to be earned.
    challenge: str = ""


class Component(BaseModel):
    name: str
    responsibility: str
    by: str


class Task(BaseModel):
    id: str
    title: str
    details: str
    status: str = "todo"


class Finding(BaseModel):
    id: str
    text: str
    # "open" (waiting for the Architect's ruling), "to fix", "fixed" (by the Coder), "resolved" (the
    # Reviewer confirmed the fix) or "waived" (the Architect accepted the Worker's reason to keep it).
    status: str = "open"
    # Why the Architect waived it.
    reason: str = ""


class AcceptanceResult(BaseModel):
    criterion_id: str
    criterion: str
    test: str
    result: str


class Job(BaseModel):
    id: str
    kind: str
    detail: str
    status: str = "queued"
    summary: str = ""
    # The tests' result when the job ran them (a develop job), such as "2 failed, 5 passed in 0.9s".
    tests: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class SharedFile(BaseModel):
    """A file the client shared during the call, saved in the project's shared/ folder."""

    name: str
    size: int
    by: str
    at: float


class LogEntry(BaseModel):
    at: float
    who: str
    what: str
    # "record" for something the team recorded or did (a requirement, a decision, a job), "progress"
    # for an activity note (the Coder wrote a file, a question was put to the client). Only records
    # count as the team moving on (``ProjectStore.last_recorded_at``).
    kind: str = "record"


class ProjectState(BaseModel):
    name: str
    goal: str = ""
    requirements: list[Requirement] = Field(default_factory=list)
    # Requirement ids are numbered from this count, never from the list: a dropped id is not used again.
    requirements_issued: int = 0
    assumptions: list[Note] = Field(default_factory=list)
    questions: list[Note] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    components: list[Component] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    acceptance: list[AcceptanceResult] = Field(default_factory=list)
    acceptance_checked_at: Optional[float] = None
    shared_files: list[SharedFile] = Field(default_factory=list)
    jobs: list[Job] = Field(default_factory=list)
    progress: list[LogEntry] = Field(default_factory=list)
    log: list[LogEntry] = Field(default_factory=list)
    requirements_confirmed: bool = False
    design_written: bool = False
    # Rounds of the design debate: each time the Architect hands the Worker the design to answer.
    design_rounds: int = 0
    # Choices the Coder made that the design did not settle, with the reason: what the Worker answers
    # "why is it done this way?" from in the code review.
    build_notes: list[Note] = Field(default_factory=list)
    # Rounds of the code review conversation on the findings not ruled on yet.
    review_rounds: int = 0
    # When the Architect approved the code, after the review.
    code_approved_at: Optional[float] = None
    # When the Reviewer last submitted a review (``submit_review``): a review job checks it moved.
    review_submitted_at: Optional[float] = None
    test_output: str = ""
    test_summary: str = ""
    delivered: bool = False
    # When it was delivered: a delivery is done again only for code built since.
    delivered_at: Optional[float] = None
