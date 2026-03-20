from dataclasses import dataclass
from typing import TypedDict


class PromptParams(TypedDict):
    is_resume: bool

@dataclass(frozen=True)
class WorkflowSuccess:
    @property
    def exit_code(self) -> int:
        return 0

@dataclass(frozen=True)
class WorkflowFailure:
    @property
    def exit_code(self) -> int:
        return 1

@dataclass(frozen=True)
class WorkflowCrash:
    resume_work_key: str | None
    error: Exception

    @property
    def exit_code(self) -> int:
        return 1

type WorkflowResult = WorkflowSuccess | WorkflowFailure | WorkflowCrash
