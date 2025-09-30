from typing import TypeVar, Generic

StateT = TypeVar("StateT")

class SummaryConfig(Generic[StateT]):
    def __init__(self, max_messages: int = ..., keep_last: int = ..., enabled: bool = ...):
        ...

    def get_summarization_prompt(self, state: StateT) -> str:
        ...

    def get_resume_prompt(self, state: StateT, summary: str) -> str:
        ...

