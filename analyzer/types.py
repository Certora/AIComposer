from typing import Protocol

class AnalysisArgs(Protocol):
    @property
    def folder(self) -> str:
        ...

    @property
    def rule(self) -> str:
        ...

    @property
    def method(self) -> str | None:
        ...

    @property
    def quiet(self) -> bool:
        ...

    @property
    def recursion_limit(self) -> int:
        ...
    
    @property
    def thread_id(self) -> str | None:
        ...

    @property
    def checkpoint_id(self) -> str | None:
        ...

    @property
    def thinking_tokens(self) -> int:
        ...
    
    @property
    def tokens(self) -> int:
        ...