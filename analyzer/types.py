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