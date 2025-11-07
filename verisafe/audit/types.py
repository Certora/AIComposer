from typing import TypedDict, NotRequired, Protocol

class InputFileLike(Protocol):
    @property
    def basename(self) -> str:
        ...
    
    @property
    def bytes_contents(self) -> bytes:
        ...

    @property
    def string_contents(self) -> str:
        ...


class RuleResult(TypedDict):
    analysis: NotRequired[str]
    status: str
    rule: str

class ManualResult(TypedDict):
    content: str
    header: str
    similarity: float

class RunInput(TypedDict):
    spec: InputFileLike
    interface: InputFileLike
    system: InputFileLike
    reqs: list[str] | None
