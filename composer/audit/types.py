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

class SpecRunEntry(TypedDict):
    """A single spec file as surfaced to ``RunInput`` callers.

    ``vfs_path`` is the path at which this spec is materialized in the
    workflow's VFS (greenfield: ``certora/<basename>``; from-source:
    path relative to the workspace root).
    """
    vfs_path: str
    basename: str
    contents: str

class RunInput(TypedDict):
    specs: list[SpecRunEntry]
    interface: InputFileLike
    system: InputFileLike
    reqs: list[str] | None
