from typing import TypedDict, NotRequired

# ``InputFileLike`` and ``TextInputFile`` live in ``composer.input.types``;
# imported here for use in TypedDicts below.
from composer.input.types import InputFileLike, TextInputFile


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
    interface: TextInputFile
    system: InputFileLike
    reqs: list[str] | None
