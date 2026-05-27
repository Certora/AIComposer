from typing import TypedDict, NotRequired

from composer.input.files import TextUploadable, Uploadable


# ``InputFileLike`` used to be a separate Protocol defined here; it has
# been collapsed into ``TextUploadable`` (the rich, text-guaranteed
# shape from ``composer/input/files.py``). These aliases keep existing
# import sites working through the audit cleanup; once ``audit/db.py``
# is removed, callers should switch to importing ``TextUploadable`` /
# ``Uploadable`` directly and the aliases can be deleted.
InputFileLike = TextUploadable
InputFileLikeMaybeText = Uploadable


class RuleResult(TypedDict):
    analysis: NotRequired[str]
    status: str
    rule: str


class ManualResult(TypedDict):
    content: str
    header: str
    similarity: float


class SpecRunEntry(TypedDict):
    """The spec file as surfaced to ``RunInput`` callers.

    ``vfs_path`` is the path at which the spec is materialized in the
    workflow's VFS (codegen's historical convention is ``rules.spec``)."""
    vfs_path: str
    basename: str
    contents: str


class RunInput(TypedDict):
    # Audit-restored file fields are ``Uploadable`` — they're not
    # rendered as content blocks themselves; the executor rehydrates
    # them through its ``FileUploader`` to get ``Document`` /
    # ``TextDocument`` instances when needed.
    spec: SpecRunEntry
    interface: TextUploadable
    system: Uploadable
    reqs: list[str] | None
