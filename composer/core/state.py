from typing import NotRequired, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import MessagesState

from graphcore.tools.vfs import VFSState, VFSInput

class ResultStateSchema(BaseModel):
    source: list[str] = Field(description="The relative filenames in the virtual FS to present to the user. IMPORTANT: "
              "the filenames here must have been populated by prior put_file tool calls")
    comments: str = Field(description="Any comments or notes on the generated implementation, and a summary of your reasoning, along with any lessons "
              "learned from iterating with the prover.")

def merge_validation(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    to_ret = left.copy()
    to_ret.update(right)
    return to_ret

def merge_skips(left: set[int], right: set[int]) -> set[int]:
    ret = left.copy()
    ret.update(right)
    return ret

class AIComposerExtra(TypedDict):
    validation: Annotated[dict[str, str], merge_validation]
    skipped_reqs: Annotated[set[int], merge_skips]
    # Single temporary spec draft. Conceptually a scratch file the agent is
    # iterating on; it has no "name" in the VFS until committed. Commit is
    # essentially `mv <working_spec> <target_path>` and is the only point at
    # which the draft gets bound to a concrete spec file.
    working_spec: str | None

class AIComposerInput(VFSInput, AIComposerExtra):
    pass

class AIComposerState(VFSState, MessagesState, AIComposerExtra):
    generated_code: NotRequired[ResultStateSchema]
