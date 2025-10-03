from langchain_core.tools.base import BaseTool
from langgraph.graph import MessagesState
from typing import Annotated, ContextManager, NotRequired, Protocol, TypeVar, Iterator
from typing_extensions import TypedDict

def merge_vfs(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    ...

class VFSState(MessagesState):
    vfs: Annotated[dict[str, str], merge_vfs]

InputType = TypeVar('InputType', bound=VFSState)
StateVar = TypeVar('StateVar', contravariant=True)

class VFSAccessor(Protocol[StateVar]):
    def materialize(self, state: StateVar, debug: bool = ...) -> ContextManager[str]:
        ...

    def iterate(self, state: StateVar) -> Iterator[tuple[str, bytes]]:
        ...

    def get(self, state: StateVar, file: str) -> bytes | None:
        ...

class VFSToolConfig(TypedDict):
    immutable: bool
    fs_layer: NotRequired[str | None]
    forbidden_read: NotRequired[str]
    forbidden_write: NotRequired[str]
    put_doc_extra: NotRequired[str]
    get_doc_extra: NotRequired[str]

def vfs_tools(conf: VFSToolConfig, ty: type[InputType]) -> tuple[list[BaseTool], VFSAccessor[InputType]]:
    ...
