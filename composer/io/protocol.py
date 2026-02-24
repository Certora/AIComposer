from typing import Protocol, Callable

from graphcore.tools.vfs import VFSAccessor

from composer.diagnostics.stream import ProgressUpdate
from composer.human.types import HumanInteractionType
from composer.core.state import ResultStateSchema, AIComposerState

class IOHandler(Protocol):
    async def log_thread_id(self, tid: str, chosen: bool): ...
    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str): ...

    async def log_state_update(self, path: list[str], st: dict): ...

    async def progress_update(self, path: list[str], upd: ProgressUpdate): ...

    async def log_start(self, *, path: list[str], tool_id: str | None): ...

    async def log_end(self, path: list[str]): ...

    async def human_interaction(
        self,
        ty: HumanInteractionType,
        debug_thunk: Callable[[], None]
    ) -> str: ...

    async def output(
        self,
        res: ResultStateSchema,
        mat: VFSAccessor[AIComposerState],
        st: AIComposerState
    ): ...
