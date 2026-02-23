from typing import Protocol, Callable

from graphcore.tools.vfs import VFSAccessor

from composer.diagnostics.handlers import ProgressUpdate
from composer.human.types import HumanInteractionType
from composer.core.state import ResultStateSchema, AIComposerState

class IOHandler(Protocol):
    async def log_thread_id(self, tid: str, chosen: bool): ...
    async def log_checkpoint_id(self, *, tid: str, checkpoint_id: str): ...

    async def log_state_update(self, tid: str, st: dict): ...

    async def progress_update(self, tid: str, upd: ProgressUpdate): ...

    async def log_start(self, *, parent: str, child: str, tool_id: str | None): ...

    async def log_end(self, parent: str, child: str): ...

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
