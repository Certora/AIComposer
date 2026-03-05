"""
Handler protocols for graph execution events.

``IOHandler[H, P]`` is the core protocol — it receives structural
events (start/end, state updates, checkpoints) from the background
drainer and handles HITL interrupts.  Type parameters let each
workflow define its own human-interaction and progress schemas.

Domain-specific sub-protocols (``CodeGenIOHandler``,
``NatSpecIOHandler``) extend ``IOHandler`` with workflow-specific
output methods called after the graph completes.

See ``DESIGN.md`` in this directory for the full event flow.
"""

from typing import Any, Protocol, Callable

from graphcore.tools.vfs import VFSAccessor

from composer.diagnostics.stream import ProgressUpdate
from composer.human.types import HumanInteractionType
from composer.core.state import ResultStateSchema, AIComposerState
from composer.spec.ptypes import NatSpecState, HumanQuestionSchema


class IOHandler[H, P](Protocol):
    """Protocol for consuming graph execution events.

    One ``IOHandler`` is active per ``with_handler()`` scope.  The
    background drainer calls its methods as events arrive from the
    ``EventQueue``.

    Type parameters:

    - ``H`` — the human-interaction schema (e.g.
      ``HumanQuestionSchema``).  Determines what the graph can ask
      the user.
    - ``P`` — the progress-update type.  Domain code calls
      ``progress_update()`` directly (not via the event queue).

    The ``path`` parameter on most methods is a list of thread IDs
    from outermost to innermost, reconstructed from ``Nested``
    event wrappers.  A top-level execution has ``len(path) == 1``.
    """

    async def log_thread_id(self, tid: str, chosen: bool): ...

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str): ...

    async def log_state_update(self, path: list[str], st: dict):
        """A graph node emitted new state (messages, tool results, etc.)."""
        ...

    async def progress_update(self, path: list[str], upd: P):
        """Domain-specific progress notification.  Called directly, not via the event queue."""
        ...

    async def log_start(self, *, path: list[str], description: str, tool_id: str | None):
        """Graph execution began.  ``tool_id`` is set when the graph runs inside a tool call."""
        ...

    async def log_end(self, path: list[str]):
        """Graph execution ended (success or failure)."""
        ...

    async def human_interaction(
        self,
        ty: H,
        debug_thunk: Callable[[], None]
    ) -> str:
        """Handle a HITL interrupt.  Block until the user responds.

        Called synchronously from ``run_graph()`` (not from the
        drainer) — the graph pauses until this returns.
        """
        ...


class CodeGenIOHandler(IOHandler[HumanInteractionType, ProgressUpdate], Protocol):
    """Extended handler for the code-generation workflow."""
    async def output(
        self,
        res: ResultStateSchema,
        mat: VFSAccessor[AIComposerState],
        st: AIComposerState
    ): ...


class NatSpecIOHandler(IOHandler[HumanQuestionSchema, Any], Protocol):
    """Extended handler for the single-agent NatSpec workflow."""
    async def display_result(self, final_state: NatSpecState) -> None: ...
