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

import enum
from typing import Protocol, Callable

from graphcore.tools.vfs import VFSAccessor

from composer.diagnostics.stream import ProgressUpdate
from composer.human.types import HumanInteractionType
from composer.core.state import ResultStateSchema, AIComposerState


class IOHandler[H](Protocol):
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

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str): ...

    async def log_state_update(self, path: list[str], st: dict):
        """A graph node emitted new state (messages, tool results, etc.)."""
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


class WorkflowPurpose(enum.Enum):
    """Identifies which sub-workflow a thread belongs to."""
    CODEGEN = "codegen"
    NATREQ = "natreq"


class CodeGenIOHandler(IOHandler[HumanInteractionType], Protocol):
    """Extended handler for the code-generation workflow."""

    async def log_workflow_thread(self, purpose: WorkflowPurpose, thread_id: str) -> None:
        """Record a thread ID for a specific sub-workflow purpose."""
        ...

    async def show_error(self, error: Exception) -> None:
        """Display a fatal workflow error (crash, recursion limit, etc.)."""
        ...


    async def progress_update(self, path: list[str], upd: ProgressUpdate) -> None:
        """Domain-specific progress notification.  Called directly, not via the event queue."""
        ...


    async def output(
        self,
        res: ResultStateSchema,
        mat: VFSAccessor[AIComposerState],
        st: AIComposerState
    ): ...
