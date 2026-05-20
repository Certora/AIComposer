"""Live-display handler for the auto-prove pipeline.

Drives an inline ``rich.live.Live`` region that shows the tree of
active agents (one root per phase task; sub-agents nest below).
Scrollback above the Live region accumulates phase headers, prover
lifecycle terminals, and errors. Per-line subprocess output streams
(``prover_output``, ``cloud_polling``, ``pre_audit_output``) are
dropped — there's no real way to surface a continuous stdout stream
in a Live region without fighting its redraw cycle.

Sister to:

  - ``AutoProveApp`` (Textual full-screen TUI) — for users who want
    per-task panels and scrollable log widgets.
  - ``AutoProveConsoleHandler`` (plain ``print``) — for users who
    want raw log output (CI, redirected pipes, etc.).

Pattern follows ``AutoProveConsoleHandler``: one handler instance per
pipeline run, ``make_handler`` returns a ``TaskHandle`` that wraps the
same instance for every phase so all roots feed into a single Live
region.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, cast, override

from rich.console import Console, RenderableType
from rich.text import Text

from composer.io.conversation import ConversationClient
from composer.io.event_handler import NullEventHandler
from composer.io.multi_job import TaskHandle, TaskInfo
from composer.spec.source.prover import ProverEvents
from composer.spec.source.preaudit_setup import PreAuditEvents
from composer.ui.autoprove_app import AUTOPROVE_PHASE_LABELS, AutoProvePhase
from composer.ui.conversation_client import RichConsoleConversationClient
from composer.ui.live_display import LiveDisplayHandler
from composer.ui.tool_display import ToolDisplayConfig


class AutoProveLiveHandler(LiveDisplayHandler[None], NullEventHandler):
    """``IOHandler[None]`` + ``EventHandler`` + ``HandlerFactory`` for
    the auto-prove pipeline, rendered as a single rich.live region.

    Use as::

        async with AutoProveLiveHandler() as handler:
            await executor(handler.make_handler)
    """

    def __init__(
        self,
        *,
        tool_display_config: ToolDisplayConfig | None = None,
        console: Console | None = None,
    ) -> None:
        super().__init__(
            tool_display_config=tool_display_config or ToolDisplayConfig(),
            console=console,
        )
        # Tracks which phases have already produced a header. The first
        # task within a phase to fire ``on_start`` emits the banner;
        # subsequent parallel tasks in the same phase don't repeat it.
        # Safe without a lock: ``on_start`` is sync and asyncio is
        # single-threaded, so the check-and-add doesn't race.
        self._phases_seen: set[AutoProvePhase] = set()

    # ── IOHandler hook: HITL — autoprove has none ────────────────────

    @override
    async def handle_human_interaction(
        self, ty: None, debug_thunk: Callable[[], None]
    ) -> str:
        raise RuntimeError(
            "Unexpected HITL interrupt in auto-prove live handler"
        )

    # ── EventHandler hooks ──────────────────────────────────────────

    @override
    async def handle_event(
        self, payload: dict, path: list[str], checkpoint_id: str
    ) -> None:
        evt = cast(ProverEvents, payload)
        thread_id = path[-1]
        match evt["type"]:
            case "prover_run":
                self._update_progress(thread_id, "running prover")
            case "prover_result":
                self._update_progress(thread_id, None)
                self._scrollback(
                    Text(f"[{self._label(path)}] prover complete", style="green")
                )
            case "cex_analysis":
                self._update_progress(
                    thread_id, f"analyzing CEX: {evt['rule_name']}"
                )
            case "rule_analysis":
                self._scrollback(
                    Text(
                        f"[{self._label(path)}] rule analysis → {evt['rule']}",
                        style="dim",
                    )
                )
            case "prover_output" | "cloud_polling":
                # Per-line subprocess stream — no good way to surface a
                # continuous stdout flow in a Live region. Use AutoProveApp
                # or AutoProveConsoleHandler for raw prover output.
                pass

    @override
    async def handle_progress_event(self, payload: dict) -> None:
        evt = cast(PreAuditEvents, payload)
        match evt["type"]:
            case "pre_audit_start":
                self._scrollback(Text("PreAudit: started", style="dim"))
            case "pre_audit_complete":
                self._scrollback(
                    Text(
                        f"PreAudit: complete (rc={evt['return_code']})",
                        style="dim",
                    )
                )
            case "pre_audit_output":
                # Per-line subprocess stream — same reasoning as
                # prover_output / cloud_polling above.
                pass

    # ── ConversationProvider ────────────────────────────────────────

    @asynccontextmanager
    async def _start_conversation(
        self, initial: RenderableType
    ) -> AsyncIterator[ConversationClient]:
        # Pause the Live region for the duration of the conversation so
        # ``prompt_toolkit`` owns the terminal; restore on exit. The
        # client itself is shared with ``AutoProveConsoleHandler`` —
        # same prompt-toolkit prompt, same progress-event drainer.
        #
        # The status provider closes over our live agent table so the
        # prompt's bottom toolbar shows "Background: N agents running"
        # while the user composes input. prompt-toolkit's
        # ``refresh_interval`` keeps it ticking.
        async with self._paused_live():
            client = RichConsoleConversationClient(
                initial, status_provider=self._background_status
            )
            async with client:
                yield client

    def _background_status(self) -> str | None:
        n = len(self._agents)
        if n == 0:
            return None
        return f"Background: {n} agent{'' if n == 1 else 's'} running"

    # ── HandlerFactory ──────────────────────────────────────────────

    async def make_handler(
        self, info: TaskInfo[AutoProvePhase]
    ) -> TaskHandle[None]:
        # ``run_task`` fires ``on_start`` / ``on_done`` per-task, not
        # per-phase — a single phase like CVL_GEN spawns one
        # "Invariant CVL" task plus one per-component batch, all
        # sharing the same ``AutoProvePhase``. Use ``_phases_seen`` to
        # collapse the per-task callback into a single phase header
        # emitted the first time any task of that phase actually starts
        # work. Subsequent tasks in the same phase don't repeat it.
        # No ``on_done`` — top-level task completion is already
        # captured by ``format_end``.

        def _on_start() -> None:
            if info.phase in self._phases_seen:
                return
            self._phases_seen.add(info.phase)
            label = AUTOPROVE_PHASE_LABELS.get(info.phase, info.phase.name)
            self._scrollback(
                Text(
                    "─" * 60 + f"\nPhase: {label}\n" + "─" * 60,
                    style="bold",
                )
            )

        async def _on_error(exc: Exception, tb: str) -> None:
            self._scrollback(
                Text(
                    f"[{info.label}] ERROR — {type(exc).__name__}: {exc}\n{tb}",
                    style="red",
                )
            )

        return TaskHandle(
            handler=self,
            event_handler=self,
            on_start=_on_start,
            on_error=_on_error,
            conversation_provider=self._start_conversation,
        )
