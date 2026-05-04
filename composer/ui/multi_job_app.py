"""
Generic multi-job TUI base class.

``MultiJobApp[P, T]`` manages multiple concurrent tasks with a summary
panel, per-task detail drill-down, HITL routing, and token tracking.

Subclasses provide domain behavior by overriding ``create_task_handler``
and optionally ``create_event_handler``.

Task handlers interact with the app through the ``TaskHost`` protocol,
providing a clean boundary between domain-specific handler code and
generic app infrastructure.
"""

import asyncio
import enum
import traceback
from collections.abc import Callable, Coroutine, Awaitable
from contextlib import asynccontextmanager, AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol, AsyncIterator
import asyncio

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static, Input, Collapsible, ContentSwitcher
from textual.binding import Binding

from rich.syntax import Syntax
from rich.spinner import Spinner
from rich.text import Text

from textual.timer import Timer

from langchain_core.messages import AIMessage

from composer.ui.message_renderer import MessageRenderer, MountFn, TokenStats, dot, KNOWN_NODES
from composer.ui.tool_call_renderer import ToolCallRenderer
from composer.ui.tool_display import ToolDisplayConfig
from composer.io.event_handler import EventHandler, NullEventHandler
from composer.io.protocol import IOHandler
from composer.ui.ide_bridge import IDEBridge
from composer.ui.ide_content import IDEContentMixin
from composer.ui.log_screen import LogViewerMixin
from composer.io.context import with_handler
from composer.io.conversation import ConversationClient, AIYapping, ToolComplete, ThinkingStart, ToolBatch, ProgressPayload
from composer.io.stream import AsyncDataQueue


# ---------------------------------------------------------------------------
# Task status
# ---------------------------------------------------------------------------

class TaskStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_HITL = "waiting_hitl"
    DONE = "done"
    ERROR = "error"


_STATUS_INDICATORS: dict[TaskStatus, tuple[str, str]] = {
    TaskStatus.PENDING:      ("\u25cc", "dim"),         # ◌
    TaskStatus.RUNNING:      ("\u25cf", "green"),       # ●
    TaskStatus.WAITING_HITL: ("??", "yellow"),
    TaskStatus.DONE:         ("\u2713", "green"),       # ✓
    TaskStatus.ERROR:        ("\u2717", "red"),          # ✗
}

_ACTIVE_STATUSES = {TaskStatus.RUNNING, TaskStatus.WAITING_HITL}
_TERMINAL_STATUSES = {TaskStatus.DONE, TaskStatus.ERROR}

_STATUS_SORT_KEY: dict[TaskStatus, int] = {
    TaskStatus.RUNNING: 0,
    TaskStatus.WAITING_HITL: 0,
    TaskStatus.PENDING: 1,
    TaskStatus.DONE: 2,
    TaskStatus.ERROR: 2,
}


def _render_row(label: str, status: TaskStatus) -> Text:
    indicator, style = _STATUS_INDICATORS[status]
    row = Text()
    row.append(f"{indicator} ", style=style)
    row.append(label)
    row.append(f"  ({status.value})", style="dim")
    return row


# ---------------------------------------------------------------------------
# Conversation rendering (refinement loop)
# ---------------------------------------------------------------------------


class _ThinkingSpinner(Static):
    """Animated dots spinner shown while the agent is thinking."""

    def __init__(self, message: str = "thinking\u2026"):
        super().__init__("")
        self._spinner = Spinner("dots", message)
        self._timer: Timer | None = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / 12, self._tick)

    def _tick(self) -> None:
        self.update(self._spinner)

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None


@dataclass
class _EndConversation:
    """Internal sentinel pushed to the progress queue to stop the reader."""
    pass


@dataclass
class _Checkpoint:
    """Internal sentinel that signals an event when the reader reaches it.

    Used to wait for the reader to drain all prior progress events
    before mounting widgets inline (e.g. the human-input row).
    """
    done: asyncio.Event

class _ConversationSession(ToolCallRenderer):
    """Runs one refinement conversation inside a task panel.

    Lifecycle:

    - ``__aenter__`` mounts the opening banner and starts the progress
      reader background task.
    - ``progress_update(payload)`` is called (sync) by ``refinement_loop``
      for each ``ProgressPayload``.  The reader drains and renders.
    - ``human_turn(ai_response)`` renders the final AI turn text (if any),
      resets tool grouping (dialogue boundary), mounts an ``Input``, and
      returns the user's reply.
    - ``__aexit__`` pushes a sentinel, waits for the reader to drain, then
      mounts a closing banner.
    """

    def __init__(
        self,
        task_id: str,
        panel: VerticalScroll,
        host: "TaskHost",
        tool_config: ToolDisplayConfig,
        mount_to: MountFn,
        set_status: Callable[[TaskStatus], None],
        opening: str,
    ):
        super().__init__(tool_config)
        self._task_id = task_id
        self._panel = panel
        self._host = host
        self._mount = mount_to
        self._set_status = set_status
        self._opening = opening

        self._queue: AsyncDataQueue[ProgressPayload | _EndConversation | _Checkpoint] = AsyncDataQueue(
            _ready=asyncio.Event(), _event_stream=[]
        )
        self._render_task: asyncio.Task | None = None

        self._spinner: _ThinkingSpinner | None = None

    # ── Progress handling ───────────────────────────────────────

    async def _progress_reader(self) -> None:
        async for ev in self._queue.stream_events():
            if isinstance(ev, _EndConversation):
                return
            if isinstance(ev, _Checkpoint):
                ev.done.set()
                continue
            await self._handle_event(ev)

    async def _drain(self) -> None:
        """Wait for the reader to process every event queued so far."""
        done = asyncio.Event()
        self._queue.push(_Checkpoint(done))
        await done.wait()
    
    def render_ai_yapping(self, text: str) -> Static:
        return Static(dot("blue", Text.assemble(("AI: ", "bold blue"), text)))


    async def _handle_event(self, ev: ProgressPayload) -> None:
        match ev:
            case ThinkingStart():
                await self._mount_spinner()
            case AIYapping(yap_content=text):
                await self._remove_spinner()
                self.reset_tool_collapsing()
                await self._mount(self._panel, self.render_ai_yapping(text))
            case ToolBatch(calls=calls):
                await self._remove_spinner()
                for call in calls:
                    w = self.render_tool_call(
                        name=call["name"],
                        input_args=call.get("args") or {},
                        tool_call_id=call.get("id"),
                    )
                    if w is not None:
                        await self._mount(self._panel, w)
            case ToolComplete(tid=_tid):
                # Results are suppressed in refinement mode.  Anchor is
                # available via renderer.get_tool_call_anchor(tid) if a
                # future change wants to flip its styling.
                pass

    async def _mount_spinner(self) -> None:
        if self._spinner is not None:
            return
        self._spinner = _ThinkingSpinner()
        await self._mount(self._panel, self._spinner)

    async def _remove_spinner(self) -> None:
        if self._spinner is None:
            return
        self._spinner.stop()
        await self._spinner.remove()
        self._spinner = None

    # ── ConversationClient protocol ─────────────────────────────

    def progress_update(self, progress: ProgressPayload) -> None:
        self._queue.push(progress)

    async def human_turn(self, ai_response: str | None) -> str:
        # Wait for the reader to render every progress event that was
        # pushed before this call so that the input widget is mounted
        # strictly below the agent's output.
        await self._drain()

        await self._remove_spinner()

        self.reset_tool_collapsing()

        if ai_response:
            await self._mount(
                self._panel,
                Static(dot("blue", Text.assemble(("AI: ", "bold blue"), ai_response))),
            )

        self._set_status(TaskStatus.WAITING_HITL)
        input_widget = Input(placeholder="Type here...", validate_on=["submitted"])
        hint_widget = Static(
            "Type your response and press Enter", classes="interaction-hint"
        )
        await self._mount(self._panel, input_widget)
        await self._mount(self._panel, hint_widget)
        input_widget.focus()

        async with self._host.hitl_input(self._task_id, input_widget) as queue:
            response = await queue.get()

        await input_widget.remove()
        await hint_widget.remove()
        await self._mount(
            self._panel,
            Static(dot("green", Text.assemble(("You: ", "bold green"), response))),
        )

        self._set_status(TaskStatus.RUNNING)
        return response

    # ── Context manager ─────────────────────────────────────────

    async def __aenter__(self) -> "_ConversationSession":
        await self._mount(
            self._panel,
            Static(dot("cyan", Text(self._opening, style="bold cyan"))),
        )
        self._render_task = asyncio.create_task(self._progress_reader())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._render_task is not None:
            if exc is None:
                # Drain any pending events, then stop cleanly.
                self._queue.push(_EndConversation())
                await self._render_task
            else:
                # Abort on exception — don't try to render further events.
                self._render_task.cancel()
                try:
                    await self._render_task
                except asyncio.CancelledError:
                    pass

        await self._remove_spinner()
        await self._mount(
            self._panel,
            Static(dot("dim", Text("— end of refinement —", style="dim"))),
        )


# ---------------------------------------------------------------------------
# TaskHost protocol
# ---------------------------------------------------------------------------

class TaskHost(Protocol):
    """Narrow interface that task handlers use to interact with the app.

    ``MultiJobApp`` implements this protocol.  Handlers hold a
    ``TaskHost`` reference rather than a concrete app reference,
    keeping the dependency boundary clean.
    """

    def on_task_status_change(self, task_id: str, label: str, status: TaskStatus) -> None: ...
    def update_tokens(self, msg: AIMessage) -> None: ...
    def make_content_link(self, label: str, content: str, filename: str) -> Static: ...
    def hitl_input(self, task_id: str, input: Input) -> AbstractAsyncContextManager[asyncio.Queue[str]]: ...


# ---------------------------------------------------------------------------
# Handler factory types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskInfo[P]:
    task_id: str
    label: str
    phase: P


@dataclass(frozen=True)
class TaskHandle[H]:
    """Bundles an IOHandler with lifecycle callbacks."""
    handler: IOHandler[H]
    event_handler: EventHandler
    on_error: Callable[[Exception, str], Awaitable[None]]
    on_start: Callable[[], None] = lambda: None
    on_done: Callable[[], None] = lambda: None


type HandlerFactory[P, H] = Callable[[TaskInfo[P]], Awaitable[TaskHandle[H]]]


# ---------------------------------------------------------------------------
# run_task helper
# ---------------------------------------------------------------------------

async def run_task[P, T, H](
    factory: HandlerFactory[P, H],
    info: TaskInfo[P],
    fn: Callable[[], Awaitable[T]],
    semaphore: asyncio.Semaphore | None = None,
) -> T:
    """Create a handler via *factory* and run *fn* in its ``with_handler`` scope.

    Manages lifecycle callbacks (on_start/on_done/on_error).  If
    *semaphore* is provided, the task waits for acquisition before
    transitioning to RUNNING.
    """
    handle = await factory(info)
    try:
        if semaphore is not None:
            async with semaphore:
                handle.on_start()
                async with with_handler(handle.handler, handle.event_handler):
                    result = await fn()
        else:
            handle.on_start()
            async with with_handler(handle.handler, handle.event_handler):
                result = await fn()
    except Exception as exc:
        await handle.on_error(exc, traceback.format_exc())
        raise
    else:
        handle.on_done()
        return result


# ---------------------------------------------------------------------------
# MultiJobTaskHandler — per-task IOHandler
# ---------------------------------------------------------------------------

class MultiJobTaskHandler[H]:
    """Per-task ``IOHandler[H]`` that renders LLM messages, handles
    HITL input, and manages task status.

    ``H`` is the human-interaction schema type.

    Interacts with the app via the ``TaskHost`` protocol — never holds
    a direct reference to ``MultiJobApp``.

    Domain-specific behavior is provided via two hooks:

    - ``on_node_state`` — process non-message state
    - ``format_hitl_prompt`` — format domain-specific HITL interactions
    """

    def __init__(
        self,
        task_id: str,
        label: str,
        panel: VerticalScroll,
        host: TaskHost,
        tool_config: ToolDisplayConfig,
    ):
        self._task_id = task_id
        self._label = label
        self._panel = panel
        self._host = host
        self._renderer = MessageRenderer(
            tool_config,
            mount_to=self._mount_to,
            on_tokens=lambda msg: self._host.update_tokens(msg),
        )
        self._status = TaskStatus.PENDING

    # ── Status management ─────────────────────────────────────

    def _set_status(self, status: TaskStatus) -> None:
        self._status = status
        self._host.on_task_status_change(self._task_id, self._label, status)

    def mark_running(self) -> None:
        self._set_status(TaskStatus.RUNNING)

    def mark_done(self) -> None:
        self._set_status(TaskStatus.DONE)

    async def mark_error(self, exc: Exception, tb: str) -> None:
        self._set_status(TaskStatus.ERROR)
        error_text = Text()
        error_text.append(f"\n{type(exc).__name__}: {exc}\n\n", style="bold red")
        error_text.append(tb, style="red dim")
        await self._mount_to(self._panel, Static(error_text))

    @asynccontextmanager
    async def start_conversation(
        self, opening: str
    ) -> AsyncIterator[ConversationClient]:
        session = _ConversationSession(
            task_id=self._task_id,
            panel=self._panel,
            host=self._host,
            tool_config=self._renderer.tool_config,
            mount_to=self._mount_to,
            set_status=self._set_status,
            opening=opening,
        )
        async with session:
            yield session

    # ── Mounting helpers ──────────────────────────────────────

    async def _mount_to(self, target: VerticalScroll, *widgets: Widget) -> None:
        await target.mount_all(widgets)
        if target.max_scroll_y - target.scroll_y <= 3:
            target.scroll_end(animate=False)

    # ── Content links ───────────────────────────────────────

    async def render_content_link(self, label: str, content: str, filename: str) -> None:
        """Mount a clickable content link in the task panel."""
        widget = self._host.make_content_link(label, content, filename)
        await self._mount_to(self._panel, widget)

    # ── Subclass hooks ────────────────────────────────────────

    async def on_node_state(self, path: list[str], node_name: str, values: dict) -> None:
        """Called for each node's state values during ``log_state_update``.

        Override for domain-specific state processing (e.g. detecting
        working copy spec updates).
        """
        pass

    def format_hitl_prompt(self, ty: H) -> list[Text | str]:
        """Format a HITL interaction into prompt content.

        Must be overridden — there is no sensible generic default.
        """
        raise NotImplementedError

    # ── IOHandler protocol ────────────────────────────────────

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str) -> None:
        pass

    async def log_start(self, *, path: list[str], description: str, tool_id: str | None) -> None:
        await self._renderer.render_start(self._panel, path=path, description=description)

    async def log_end(self, path: list[str]) -> None:
        await self._renderer.render_end(self._panel, path=path)

    async def log_state_update(self, path: list[str], st: dict) -> None:
        target = self._renderer.get_mount_target(self._panel, path)

        for node_name, v in st.items():
            if node_name not in KNOWN_NODES:
                continue
            if "messages" in v:
                await self._renderer.render_messages(target, v["messages"])
            await self.on_node_state(path, node_name, v)

    async def human_interaction(
        self,
        ty: H,
        debug_thunk: Callable[[], None],
    ) -> str:
        self._set_status(TaskStatus.WAITING_HITL)

        prompt_parts = self.format_hitl_prompt(ty)

        prompt_widget = Static(Text.assemble(*prompt_parts))
        hint_widget = Static("Type your response and press Enter", classes="interaction-hint")
        input_widget = Input(placeholder="Type here...", validate_on=["submitted"])

        await self._mount_to(self._panel, prompt_widget, input_widget, hint_widget)
        input_widget.focus()

        async with self._host.hitl_input(self._task_id, input_widget) as queue:
            response = await queue.get()

        await prompt_widget.remove()
        await input_widget.remove()
        await hint_widget.remove()
        await self._mount_to(
            self._panel,
            Static(dot("green", Text.assemble(("You: ", "bold green"), response))),
        )

        self._set_status(TaskStatus.RUNNING)
        return response


# ---------------------------------------------------------------------------
# MultiJobApp — generic multi-job Textual app
# ---------------------------------------------------------------------------

class MultiJobApp[P, T: MultiJobTaskHandler](LogViewerMixin, IDEContentMixin, App):
    """Generic multi-job TUI with summary panel, task drill-down, and HITL routing.

    Implements ``TaskHost`` so that task handlers interact through a
    narrow protocol rather than holding a concrete app reference.

    Type parameters:

    - ``P`` — the phase type (e.g. an ``Enum``)
    - ``T`` — the task handler type (``MultiJobTaskHandler`` subclass)
    """

    CSS = """
    #header { dock: top; height: 1; background: $surface; padding: 0 1; }
    #token-bar { dock: top; height: 1; background: $surface; padding: 0 1; }
    #summary { height: 1fr; padding: 0 1; }
    #summary > * { margin-bottom: 0; }
    .task-row { padding: 0 1; }
    .task-row:hover { background: $surface; }
    .task-panel { height: 1fr; padding: 0 1; }
    .task-panel > * { margin-bottom: 1; }
    .content-pane { height: 1fr; padding: 0 1; }
    .nested-workflow { margin-left: 2; border-left: solid $secondary; padding-left: 1; }
    .interaction-hint { color: $text-muted; padding: 0 1; }
    Collapsible { background: transparent; border: none; padding: 0; }
    CollapsibleTitle { padding: 0 1; }
    Collapsible Contents { padding: 0 0 0 3; }
    """

    BINDINGS = [
        Binding("escape", "go_back", "Back to summary", show=True),
        Binding("q", "quit_app", "Quit", show=False),
    ]

    def __init__(
        self,
        *,
        phase_labels: dict[P, str],
        section_order: list[str],
        header_text: str,
        ide: IDEBridge | None = None,
    ):
        super().__init__()
        self._init_log_viewer()
        self._init_ide_content(ide)
        self._phase_labels = phase_labels
        self._section_order = section_order
        self._header_text = header_text

        self._active_inputs: dict[Input, asyncio.Queue[str]] = {}
        self._hitl_inputs: dict[str, Input] = {}
        self._work_fn: Callable[[], Coroutine[None, None, None]] | None = None
        self._pipeline_done = False
        self._previous_view: str | None = None
        self._content_pane_ids: set[str] = set()

        self._phase_sections: dict[str, Collapsible] = {}
        self._task_labels: dict[str, str] = {}
        self._task_sections: dict[str, str] = {}
        self._task_statuses: dict[str, TaskStatus] = {}

    def compose(self) -> ComposeResult:
        yield Static(self._header_text, id="header")
        yield Static("", id="token-bar")
        with ContentSwitcher(id="switcher", initial="summary"):
            yield VerticalScroll(id="summary")

    def set_work(self, fn: Callable[[], Coroutine[None, None, None]]) -> None:
        self._work_fn = fn

    def on_mount(self) -> None:
        self._tokens = TokenStats(self.query_one("#token-bar", Static))
        if self._work_fn is not None:
            self.run_worker(self._work_fn(), thread=False)

    # ── Subclass hooks ───────────────────────────────────────

    def create_task_handler(self, panel: VerticalScroll, info: TaskInfo[P]) -> T:
        """Create the per-task handler. Subclass determines tool config from ``info.phase``."""
        raise NotImplementedError

    def create_event_handler(self, handler: T, info: TaskInfo[P]) -> EventHandler:
        """Create the per-task event handler. Override for domain-specific event routing."""
        return NullEventHandler()

    # ── TaskHost implementation ──────────────────────────────

    def on_task_status_change(self, task_id: str, label: str, status: TaskStatus) -> None:
        self._task_statuses[task_id] = status
        try:
            row = self.query_one(f"#row-{task_id}", Static)
        except Exception:
            return
        row.update(_render_row(label, status))
        self._reorder_summary()

    def update_tokens(self, msg: AIMessage) -> None:
        self._tokens.update(msg)

    def make_content_link(self, label: str, content: str, filename: str) -> Static:
        snap_id = self._store_snapshot(label, content, filename)
        return self._make_content_link_widget(snap_id, label, filename)

    @asynccontextmanager
    async def hitl_input(self, task_id: str, input: Input):
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._active_inputs[input] = queue
        self._hitl_inputs[task_id] = input
        try:
            yield queue
        finally:
            self._active_inputs.pop(input, None)
            self._hitl_inputs.pop(task_id, None)

    # ── Key bindings ──────────────────────────────────────────

    def action_go_back(self) -> None:
        switcher = self.query_one("#switcher", ContentSwitcher)
        current = switcher.current

        if current is not None and current in self._content_pane_ids:
            pane = switcher.query_one(f"#{current}")
            self._content_pane_ids.discard(current)
            switcher.current = self._previous_view or "summary"
            self._previous_view = None
            pane.remove()
            return

        switcher.current = "summary"

    def _show_content_fallback(
        self, snap_id: int, label: str, content: str, filename: str,
    ) -> None:
        self.run_worker(self._mount_content_pane(label, content, filename), thread=False)

    async def _mount_content_pane(self, label: str, content: str, filename: str) -> None:
        switcher = self.query_one("#switcher", ContentSwitcher)

        pane_id = f"snap-{self._next_snap_id}"
        self._content_pane_ids.add(pane_id)

        lang = self._guess_lang(filename) or "text"
        syntax = Syntax(content, lang, theme="monokai", line_numbers=True)

        pane = VerticalScroll(id=pane_id, classes="content-pane")
        pane.display = False
        await switcher.mount(pane)
        await pane.mount(
            Static(Text.assemble(
                (f"{label} ", "bold"),
                (filename, "cyan"),
                ("  (ESC to go back)", "dim"),
            )),
            Static(syntax),
        )

        self._previous_view = switcher.current
        switcher.current = pane_id

    def action_quit_app(self) -> None:
        if self._pipeline_done:
            self.exit()

    # ── HandlerFactory implementation ─────────────────────────

    async def make_handler(self, info: TaskInfo[P]) -> TaskHandle[Any]:
        """Create per-task panel, handler, summary row, and return a ``TaskHandle``."""
        task_id = info.task_id
        label = info.label
        phase = info.phase

        section_label = self._phase_labels[phase]
        section = await self._ensure_phase_section(section_label)

        row = Static(
            _render_row(label, TaskStatus.PENDING),
            id=f"row-{task_id}",
            classes="task-row",
        )
        await section.query_one("Contents").mount(row)

        panel = VerticalScroll(id=task_id, classes="task-panel")
        panel.display = False
        switcher = self.query_one("#switcher", ContentSwitcher)
        await switcher.mount(panel)

        handler = self.create_task_handler(panel, info)
        event_handler = self.create_event_handler(handler, info)

        self._task_labels[task_id] = label
        self._task_sections[task_id] = section_label
        self._task_statuses[task_id] = TaskStatus.PENDING

        return TaskHandle(
            handler=handler,
            event_handler=event_handler,
            on_start=handler.mark_running,
            on_done=handler.mark_done,
            on_error=handler.mark_error,
        )

    # ── HITL routing ──────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return

        queue = self._active_inputs.get(event.input)
        if queue is None:
            return

        if event.validation_result and not event.validation_result.is_valid:
            for desc in event.validation_result.failure_descriptions:
                self.notify(desc, severity="error")
            return

        event.input.disabled = True
        queue.put_nowait(value)

    # ── Navigation ────────────────────────────────────────────

    def _drill_to(self, task_id: str) -> None:
        switcher = self.query_one("#switcher", ContentSwitcher)
        switcher.current = task_id

        inp = self._hitl_inputs.get(task_id)
        if inp is not None:
            inp.focus()

    # ── Summary management ────────────────────────────────────

    async def _ensure_phase_section(self, section_label: str) -> Collapsible:
        if section_label in self._phase_sections:
            return self._phase_sections[section_label]

        section = Collapsible(title=section_label, collapsed=False)
        self._phase_sections[section_label] = section

        summary = self.query_one("#summary", VerticalScroll)

        existing_indices = {
            lbl: self._section_order.index(lbl)
            for lbl in self._phase_sections
            if lbl in self._section_order and lbl != section_label
        }
        new_idx = self._section_order.index(section_label) if section_label in self._section_order else len(self._section_order)

        insert_before = None
        for lbl, idx in sorted(existing_indices.items(), key=lambda x: x[1]):
            if idx > new_idx:
                insert_before = self._phase_sections[lbl]
                break

        if insert_before is not None:
            await summary.mount(section, before=insert_before)
        else:
            await summary.mount(section)
        return section

    def _reorder_summary(self) -> None:
        if len(self._phase_sections) <= 1:
            return

        summary = self.query_one("#summary", VerticalScroll)

        def section_has_active(label: str) -> bool:
            return any(
                self._task_statuses.get(tid) in _ACTIVE_STATUSES
                for tid, slabel in self._task_sections.items()
                if slabel == label
            )

        ordered = sorted(
            self._phase_sections.keys(),
            key=lambda lbl: (
                0 if section_has_active(lbl) else 1,
                self._section_order.index(lbl) if lbl in self._section_order else len(self._section_order),
            ),
        )

        children = list(summary.children)
        first_section = children[0] if children else None
        for i, label in enumerate(ordered):
            section = self._phase_sections[label]
            if i == 0:
                if first_section is not None and section is not first_section:
                    summary.move_child(section, before=first_section)
            else:
                prev = self._phase_sections[ordered[i - 1]]
                summary.move_child(section, after=prev)

        for label, section in self._phase_sections.items():
            task_ids = [
                tid for tid, slabel in self._task_sections.items()
                if slabel == label
            ]
            if len(task_ids) <= 1:
                continue

            task_ids.sort(key=lambda tid: _STATUS_SORT_KEY.get(self._task_statuses.get(tid, TaskStatus.PENDING), 1))

            contents = section.query_one("Contents")
            for i, tid in enumerate(task_ids):
                row = self.query_one(f"#row-{tid}", Static)
                if i == 0:
                    continue
                prev_row = self.query_one(f"#row-{task_ids[i - 1]}", Static)
                contents.move_child(row, after=prev_row)

            all_terminal = all(
                self._task_statuses.get(tid) in _TERMINAL_STATUSES
                for tid in task_ids
            )
            if all_terminal:
                section.collapsed = True

    # ── Click handling ───────────────────────────────────────

    def on_click(self, event: Any) -> None:
        widget = event.widget if hasattr(event, "widget") else None
        while widget is not None:
            if isinstance(widget, Static) and widget.has_class("task-row"):
                if widget.id and widget.id.startswith("row-"):
                    task_id = widget.id.removeprefix("row-")
                    self._drill_to(task_id)
                return
            widget = widget.parent
