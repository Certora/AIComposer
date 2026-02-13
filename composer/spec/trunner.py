from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import traceback as tb_mod
import uuid
from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar

import composer.certora as _

from typing import Iterable, Literal, TypeVar, Any

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Horizontal, Vertical
from textual.widgets import Button, Static, Header, Footer, Collapsible, RichLog
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.markup import escape as rich_escape

from typing import cast

from langchain_core.runnables import RunnableConfig

from langgraph._internal._typing import StateLike
from langgraph.graph.state import CompiledStateGraph

from composer.spec.events import (
    BufferEvent, CheckpointEvt, NodeUpdateEvt, ProverOutputEvt, CloudPollingEvt,
    NestedStart, NestedEnd, NestedEvt, SilentStart, SilentEnd, ErrorEvt,
    parse_custom_event, dom_id, TOOL_CALL, PROVER_OUTPUT, SILENT_STATUS,
)

T = TypeVar('T')

_LEVEL_STYLES: dict[int, str] = {
    logging.DEBUG: "dim",
    logging.INFO: "",
    logging.WARNING: "bold yellow",
    logging.ERROR: "bold red",
    logging.CRITICAL: "bold red",
}

_LOG_NAMESPACE = "composer.spec"


class _TUILogHandler(logging.Handler):
    """Synchronous handler that buffers records and signals an asyncio event.

    Installed on the ``composer.spec`` logger while the TUI is alive.
    A background task in :class:`GraphRunnerApp` waits on *notify*, drains
    the buffer, and writes to the :class:`LogPanel`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.notify: asyncio.Event = asyncio.Event()

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.notify.set()

    def drain(self) -> list[logging.LogRecord]:
        """Return all buffered records and clear the buffer."""
        out = self.records
        self.records = []
        return out


# Context variables — routing execution to the correct buffer/depth
_nesting_depth: ContextVar[int] = ContextVar('_nesting_depth', default=0)


# ---------------------------------------------------------------------------
# Buffer infrastructure for parallel job execution
# ---------------------------------------------------------------------------

@dataclass
class JobEventBuffer:
    """Append-only event log for a single workflow job.

    Producers (``_run_depth0`` and its nested variants) append typed
    ``BufferEvent`` instances and set ``notify``.  Consumers
    (``GraphRunnerApp``, ``JobDisplay``) read from a cursor position,
    clearing ``notify`` before draining and waiting again when caught up.
    """
    name: str
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[BufferEvent] = field(default_factory=list)
    notify: asyncio.Event = field(default_factory=asyncio.Event)
    resume: asyncio.Event = field(default_factory=lambda: _make_set_event())
    status: Literal["waiting", "running", "done", "error"] = "running"

    def append(self, event: BufferEvent) -> None:
        """Append an event and wake any waiting consumer."""
        self.events.append(event)
        self.notify.set()


def _make_set_event() -> asyncio.Event:
    e = asyncio.Event()
    e.set()
    return e


_current_event_buffer: ContextVar["JobEventBuffer | None"] = ContextVar(
    '_current_event_buffer', default=None
)

_buffer_collection: ContextVar["list[JobEventBuffer] | None"] = ContextVar(
    '_buffer_collection', default=None
)


@contextmanager
def buffer_collection():
    """Set up a shared buffer collection visible to ``fresh_buffer`` calls.

    Yields the live list so callers can pass it to ``JobManagerApp``.
    """
    buffers: list[JobEventBuffer] = []
    token = _buffer_collection.set(buffers)
    try:
        yield buffers
    finally:
        _buffer_collection.reset(token)


@contextmanager
def fresh_buffer(
    name: str,
    *,
    status: Literal["waiting", "running"] = "running",
):
    """Create a ``JobEventBuffer``, register it in the collection, and activate it.

    Must be called inside a ``buffer_collection`` context.
    """
    buffers = _buffer_collection.get()
    if buffers is None:
        raise RuntimeError("fresh_buffer used outside buffer_collection")
    buf = JobEventBuffer(name=name, status=status)
    buffers.append(buf)
    with event_buffer_context(buf):
        yield buf


@contextmanager
def event_buffer_context(buffer: JobEventBuffer):
    """Public context manager: run code with *buffer* as the active event sink.

    Sets ``_current_event_buffer`` and ``_nesting_depth`` for the calling
    task.  On normal exit, marks the buffer ``"done"``; on exception,
    marks it ``"error"`` and re-raises.  Always fires ``notify`` so
    consumers see the terminal status.
    """
    buf_token = _current_event_buffer.set(buffer)
    depth_token = _nesting_depth.set(0)
    try:
        yield
    except Exception as exc:
        buffer.status = "error"
        buffer.append(ErrorEvt(message=str(exc), traceback=tb_mod.format_exc()))
        raise
    else:
        if buffer.status in ("waiting", "running"):
            buffer.status = "done"
    finally:
        buffer.notify.set()
        _current_event_buffer.reset(buf_token)
        _nesting_depth.reset(depth_token)


MAX_TOOL_CONTENT_LENGTH = 500
PREVIEW_LENGTH = 80


def _collapsible_message(content: str, title: str, border_style: str, collapsed: bool = True) -> Collapsible:
    """Wrap a message in a Collapsible with a preview in the title."""
    preview = content[:PREVIEW_LENGTH].replace('\n', ' ').strip()
    if len(content) > PREVIEW_LENGTH:
        preview += "..."
    return Collapsible(
        Static(Panel(_format_text_content(content), title=title, border_style=border_style)),
        title=f"{title}: {rich_escape(preview)}",
        collapsed=collapsed,
    )


def _normalize_content(s: str | list[str | dict]) -> list[dict]:
    """Normalize message content to a list of typed blocks."""
    l: list[str | dict]
    if isinstance(s, str):
        l = [s]
    else:
        l = s
    to_ret = []
    for r in l:
        if isinstance(r, str):
            to_ret.append({"type": "text", "text": r})
        else:
            to_ret.append(r)
    return to_ret


class _SafeMarkdown:
    """Markdown renderable that falls back to plain Text on render errors."""

    def __init__(self, content: str) -> None:
        self._md = Markdown(content)
        self._fallback = Text(content)

    def __rich_console__(self, console, options):
        try:
            yield from self._md.__rich_console__(console, options)
        except Exception:
            yield from self._fallback.__rich_console__(console, options)


def _format_text_content(content: str) -> Text | _SafeMarkdown:
    """Format string content with markdown if appropriate."""
    import re
    html_tag_pattern = r'<[^>]+>'
    if re.search(html_tag_pattern, content):
        return Text(content)

    markdown_markers = ['#', '*', '`', '```', '- ', '* ', '1. ', '## ', '### ']
    if any(marker in content for marker in markdown_markers):
        return _SafeMarkdown(content)
    else:
        return Text(content)


class MessageWidget(Static):
    """Widget for displaying a single message."""

    def __init__(self, message: BaseMessage, **kwargs) -> None:
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        match self.message:
            case AIMessage():
                yield from self._render_ai_message(self.message)
            case ToolMessage():
                yield from self._render_tool_message(self.message)
            case HumanMessage():
                yield from self._render_human_message(self.message)
            case SystemMessage():
                yield from self._render_system_message(self.message)
            case _:
                content = _normalize_content(self.message.content)
                text = "\n".join(c.get("text", str(c)) for c in content)
                yield Static(Panel(text, title=type(self.message).__name__))

    def _render_ai_message(self, msg: AIMessage) -> Iterable[Collapsible | ToolCallWidget]:
        content_parts: list[str] = []

        for block in _normalize_content(msg.content):
            block_type = block.get("type", "unknown")

            if block_type == "thinking":
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    yield Collapsible(
                        Static(_format_text_content(thinking_text)),
                        title="Thinking",
                        collapsed=True
                    )
            elif block_type == "text":
                text = block.get("text", "")
                if text:
                    content_parts.append(text)
            elif block_type == "tool_use":
                tool_name = block.get("name", "unknown")
                tool_call_id = block.get("id", "")
                tool_input = block.get("input", {})
                args_str = json.dumps(tool_input, indent=2) if tool_input else ""
                yield ToolCallWidget(tool_name, args_str, tool_call_id)

        if content_parts:
            yield _collapsible_message("\n".join(content_parts), "Assistant", "magenta", collapsed=False)

    def _render_tool_message(self, msg: ToolMessage) -> Iterable[Collapsible]:
        content = str(msg.content)
        tool_name = getattr(msg, 'name', 'Tool Result')
        yield _collapsible_message(content, tool_name, "cyan")

    def _render_human_message(self, msg: HumanMessage) -> Iterable[Collapsible]:
        content_parts: list[str] = []

        for block in _normalize_content(msg.content):
            block_type = block.get("type", "unknown")

            if block_type == "text":
                text = block.get("text", "")
                if text:
                    content_parts.append(text)
            elif block_type == "image" or block_type == "image_url":
                content_parts.append("[Image content]")
            elif block_type == "document":
                content_parts.append("[Document content]")
            else:
                content_parts.append(f"[{block_type}]")

        if content_parts:
            combined = "\n".join(content_parts)
            yield _collapsible_message(combined, "Human", "green")

    def _render_system_message(self, msg: SystemMessage) -> Iterable[Collapsible]:
        content_parts: list[str] = []

        for block in _normalize_content(msg.content):
            block_type = block.get("type", "unknown")

            if block_type == "text":
                text = block.get("text", "")
                if text:
                    content_parts.append(text)
            else:
                content_parts.append(f"[{block_type}]")

        if content_parts:
            combined = "\n".join(content_parts)
            yield _collapsible_message(combined, "System", "blue")


class UpdateWidget(Static):
    """Widget for displaying a state update."""

    def __init__(self, node_name: str, state_update: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.node_name = node_name
        self.state_update = state_update

    def compose(self) -> ComposeResult:
        yield Static(f"[bold]Node: {self.node_name}[/bold]", classes="update-header")

        if "messages" in self.state_update:
            for msg in self.state_update["messages"]:
                if isinstance(msg, BaseMessage):
                    yield MessageWidget(msg)

        other_keys = {k: v for k, v in self.state_update.items() if k != "messages"}
        if other_keys:
            summary = ", ".join(f"{k}={type(v).__name__}" for k, v in other_keys.items())
            yield Static(f"[dim]State updates: {summary}[/dim]")


class NestedWorkflowPanel(Vertical):
    """Collapsible panel for nested workflow execution - reuses UpdateWidget for content."""

    DEFAULT_CSS = """
    NestedWorkflowPanel {
        height: auto;
        margin: 1 0;
        border: solid $secondary;
    }

    NestedWorkflowPanel .nested-status {
        height: 1;
        padding: 0 1;
        background: $surface;
    }

    NestedWorkflowPanel #nested-messages {
        height: auto;
        max-height: 20;
        padding: 0 1;
    }
    """

    def __init__(self, workflow_name: str, thread_id: str, **kwargs):
        super().__init__(**kwargs)
        self.workflow_name = workflow_name
        self.workflow_thread_id = thread_id

    def compose(self) -> ComposeResult:
        yield Collapsible(
            Static("Checkpoint: -", id="nested-checkpoint", classes="nested-status"),
            Static("Node: -", id="nested-node", classes="nested-status"),
            ScrollableContainer(id="nested-messages"),
            title=f"⚙ {self.workflow_name} [{self.workflow_thread_id[:8]}...]",
            collapsed=False,
        )

    def update_checkpoint(self, checkpoint_id: str) -> None:
        self.query_one("#nested-checkpoint", Static).update(f"Checkpoint: {checkpoint_id[:12]}...")

    def update_node(self, node_name: str) -> None:
        self.query_one("#nested-node", Static).update(f"Node: {node_name}")

    async def add_update(self, widget: Static) -> None:
        """Mount an UpdateWidget (or any widget) into the nested message area."""
        area = self.query_one("#nested-messages", ScrollableContainer)
        await area.mount(widget)
        area.scroll_end(animate=False)

    def mark_complete(self) -> None:
        collapsible = self.query_one(Collapsible)
        collapsible.collapsed = True


class LogPanel(Vertical):
    """Always-visible collapsible log panel.

    Starts collapsed; automatically expands when an ERROR-level record arrives.
    """

    DEFAULT_CSS = """
    LogPanel {
        width: 1fr;
        height: 1fr;
        border: solid $secondary;
    }

    LogPanel RichLog {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Collapsible(
            RichLog(id="log-output", wrap=True, markup=True),
            title="Logs",
            collapsed=True,
        )

    def write_record(self, record: logging.LogRecord) -> None:
        """Render a log record.  Expands the panel on first error."""
        style = _LEVEL_STYLES.get(record.levelno, "")
        prefix = record.levelname.ljust(7)
        message = record.getMessage()

        log_widget = self.query_one("#log-output", RichLog)
        if style:
            log_widget.write(Text.from_markup(f"[{style}]{prefix} {rich_escape(message)}[/{style}]"))
        else:
            log_widget.write(f"{prefix} {message}")

        if record.levelno >= logging.ERROR:
            self.query_one(Collapsible).collapsed = False


class ToolCallWidget(Vertical):
    """Container for a tool call display, identified by tool_call_id for later lookup."""

    DEFAULT_CSS = """
    ToolCallWidget {
        height: auto;
        border: solid yellow;
        padding: 0 1;
    }
    """

    def __init__(self, tool_name: str, args_str: str, tool_call_id: str):
        super().__init__(id=dom_id(TOOL_CALL, tool_call_id))
        self._tool_name = tool_name
        self._args_str = args_str

    def compose(self) -> ComposeResult:
        yield Static(f"[bold yellow]Tool Call[/bold yellow]: [bold]{rich_escape(self._tool_name)}[/bold]")
        if self._args_str:
            yield Collapsible(
                Static(Text(self._args_str)),
                title="Input",
                collapsed=True,
            )


class ProverOutputPanel(Vertical):
    """Collapsible panel for real-time prover stdout and cloud polling status."""

    DEFAULT_CSS = """
    ProverOutputPanel { height: auto; }
    ProverOutputPanel RichLog { height: auto; max-height: 20; }
    """

    def compose(self) -> ComposeResult:
        yield Collapsible(
            RichLog(id="prover-log", wrap=True),
            title="Prover Output",
            collapsed=False,
        )

    def append_line(self, line: str) -> None:
        self.query_one("#prover-log", RichLog).write(line)


# ---------------------------------------------------------------------------
# Shared event renderer — single rendering path for both TUI consumers
# ---------------------------------------------------------------------------

class EventRenderer:
    """Stateful renderer: consumes ``BufferEvent`` instances and mounts widgets.

    Used by both ``GraphRunnerApp`` (single-workflow TUI) and ``JobDisplay``
    (job manager detail view).  Callers pass a Textual *app* (for DOM queries)
    and a *container* (``ScrollableContainer``) into which widgets are mounted.

    Set ``live = True`` to enable auto-scroll (live follow mode);
    ``live = False`` suppresses scrolling (batch replay mode).

    Populate ``skip_nested`` with thread IDs of already-completed nested
    workflows before replaying history — their panels will be pre-collapsed.
    """

    def __init__(self, app: App, container: ScrollableContainer, *, live: bool = True) -> None:
        self._app = app
        self._container = container
        self.live = live
        self.skip_nested: set[str] = set()
        self._nested_panels: dict[str, NestedWorkflowPanel] = {}
        self._prover_panels: dict[str, ProverOutputPanel] = {}
        self._pending_prover_lines: dict[str, list[str]] = {}

    def _scroll(self) -> None:
        if self.live:
            self._container.scroll_end(animate=False)

    async def render(self, event: BufferEvent) -> None:
        """Render a single typed event.  Checkpoint events are no-ops (callers
        that need status-bar updates should inspect the event separately)."""
        match event:
            case CheckpointEvt():
                pass
            case NodeUpdateEvt(node_name=name, state_update=update):
                widget = UpdateWidget(name, update)
                await self._container.mount(widget)
                await self._flush_pending_prover()
                self._scroll()
            case ProverOutputEvt(tool_call_id=tc_id, line=line):
                await self._route_prover(tc_id, line)
            case CloudPollingEvt(tool_call_id=tc_id, message=msg):
                await self._route_prover(tc_id, msg)
            case NestedStart(thread_id=tid, workflow_name=wf_name):
                panel = NestedWorkflowPanel(wf_name, tid, id=dom_id("nested", tid))
                if tid in self.skip_nested:
                    panel.mark_complete()
                self._nested_panels[tid] = panel
                await self._container.mount(panel)
                self._scroll()
            case NestedEvt(thread_id=tid, inner=inner):
                panel = self._nested_panels.get(tid)
                if panel is not None:
                    await self._render_nested(inner, panel)
            case NestedEnd(thread_id=tid):
                panel = self._nested_panels.pop(tid, None)
                if panel is not None:
                    panel.mark_complete()
            case SilentStart(thread_id=tid, workflow_name=wf_name):
                status = Static(
                    f"[dim]Running: {wf_name}...[/dim]",
                    id=dom_id(SILENT_STATUS, tid),
                    classes="update-header",
                )
                await self._container.mount(status)
            case SilentEnd(thread_id=tid):
                results = self._app.query(f"#{dom_id(SILENT_STATUS, tid)}")
                if results:
                    w = results[0]
                    assert isinstance(w, Static)
                    w.update("[dim]Completed[/dim]")
            case ErrorEvt(message=msg, traceback=traceback):
                await self._container.mount(
                    Static(f"[bold red]Error:[/bold red] {rich_escape(msg)}")
                )
                if traceback:
                    await self._container.mount(Collapsible(
                        Static(Text(traceback)),
                        title="Traceback",
                        collapsed=False,
                    ))
                self._scroll()

    async def _render_nested(self, event: BufferEvent, panel: NestedWorkflowPanel) -> None:
        match event:
            case CheckpointEvt(checkpoint_id=cp_id):
                panel.update_checkpoint(cp_id)
            case NodeUpdateEvt(node_name=name, state_update=update):
                panel.update_node(name)
                widget = UpdateWidget(name, update)
                await panel.add_update(widget)
                await self._flush_pending_prover()
            case ProverOutputEvt(tool_call_id=tc_id, line=line):
                await self._route_prover(tc_id, line)
            case CloudPollingEvt(tool_call_id=tc_id, message=msg):
                await self._route_prover(tc_id, msg)

    async def _route_prover(self, tool_call_id: str, line: str) -> None:
        if tool_call_id in self._prover_panels:
            self._prover_panels[tool_call_id].append_line(line)
            self._scroll()
            return
        self._pending_prover_lines.setdefault(tool_call_id, []).append(line)
        await self._try_attach_prover(tool_call_id)

    async def _try_attach_prover(self, tool_call_id: str) -> None:
        results = self._app.query(f"#{dom_id(TOOL_CALL, tool_call_id)}")
        if not results:
            return
        panel = ProverOutputPanel(id=dom_id(PROVER_OUTPUT, tool_call_id))
        self._prover_panels[tool_call_id] = panel
        await results[0].mount(panel)
        for buffered_line in self._pending_prover_lines.pop(tool_call_id, []):
            panel.append_line(buffered_line)

    async def _flush_pending_prover(self) -> None:
        for tc_id in list(self._pending_prover_lines.keys()):
            await self._try_attach_prover(tc_id)


class GraphRunnerApp(App):
    """TUI for a single workflow: consumes typed events from a ``JobEventBuffer``.

    The graph producer runs as a separate asyncio task (spawned by
    ``_run_top_level_buffered``).  This app renders events, manages
    pause/resume via ``buffer.resume``, and auto-exits on completion.
    """

    CSS = """
    #status-bar {
        dock: top;
        height: 3;
        background: $surface;
        padding: 0 1;
    }

    #status-bar Static {
        width: auto;
    }

    #main-content {
        height: 1fr;
    }

    #message-area {
        width: 2fr;
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #controls {
        dock: bottom;
        height: 3;
        align: center middle;
    }

    .update-header {
        background: $boost;
        padding: 0 1;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("p", "toggle_pause", "Pause/Resume"),
        ("q", "quit", "Quit"),
        ("ctrl+c", "interrupt", "Abort (2x)"),
    ]

    is_paused: reactive[bool] = reactive(False)
    is_complete: reactive[bool] = reactive(False)

    def __init__(self, buffer: JobEventBuffer, display_thread_id: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._buffer = buffer
        self._display_thread_id = display_thread_id
        self._follow_task: asyncio.Task | None = None
        self._error: Exception | None = None
        self._last_ctrl_c_time: float | None = None
        self._renderer: EventRenderer | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="status-bar"):
            yield Static(f"Thread: {self._display_thread_id}", id="thread-display")
            yield Static(" | ", classes="separator")
            yield Static("Checkpoint: -", id="checkpoint-display")
            yield Static(" | ", classes="separator")
            yield Static("Node: -", id="node-display")
        with Horizontal(id="main-content"):
            yield ScrollableContainer(id="message-area")
            yield LogPanel(id="log-panel")
        with Horizontal(id="controls"):
            yield Button("Pause", id="pause-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._log_handler = _TUILogHandler()
        logger = logging.getLogger(_LOG_NAMESPACE)
        logger.addHandler(self._log_handler)
        self._renderer = EventRenderer(
            self, self.query_one("#message-area", ScrollableContainer)
        )
        self._log_task = asyncio.create_task(self._drain_logs())
        self._follow_task = asyncio.create_task(self._follow_buffer())

    async def _drain_logs(self) -> None:
        """Background task: wait for log records, render them in the LogPanel."""
        panel = self.query_one("#log-panel", LogPanel)
        handler = self._log_handler
        while True:
            await handler.notify.wait()
            handler.notify.clear()
            for record in handler.drain():
                panel.write_record(record)

    # -- Buffer consumer -------------------------------------------------------

    async def _follow_buffer(self) -> None:
        """Main consumer loop: drain events from buffer, render via EventRenderer."""
        buf = self._buffer
        cursor = 0
        assert self._renderer is not None

        try:
            while True:
                buf.notify.clear()
                events = buf.events[cursor:]
                cursor = len(buf.events)

                for event in events:
                    # Status bar updates (specific to GraphRunnerApp)
                    if isinstance(event, CheckpointEvt):
                        self.query_one("#checkpoint-display", Static).update(
                            f"Checkpoint: {event.checkpoint_id[:12]}..."
                        )
                    elif isinstance(event, NodeUpdateEvt):
                        self.query_one("#node-display", Static).update(
                            f"Node: {event.node_name}"
                        )
                    await self._renderer.render(event)

                if buf.status != "running":
                    self.is_complete = True
                    self.query_one("#pause-btn", Button).label = "Complete (15s)"
                    self.query_one("#pause-btn", Button).disabled = True
                    self.set_timer(15, self._auto_exit)
                    return

                await buf.notify.wait()

        except Exception as e:
            self._error = e
            self.exit()

    # -- Controls --------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pause-btn":
            self.action_toggle_pause()

    def action_toggle_pause(self) -> None:
        if self.is_complete:
            return

        self.is_paused = not self.is_paused
        btn = self.query_one("#pause-btn", Button)

        if self.is_paused:
            btn.label = "Resume"
            btn.variant = "success"
            self._buffer.resume.clear()
        else:
            btn.label = "Pause"
            btn.variant = "primary"
            self._buffer.resume.set()

    def on_unmount(self) -> None:
        logging.getLogger(_LOG_NAMESPACE).removeHandler(self._log_handler)
        self._log_task.cancel()
        if self._follow_task:
            self._follow_task.cancel()

    def _auto_exit(self) -> None:
        """Called by timer after completion to auto-exit."""
        self.exit()

    async def action_quit(self) -> None:
        if self.is_complete:
            self.exit()

    def action_interrupt(self) -> None:
        """Handle Ctrl+C - double press within 2 seconds to abort."""
        current_time = time.time()

        if self._last_ctrl_c_time is not None and (current_time - self._last_ctrl_c_time) < 2.0:
            sys.exit(1)

        self._last_ctrl_c_time = current_time
        self.notify("Press Ctrl+C again to abort", severity="warning")


async def run_to_completion[I: StateLike, S: StateLike](
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    *,
    workflow_name: str = "Nested Workflow",
    checkpoint_id: str | None = None,
    recursion_limit: int = 100,
) -> S:
    """Run a compiled state graph to completion.

    Routing:
    - If a ``JobEventBuffer`` is active (set by ``event_buffer_context`` or
      ``_run_top_level_buffered``), events are written to the buffer at the
      appropriate depth.
    - Otherwise, creates a fresh buffer, spawns the graph as a producer
      task, and runs ``GraphRunnerApp`` as the consumer TUI.
    """
    buffer = _current_event_buffer.get()
    if buffer is not None:
        depth = _nesting_depth.get()
        if depth == 0:
            return await _run_depth0(
                buffer, graph, input, thread_id, checkpoint_id, recursion_limit
            )
        elif depth == 1:
            return await _run_depth1(
                buffer, graph, input, thread_id, workflow_name, checkpoint_id, recursion_limit
            )
        else:
            return await _run_depth2(
                buffer, graph, input, thread_id, workflow_name, checkpoint_id, recursion_limit
            )

    # Top-level: create buffer, run producer + TUI consumer
    return await _run_top_level_buffered(
        graph, input, thread_id, checkpoint_id, recursion_limit
    )


async def _run_top_level_buffered[I: StateLike, S: StateLike](
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    checkpoint_id: str | None,
    recursion_limit: int,
) -> S:
    """Create a buffer, spawn graph producer, run GraphRunnerApp as consumer."""
    buf = JobEventBuffer(name="main")

    async def producer():
        with event_buffer_context(buf):
            try:
                await _run_depth0(buf, graph, input, thread_id, checkpoint_id, recursion_limit)
            except Exception as e:
                buf.append(ErrorEvt(message=str(e), traceback=tb_mod.format_exc()))
                raise

    app = GraphRunnerApp(buffer=buf, display_thread_id=thread_id)
    task = asyncio.create_task(producer())
    await app.run_async()

    if not task.done():
        task.cancel()
    elif (exc := task.exception()) is not None:
        raise exc

    return cast(S, graph.get_state({"configurable": {"thread_id": thread_id}}).values)


# ---------------------------------------------------------------------------
# Buffer-based producers: stream graph events into a JobEventBuffer
# ---------------------------------------------------------------------------

async def _run_depth0[I: StateLike, S: StateLike](
    buffer: JobEventBuffer,
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    checkpoint_id: str | None,
    recursion_limit: int,
) -> S:
    """Top-level producer: streams graph, translates LangGraph tuples to typed events."""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    if checkpoint_id is not None:
        config["configurable"]["checkpoint_id"] = checkpoint_id

    stream_input = None if checkpoint_id is not None else input

    depth_token = _nesting_depth.set(1)
    try:
        async for (tag, payload) in graph.astream(
            input=stream_input,
            config=config,
            stream_mode=["checkpoints", "updates", "custom"],
        ):
            await buffer.resume.wait()
            assert isinstance(payload, dict)

            match tag:
                case "checkpoints":
                    cp_id = payload["config"]["configurable"]["checkpoint_id"]
                    buffer.append(CheckpointEvt(checkpoint_id=cp_id))
                case "updates":
                    for node_name, update in payload.items():
                        if node_name == "__interrupt__":
                            continue
                        if isinstance(update, dict):
                            buffer.append(NodeUpdateEvt(node_name=node_name, state_update=update))
                case "custom":
                    evt = parse_custom_event(payload)
                    if evt is not None:
                        buffer.append(evt)
    finally:
        _nesting_depth.reset(depth_token)

    return cast(S, graph.get_state({"configurable": {"thread_id": thread_id}}).values)


async def _run_depth1[I: StateLike, S: StateLike](
    buffer: JobEventBuffer,
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    workflow_name: str,
    checkpoint_id: str | None,
    recursion_limit: int,
) -> S:
    """Nested producer (depth 1): wraps each inner event in NestedEvt."""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    if checkpoint_id is not None:
        config["configurable"]["checkpoint_id"] = checkpoint_id

    stream_input = None if checkpoint_id is not None else input

    buffer.append(NestedStart(thread_id=thread_id, workflow_name=workflow_name))

    depth_token = _nesting_depth.set(2)
    try:
        async for (tag, payload) in graph.astream(
            input=stream_input,
            config=config,
            stream_mode=["checkpoints", "updates", "custom"],
        ):
            await buffer.resume.wait()
            assert isinstance(payload, dict)

            match tag:
                case "checkpoints":
                    cp_id = payload["config"]["configurable"]["checkpoint_id"]
                    buffer.append(NestedEvt(
                        thread_id=thread_id,
                        inner=CheckpointEvt(checkpoint_id=cp_id),
                    ))
                case "updates":
                    for node_name, update in payload.items():
                        if node_name == "__interrupt__":
                            continue
                        if isinstance(update, dict):
                            buffer.append(NestedEvt(
                                thread_id=thread_id,
                                inner=NodeUpdateEvt(node_name=node_name, state_update=update),
                            ))
                case "custom":
                    evt = parse_custom_event(payload)
                    if evt is not None:
                        buffer.append(NestedEvt(thread_id=thread_id, inner=evt))
    finally:
        _nesting_depth.reset(depth_token)

    buffer.append(NestedEnd(thread_id=thread_id))

    return cast(S, graph.get_state({"configurable": {"thread_id": thread_id}}).values)


async def _run_depth2[I: StateLike, S: StateLike](
    buffer: JobEventBuffer,
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    workflow_name: str,
    checkpoint_id: str | None,
    recursion_limit: int,
) -> S:
    """Deeply nested producer (depth >= 2): only lifecycle events."""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    if checkpoint_id is not None:
        config["configurable"]["checkpoint_id"] = checkpoint_id

    stream_input = None if checkpoint_id is not None else input

    buffer.append(SilentStart(thread_id=thread_id, workflow_name=workflow_name))

    async for _ in graph.astream(
        input=stream_input,
        config=config,
        stream_mode=["checkpoints"],
    ):
        await buffer.resume.wait()

    buffer.append(SilentEnd(thread_id=thread_id))

    return cast(S, graph.get_state({"configurable": {"thread_id": thread_id}}).values)


def run_to_completion_sync[I: StateLike, S: StateLike](
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    *,
    workflow_name: str = "Workflow",
    checkpoint_id: str | None = None,
    recursion_limit: int = 100,
) -> S:
    """
    Synchronous wrapper for run_to_completion.

    Use this for top-level calls from non-async contexts (e.g., CLI entry points).
    """
    return asyncio.run(run_to_completion(
        graph, input, thread_id,
        workflow_name=workflow_name,
        checkpoint_id=checkpoint_id,
        recursion_limit=recursion_limit,
    ))
