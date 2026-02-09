
import asyncio
import json
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal, TypedDict

import composer.certora as _

from typing import Iterable, TypeVar, Any

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Horizontal, Vertical
from textual.widgets import Button, Static, Header, Footer, Collapsible, RichLog
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown

from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from langgraph._internal._typing import StateLike
from langgraph.graph.state import CompiledStateGraph

T = TypeVar('T')

LogLevel = Literal["info", "warning", "error", "debug"]

LOG_LEVEL_STYLES: dict[LogLevel, str] = {
    "debug": "dim",
    "info": "",
    "warning": "bold yellow",
    "error": "bold red",
}


class LogEvent(TypedDict):
    type: Literal["log"]
    message: str
    level: LogLevel


def log_message(message: str, level: LogLevel = "info") -> None:
    """Emit a log message via the LangGraph stream writer.

    Call this from within a graph node to send a log entry to the TUI log panel.
    """
    writer = get_stream_writer()
    event: LogEvent = {"type": "log", "message": message, "level": level}
    writer(event)


# Context variables for nested workflow detection
_current_runner_app: ContextVar["GraphRunnerApp | None"] = ContextVar('_current_runner_app', default=None)
_nesting_depth: ContextVar[int] = ContextVar('_nesting_depth', default=0)


@contextmanager
def _workflow_context(app: "GraphRunnerApp | None", depth: int):
    """Context manager for setting workflow nesting context."""
    app_token = _current_runner_app.set(app)
    depth_token = _nesting_depth.set(depth)
    try:
        yield
    finally:
        _current_runner_app.reset(app_token)
        _nesting_depth.reset(depth_token)


MAX_TOOL_CONTENT_LENGTH = 500


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


def _format_text_content(content: str) -> Text | Markdown:
    """Format string content with markdown if appropriate."""
    import re
    html_tag_pattern = r'<[^>]+>'
    if re.search(html_tag_pattern, content):
        return Text(content)

    markdown_markers = ['#', '*', '`', '```', '- ', '* ', '1. ', '## ', '### ']
    if any(marker in content for marker in markdown_markers):
        return Markdown(content)
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

    def _render_ai_message(self, msg: AIMessage) -> Iterable[Static | Collapsible]:
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
                tool_input = block.get("input", {})
                args_str = json.dumps(tool_input, indent=2) if tool_input else ""
                yield Static(Panel(
                    f"[bold]{tool_name}[/bold]\n{args_str}",
                    title="Tool Call",
                    border_style="yellow"
                ))

        if content_parts:
            yield Static(Panel(
                _format_text_content("\n".join(content_parts)),
                title="Assistant",
                border_style="magenta"
            ))

    def _render_tool_message(self, msg: ToolMessage) -> Iterable[Static]:
        content = str(msg.content)
        tool_name = getattr(msg, 'name', 'Tool Result')

        if len(content) > MAX_TOOL_CONTENT_LENGTH:
            truncated = content[:MAX_TOOL_CONTENT_LENGTH] + f"\n... [truncated {len(content) - MAX_TOOL_CONTENT_LENGTH} chars]"
            yield Static(Panel(truncated, title=tool_name, border_style="cyan"))
        else:
            yield Static(Panel(content, title=tool_name, border_style="cyan"))

    def _render_human_message(self, msg: HumanMessage) -> Iterable[Static]:
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
            if len(combined) > MAX_TOOL_CONTENT_LENGTH:
                truncated = combined[:MAX_TOOL_CONTENT_LENGTH] + f"\n... [truncated {len(combined) - MAX_TOOL_CONTENT_LENGTH} chars]"
                yield Static(Panel(truncated, title="Human", border_style="green"))
            else:
                yield Static(Panel(_format_text_content(combined), title="Human", border_style="green"))

    def _render_system_message(self, msg: SystemMessage) -> Iterable[Static]:
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
            if len(combined) > MAX_TOOL_CONTENT_LENGTH:
                truncated = combined[:MAX_TOOL_CONTENT_LENGTH] + f"\n... [truncated {len(combined) - MAX_TOOL_CONTENT_LENGTH} chars]"
                yield Static(Panel(truncated, title="System", border_style="blue"))
            else:
                yield Static(Panel(_format_text_content(combined), title="System", border_style="blue"))


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
    """Collapsible panel that displays log messages emitted via stream writers.

    Hidden by default; automatically appears (expanded) when an error-level
    log is received.  All log levels are buffered so earlier context is
    visible once the panel opens.
    """

    DEFAULT_CSS = """
    LogPanel {
        width: 1fr;
        height: 1fr;
        border: solid $secondary;
        display: none;
    }

    LogPanel RichLog {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._buffer: list[tuple[str, LogLevel]] = []

    def compose(self) -> ComposeResult:
        yield Collapsible(
            RichLog(id="log-output", wrap=True, markup=True),
            title="Logs",
            collapsed=False,
        )

    def write_log(self, message: str, level: LogLevel = "info") -> None:
        """Write a log entry.  Shows the panel on first error."""
        self._buffer.append((message, level))

        style = LOG_LEVEL_STYLES.get(level, "")
        prefix = level.upper().ljust(7)

        log_widget = self.query_one("#log-output", RichLog)
        if style:
            log_widget.write(Text.from_markup(f"[{style}]{prefix} {message}[/{style}]"))
        else:
            log_widget.write(f"{prefix} {message}")

        if level == "error" and self.styles.display == "none":
            self.styles.display = "block"


class GraphRunnerApp(App):
    """TUI for running a LangGraph to completion with pause/resume."""

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

    def __init__(
        self,
        graph: CompiledStateGraph,
        input: Any,
        thread_id: str,
        initial_checkpoint_id: str | None,
        recursion_limit: int,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.graph_input = input
        self.thread_id = thread_id
        self.checkpoint_id = initial_checkpoint_id
        self.recursion_limit = recursion_limit
        self._stream_task: asyncio.Task | None = None
        self._error: Exception | None = None
        self._last_ctrl_c_time: float | None = None
        self._resume_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()  # Initially not paused
        self._nested_events: dict[str, asyncio.Event] = {}
        self._nested_panels: dict[str, "NestedWorkflowPanel"] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="status-bar"):
            yield Static(f"Thread: {self.thread_id}", id="thread-display")
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
        self._start_streaming()

    def _start_streaming(self) -> None:
        self._stream_task = asyncio.create_task(self._run_stream())

    def _handle_custom_event(self, payload: dict) -> None:
        """Route a custom stream event to the appropriate handler."""
        if payload.get("type") == "log":
            log_panel = self.query_one("#log-panel", LogPanel)
            log_panel.write_log(payload.get("message", ""), payload.get("level", "info"))

    async def _run_stream(self) -> None:
        config: RunnableConfig = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.recursion_limit,
        }

        if self.checkpoint_id is not None:
            config["configurable"]["checkpoint_id"] = self.checkpoint_id

        stream_input = None if self.checkpoint_id is not None else self.graph_input

        try:
            async for (tag, payload) in self.graph.astream(
                input=stream_input,
                config=config,
                stream_mode=["checkpoints", "updates", "custom"]
            ):
                # Wait here if paused - blocks until event is set
                await self._resume_event.wait()

                if tag == "custom":
                    if isinstance(payload, dict):
                        self._handle_custom_event(payload)
                    continue

                assert isinstance(payload, dict)

                if tag == "checkpoints":
                    new_checkpoint = payload["config"]["configurable"]["checkpoint_id"]
                    self.checkpoint_id = new_checkpoint
                    self.query_one("#checkpoint-display", Static).update(f"Checkpoint: {new_checkpoint[:12]}...")
                else:
                    for node_name, update in payload.items():
                        if node_name == "__interrupt__":
                            continue
                        self.query_one("#node-display", Static).update(f"Node: {node_name}")

                        if isinstance(update, dict):
                            message_area = self.query_one("#message-area", ScrollableContainer)
                            widget = UpdateWidget(node_name, update)
                            await message_area.mount(widget)
                            message_area.scroll_end(animate=False)

            self.is_complete = True
            self.query_one("#pause-btn", Button).label = "Complete (15s)"
            self.query_one("#pause-btn", Button).disabled = True
            self.set_timer(15, self._auto_exit)

        except Exception as e:
            self._error = e
            self.exit()

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
            self._resume_event.clear()  # Block workflows
        else:
            btn.label = "Pause"
            btn.variant = "primary"
            self._resume_event.set()  # Unblock workflows

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
            # Double Ctrl+C within 2 seconds - abort immediately
            sys.exit(1)

        self._last_ctrl_c_time = current_time
        self.notify("Press Ctrl+C again to abort", severity="warning")

    @contextmanager
    def nested_workflow(self, workflow_name: str, thread_id: str):
        """Create, register, and manage a nested workflow panel."""
        panel = NestedWorkflowPanel(workflow_name, thread_id)
        event = asyncio.Event()
        event.set()  # Initially not paused
        self._nested_events[thread_id] = event
        self._nested_panels[thread_id] = panel
        try:
            yield panel, event
        finally:
            self._nested_events.pop(thread_id, None)
            self._nested_panels.pop(thread_id, None)

    def toggle_nested_pause(self, thread_id: str) -> bool:
        """Toggle pause state for a specific nested workflow. Returns new paused state."""
        if thread_id not in self._nested_events:
            return False
        event = self._nested_events[thread_id]
        if event.is_set():
            event.clear()
            return True  # Now paused
        else:
            event.set()
            return False  # Now running


async def run_to_completion[I: StateLike, S: StateLike](
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    *,
    workflow_name: str = "Nested Workflow",
    checkpoint_id: str | None = None,
    recursion_limit: int = 100,
) -> S:
    """
    Run a compiled state graph to completion using a TUI with pause/resume support.

    Supports nested execution: when called from within a parent workflow's tool,
    the nested workflow's progress is displayed in a collapsible panel within
    the parent's TUI.

    Args:
        graph: The compiled state graph to execute
        input: The input to the graph (ignored if checkpoint_id is set)
        thread_prefix: Prefix for auto-generated thread IDs.
        workflow_name: Display name for nested workflows.
        thread_id: Optional thread ID for checkpointing. Auto-generated if None.
        checkpoint_id: Optional checkpoint ID to resume from.
        recursion_limit: Maximum recursion depth (default 100).

    Returns:
        The final state after graph completion.
    """
    parent_app = _current_runner_app.get()
    depth = _nesting_depth.get()

    if parent_app is None:
        # Top-level: run with full TUI
        return await _run_top_level(graph, input, thread_id, checkpoint_id, recursion_limit)
    elif depth == 1:
        # One level of nesting: show in parent's TUI
        return await _run_nested(
            parent_app, graph, input, thread_id, workflow_name, checkpoint_id, recursion_limit
        )
    else:
        # Too deep: run silently with just a status message
        return await _run_silent(
            parent_app, graph, input, thread_id, workflow_name, checkpoint_id, recursion_limit
        )


async def _run_top_level[I: StateLike, S: StateLike](
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    checkpoint_id: str | None,
    recursion_limit: int,
) -> S:
    """Run with full TUI at the top level."""
    app = GraphRunnerApp(
        graph=graph,
        input=input,
        thread_id=thread_id,
        initial_checkpoint_id=checkpoint_id,
        recursion_limit=recursion_limit,
    )

    with _workflow_context(app, depth=1):
        await app.run_async()

        if app._error is not None:
            raise app._error

        return cast(S, graph.get_state({"configurable": {"thread_id": thread_id}}).values)


async def _run_nested[I: StateLike, S: StateLike](
    parent_app: GraphRunnerApp,
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    workflow_name: str,
    checkpoint_id: str | None,
    recursion_limit: int,
) -> S:
    """Run nested workflow, streaming into a panel in the parent's TUI."""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }

    if checkpoint_id is not None:
        config["configurable"]["checkpoint_id"] = checkpoint_id

    stream_input = None if checkpoint_id is not None else input

    with (
        _workflow_context(parent_app, depth=2),
        parent_app.nested_workflow(workflow_name, thread_id) as (panel, nested_event),
    ):
        # Mount the panel created by the context manager
        message_area = parent_app.query_one("#message-area", ScrollableContainer)
        await message_area.mount(panel)
        message_area.scroll_end(animate=False)

        async for (tag, payload) in graph.astream(
            input=stream_input,
            config=config,
            stream_mode=["checkpoints", "updates", "custom"]
        ):
            # Wait on global pause first, then individual pause
            await parent_app._resume_event.wait()
            await nested_event.wait()

            if tag == "custom":
                if isinstance(payload, dict):
                    parent_app._handle_custom_event(payload)
                continue

            assert isinstance(payload, dict)

            if tag == "checkpoints":
                new_checkpoint = payload["config"]["configurable"]["checkpoint_id"]
                panel.update_checkpoint(new_checkpoint)
            else:
                for node_name, update in payload.items():
                    if node_name == "__interrupt__":
                        continue
                    panel.update_node(node_name)

                    if isinstance(update, dict):
                        widget = UpdateWidget(node_name, update)
                        await panel.add_update(widget)

        # Mark complete and collapse
        panel.mark_complete()

    return cast(S, graph.get_state({"configurable": {"thread_id": thread_id}}).values)


async def _run_silent[I: StateLike, S: StateLike](
    parent_app: GraphRunnerApp,
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    workflow_name: str,
    checkpoint_id: str | None,
    recursion_limit: int,
) -> S:
    """Run deeply nested workflow silently, just showing a status message."""
    # Show simple status in parent
    message_area = parent_app.query_one("#message-area", ScrollableContainer)
    status = Static(f"[dim]Running child workflow: {workflow_name}...[/dim]", classes="update-header")
    await message_area.mount(status)

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }

    if checkpoint_id is not None:
        config["configurable"]["checkpoint_id"] = checkpoint_id

    stream_input = None if checkpoint_id is not None else input

    # Run without detailed UI updates (depth stays at current level)
    async for (_tag, _payload) in graph.astream(
        input=stream_input,
        config=config,
        stream_mode=["checkpoints", "custom"]
    ):
        # Wait here if paused - blocks until event is set
        await parent_app._resume_event.wait()

        if _tag == "custom" and isinstance(_payload, dict):
            parent_app._handle_custom_event(_payload)

    status.update(f"[dim]Completed: {workflow_name}[/dim]")

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
