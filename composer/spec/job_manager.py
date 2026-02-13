"""Job manager TUI for parallel workflow execution.

Provides a two-view Textual app:
  - **Job list**: shows all active ``JobEventBuffer`` instances with status.
  - **Job detail**: replays/follows a single buffer using the same widget
    types as ``GraphRunnerApp`` (UpdateWidget, ToolCallWidget,
    ProverOutputPanel, NestedWorkflowPanel).

Navigation: Enter drills in, Esc goes back. ``p`` toggles per-job pause.
"""

from __future__ import annotations

import asyncio
import time
import sys

from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Static, Header, Footer, ListView, ListItem, Label
from textual.reactive import reactive
from rich.markup import escape as rich_escape

from composer.spec.trunner import (
    JobEventBuffer,
    EventRenderer,
)
from composer.spec.events import NestedStart, NestedEnd


# ---------------------------------------------------------------------------
# JobDisplay — replays / follows one JobEventBuffer
# ---------------------------------------------------------------------------

class JobDisplay(Vertical):
    """Renders events from a ``JobEventBuffer`` via ``EventRenderer``.

    On mount, replays all existing events (with scroll suppressed), then
    switches to live-follow mode.  Destroyed when the user navigates back.
    """

    DEFAULT_CSS = """
    JobDisplay {
        height: 1fr;
    }

    JobDisplay #job-message-area {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    """

    def __init__(self, buffer: JobEventBuffer, **kwargs) -> None:
        super().__init__(**kwargs)
        self._buffer = buffer
        self._cursor = 0
        self._follow_task: asyncio.Task | None = None
        self._renderer: EventRenderer | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold]{rich_escape(self._buffer.name)}[/bold]",
            id="job-title",
        )
        yield ScrollableContainer(id="job-message-area")

    def on_mount(self) -> None:
        area = self.query_one("#job-message-area", ScrollableContainer)
        self._renderer = EventRenderer(self.app, area, live=False)
        self._follow_task = asyncio.create_task(self._follow())

    def cancel(self) -> None:
        """Cancel the follow loop permanently."""
        if self._follow_task is not None:
            self._follow_task.cancel()
            self._follow_task = None

    def pause_follow(self) -> None:
        """Cancel follow task, preserving cursor and widget state."""
        if self._follow_task is not None:
            self._follow_task.cancel()
            self._follow_task = None

    def resume_follow(self) -> None:
        """Restart following from the current cursor (no replay)."""
        if self._follow_task is None:
            self._follow_task = asyncio.create_task(self._follow_from_cursor())

    async def _follow_from_cursor(self) -> None:
        """Resume live follow from the current cursor position."""
        buf = self._buffer
        assert self._renderer is not None
        renderer = self._renderer
        renderer.live = True

        while True:
            buf.notify.clear()
            while self._cursor < len(buf.events):
                await renderer.render(buf.events[self._cursor])
                self._cursor += 1
            if buf.status != "running":
                break
            await buf.notify.wait()

    # -- event consumption --------------------------------------------------

    async def _follow(self) -> None:
        """Drain-then-wait loop.  Never clears or mutates the buffer."""
        buf = self._buffer
        assert self._renderer is not None
        renderer = self._renderer

        # Pre-scan: find nested workflows already completed in the buffer
        # so we can render them collapsed during replay.
        started: set[str] = set()
        for evt in buf.events:
            if isinstance(evt, NestedStart):
                started.add(evt.thread_id)
            elif isinstance(evt, NestedEnd) and evt.thread_id in started:
                renderer.skip_nested.add(evt.thread_id)

        # Replay existing events (batch mode — scroll suppressed)
        while self._cursor < len(buf.events):
            await renderer.render(buf.events[self._cursor])
            self._cursor += 1

        # Switch to live mode
        renderer.live = True
        renderer.skip_nested.clear()
        area = self.query_one("#job-message-area", ScrollableContainer)
        area.scroll_end(animate=False)

        if buf.status != "running":
            return

        # Live follow
        while True:
            buf.notify.clear()
            while self._cursor < len(buf.events):
                await renderer.render(buf.events[self._cursor])
                self._cursor += 1
            if buf.status != "running":
                break
            await buf.notify.wait()


# ---------------------------------------------------------------------------
# JobListItem — a single row in the job list
# ---------------------------------------------------------------------------

class JobListItem(ListItem):
    """Wraps a ``JobEventBuffer`` reference for display in the list."""

    def __init__(self, buffer: JobEventBuffer, index: int) -> None:
        super().__init__()
        self.buffer = buffer
        self.index = index
        self._last_rendered: str = ""

    def compose(self) -> ComposeResult:
        yield Label(id="job-label")

    def on_mount(self) -> None:
        self.refresh_label()

    def refresh_label(self) -> None:
        status_icon = {"waiting": "◌", "running": "●", "done": "✓", "error": "✗"}.get(
            self.buffer.status, "?"
        )
        paused = " [paused]" if not self.buffer.resume.is_set() else ""
        text = f" {status_icon}  {rich_escape(self.buffer.name)}{paused}"
        if text != self._last_rendered:
            self.query_one("#job-label", Label).update(text)
            self._last_rendered = text


# ---------------------------------------------------------------------------
# JobManagerApp — top-level Textual app with two views
# ---------------------------------------------------------------------------

class JobManagerApp(App):
    """Two-view job manager: job list ↔ job detail (full-screen swap)."""

    CSS = """
    #job-list-view {
        height: 1fr;
    }

    #job-list-view ListView {
        height: 1fr;
        border: solid $primary;
    }

    #job-list-header {
        height: 3;
        background: $surface;
        padding: 0 1;
    }

    .update-header {
        background: $boost;
        padding: 0 1;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("p", "toggle_pause", "Pause/Resume"),
        ("q", "quit_app", "Quit"),
        ("ctrl+c", "interrupt", "Abort (2x)"),
    ]

    # Track which view is active
    in_detail: reactive[bool] = reactive(False)

    def __init__(self, buffers: list[JobEventBuffer], **kwargs) -> None:
        super().__init__(**kwargs)
        self._buffers = buffers
        self._rendered: set[str] = set()
        self._displays: dict[str, JobDisplay] = {}
        self._active_uid: str | None = None
        self._current_buffer: JobEventBuffer | None = None
        self._refresh_task: asyncio.Task | None = None
        self._last_ctrl_c_time: float | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="job-list-view"):
            yield Static("[bold]Jobs[/bold]", id="job-list-header")
            yield ListView(id="job-list")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def _refresh_loop(self) -> None:
        """Periodically refresh the job list to pick up new buffers and status changes."""
        while True:
            if not self.in_detail:
                await self._sync_list()
            await asyncio.sleep(0.5)

    async def _sync_list(self) -> None:
        """Append new buffers and update existing item labels in-place."""
        lv = self.query_one("#job-list", ListView)
        for i, buf in enumerate(self._buffers):
            if buf.uid not in self._rendered:
                await lv.append(JobListItem(buf, i))
                self._rendered.add(buf.uid)
        for child in lv.children:
            if isinstance(child, JobListItem):
                child.refresh_label()

    # -- navigation ---------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if self.in_detail:
            return
        item = event.item
        if not isinstance(item, JobListItem):
            return
        self._current_buffer = item.buffer
        asyncio.create_task(self._enter_detail(item.buffer))

    async def _enter_detail(self, buffer: JobEventBuffer) -> None:
        uid = buffer.uid

        if uid not in self._displays:
            display = JobDisplay(buffer, id=f"job-detail-{uid}")
            self._displays[uid] = display
            await self.mount(display, before=self.query_one(Footer))
        else:
            display = self._displays[uid]
            display.display = True
            display.resume_follow()

        self.query_one("#job-list-view", Vertical).display = False
        self._active_uid = uid
        self._current_buffer = buffer
        self.in_detail = True

    def action_go_back(self) -> None:
        if not self.in_detail:
            return
        asyncio.create_task(self._exit_detail())

    async def _exit_detail(self) -> None:
        if self._active_uid and self._active_uid in self._displays:
            display = self._displays[self._active_uid]
            display.pause_follow()
            display.display = False
        self._active_uid = None
        self._current_buffer = None

        self.query_one("#job-list-view", Vertical).display = True
        self.in_detail = False

    # -- pause / resume -----------------------------------------------------

    def action_toggle_pause(self) -> None:
        if self.in_detail and self._current_buffer is not None:
            buf = self._current_buffer
        else:
            # In list view, pause the highlighted job
            lv = self.query_one("#job-list", ListView)
            if lv.highlighted_child is None:
                return
            item = lv.highlighted_child
            if not isinstance(item, JobListItem):
                return
            buf = item.buffer

        if buf.resume.is_set():
            buf.resume.clear()
        else:
            buf.resume.set()

    # -- quit / interrupt ---------------------------------------------------

    async def action_quit_app(self) -> None:
        if self.in_detail:
            await self._exit_detail()
        else:
            self.exit()

    def action_interrupt(self) -> None:
        current_time = time.time()
        if self._last_ctrl_c_time is not None and (current_time - self._last_ctrl_c_time) < 2.0:
            sys.exit(1)
        self._last_ctrl_c_time = current_time
        self.notify("Press Ctrl+C again to abort", severity="warning")

    def on_unmount(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
        for display in self._displays.values():
            display.cancel()
