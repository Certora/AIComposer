"""
NatSpec pipeline TUI.

Provides ``PipelineApp`` — a Textual app that drives
``run_natspec_pipeline`` by implementing its ``HandlerFactory``.
Each pipeline task gets its own event stream, HITL queue, and
detail panel.  A summary view groups tasks by phase into
collapsible sections.
"""

import asyncio
import enum
import pathlib
from collections.abc import Callable, Coroutine
from typing import Any, cast

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Input, Collapsible, ContentSwitcher
from textual.binding import Binding

from rich.syntax import Syntax
from rich.text import Text

from langchain_core.messages import (
    AIMessage, ToolMessage, HumanMessage, SystemMessage,
)

from composer.io.message_renderer import MessageRenderer, TokenStats, dot, KNOWN_NODES
from composer.io.tool_display import ToolDisplayConfig, ToolDisplay, CommonTools, _suppress_ack
from composer.io.event_handler import EventHandler
from composer.io.ide_bridge import IDEBridge
from composer.io.ide_content import IDEContentMixin
from composer.spec.pipeline import Phase, TaskInfo, TaskHandle, PipelineResult
from composer.spec.pipeline_events import NatspecEvent
from composer.spec.ptypes import HumanQuestionSchema


# ---------------------------------------------------------------------------
# Task status
# ---------------------------------------------------------------------------

class TaskStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_HITL = "waiting_hitl"
    DONE = "done"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Phase labels and tool configs
# ---------------------------------------------------------------------------

PHASE_LABELS: dict[Phase, str] = {
    "component_analysis": "Component Analysis",
    "bug_analysis": "Property Extraction",
    "interface_gen": "Interface & Stub Generation",
    "stub_gen": "Interface & Stub Generation",
    "cvl_gen": "CVL Generation",
}

# Ordered list of unique section labels for consistent display order.
_SECTION_ORDER: list[str] = [
    "Component Analysis",
    "Property Extraction",
    "Interface & Stub Generation",
    "CVL Generation",
]


def tool_config_for_phase(phase: Phase) -> ToolDisplayConfig:
    """Return the appropriate ``ToolDisplayConfig`` for *phase*."""
    match phase:
        case "component_analysis":
            return ToolDisplayConfig(tool_display={
                "result": CommonTools.result,
                "memory": CommonTools.memory,
            })
        case "bug_analysis":
            return ToolDisplayConfig(tool_display={
                "result": CommonTools.result,
                "write_rough_draft": CommonTools.write_rough_draft,
                "read_rough_draft": CommonTools.read_rough_draft,
            })
        case "interface_gen" | "stub_gen":
            return ToolDisplayConfig(tool_display={
                "result": CommonTools.result,
            })
        case "cvl_gen":
            return ToolDisplayConfig(tool_display={
                "put_cvl": ToolDisplay("Writing spec", _suppress_ack("Spec write result")),
                "put_cvl_raw": ToolDisplay("Writing spec", _suppress_ack("Spec write result")),
                "get_cvl": ToolDisplay("Reading spec", None),
                "feedback_tool": ToolDisplay("Getting feedback", "Feedback"),
                "cvl_research": ToolDisplay(lambda p: f"Researching CVL: {p.get('question', '?')}", "Research result"),
                "extended_reasoning": ToolDisplay("Reasoning", None),
                "cvl_manual_search": CommonTools.cvl_manual,
                "get_cvl_manual_section": ToolDisplay(lambda p: f"Read CVL Manual: {" / ".join(p.get("headers", []))}", None),
                "cvl_keyword_search": ToolDisplay(lambda p: f"CVL Manual Search: {p.get("query")}", "CVL Matching Sections"),
                "publish_spec": ToolDisplay("Publishing to master spec", _suppress_ack("Publish result")),
                "give_up": ToolDisplay("Giving up on property", _suppress_ack("Give up result")),
                "record_skip": ToolDisplay(
                    lambda p: f"Skipping property #{p.get('property_index', '?')}",
                    _suppress_ack("Skip result", ("Recorded skip",)),
                ),
                "unskip_property": ToolDisplay(
                    lambda p: f"Un-skipping property #{p.get('property_index', '?')}",
                    _suppress_ack("Unskip result", ("Removed skip",)),
                ),
                "read_stub": ToolDisplay("Reading verification stub", None),
                "request_stub_field": ToolDisplay("Requesting stub field", "Stub field result"),
                "advisory_typecheck": ToolDisplay("Type-checking spec", "Type-check result"),
                "scan_knowledge_base": ToolDisplay("Scanning knowledge base", "KB scan results"),
                "get_knowledge_base_article": ToolDisplay("Reading KB article", "KB article"),
                "knowledge_base_contribute": ToolDisplay("Contributing to KB", "KB contribution"),
                "result": CommonTools.result,
                "write_rough_draft": CommonTools.write_rough_draft,
                "read_rough_draft": CommonTools.read_rough_draft,
                "memory": CommonTools.memory,
            })


# ---------------------------------------------------------------------------
# Status rendering helpers
# ---------------------------------------------------------------------------

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
    """Build the Rich ``Text`` for a summary row."""
    indicator, style = _STATUS_INDICATORS[status]
    row = Text()
    row.append(f"{indicator} ", style=style)
    row.append(label)
    row.append(f"  ({status.value})", style="dim")
    return row


# ---------------------------------------------------------------------------
# PipelineEventHandler — routes stream writer events to the task panel
# ---------------------------------------------------------------------------

class PipelineEventHandler(EventHandler):
    """Routes ``NatspecEvent`` payloads to the task panel as content links."""

    def __init__(self, handler: "PipelineTaskHandler"):
        self._handler = handler

    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None:
        evt = cast(NatspecEvent, payload)
        match evt["type"]:
            case "master_spec_update":
                await self._handler.render_content_link(
                    "Master spec updated", evt["spec"], "input.spec",
                )
            case "stub_update":
                await self._handler.render_content_link(
                    "Stub updated", evt["stub"], "Impl.sol",
                )


# ---------------------------------------------------------------------------
# PipelineTaskHandler — per-task IOHandler
# ---------------------------------------------------------------------------

class PipelineTaskHandler:
    """Per-task ``IOHandler[HumanQuestionSchema, Any]`` created by ``PipelineApp.make_handler``."""

    def __init__(
        self,
        task_id: str,
        label: str,
        panel: VerticalScroll,
        app: "PipelineApp",
        tool_config: ToolDisplayConfig,
    ):
        self._task_id = task_id
        self._label = label
        self._panel = panel
        self._app = app
        self._renderer = MessageRenderer(tool_config)
        self._input_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._status = TaskStatus.PENDING

    # ── Status management ─────────────────────────────────────

    def _set_status(self, status: TaskStatus) -> None:
        self._status = status
        self._app._on_task_status_change(self._task_id, self._label, status)

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

    # ── Mounting helpers ──────────────────────────────────────

    async def _mount_to(self, target: VerticalScroll, *widgets: Any) -> None:
        await target.mount_all(widgets)
        # Auto-scroll the panel
        if target.max_scroll_y - target.scroll_y <= 3:
            target.scroll_end(animate=False)

    # ── Content links ───────────────────────────────────────

    async def render_content_link(self, label: str, content: str, filename: str) -> None:
        """Mount a clickable content link in the task panel."""
        snap_id = self._app._store_snapshot(label, content, filename)
        widget = self._app._make_content_link_widget(snap_id, label, filename)
        await self._mount_to(self._panel, widget)

    # ── IOHandler protocol ────────────────────────────────────

    async def log_thread_id(self, tid: str, chosen: bool) -> None:
        pass  # no-op: pipeline tasks don't display session IDs

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str) -> None:
        pass  # no-op

    async def log_start(self, *, path: list[str], description: str, tool_id: str | None) -> None:
        target = self._renderer.get_mount_target(self._panel, path)

        if len(path) == 1:
            banner = Static(Text(f"━━ {description} ━━", style="bold"))
            await self._mount_to(target, banner)
        else:
            inner = VerticalScroll(classes="nested-workflow")
            coll = Collapsible(inner, title=description, collapsed=True)
            self._renderer.nested_containers[path[-1]] = inner
            await self._mount_to(target, coll)

    async def log_end(self, path: list[str]) -> None:
        if len(path) == 1:
            target = self._renderer.get_mount_target(self._panel, path)
            banner = Static(Text(f"━━ end ━━", style="bold dim"))
            await self._mount_to(target, banner)
        else:
            tid = path[-1]
            if tid in self._renderer.nested_containers:
                container = self._renderer.nested_containers.pop(tid)
                parent_coll = container.parent
                if isinstance(parent_coll, Collapsible):
                    parent_coll.collapsed = True

    async def log_state_update(self, path: list[str], st: dict) -> None:
        target = self._renderer.get_mount_target(self._panel, path)

        for node_name, v in st.items():
            if node_name not in KNOWN_NODES:
                continue

            if "messages" not in v:
                continue

            for m in v["messages"]:
                match m:
                    case AIMessage():
                        widgets = self._renderer.render_ai_turn(m)
                        if widgets:
                            await self._mount_to(target, *widgets)
                        self._app._tokens.update(m)
                    case SystemMessage():
                        self._renderer.reset_tool_collapsing()
                        coll = Collapsible(Static(m.text()), title="System prompt", collapsed=True)
                        await self._mount_to(target, coll)
                    case HumanMessage():
                        self._renderer.reset_tool_collapsing()
                        coll = Collapsible(Static(m.text()), title="User input", collapsed=True)
                        await self._mount_to(target, coll)
                    case ToolMessage():
                        coll = self._renderer.render_tool_result(m)
                        if coll is None:
                            continue
                        await self._mount_to(target, coll)
                    case _:
                        self._renderer.reset_tool_collapsing()
                        await self._mount_to(
                            target,
                            Static(Text(f"[Message: {type(m).__name__}]", style="dim")),
                        )

            # Detect working copy spec updates
            if "curr_spec" in v and isinstance(v["curr_spec"], str) and len(path) == 1:
                await self.render_content_link(
                    "Working copy updated", v["curr_spec"], "working.spec",
                )

    async def progress_update(self, path: list[str], upd: Any) -> None:
        pass  # no-op for pipeline tasks

    async def human_interaction(
        self,
        ty: HumanQuestionSchema,
        debug_thunk: Callable[[], None],
    ) -> str:
        self._set_status(TaskStatus.WAITING_HITL)

        # Build prompt from the interaction type
        prompt_parts: list[Text | str] = [Text("Question: ", style="bold yellow"), ty.question]
        if ty.context:
            prompt_parts.append(f"\n  Context: {ty.context}")

        prompt_widget = Static(Text.assemble(*prompt_parts))
        hint_widget = Static("Type your response and press Enter", classes="interaction-hint")
        input_widget = Input(placeholder="Type here...", validate_on=["submitted"])

        # Register for input routing
        self._app._active_inputs[input_widget] = self._input_queue
        self._app._hitl_inputs[self._task_id] = input_widget

        await self._mount_to(self._panel, prompt_widget, input_widget, hint_widget)
        input_widget.focus()

        # Block until the user submits a response
        response = await self._input_queue.get()

        # Deregister and clean up
        del self._app._active_inputs[input_widget]
        self._app._hitl_inputs.pop(self._task_id, None)
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
# PipelineApp — top-level Textual app
# ---------------------------------------------------------------------------

class PipelineApp(IDEContentMixin, App):
    """Textual TUI for the NatSpec multi-agent pipeline."""

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

    def __init__(self, ide: IDEBridge | None = None):
        super().__init__()
        self._init_ide_content(ide)
        self._handlers: dict[str, PipelineTaskHandler] = {}
        self._active_inputs: dict[Input, asyncio.Queue[str]] = {}
        self._hitl_inputs: dict[str, Input] = {}
        self._work_fn: Callable[[], Coroutine[None, None, None]] | None = None
        self._pipeline_done = False
        self._previous_view: str | None = None
        self._content_pane_ids: set[str] = set()

        # Phase section tracking: section_label -> Collapsible widget
        self._phase_sections: dict[str, Collapsible] = {}
        # Task label cache for summary row updates
        self._task_labels: dict[str, str] = {}
        # Task → section and status tracking for reordering
        self._task_sections: dict[str, str] = {}
        self._task_statuses: dict[str, TaskStatus] = {}

    def compose(self) -> ComposeResult:
        yield Static("NatSpec Pipeline | ESC: summary | q: quit (when done)", id="header")
        yield Static("", id="token-bar")
        with ContentSwitcher(id="switcher", initial="summary"):
            yield VerticalScroll(id="summary")

    def set_work(self, fn: Callable[[], Coroutine[None, None, None]]) -> None:
        self._work_fn = fn

    def on_mount(self) -> None:
        self._tokens = TokenStats(self.query_one("#token-bar", Static))
        if self._work_fn is not None:
            self.run_worker(self._work_fn(), thread=False)

    # ── Key bindings ──────────────────────────────────────────

    def action_go_back(self) -> None:
        switcher = self.query_one("#switcher", ContentSwitcher)
        current = switcher.current

        # If viewing a content pane, go back to where we came from and clean up
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
        """No-IDE fallback: open content in a temporary ContentSwitcher pane."""
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

    async def make_handler(self, info: TaskInfo) -> TaskHandle:
        """``HandlerFactory`` implementation — creates per-task panel, handler, summary row."""
        task_id = info.task_id
        label = info.label
        phase = info.phase

        # 1. Ensure the phase section exists in the summary
        section_label = PHASE_LABELS[phase]
        section = await self._ensure_phase_section(section_label)

        # 2. Create summary row (starts PENDING)
        row = Static(
            _render_row(label, TaskStatus.PENDING),
            id=f"row-{task_id}",
            classes="task-row",
        )
        await section.query_one("Contents").mount(row)

        # 3. Create detail panel and add to content switcher
        #    ContentSwitcher doesn't auto-hide dynamically mounted children,
        #    so we must hide the panel explicitly. watch_current will show it
        #    when the user drills in.
        panel = VerticalScroll(id=task_id, classes="task-panel")
        panel.display = False
        switcher = self.query_one("#switcher", ContentSwitcher)
        await switcher.mount(panel)

        # 4. Create handler with phase-appropriate tool config
        tc = tool_config_for_phase(phase)
        handler = PipelineTaskHandler(task_id, label, panel, self, tc)

        self._handlers[task_id] = handler
        self._task_labels[task_id] = label
        self._task_sections[task_id] = section_label
        self._task_statuses[task_id] = TaskStatus.PENDING

        return TaskHandle(
            handler=handler,
            event_handler=PipelineEventHandler(handler),
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
        """Switch to a task's detail panel."""
        switcher = self.query_one("#switcher", ContentSwitcher)
        switcher.current = task_id

        # If the task is waiting for HITL, focus its input
        inp = self._hitl_inputs.get(task_id)
        if inp is not None:
            inp.focus()

    # ── Summary management ────────────────────────────────────

    async def _ensure_phase_section(self, section_label: str) -> Collapsible:
        """Lazily create a ``Collapsible`` section for a phase group."""
        if section_label in self._phase_sections:
            return self._phase_sections[section_label]

        section = Collapsible(title=section_label, collapsed=False)

        # Register BEFORE awaiting mount — concurrent make_handler calls for
        # the same phase will see the section and reuse it rather than racing
        # past this check while mount is in progress.
        self._phase_sections[section_label] = section

        summary = self.query_one("#summary", VerticalScroll)

        # Insert in canonical order (exclude the section we just registered)
        existing_indices = {
            lbl: _SECTION_ORDER.index(lbl)
            for lbl in self._phase_sections
            if lbl in _SECTION_ORDER and lbl != section_label
        }
        new_idx = _SECTION_ORDER.index(section_label) if section_label in _SECTION_ORDER else len(_SECTION_ORDER)

        # Find the correct insertion point
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

    def _on_task_status_change(self, task_id: str, label: str, status: TaskStatus) -> None:
        """Update the summary row for a task and reorder the summary."""
        self._task_statuses[task_id] = status
        try:
            row = self.query_one(f"#row-{task_id}", Static)
        except Exception:
            return
        row.update(_render_row(label, status))
        self._reorder_summary()

    def _reorder_summary(self) -> None:
        """Reorder sections (active first) and rows within sections (running first)."""
        if len(self._phase_sections) <= 1:
            return

        summary = self.query_one("#summary", VerticalScroll)

        def section_has_active(label: str) -> bool:
            return any(
                self._task_statuses.get(tid) in _ACTIVE_STATUSES
                for tid, slabel in self._task_sections.items()
                if slabel == label
            )

        # Sort sections: active first (in canonical order), then inactive
        ordered = sorted(
            self._phase_sections.keys(),
            key=lambda lbl: (
                0 if section_has_active(lbl) else 1,
                _SECTION_ORDER.index(lbl) if lbl in _SECTION_ORDER else len(_SECTION_ORDER),
            ),
        )

        # Reorder sections within the summary
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

        # Reorder rows within each section
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

            # Auto-collapse sections where all tasks are terminal
            all_terminal = all(
                self._task_statuses.get(tid) in _TERMINAL_STATUSES
                for tid in task_ids
            )
            if all_terminal:
                section.collapsed = True

    # ── Pipeline completion ───────────────────────────────────

    async def on_pipeline_done(self, result: PipelineResult) -> None:
        """Show completion banner, preview results, and enable quit."""
        self._pipeline_done = True

        summary = self.query_one("#summary", VerticalScroll)
        switcher = self.query_one("#switcher", ContentSwitcher)
        switcher.current = "summary"

        n_fail = len(result.failures)

        banner_text = Text()
        banner_text.append("\n━━ Pipeline Complete ━━\n", style="bold green")
        banner_text.append(f"Contract: {result.contract_name} (solc {result.solc_version})\n")
        banner_text.append(f"Failures: {n_fail}\n" if n_fail else "All properties succeeded\n")
        if n_fail:
            for f in result.failures:
                banner_text.append(f"  \u2717 {f.prop.description}: {f.reason}\n", style="red")

        await summary.mount(Static(banner_text))

        files: dict[str, str] = {
            "input.spec": result.spec,
            "Impl.sol": result.stub,
            "Intf.sol": result.interface,
        }

        if self._ide is not None:
            preview_id: str | None = None
            try:
                preview_id = await self._ide.preview_results(files)
            except Exception:
                self.notify("Failed to preview results in VS Code", severity="warning")

            if preview_id is not None:
                await self._show_accept_reject_prompt(summary, preview_id)
            else:
                await summary.mount(
                    Static(Text("Preview unavailable.", style="dim"))
                )
        else:
            # No IDE — write files to disk and show expanded
            out_dir = pathlib.Path.cwd()
            for path, content in files.items():
                (out_dir / path).write_text(content)
                lexer = self._guess_lang(path) or "text"
                syntax = Syntax(content, lexer, theme="monokai", line_numbers=True)
                coll = Collapsible(Static(syntax), title=path, collapsed=False)
                await summary.mount(coll)
            await summary.mount(
                Static(Text(f"Wrote {len(files)} file(s) to {out_dir}", style="bold green"))
            )

        await summary.mount(Static(Text("Press q to quit.", style="dim")))

    async def _show_accept_reject_prompt(
        self,
        summary: VerticalScroll,
        preview_id: str,
    ) -> None:
        """Show ACCEPT/REJECT prompt and handle the IDE preview lifecycle."""
        assert self._ide is not None

        prompt_widget = Static(Text.assemble(
            ("Results previewed in VS Code.\n", "bold"),
            ("Type ACCEPT to write files or REJECT to discard.", "dim"),
        ))
        hint_widget = Static("Response must be ACCEPT or REJECT", classes="interaction-hint")
        input_widget = Input(placeholder="ACCEPT / REJECT", validate_on=["submitted"])

        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._active_inputs[input_widget] = queue

        await summary.mount(prompt_widget, input_widget, hint_widget)
        input_widget.focus()

        response = await queue.get()
        del self._active_inputs[input_widget]

        await prompt_widget.remove()
        await input_widget.remove()
        await hint_widget.remove()

        decision = response.strip().upper()
        if decision == "ACCEPT":
            try:
                written = await self._ide.accept_results(preview_id)
                await summary.mount(
                    Static(Text(f"Results accepted — wrote {len(written)} file(s).", style="bold green"))
                )
            except Exception:
                self.notify("Failed to accept results in VS Code", severity="warning")
        else:
            try:
                await self._ide.reject_results(preview_id)
            except Exception:
                pass
            await summary.mount(Static(Text("Results rejected.", style="yellow")))

    def on_click(self, event: Any) -> None:
        """Handle clicks on summary rows to drill into task panels."""
        # Walk up from the click target to find a .task-row Static
        widget = event.widget if hasattr(event, "widget") else None
        while widget is not None:
            if isinstance(widget, Static) and widget.has_class("task-row"):
                if widget.id and widget.id.startswith("row-"):
                    task_id = widget.id.removeprefix("row-")
                    self._drill_to(task_id)
                return
            widget = widget.parent
