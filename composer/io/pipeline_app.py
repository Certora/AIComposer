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
from collections.abc import Callable, Coroutine
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Input, Collapsible, ContentSwitcher
from textual.binding import Binding

from rich.text import Text

from langchain_core.messages import (
    AIMessage, ToolMessage, HumanMessage, SystemMessage,
)

from composer.io.message_renderer import MessageRenderer, dot, KNOWN_NODES
from composer.io.tool_display import ToolDisplayConfig
from composer.io.event_handler import NullEventHandler
from composer.spec.pipeline import Phase, TaskInfo, TaskHandle, PipelineResult
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
            return ToolDisplayConfig(
                tool_display={
                    "result": "Delivering result",
                    "memory": "Accessing memory",
                },
                collapse_groups={"memory": "memory"},
                suppress_results={"memory"},
            )
        case "bug_analysis":
            return ToolDisplayConfig(
                tool_display={
                    "result": "Delivering result",
                },
            )
        case "interface_gen" | "stub_gen":
            return ToolDisplayConfig(
                tool_display={
                    "result": "Delivering result",
                },
            )
        case "cvl_gen":
            return ToolDisplayConfig(
                tool_display={
                    "put_cvl": "Writing spec",
                    "put_cvl_raw": "Writing spec",
                    "get_cvl": "Reading spec",
                    "feedback_tool": "Getting feedback",
                    "cvl_research": "Researching CVL",
                    "extended_reasoning": "Reasoning",
                    "publish_spec": "Publishing to master spec",
                    "give_up": "Giving up on property",
                    "read_stub": "Reading verification stub",
                    "request_stub_field": "Requesting stub field",
                    "advisory_typecheck": "Type-checking spec",
                    "scan_knowledge_base": "Scanning knowledge base",
                    "get_knowledge_base_article": "Reading KB article",
                    "knowledge_base_contribute": "Contributing to KB",
                    "result": "Delivering result",
                },
                collapse_groups={
                    "scan_knowledge_base": "kb",
                    "get_knowledge_base_article": "kb",
                },
                suppress_results={
                    "get_cvl", "read_stub", "extended_reasoning",
                },
            )


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


def _render_row(label: str, status: TaskStatus) -> Text:
    """Build the Rich ``Text`` for a summary row."""
    indicator, style = _STATUS_INDICATORS[status]
    row = Text()
    row.append(f"{indicator} ", style=style)
    row.append(label)
    row.append(f"  ({status.value})", style="dim")
    return row


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

    def mark_error(self, reason: str) -> None:
        self._set_status(TaskStatus.ERROR)

    # ── Mounting helpers ──────────────────────────────────────

    async def _mount_to(self, target: VerticalScroll, *widgets: Any) -> None:
        await target.mount_all(widgets)
        # Auto-scroll the panel
        if target.max_scroll_y - target.scroll_y <= 3:
            target.scroll_end(animate=False)

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
        tc = self._renderer.tool_config

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
                    case SystemMessage():
                        self._renderer.reset_tool_collapsing()
                        coll = Collapsible(Static(m.text()), title="System prompt", collapsed=True)
                        await self._mount_to(target, coll)
                    case HumanMessage():
                        self._renderer.reset_tool_collapsing()
                        coll = Collapsible(Static(m.text()), title="User input", collapsed=True)
                        await self._mount_to(target, coll)
                    case ToolMessage():
                        name = getattr(m, "name", None) or "Tool result"
                        if name in tc.collapse_groups:
                            continue
                        content = m.text()
                        if not tc.should_show_result(name, content):
                            continue
                        self._renderer.reset_tool_collapsing()
                        friendly = tc.tool_result_display.get(name, name)
                        coll = Collapsible(Static(content), title=friendly, collapsed=True)
                        await self._mount_to(target, coll)
                    case _:
                        self._renderer.reset_tool_collapsing()
                        await self._mount_to(
                            target,
                            Static(Text(f"[Message: {type(m).__name__}]", style="dim")),
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

        # Register for HITL routing — direct object reference
        self._app._active_inputs[input_widget] = self

        await self._mount_to(self._panel, prompt_widget, input_widget, hint_widget)
        input_widget.focus()

        # Block until the user submits a response
        response = await self._input_queue.get()

        # Deregister and clean up
        del self._app._active_inputs[input_widget]
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

class PipelineApp(App):
    """Textual TUI for the NatSpec multi-agent pipeline."""

    CSS = """
    #header { dock: top; height: 1; background: $surface; padding: 0 1; }
    #summary { height: 1fr; padding: 0 1; }
    #summary > * { margin-bottom: 0; }
    .task-row { padding: 0 1; }
    .task-row:hover { background: $surface; }
    .task-panel { height: 1fr; padding: 0 1; }
    .task-panel > * { margin-bottom: 1; }
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

    def __init__(self):
        super().__init__()
        self._handlers: dict[str, PipelineTaskHandler] = {}
        self._active_inputs: dict[Input, PipelineTaskHandler] = {}
        self._work_fn: Callable[[], Coroutine[None, None, None]] | None = None
        self._pipeline_done = False

        # Phase section tracking: section_label -> Collapsible widget
        self._phase_sections: dict[str, Collapsible] = {}
        # Task label cache for summary row updates
        self._task_labels: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static("NatSpec Pipeline | ESC: summary | q: quit (when done)", id="header")
        with ContentSwitcher(id="switcher", initial="summary"):
            yield VerticalScroll(id="summary")

    def set_work(self, fn: Callable[[], Coroutine[None, None, None]]) -> None:
        self._work_fn = fn

    def on_mount(self) -> None:
        if self._work_fn is not None:
            self.run_worker(self._work_fn(), thread=False)

    # ── Key bindings ──────────────────────────────────────────

    def action_go_back(self) -> None:
        switcher = self.query_one("#switcher", ContentSwitcher)
        switcher.current = "summary"

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

        return TaskHandle(
            handler=handler,
            event_handler=NullEventHandler(),
            on_start=handler.mark_running,
            on_done=handler.mark_done,
            on_error=handler.mark_error,
        )

    # ── HITL routing ──────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return

        handler = self._active_inputs.get(event.input)
        if handler is None:
            return

        if event.validation_result and not event.validation_result.is_valid:
            for desc in event.validation_result.failure_descriptions:
                self.notify(desc, severity="error")
            return

        event.input.disabled = True
        handler._input_queue.put_nowait(value)

    # ── Navigation ────────────────────────────────────────────

    def _drill_to(self, task_id: str) -> None:
        """Switch to a task's detail panel."""
        switcher = self.query_one("#switcher", ContentSwitcher)
        switcher.current = task_id

        # If the task is waiting for HITL, focus its input
        handler = self._handlers.get(task_id)
        if handler and handler._status == TaskStatus.WAITING_HITL:
            for inp, h in self._active_inputs.items():
                if h is handler:
                    inp.focus()
                    break

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
        """Update the summary row for a task."""
        try:
            row = self.query_one(f"#row-{task_id}", Static)
        except Exception:
            return
        row.update(_render_row(label, status))

    # ── Pipeline completion ───────────────────────────────────

    async def on_pipeline_done(self, result: PipelineResult) -> None:
        """Show completion banner and enable quit."""
        self._pipeline_done = True

        summary = self.query_one("#summary", VerticalScroll)

        n_fail = len(result.failures)

        banner_text = Text()
        banner_text.append("\n━━ Pipeline Complete ━━\n", style="bold green")
        banner_text.append(f"Contract: {result.contract_name} (solc {result.solc_version})\n")
        banner_text.append(f"Failures: {n_fail}\n" if n_fail else "All properties succeeded\n")
        if n_fail:
            for f in result.failures:
                banner_text.append(f"  \u2717 {f.prop.description[:60]}: {f.reason[:40]}\n", style="red")
        banner_text.append("\nPress q to quit.", style="dim")

        await summary.mount(Static(banner_text))

        # Switch back to summary so user sees the result
        switcher = self.query_one("#switcher", ContentSwitcher)
        switcher.current = "summary"

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
