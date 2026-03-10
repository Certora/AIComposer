import asyncio
import traceback
from abc import abstractmethod
from collections.abc import Coroutine
from typing import Callable

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Input, Collapsible
from textual.binding import Binding
from textual.validation import Validator

from rich.text import Text

from composer.io.ide_bridge import IDEBridge
from composer.io.ide_content import IDEContentMixin
from composer.io.tool_display import ToolDisplayConfig
from composer.io.message_renderer import MessageRenderer, TokenStats, dot, KNOWN_NODES

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage


class BaseRichConsoleApp[H, P](IDEContentMixin, App):
    """Base Textual TUI for workflow IO, parameterized by human interaction (H) and progress (P) types."""

    CSS = """
    #status-bar { dock: top; height: 1; background: $surface; padding: 0 1; }
    #token-bar { dock: top; height: 1; background: $surface; padding: 0 1; }
    #event-log { height: 1fr; padding: 0 1; }
    #event-log > * { margin-bottom: 1; }
    .interaction-hint { color: $text-muted; padding: 0 1; }
    .nested-workflow { margin-left: 2; border-left: solid $secondary; padding-left: 1; }
    .vfs-change { color: cyan; }
    Collapsible { background: transparent; border: none; padding: 0; }
    CollapsibleTitle { padding: 0 1; }
    Collapsible Contents { padding: 0 0 0 3; }
    """

    BINDINGS = [
        Binding("q", "quit_app", "Quit", show=True),
    ]

    def __init__(
        self,
        tool_config: ToolDisplayConfig,
        show_checkpoints: bool = False,
        ide: IDEBridge | None = None,
    ):
        super().__init__()

        self._renderer = MessageRenderer(tool_config)
        self._input_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._mounted: asyncio.Event = asyncio.Event()
        self._graph_done = False
        self._session_id = "—"
        self._checkpoint_id = "—"
        self._work_fn: Callable[[], Coroutine[None, None, None]] | None = None
        self._show_checkpoints = show_checkpoints
        self._init_ide_content(ide)

    def compose(self) -> ComposeResult:
        yield Static("Session: — | Checkpoint: —", id="status-bar")
        yield Static("", id="token-bar")
        yield VerticalScroll(id="event-log")

    def set_work(self, fn: Callable[[], Coroutine[None, None, None]]):
        self._work_fn = fn

    def on_mount(self):
        self._tokens = TokenStats(self.query_one("#token-bar", Static))
        self._mounted.set()
        if self._work_fn is not None:
            self.run_worker(self._work_fn(), thread=False)

    async def show_error(self, error: Exception) -> None:
        """Display a fatal error in the event log and enable quit."""
        await self._mounted.wait()
        target = self.query_one("#event-log", VerticalScroll)
        tb = "".join(traceback.format_exception(error))
        error_text = Text()
        error_text.append("\n━━ WORKFLOW ERROR ━━\n\n", style="bold red")
        error_text.append(f"{type(error).__name__}: {error}\n\n", style="red")
        error_text.append(tb, style="red dim")
        error_text.append("\nPress q to quit.", style="dim")
        await self._mount_to(target, Static(error_text))
        self._graph_done = True

    # ── Key bindings ──────────────────────────────────────────

    def action_quit_app(self):
        if self._graph_done:
            self.exit()

    # ── Helpers ───────────────────────────────────────────────

    def _get_mount_target(self, path: list[str]) -> VerticalScroll:
        return self._renderer.get_mount_target(self.query_one("#event-log", VerticalScroll), path)

    async def _auto_scroll(self):
        log = self.query_one("#event-log", VerticalScroll)
        if log.max_scroll_y - log.scroll_y <= 3:
            log.scroll_end(animate=False)

    async def _mount_to(self, target: VerticalScroll, *widgets):
        await target.mount_all(widgets)
        await self._auto_scroll()

    def _reset_tool_collapsing(self):
        """Reset consecutive tool call collapsing state."""
        self._renderer.reset_tool_collapsing()

    def _render_ai_turn(self, msg: AIMessage) -> list[Static | Collapsible]:
        """Render an AI turn as a list of widgets (not a single collapsible)."""
        widgets = self._renderer.render_ai_turn(msg)
        self._tokens.update(msg)
        return widgets

    # ── Abstract / overridable methods ────────────────────────

    @abstractmethod
    def build_interaction(self, ty: H) -> tuple[Text, str, list[Validator]]:
        """Return (prompt_renderable, hint_text, validators) for the interaction type."""
        ...

    @abstractmethod
    async def render_progress(self, target: VerticalScroll, path: list[str], upd: P) -> None:
        """Render a progress update into the target container."""
        ...

    async def render_state_extras(self, target: VerticalScroll, node_name: str, node_data: dict) -> None:
        """Handle non-message state data (e.g. VFS changes). Override in subclasses."""
        pass

    def classify_human_message(self, m: HumanMessage) -> tuple[str, bool]:
        """Return (title, collapsed) for a HumanMessage. Override for workflow-specific classification."""
        return ("User input", True)

    # ── IOHandler protocol ────────────────────────────────────

    def _update_status_bar(self):
        bar = self.query_one("#status-bar", Static)
        bar.update(f"Session: {self._session_id} | Checkpoint: {self._checkpoint_id}")

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str):
        await self._mounted.wait()
        self._checkpoint_id = checkpoint_id
        self._update_status_bar()
        if self._show_checkpoints:
            target = self._get_mount_target(path)
            await self._mount_to(
                target,
                Static(Text(f"checkpoint: {checkpoint_id}", style="dim"))
            )

    async def log_start(self, *, path: list[str], description: str, tool_id: str | None):
        await self._mounted.wait()
        target = self._get_mount_target(path)

        if len(path) == 1:
            banner = Static(
                Text(f"━━ {description} ━━", style="bold"),
            )
            await self._mount_to(target, banner)
        else:
            inner = VerticalScroll(classes="nested-workflow")
            coll = Collapsible(inner, title=description, collapsed=True)
            self._renderer.nested_containers[path[-1]] = inner
            await self._mount_to(target, coll)

    async def log_end(self, path: list[str]):
        await self._mounted.wait()
        target = self._get_mount_target(path)

        if len(path) == 1:
            banner = Static(
                Text(f"━━ Workflow end: {path[0]} ━━", style="bold"),
            )
            await self._mount_to(target, banner)
        else:
            # Collapse the nested workflow
            tid = path[-1]
            if tid in self._renderer.nested_containers:
                container = self._renderer.nested_containers.pop(tid)
                parent_coll = container.parent
                if isinstance(parent_coll, Collapsible):
                    parent_coll.collapsed = True

    async def log_state_update(self, path: list[str], st: dict):
        await self._mounted.wait()
        target = self._get_mount_target(path)

        for node_name, v in st.items():
            if node_name not in KNOWN_NODES:
                continue

            if "messages" in v:
                for m in v["messages"]:
                    match m:
                        case AIMessage():
                            widgets = self._render_ai_turn(m)
                            if widgets:
                                await self._mount_to(target, *widgets)
                        case SystemMessage():
                            self._reset_tool_collapsing()
                            coll = Collapsible(Static(m.text()), title="System prompt", collapsed=True)
                            await self._mount_to(target, coll)
                        case HumanMessage():
                            self._reset_tool_collapsing()
                            title, collapsed = self.classify_human_message(m)
                            content = m.text()
                            coll = Collapsible(Static(content), title=title, collapsed=collapsed)
                            await self._mount_to(target, coll)
                        case ToolMessage():
                            coll = self._renderer.render_tool_result(m)
                            if coll is None:
                                continue
                            await self._mount_to(target, coll)
                        case _:
                            self._reset_tool_collapsing()
                            await self._mount_to(target, Static(Text(f"[Message: {type(m).__name__}]", style="dim")))

            await self.render_state_extras(target, node_name, v)

    async def progress_update(self, path: list[str], upd: P):
        await self._mounted.wait()
        target = self._get_mount_target(path)
        await self.render_progress(target, path, upd)

    async def human_interaction(
        self,
        ty: H,
        debug_thunk: Callable[[], None]
    ) -> str:
        await self._mounted.wait()
        target = self.query_one("#event-log", VerticalScroll)

        # Mount directly from worker — post_message races with state update mounts
        prompt_content, hint_text, validators = self.build_interaction(ty)

        prompt_widget = Static(prompt_content)
        hint_widget = Static(hint_text, classes="interaction-hint")
        input_widget = Input(placeholder="Type here...", validate_on=["submitted"])
        input_widget.validators = validators

        await self._mount_to(target, prompt_widget, input_widget, hint_widget)
        input_widget.focus()

        response = await self._input_queue.get()

        # Replace interaction widgets with compact summary
        await prompt_widget.remove()
        await input_widget.remove()
        await hint_widget.remove()
        await self._mount_to(
            target,
            Static(dot("green", Text.assemble(("You: ", "bold green"), response)))
        )

        return response

    def on_input_submitted(self, event: Input.Submitted):
        value = event.value.strip()
        if not value:
            return

        if event.validation_result and not event.validation_result.is_valid:
            for desc in event.validation_result.failure_descriptions:
                self.notify(desc, severity="error")
            return

        # Disable input to prevent double-submit
        event.input.disabled = True
        self._input_queue.put_nowait(value)
