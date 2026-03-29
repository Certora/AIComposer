"""
Extracted message rendering logic shared by BaseRichConsoleApp and MultiJobTaskHandler.

``MessageRenderer`` holds per-stream rendering state (tool collapsing, nested
containers) and exposes both widget-producing and widget-mounting methods.

``TokenStats`` accumulates token usage from AI messages and updates a display widget.
"""

from typing import Callable, Protocol

from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static, Collapsible

from rich.text import Text

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from composer.diagnostics.handlers import normalize_content
from composer.ui.tool_display import ToolDisplayConfig

from graphcore.graph import INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE

KNOWN_NODES: set[str] = {INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE}

_DOT = "\u25cf "  # ● filled circle

from logging import getLogger

_logger = getLogger(__name__)

def dot(style: str, text: Text | str) -> Text:
    """Prepend a colored dot to a Text or string for visual structure."""
    if isinstance(text, str):
        text = Text(text)
    result = Text()
    result.append(_DOT, style=style)
    result.append_text(text)
    return result


class TokenStats:
    """Accumulates token usage across AI messages and updates a display widget."""

    def __init__(self, display: Static):
        self._display = display
        self.input: int = 0
        self.output: int = 0
        self.cache_read: int = 0
        self.cache_write: int = 0

    # Prices in $/MTok
    _PRICE_INPUT = 5.0
    _PRICE_OUTPUT = 25.0
    _PRICE_CACHE_READ = 0.50
    _PRICE_CACHE_WRITE = 6.25

    def _cost(self) -> float:
        """Estimated cost in dollars."""
        return (
            self.input * self._PRICE_INPUT
            + self.output * self._PRICE_OUTPUT
            + self.cache_read * self._PRICE_CACHE_READ
            + self.cache_write * self._PRICE_CACHE_WRITE
        ) / 1_000_000

    def update(self, msg: AIMessage) -> None:
        """Extract usage from the message and refresh the display widget."""
        if not isinstance(msg.response_metadata, dict):
            return
        usage = msg.response_metadata.get("usage")
        if usage is None:
            return
        self.input += usage.get("input_tokens", 0)
        self.output += usage.get("output_tokens", 0)
        self.cache_read += usage.get("cache_read_input_tokens", 0)
        self.cache_write += usage.get("cache_creation_input_tokens", 0)
        cost = self._cost()
        self._display.update(
            f"in:{self.input:,} out:{self.output:,} "
            f"cache_read:{self.cache_read:,} cache_write:{self.cache_write:,} "
            f"| ${cost:.2f}"
        )


class MountFn(Protocol):
    """Callback for mounting widgets into a scrollable container."""
    async def __call__(self, target: VerticalScroll, *widgets: Widget) -> None: ...


_HUMAN_TAG_DISPLAY: dict[str, tuple[str, bool]] = {
    "initial_prompt": ("Initial prompt", True),
    "resume": ("Resume context", True),
    "summarization": ("Summarization", True),
    "scolding": ("System correction", True),
    "prover_summary": ("Prover violation summary", False),
}


class MessageRenderer:
    """Per-stream rendering state, widget production, and mounting.

    Used by both ``BaseRichConsoleApp`` (single-stream) and
    ``MultiJobTaskHandler`` (per-task stream).
    """

    def __init__(
        self,
        tool_config: ToolDisplayConfig,
        mount_to: MountFn,
        on_tokens: Callable[[AIMessage], None],
    ):
        self.tool_config = tool_config
        self._mount_to = mount_to
        self._on_tokens = on_tokens
        self.nested_containers: dict[str, VerticalScroll] = {}

        # Consecutive tool call collapsing state
        self._last_tool_group: str | None = None
        self._last_tool_items: list[str] = []
        self._last_tool_widget: Static | None = None

        # Tool call anchor tracking for live output panes
        self._tool_call_anchors: dict[str, Static] = {}

    def render_ai_turn(self, msg: AIMessage) -> list[Static | Collapsible]:
        """Render an AI turn as a list of widgets."""
        widgets: list[Static | Collapsible] = []
        tc = self.tool_config

        for c in normalize_content(msg.content):
            match c["type"]:
                case "thinking":
                    full_text = c.get("thinking", "")
                    widgets.append(
                        Collapsible(Static(full_text, markup=False), title="Thinking...", collapsed=True)
                    )
                case "text":
                    text = c["text"]
                    if (stripped := text.strip()):
                        widgets.append(Static(dot("blue", stripped)))
                case "tool_use":
                    tool_call_id = c.get("id")
                    name = c["name"]
                    input_args = c.get("input", {})
                    grouped = tc.get_group(name)

                    if grouped is not None:
                        raw = grouped.extract_group_items(input_args)
                        new_items = [raw] if isinstance(raw, str) else list(raw)

                        if grouped.group_id == self._last_tool_group:
                            # Same group — update existing widget
                            self._last_tool_items.extend(new_items)
                            new_text = grouped.render_group(self._last_tool_items)
                            if self._last_tool_widget is not None:
                                self._last_tool_widget.update(dot("green", Text(new_text, style="dim")))
                            w = self._last_tool_widget
                        else:
                            # New group
                            self._last_tool_group = grouped.group_id
                            self._last_tool_items = new_items
                            display_text = grouped.render_group(self._last_tool_items)
                            w = Static(dot("green", Text(display_text, style="dim")))
                            self._last_tool_widget = w
                            widgets.append(w)

                        if tool_call_id is not None and w is not None:
                            self._tool_call_anchors[tool_call_id] = w
                    else:
                        # Non-grouped tool — reset and emit standalone
                        self.reset_tool_collapsing()
                        friendly = tc.format_tool_call(name, input_args)
                        w = Static(dot("green", Text(friendly, style="dim")))
                        if tool_call_id is not None:
                            self._tool_call_anchors[tool_call_id] = w
                        widgets.append(w)
                case other:
                    widgets.append(Static(f"Unknown block: {other}"))

        return widgets

    def render_tool_result(self, msg: ToolMessage) -> Collapsible | None:
        """Render a tool result as a collapsible, or ``None`` to suppress."""
        name = getattr(msg, "name", None) or "Tool result"
        result_info = self.tool_config.format_result(name, msg)
        if result_info is None:
            return None
        self.reset_tool_collapsing()
        label, body = result_info
        return Collapsible(Static(body, markup=False), title=label, collapsed=True)

    def get_tool_call_anchor(self, tool_call_id: str) -> Static | None:
        """Return the widget for a tool call, if tracked."""
        return self._tool_call_anchors.get(tool_call_id)

    def reset_tool_collapsing(self):
        """Reset consecutive tool call collapsing state."""
        self._last_tool_group = None
        self._last_tool_items = []
        self._last_tool_widget = None

    def get_mount_target(self, root: VerticalScroll, path: list[str]) -> VerticalScroll:
        """Resolve the mount target for a given path.

        If the path references a nested container, returns it; otherwise
        falls back to ``root``.
        """
        if len(path) > 1 and path[-1] in self.nested_containers:
            return self.nested_containers[path[-1]]
        return root

    # ── Shared rendering methods ─────────────────────────────

    def classify_human(self, m: HumanMessage) -> tuple[str, bool]:
        """Classify a human message for display. Returns (title, collapsed)."""
        tag = getattr(m, "display_tag", None)
        if tag is not None:
            return _HUMAN_TAG_DISPLAY.get(tag, ("User input", True))
        return ("User input", True)
    
    def get_flow_target(self, root: VerticalScroll, path: list[str]) -> VerticalScroll:
        if len(path) > 1 and path[-2] in self.nested_containers:
            return self.nested_containers[path[-1]]
        return root

    async def render_start(self, root: VerticalScroll, *, path: list[str], description: str) -> None:
        """Render a workflow start banner or nested collapsible."""
        target = self.get_flow_target(root, path)
        if len(path) == 1:
            _logger.debug("Starting top level workflow: %s", description)
            banner = Static(Text(f"━━ {description} ━━", style="bold"))
            await self._mount_to(target, banner)
        else:
            inner = VerticalScroll(classes="nested-workflow")
            coll = Collapsible(inner, title=description, collapsed=True)
            self.nested_containers[path[-1]] = inner
            await self._mount_to(target, coll)

    async def render_end(self, root: VerticalScroll, *, path: list[str]) -> None:
        """Render a workflow end banner or collapse a nested workflow."""
        if len(path) == 1:
            target = self.get_mount_target(root, path)
            banner = Static(Text("━━ end ━━", style="bold dim"))
            await self._mount_to(target, banner)
        else:
            tid = path[-1]
            if tid in self.nested_containers:
                container = self.nested_containers.pop(tid)
                parent_coll = container.parent
                if isinstance(parent_coll, Collapsible):
                    parent_coll.collapsed = True

    async def render_messages(self, target: VerticalScroll, messages: list) -> None:
        """Render a list of LangChain messages, mounting widgets to *target*."""
        for m in messages:
            match m:
                case AIMessage():
                    widgets = self.render_ai_turn(m)
                    if widgets:
                        await self._mount_to(target, *widgets)
                    self._on_tokens(m)
                case SystemMessage():
                    self.reset_tool_collapsing()
                    coll = Collapsible(Static(m.text(), markup=False), title="System prompt", collapsed=True)
                    await self._mount_to(target, coll)
                case HumanMessage():
                    self.reset_tool_collapsing()
                    title, collapsed = self.classify_human(m)
                    content = m.text()
                    coll = Collapsible(Static(content, markup=False), title=title, collapsed=collapsed)
                    await self._mount_to(target, coll)
                case ToolMessage():
                    coll = self.render_tool_result(m)
                    if coll is None:
                        continue
                    await self._mount_to(target, coll)
                case _:
                    self.reset_tool_collapsing()
                    await self._mount_to(
                        target,
                        Static(Text(f"[Message: {type(m).__name__}]", style="dim")),
                    )
