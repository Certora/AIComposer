"""
Extracted message rendering logic shared by BaseRichConsoleApp and MultiJobTaskHandler.

``MessageRenderer`` holds per-stream rendering state (tool collapsing, nested
containers) and exposes both widget-producing and widget-mounting methods.

``TokenStats`` accumulates token usage from AI messages and updates a display widget.
"""

from typing import Callable, Protocol

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static, Collapsible

from rich.spinner import Spinner
from rich.text import Text

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from composer.diagnostics.handlers import normalize_content
from composer.ui.tool_display import ToolDisplayConfig, ToolDisplayClassifier
from composer.ui.tool_call_renderer import ToolCallRenderer

from graphcore.graph import INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE
from graphcore.utils import get_token_usage

KNOWN_NODES: set[str] = {INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE}

import logging
logger = logging.getLogger(__name__)

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
        usage = get_token_usage(msg)
        self.input += usage["input_tokens"]
        self.output += usage["output_tokens"]
        self.cache_read += usage["cache_read_input_tokens"]
        self.cache_write += usage["cache_creation_input_tokens"]
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


_CORNER = "╰─"  # L-corner glyph rendered as a sibling Static to the left


class _ToolCallSpinner(Static):
    """Animated spinner mounted under a long-running tool call. Replaced
    by the tool's result widget when the result lands."""

    def __init__(self, message: str = "Running"):
        super().__init__("")
        self._spinner = Spinner("dots", message)
        self._timer: Timer | None = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / 12, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        self.update(self._spinner)


def _attachment_row(content: Widget, *, placeholder: bool = False) -> Horizontal:
    """An attachment row: a 2-cell Static holding the L-corner glyph,
    sitting to the left of the actual content widget. Keeps the corner
    *outside* a Collapsible's chevron so the visual order reads
    ``╰─ ▶ Title`` rather than ``▶ ╰─ Title``.

    ``placeholder=True`` tags the row as transient (spinner / "Pending
    sub-agent…") so ``mount_tool_result`` can selectively remove it
    when the result lands without disturbing non-placeholder rows
    (e.g. a sub-agent's nested-workflow Collapsible)."""
    corner = Static(_CORNER, classes="attachment-corner")
    classes = "attachment-row placeholder-row" if placeholder else "attachment-row"
    return Horizontal(corner, content, classes=classes)


def _pending_row() -> Horizontal:
    """Placeholder mounted under a ``LONG_RUNNING_AGENT`` call while the
    sub-agent works. The sub-agent renders activity in its own panel, so
    no spinner here — just a static "pending" affordance that anchors
    the attachment slot visually."""
    pending = Static("Pending sub-agent…", classes="attachment-pending")
    return _attachment_row(pending, placeholder=True)


class MessageRenderer(ToolCallRenderer):
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
        super().__init__(tool_config)
        self._mount_to = mount_to
        self._on_tokens = on_tokens
        self.nested_containers: dict[str, VerticalScroll] = {}

        # Per-tool-call attachment slots: ``Vertical`` mounted directly
        # below each non-grouped tool-call widget. Tool results land
        # inside the matching attachment instead of at the top level, so
        # each result visually nests under its call (with a "╰─"
        # L-connector). ``LONG_RUNNING`` calls pre-fill the attachment
        # with a spinner; ``LONG_RUNNING_AGENT`` calls pre-fill with a
        # static "Pending sub-agent…" placeholder. Both are cleared when
        # the result lands.
        self._tool_call_attachments: dict[str, Vertical] = {}

    def _make_attachment(
        self, tool_call_id: str, classifier: ToolDisplayClassifier,
    ) -> Vertical:
        """Build the attachment slot for a tool call. Pre-fills with a
        spinner / pending placeholder per the classifier."""
        attachment = Vertical(classes="tool-call-attachment")
        self._tool_call_attachments[tool_call_id] = attachment
        if classifier is ToolDisplayClassifier.LONG_RUNNING:
            attachment._add_child(_attachment_row(_ToolCallSpinner(), placeholder=True))
        elif classifier is ToolDisplayClassifier.LONG_RUNNING_AGENT:
            attachment._add_child(_pending_row())
        return attachment

    def render_ai_turn(self, msg: AIMessage) -> list[Widget]:
        """Render an AI turn as a list of widgets."""
        widgets: list[Widget] = []

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
                    name = c["name"]
                    tool_call_id = c.get("id")
                    w = self.render_tool_call(
                        name=name,
                        input_args=c.get("input", {}),
                        tool_call_id=tool_call_id,
                    )
                    if w is None:
                        # Folded into a grouped tool's existing widget;
                        # no per-call attachment for grouped tools (their
                        # results are suppressed anyway).
                        continue
                    if (
                        tool_call_id is not None
                        and self.tool_config.get_group(name) is None
                    ):
                        # Wrap the call line and its attachment in a
                        # single block so they mount as one unit (no
                        # inter-row gap from the parent container's
                        # ``> *`` margin rules) and so the call's
                        # tool-call-line classes apply uniformly.
                        attachment = self._make_attachment(
                            tool_call_id,
                            self.tool_config.get_classifier(name),
                        )
                        widgets.append(Vertical(w, attachment, classes="tool-call-block"))
                    else:
                        widgets.append(w)
                case other:
                    widgets.append(Static(f"Unknown block: {other}"))

        return widgets

    async def mount_tool_result(self, target: VerticalScroll, msg: ToolMessage) -> None:
        """Mount a tool result widget inside its anchoring tool-call
        attachment, replacing any spinner / pending placeholder. Falls
        back to top-level mount in *target* when no attachment is found
        (e.g. ``tool_call_id`` missing, grouped tool, or replay of a
        turn whose call we didn't render)."""
        name = getattr(msg, "name", None) or "Tool result"
        result_info = self.tool_config.format_result(name, msg)
        tool_call_id = getattr(msg, "tool_call_id", None)
        attachment = (
            self._tool_call_attachments.get(tool_call_id)
            if tool_call_id is not None else None
        )
        # Clear ONLY transient placeholder rows (spinner / pending) —
        # leaving any non-placeholder rows (e.g. a sub-agent's
        # nested-workflow Collapsible mounted via log_start) intact so
        # the activity record stays alongside the result.
        if attachment is not None and attachment.is_mounted:
            for row in attachment.query(".attachment-row.placeholder-row"):
                await row.remove()
        if result_info is None:
            return
        self.reset_tool_collapsing()
        label, body = result_info
        coll = Collapsible(Static(body, markup=False), title=label, collapsed=True)
        if attachment is not None and attachment.is_mounted:
            await attachment.mount(_attachment_row(coll))
            return
        # Fallback: no anchor found; mount top-level (no L-corner).
        await self._mount_to(target, coll)

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
        # Walk from most specific to least specific: the current flow's container
        # may not exist yet (render_start creates it), so fall back to the parent's.
        if len(path) > 1 and path[-1] in self.nested_containers:
            return self.nested_containers[path[-1]]
        if len(path) > 1 and path[-2] in self.nested_containers:
            return self.nested_containers[path[-2]]
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
                    coll = Collapsible(Static(m.text, markup=False), title="System prompt", collapsed=True)
                    await self._mount_to(target, coll)
                case HumanMessage():
                    self.reset_tool_collapsing()
                    title, collapsed = self.classify_human(m)
                    content = m.text
                    coll = Collapsible(Static(content, markup=False), title=title, collapsed=collapsed)
                    await self._mount_to(target, coll)
                case ToolMessage():
                    await self.mount_tool_result(target, m)
                case _:
                    self.reset_tool_collapsing()
                    await self._mount_to(
                        target,
                        Static(Text(f"[Message: {type(m).__name__}]", style="dim")),
                    )
