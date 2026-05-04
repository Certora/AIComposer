"""
Message rendering shared by BaseRichConsoleApp and MultiJobTaskHandler.

``MessageRenderer`` resolves mount targets per-thread, owns one
``ThreadRenderState`` per active graph thread, and exposes hooks for
``LangGraph`` lifecycle events (``render_start``/``render_end``/
``render_messages``).

Per-thread isolation matters because parallel sub-graphs (e.g. the
agentic CEX analyzer's per-rule gather) interleave AI turns onto the
shared event queue; a flat tool-grouping state would let one sub-
agent's tool calls fold into another's. Each thread carries its own
``ToolGroupState``, scoped to its lifetime.

The within-tool override is the routing hook callers use to redirect
sub-graph mounts. ``CodeGenRichApp`` installs an override on
``prover_run`` so that sub-agents launched from inside the prover tool
land inside the "Analysis Agents" collapsible instead of at the
codegen thread's root.

``TokenStats`` accumulates token usage from AI messages.
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol

from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static, Collapsible

from rich.text import Text

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from composer.diagnostics.handlers import normalize_content
from composer.ui.tool_display import ToolDisplayConfig
from composer.ui.tool_call_renderer import ToolGroupState, render_tool_call

from graphcore.graph import INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE
from graphcore.utils import get_token_usage

KNOWN_NODES: set[str] = {INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE}

import logging
logger = logging.getLogger(__name__)

_DOT = "● "  # ● filled circle


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
    "rough_draft_review_reminder": ("Rough draft review reminder", True),
    "prover_routing_reminder": ("Prover routing reminder", True),
}


@dataclass
class ThreadRenderState:
    """Per-thread renderer state.

    Created on ``render_start``, evicted on ``render_end``. Parallel
    sub-graphs each get their own — the ``tool_group`` field is the
    main reason for the per-thread split (consecutive-collapsing must
    not span sub-agent boundaries).
    """
    thread_id: str
    container: VerticalScroll
    tool_group: ToolGroupState = field(default_factory=ToolGroupState)


class MessageRenderer:
    """Per-graph-stream rendering state, widget production, and mounting.

    Used by both ``BaseRichConsoleApp`` (single event-log root) and
    ``MultiJobTaskHandler`` (per-task panel root).
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
        self.threads: dict[str, ThreadRenderState] = {}
        # Anchors are flat — tool_call ids are LLM-minted UUIDs.
        self._tool_call_anchors: dict[str, Static] = {}
        # within_tool tool_call_id -> mount target. Installed by callers
        # that want sub-graphs spawned with that within_tool to land
        # somewhere specific (e.g. the Analysis Agents collapsible).
        self._within_tool_overrides: dict[str, VerticalScroll] = {}

    # ── thread lifecycle ─────────────────────────────────────

    async def render_start(
        self,
        root: VerticalScroll,
        *,
        path: list[str],
        description: str,
        tool_id: str | None,
    ) -> None:
        """Mount the workflow start banner / nested collapsible and
        register the new thread in ``self.threads``.

        The mount target for nested threads is determined by:

        1. ``self._within_tool_overrides[tool_id]`` if a routing override
           is registered for this start's ``tool_id``.
        2. Otherwise the parent thread's container (``threads[path[-2]]``).
        """
        if len(path) == 1:
            logger.debug("Starting top-level workflow: %s", description)
            banner = Static(Text(f"━━ {description} ━━", style="bold"))
            await self._mount_to(root, banner)
            self.threads[path[-1]] = ThreadRenderState(
                thread_id=path[-1], container=root,
            )
            return

        target = (
            self._within_tool_overrides.get(tool_id) if tool_id else None
        )
        if target is None:
            parent = self.threads.get(path[-2])
            target = parent.container if parent is not None else root

        inner = VerticalScroll(classes="nested-workflow")
        coll = Collapsible(inner, title=description, collapsed=True)
        await self._mount_to(target, coll)
        self.threads[path[-1]] = ThreadRenderState(
            thread_id=path[-1], container=inner,
        )

    async def render_end(self, root: VerticalScroll, *, path: list[str]) -> None:
        """Evict the thread state and either mount the end banner
        (top-level) or collapse the nested collapsible."""
        state = self.threads.pop(path[-1], None)
        if state is None:
            return
        if len(path) == 1:
            await self._mount_to(
                state.container,
                Static(Text("━━ end ━━", style="bold dim")),
            )
            return
        parent_coll = state.container.parent
        if isinstance(parent_coll, Collapsible):
            parent_coll.collapsed = True

    # ── target lookup ────────────────────────────────────────

    def get_mount_target(self, root: VerticalScroll, path: list[str]) -> VerticalScroll:
        """Resolve the mount target for a given path.

        Used by callers that need to mount their own widgets into a
        thread's container (e.g. ``CodeGenRichApp.render_progress``
        mounting the prover output pane).
        """
        state = self.threads.get(path[-1]) if path else None
        return state.container if state is not None else root

    # ── overrides ────────────────────────────────────────────

    def install_within_tool_override(
        self, tool_call_id: str, target: VerticalScroll,
    ) -> None:
        self._within_tool_overrides[tool_call_id] = target

    def clear_within_tool_override(self, tool_call_id: str) -> None:
        self._within_tool_overrides.pop(tool_call_id, None)

    # ── anchors ──────────────────────────────────────────────

    def get_tool_call_anchor(self, tool_call_id: str) -> Static | None:
        return self._tool_call_anchors.get(tool_call_id)

    # ── shared rendering helpers ─────────────────────────────

    def classify_human(self, m: HumanMessage) -> tuple[str, bool]:
        """Classify a human message for display. Returns (title, collapsed)."""
        tag = getattr(m, "display_tag", None)
        if tag is not None:
            return _HUMAN_TAG_DISPLAY.get(tag, ("User input", True))
        return ("User input", True)

    def render_ai_turn(
        self,
        msg: AIMessage,
        *,
        tool_group: ToolGroupState,
    ) -> list[Static | Collapsible]:
        """Render an AI turn as a list of widgets, mutating ``tool_group``
        for consecutive-call collapsing and writing into the renderer's
        flat ``_tool_call_anchors``."""
        widgets: list[Static | Collapsible] = []

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
                    w = render_tool_call(
                        self.tool_config,
                        tool_group,
                        name=c["name"],
                        input_args=c.get("input", {}),
                        tool_call_id=c.get("id"),
                        anchors=self._tool_call_anchors,
                    )
                    if w is not None:
                        widgets.append(w)
                case other:
                    widgets.append(Static(f"Unknown block: {other}"))

        return widgets

    def render_tool_result(
        self,
        msg: ToolMessage,
        *,
        tool_group: ToolGroupState,
    ) -> Collapsible | None:
        """Render a tool result as a collapsible, or ``None`` to suppress."""
        name = getattr(msg, "name", None) or "Tool result"
        result_info = self.tool_config.format_result(name, msg)
        if result_info is None:
            return None
        tool_group.reset()
        label, body = result_info
        return Collapsible(Static(body, markup=False), title=label, collapsed=True)

    # ── stream-message rendering ─────────────────────────────

    async def render_messages(self, path: list[str], messages: list) -> None:
        """Render a list of LangChain messages into the thread's container.

        Resolves both target and tool-group state from
        ``self.threads[path[-1]]``; if no state is registered (e.g. a
        message arrived for a thread we never saw a Start for), this is
        a no-op.
        """
        state = self.threads.get(path[-1]) if path else None
        if state is None:
            logger.debug("render_messages: no thread state for path=%s", path)
            return
        target = state.container
        group = state.tool_group
        for m in messages:
            match m:
                case AIMessage():
                    widgets = self.render_ai_turn(m, tool_group=group)
                    if widgets:
                        await self._mount_to(target, *widgets)
                    self._on_tokens(m)
                case SystemMessage():
                    group.reset()
                    coll = Collapsible(Static(m.text, markup=False), title="System prompt", collapsed=True)
                    await self._mount_to(target, coll)
                case HumanMessage():
                    group.reset()
                    title, collapsed = self.classify_human(m)
                    coll = Collapsible(Static(m.text, markup=False), title=title, collapsed=collapsed)
                    await self._mount_to(target, coll)
                case ToolMessage():
                    coll = self.render_tool_result(m, tool_group=group)
                    if coll is None:
                        continue
                    await self._mount_to(target, coll)
                case _:
                    group.reset()
                    await self._mount_to(
                        target,
                        Static(f"Unknown message type: {type(m).__name__}"),
                    )
