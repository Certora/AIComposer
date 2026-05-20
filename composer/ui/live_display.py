"""Common base for the codegen + autoprove live-display IOHandlers.

Drives an inline ``rich.live.Live`` region whose contents are derived
from a per-thread ``AgentDisplayState`` map. Scrollback (the area
above the Live region) is append-only; ``log_start`` / ``log_end``
always land there. The Live region itself shows a graceful-degradation
view of the active agent tree.

Subclasses fill in:

- ``handle_human_interaction`` — abstract.
- ``progress_update`` / ``handle_event`` — domain event ingest. Both
  ultimately call ``_update_progress`` and ``_scrollback``.
- ``derive_status`` — usually inherited as-is; override only if a
  workflow has its own descriptor logic that the default
  composition doesn't capture.
- ``format_start`` / ``format_end`` / ``format_state_update`` /
  ``handle_overflow`` — scrollback + overflow policy hooks.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Self

from langchain_core.messages import AIMessage, ToolCall as LC_ToolCall, ToolMessage

from rich.console import Console, RenderableType
from rich.live import Live
from rich.text import Text

from composer.ui.tool_display import GroupedTool, ToolDisplayConfig


# ---------------- data model ----------------


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LiveToolGroup:
    """Per-thread tool-call grouping state.

    Mirrors ``ToolGroupState`` from ``tool_call_renderer`` but persists
    across tool *results* — only a non-matching tool call resets it.
    The live region wants accumulated history across an iteration phase
    (e.g. "Reading: a, b, c, d" across four rounds of get_file), which
    the TUI's per-result reset would erase.
    """
    group: GroupedTool | None = None
    items: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.group, self.items = None, []


@dataclass
class AgentDisplayState:
    thread_id: str
    description: str
    parent_id: str | None
    spawn_tool_call_id: str | None
    children: dict[str, "AgentDisplayState"] = field(default_factory=dict)
    pending_tool_calls: dict[str, ToolCall] = field(default_factory=dict)
    tool_group: LiveToolGroup = field(default_factory=LiveToolGroup)
    progress_override: str | None = None


# ---------------- handler ----------------


class LiveDisplayHandler[H]:
    """IOHandler[H] base that drives an inline rich.live.Live region."""

    def __init__(
        self,
        *,
        tool_display_config: ToolDisplayConfig,
        reserved_rows: int | None = None,
        console: Console | None = None,
    ) -> None:
        self._tool_display_config = tool_display_config
        self._console = console or Console()
        self._reserved_rows = reserved_rows or max(6, self._console.size.height // 3)

        self._descriptions: dict[str, str] = {}
        self._agents: dict[str, AgentDisplayState] = {}
        self._roots: list[str] = []

        self._live: Live | None = None
        self._mutate_lock = asyncio.Lock()
        self._suppress_scrollback = False

    # ---------- lifecycle ----------

    async def __aenter__(self) -> Self:
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    @asynccontextmanager
    async def _paused_live(self) -> AsyncIterator[None]:
        # Whoever runs inside _paused_live (HITL conversation, etc.)
        # owns the terminal. Suppress scrollback so background graph
        # activity in other threads doesn't scramble their rendering —
        # rich and prompt-toolkit would otherwise race for stdout.
        live = self._live
        if live is not None:
            live.stop()
        prev_suppress = self._suppress_scrollback
        self._suppress_scrollback = True
        try:
            yield
        finally:
            self._suppress_scrollback = prev_suppress
            if live is not None:
                live.start()
                live.update(self._render())

    # ---------- IOHandler[H] surface ----------

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str) -> None:
        pass

    async def log_start(
        self, *, path: list[str], description: str, tool_id: str | None
    ) -> None:
        async with self._mutate_lock:
            thread_id = path[-1]
            parent_id = path[-2] if len(path) >= 2 else None

            self._descriptions[thread_id] = description
            agent = AgentDisplayState(
                thread_id=thread_id,
                description=description,
                parent_id=parent_id,
                spawn_tool_call_id=tool_id,
            )
            self._agents[thread_id] = agent
            parent = self._agents.get(parent_id) if parent_id is not None else None
            if parent is not None:
                parent.children[thread_id] = agent
            else:
                self._roots.append(thread_id)

            self._scrollback(self.format_start(path, description, tool_id))
            self._refresh()

    async def log_end(self, path: list[str]) -> None:
        async with self._mutate_lock:
            thread_id = path[-1]
            agent = self._agents.pop(thread_id, None)
            if agent is None:
                self._scrollback(self.format_end(path))
                return

            parent = (
                self._agents.get(agent.parent_id)
                if agent.parent_id is not None
                else None
            )
            if parent is not None:
                parent.children.pop(thread_id, None)
                if agent.spawn_tool_call_id is not None:
                    parent.pending_tool_calls.pop(agent.spawn_tool_call_id, None)
            elif thread_id in self._roots:
                self._roots.remove(thread_id)

            self._scrollback(self.format_end(path))
            self._refresh()

    async def log_state_update(self, path: list[str], st: dict) -> None:
        async with self._mutate_lock:
            agent = self._agents.get(path[-1])
            if agent is not None:
                self._ingest_state_update(agent, st)

            rendered = self.format_state_update(path, st)
            if rendered is not None:
                self._scrollback(rendered)
            self._refresh()

    async def human_interaction(
        self,
        ty: H,
        debug_thunk: Callable[[], None],
    ) -> str:
        async with self._paused_live():
            return await self.handle_human_interaction(ty, debug_thunk)

    # ---------- pluggable hooks ----------

    async def handle_human_interaction(
        self,
        ty: H,
        debug_thunk: Callable[[], None],
    ) -> str:
        raise NotImplementedError

    def derive_status(self, agent: AgentDisplayState) -> str:
        if agent.progress_override is not None:
            return agent.progress_override

        parts: list[str] = []
        regular_clause = self._render_regular_slots(agent)
        if regular_clause:
            parts.append(regular_clause)
        if agent.children:
            n = len(agent.children)
            parts.append(f"waiting on {n} agent{'' if n == 1 else 's'}")
        return "; ".join(parts) if parts else "thinking"

    def format_start(
        self, path: list[str], description: str, tool_id: str | None
    ) -> RenderableType | None:
        # All transient activity is already visible in the Live region
        # as the tree mounts a new node; a per-agent "start" line in
        # scrollback just pushes interesting events off the top.
        return None

    def format_end(self, path: list[str]) -> RenderableType | None:
        # Only top-level completions go to scrollback. Sub-agents finish
        # inside their parent's tree subtree — their disappearance from
        # the Live region is the signal that they're done.
        if len(path) > 1:
            return None
        return Text(f"[{self._label(path)}] ✓", style="green")

    def format_state_update(
        self, path: list[str], st: dict
    ) -> RenderableType | None:
        return None

    def handle_overflow(
        self, roots: list[AgentDisplayState], avail: int
    ) -> RenderableType:
        keep = max(1, avail - 1)
        out = Text()
        for r in roots[:keep]:
            out.append(f"● {r.description} — {self.derive_status(r)}\n")
        extra = len(roots) - keep
        if extra > 0:
            out.append(f"+{extra} more\n", style="dim")
        return out

    # ---------- helpers for subclasses ----------

    def _update_progress(self, thread_id: str, descriptor: str | None) -> None:
        agent = self._agents.get(thread_id)
        if agent is None:
            return
        agent.progress_override = descriptor
        self._refresh()

    def _scrollback(self, renderable: RenderableType | None) -> None:
        if renderable is None or self._suppress_scrollback:
            return
        if self._live is not None:
            self._live.console.print(renderable)
        else:
            self._console.print(renderable)

    def _label(self, path: list[str]) -> str:
        return " / ".join(self._descriptions.get(tid, tid) for tid in path)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    # ---------- internal: state update ingest ----------

    def _ingest_state_update(self, agent: AgentDisplayState, st: dict) -> None:
        # ``run_graph._normalize_updates_payload`` guarantees each node's
        # value is a dict by the time it reaches us, even when langgraph's
        # ToolNode produced a list-of-Commands payload internally.
        for update in st.values():
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages", []):
                if not isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage):
                    continue
                if isinstance(msg, ToolMessage):
                    agent.pending_tool_calls.pop(msg.tool_call_id, None)
                    continue
                tc_list = msg.tool_calls
                if tc_list:
                    # New AIMessage with tool calls — the prior tool's
                    # custom progress descriptor (e.g. "running prover",
                    # "analyzing CEX: ...") is stale by definition.
                    agent.progress_override = None
                    for tc in tc_list:
                        self._ingest_tool_call(agent, tc)

    def _ingest_tool_call(self, agent: AgentDisplayState, tc: LC_ToolCall) -> None:
        name = tc["name"]
        args = tc.get("args", {}) or {}
        tcid = tc["id"]

        if tcid is None:
            return

        agent.pending_tool_calls[tcid] = ToolCall(id=tcid, name=name, args=args)

        grouped = self._tool_display_config.get_group(name)
        group = agent.tool_group
        if grouped is None:
            group.reset()
            return

        raw = grouped.extract_group_items(args)
        new_items = [raw] if isinstance(raw, str) else list(raw)
        if group.group is not None and grouped.group_id == group.group.group_id:
            group.items.extend(new_items)
        else:
            group.group = grouped
            group.items = list(new_items)

    # ---------- internal: regular-slot rendering ----------

    def _render_regular_slots(self, agent: AgentDisplayState) -> str:
        spawn_ids = {c.spawn_tool_call_id for c in agent.children.values()}
        parts: list[str] = []

        group = agent.tool_group
        if group.group is not None and group.items:
            parts.append(group.group.render_group(group.items))

        for tc in agent.pending_tool_calls.values():
            if tc.id in spawn_ids:
                continue
            if self._tool_display_config.get_group(tc.name) is not None:
                continue
            parts.append(self._tool_display_config.format_tool_call(tc.name, tc.args, short=True))

        return ", ".join(parts)

    # ---------- internal: live region rendering ----------

    def _root_list(self) -> list[AgentDisplayState]:
        return [self._agents[t] for t in self._roots if t in self._agents]

    def _render(self) -> RenderableType:
        roots = self._root_list()
        if not roots:
            return Text("(no active agents)\n", style="dim")

        avail = self._reserved_rows
        if (expanded := self._render_expanded(roots, avail)) is not None:
            return expanded
        if (tight := self._render_tight(roots, avail)) is not None:
            return tight
        return self.handle_overflow(roots, avail)

    def _render_expanded(
        self, roots: list[AgentDisplayState], budget: int
    ) -> Text | None:
        if self._expanded_line_count(roots) > budget:
            return None
        out = Text()
        def walk(agent: AgentDisplayState, depth: int) -> None:
            prefix = ("  " * (depth - 1) + "└─ ") if depth else ""
            out.append(
                f"{prefix}● {agent.description} — {self.derive_status(agent)}\n"
            )
            for child in agent.children.values():
                walk(child, depth + 1)
        for r in roots:
            walk(r, 0)
        return out

    def _render_tight(
        self, roots: list[AgentDisplayState], budget: int
    ) -> Text | None:
        if len(roots) > budget:
            return None
        out = Text()
        for r in roots:
            out.append(f"● {r.description} — {self.derive_status(r)}\n")
        return out

    @staticmethod
    def _expanded_line_count(roots: list[AgentDisplayState]) -> int:
        def count(agent: AgentDisplayState) -> int:
            return 1 + sum(count(c) for c in agent.children.values())
        return sum(count(r) for r in roots)
