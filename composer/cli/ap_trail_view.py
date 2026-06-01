"""``ap-trail view`` — Textual TUI drill-down explorer for a run.

Tree pane (left): thread forest rooted at threads with ``from_tool_id is
None``. Children attach lazily — when a tree node is first expanded, its
timeline is loaded and scanned for tool_calls whose ids appear in the
run-wide ``from_tool_id`` index, and matching ThreadMeta records are
mounted under that node.

Right pane: a ``ContentSwitcher`` of ``ThreadRenderer`` instances, keyed
by ``thread_run_id``. Renderers are constructed on first visit and cached.

Two source modes:

- ``ap-trail view <run_id>`` — live DB. ``LiveRunSource`` opens store +
  checkpointer.
- ``ap-trail view --from-export <path>`` — replay. ``ReplayRunSource``
  reads the gzipped JSON.
"""

import argparse
import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Tree, ContentSwitcher, Static
from textual.widgets.tree import TreeNode

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from composer.io.run_index import (
    ExportedRun,
    decode_thread_timeline,
    get_run,
    list_threads_for_run,
    read_export,
)
from composer.io.thread_logging import RunMeta, ThreadMeta
from composer.io.thread_timeline import TimelineItem, load_timeline
from composer.ui.thread_renderer import DescendableToolCall, ThreadRenderer
from composer.workflow.services import checkpointer_context, store_context


# ---------------------------------------------------------------------------
# RunSource — uniform interface over live DB and exported file
# ---------------------------------------------------------------------------

class RunSource(Protocol):
    """Read-side abstraction shared by live and replay modes."""

    @property
    def run_id(self) -> str: ...

    @property
    def run(self) -> RunMeta: ...

    @property
    def threads(self) -> list[tuple[str, ThreadMeta]]:
        """All ``(thread_run_id, meta)`` pairs for the run, oldest-first."""
        ...

    async def load_timeline(
        self, thread_run_id: str
    ) -> list[tuple[TimelineItem, str | None]]:
        """Timeline for one thread segment, chain-walked + summarization-marked."""
        ...


@dataclass
class LiveRunSource:
    _run_id: str
    _run: RunMeta
    _threads: list[tuple[str, ThreadMeta]]
    _checkpointer: BaseCheckpointSaver

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run(self) -> RunMeta:
        return self._run

    @property
    def threads(self) -> list[tuple[str, ThreadMeta]]:
        return self._threads

    async def load_timeline(self, thread_run_id: str):
        meta = next((m for tid, m in self._threads if tid == thread_run_id), None)
        if meta is None:
            return []
        return await load_timeline(
            self._checkpointer,
            meta["thread_id"],
            anchor_checkpoint_id=meta["end_checkpoint_id"],
            stop_at_checkpoint_id=meta["start_checkpoint_id"],
        )


@dataclass
class ReplayRunSource:
    _exported: ExportedRun
    _by_id: dict[str, list[tuple[TimelineItem, str | None]]]

    @classmethod
    def from_exported(cls, exported: ExportedRun) -> "ReplayRunSource":
        return cls(
            _exported=exported,
            _by_id={t.thread_run_id: decode_thread_timeline(t) for t in exported.threads},
        )

    @property
    def run_id(self) -> str:
        return self._exported.run_id

    @property
    def run(self) -> RunMeta:
        return self._exported.run

    @property
    def threads(self) -> list[tuple[str, ThreadMeta]]:
        return [(t.thread_run_id, t.meta) for t in self._exported.threads]

    async def load_timeline(self, thread_run_id: str):
        return self._by_id.get(thread_run_id, [])


# ---------------------------------------------------------------------------
# Tree node payload
# ---------------------------------------------------------------------------

@dataclass
class _TreeNodeData:
    """What's attached to each Tree node so we can resolve back to a thread."""
    thread_run_id: str
    meta: ThreadMeta


@dataclass
class _ThreadCache:
    """Memoized result of loading + scanning a thread segment."""
    timeline: list[tuple[TimelineItem, str | None]]
    descendable_tool_call_ids: set[str]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class RunExplorerApp(App):
    """Tree + ContentSwitcher drill-down for one run's thread forest."""

    CSS = """
    #tree-pane { width: 1fr; min-width: 30; border: solid $primary; }
    #right-pane { width: 3fr; border: solid $primary; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("b", "jump_to_parent", "Parent", show=True),
        Binding("d", "descend", "Descend", show=True),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    def __init__(self, source: RunSource) -> None:
        super().__init__()
        self._source = source
        self._mounted_renderers: set[str] = set()
        # tree_node lookup by thread_run_id for parent/sibling navigation
        self._tree_nodes: dict[str, TreeNode[_TreeNodeData]] = {}
        # Set of tree nodes we've already lazy-expanded
        self._expanded: set[str] = set()
        # from_tool_id -> [ThreadMeta] index (one-time pass over run's ThreadMeta)
        self._by_from_tool: dict[str, list[tuple[str, ThreadMeta]]] = defaultdict(list)
        for tid, meta in source.threads:
            ft = meta.get("from_tool_id")
            if ft is not None:
                self._by_from_tool[ft].append((tid, meta))
        # Per-thread cache. Populated lazily on first access via
        # ``_get_thread_cache``; subsumes both tree expansion and right-pane
        # rendering so each thread is loaded + scanned exactly once.
        self._thread_cache: dict[str, _ThreadCache] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="tree-pane"):
                yield Tree("Threads", id="tree")
            with Vertical(id="right-pane"):
                # Placeholder + dynamic ThreadRenderers swapped into here.
                yield ContentSwitcher(
                    Static("Select a thread on the left.", id="placeholder"),
                    initial="placeholder",
                    id="switcher",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"ap-trail view  ({self._source.run_id})"
        self.sub_title = self._fmt_run_subtitle()
        tree: Tree[_TreeNodeData] = self.query_one("#tree", Tree)
        tree.root.set_label("(run)")
        tree.root.expand()
        roots = [
            (tid, m) for tid, m in self._source.threads if m.get("from_tool_id") is None
        ]
        for tid, meta in roots:
            node = tree.root.add(self._fmt_thread_label(meta), data=_TreeNodeData(tid, meta))
            self._tree_nodes[tid] = node

    def _fmt_run_subtitle(self) -> str:
        n_threads = len(self._source.threads)
        end = self._source.run.get("end_time")
        status = "in-progress" if end is None else "completed"
        return f"{status}  |  {n_threads} thread(s)"

    def _fmt_thread_label(self, meta: ThreadMeta) -> str:
        desc = meta.get("description") or "(no description)"
        tid = meta.get("thread_id") or ""
        return f"{desc}  [{tid}]"

    # ── Tree expansion & selection ───────────────────────────────────────

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        data: _TreeNodeData | None = event.node.data
        if data is None:
            return
        await self._ensure_expanded(event.node, data)
        await self._show_in_right_pane(data.thread_run_id)

    async def _get_thread_cache(self, thread_run_id: str) -> _ThreadCache:
        """Load + scan a thread once, then memoize for the lifetime of the app."""
        if (cached := self._thread_cache.get(thread_run_id)) is not None:
            return cached
        timeline = await self._source.load_timeline(thread_run_id)
        descendable: set[str] = set()
        for item, _ in timeline:
            if isinstance(item, AIMessage) and item.tool_calls:
                for tc in item.tool_calls:
                    tcid = tc.get("id") or ""
                    if tcid and tcid in self._by_from_tool:
                        descendable.add(tcid)
        cache = _ThreadCache(timeline=timeline, descendable_tool_call_ids=descendable)
        self._thread_cache[thread_run_id] = cache
        return cache

    async def _ensure_expanded(self, node: TreeNode[_TreeNodeData], data: _TreeNodeData) -> None:
        """Lazily attach child threads on first visit."""
        if data.thread_run_id in self._expanded:
            return
        self._expanded.add(data.thread_run_id)
        cache = await self._get_thread_cache(data.thread_run_id)
        if not cache.timeline:
            return
        for tcid in cache.descendable_tool_call_ids:
            for child_tid, child_meta in self._by_from_tool.get(tcid, []):
                if child_tid in self._tree_nodes:
                    continue
                child_node = node.add(
                    self._fmt_thread_label(child_meta),
                    data=_TreeNodeData(child_tid, child_meta),
                )
                self._tree_nodes[child_tid] = child_node
        node.expand()

    async def _show_in_right_pane(self, thread_run_id: str) -> None:
        switcher = self.query_one("#switcher", ContentSwitcher)
        if thread_run_id not in self._mounted_renderers:
            cache = await self._get_thread_cache(thread_run_id)
            renderer = ThreadRenderer(
                cache.timeline,
                descendable_tool_call_ids=cache.descendable_tool_call_ids,
                on_tool_descend=self._descend_to_child,
                id=f"r_{thread_run_id}",
            )
            await switcher.mount(renderer)
            self._mounted_renderers.add(thread_run_id)
        switcher.current = f"r_{thread_run_id}"
        # Land focus inside the newly-active renderer so the user can
        # immediately arrow/j/k around without first tabbing across panes.
        from textual.css.query import NoMatches
        try:
            renderer = switcher.query_one(f"#r_{thread_run_id}", ThreadRenderer)
            renderer.query_one("CollapsibleTitle").focus()
        except NoMatches:
            pass

    # ── Drill-down callback fed to ThreadRenderer ────────────────────────

    async def _descend_to_child(self, tool_call_id: str) -> None:
        children = self._by_from_tool.get(tool_call_id, [])
        if not children:
            self.notify(f"No child thread for tool_call {tool_call_id}")
            return
        # If multiple children (parallel sub-agents from the same tool_call),
        # jump to the first; the others sit as siblings in the tree and the
        # user can navigate to them manually.
        child_tid, _ = children[0]
        node = self._tree_nodes.get(child_tid)
        if node is None:
            self.notify("Child thread not yet attached to the tree.")
            return
        tree: Tree[_TreeNodeData] = self.query_one("#tree", Tree)
        tree.select_node(node)
        tree.scroll_to_node(node)

    # ── Misc bindings ────────────────────────────────────────────────────

    async def action_descend(self) -> None:
        """Walk the focused widget's parents looking for a DescendableToolCall.
        If found, fire its descend action."""
        node = self.focused
        while node is not None:
            if isinstance(node, DescendableToolCall):
                await node.action_descend()
                return
            node = node.parent
        self.notify("Focus a descendable tool_call first (↘ marker).", severity="information")

    def action_jump_to_parent(self) -> None:
        tree: Tree[_TreeNodeData] = self.query_one("#tree", Tree)
        node = tree.cursor_node
        if node is None or node.parent is None:
            return
        tree.select_node(node.parent)
        tree.scroll_to_node(node.parent)

    def action_scroll_home(self) -> None:
        switcher = self.query_one("#switcher", ContentSwitcher)
        current = switcher.current
        if current and current != "placeholder":
            renderer = switcher.query_one(f"#{current}", ThreadRenderer)
            renderer.scroll_home(animate=False)

    def action_scroll_end(self) -> None:
        switcher = self.query_one("#switcher", ContentSwitcher)
        current = switcher.current
        if current and current != "placeholder":
            renderer = switcher.query_one(f"#{current}", ThreadRenderer)
            renderer.scroll_end(animate=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("run_id", nargs="?", help="Run id to view (or 8-char prefix).")
    group.add_argument(
        "--from-export",
        dest="from_export",
        default=None,
        help="Replay mode: load a gzipped export produced by `ap-trail export`.",
    )
    parser.add_argument(
        "--uid",
        default=None,
        help="User id (live mode only). Defaults to AUTOPROVER_USER_ID / _anonymous.",
    )


async def _main(args: argparse.Namespace) -> int:
    if args.from_export is not None:
        exported = read_export(args.from_export)
        source: RunSource = ReplayRunSource.from_exported(exported)
        app = RunExplorerApp(source)
        await app.run_async()
        return 0

    # Live mode: keep checkpointer + store open for the lifetime of the app
    # so lazy timeline loads in the UI can hit the DB.
    async with store_context() as store, checkpointer_context() as checkpointer:
        run = await get_run(store, args.run_id, uid=args.uid)
        if run is None:
            print(f"Run {args.run_id} not found.", file=sys.stderr)
            return 1
        threads = await list_threads_for_run(store, args.run_id, uid=args.uid)
        source = LiveRunSource(
            _run_id=args.run_id,
            _run=run,
            _threads=threads,
            _checkpointer=checkpointer,
        )
        app = RunExplorerApp(source)
        await app.run_async()
    return 0


def main(args: argparse.Namespace) -> int:
    return asyncio.run(_main(args))
