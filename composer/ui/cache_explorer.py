"""
Base Cache & Memory Explorer TUI.

Provides a generic two-pane Textual app for browsing WorkflowContext cache
hierarchies and memory filesystems. Workflow-specific scripts supply the
tree builder and value formatter.

See scripts/cache_explorer.py for the NatSpec pipeline entry point.
"""

from typing import Callable, Awaitable, AsyncIterator, Iterator
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar

from pydantic import BaseModel

from langgraph.store.base import BaseStore

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Tree, Static, Label, TextArea
from textual.widgets.tree import TreeNode

from composer.spec.context import WorkflowContext, CacheKey, CacheTypes, Marker


# ---------------------------------------------------------------------------
# Dummy Services (all methods throw — we only use WorkflowContext for
# cache/memory namespace derivation, never for actual tool access)
# ---------------------------------------------------------------------------

class DummyServices:
    """All methods raise — we never access tools, only namespace derivation."""

    def __getattr__(self, name: str):
        raise RuntimeError(f"DummyServices: {name}() not available in explorer")


# ---------------------------------------------------------------------------
# Cache node model
# ---------------------------------------------------------------------------

from typing import TypeVar, Generic

CACHE_V = TypeVar("CACHE_V", covariant=True)

@dataclass
class OrgNode(Generic[CACHE_V]):
    label: str
    children: list["CacheTreeNode[CACHE_V]"] = field(default_factory=list)


# A storage slot in the langgraph store: ``(namespace_tuple, key)``.
type StoreSlot = tuple[tuple[str, ...], str]


@dataclass
class CacheNode(Generic[CACHE_V]):
    """A node in the typed cache hierarchy \u2014 its slot and memory namespace
    are derived from the ``WorkflowContext`` it carries. Use this for
    anything reachable via ``WorkflowContext.child(...)`` chains."""
    label: str
    ctx: WorkflowContext
    value: CACHE_V | None = None
    children: list["CacheTreeNode[CACHE_V]"] = field(default_factory=list)


@dataclass
class StoreNode(Generic[CACHE_V]):
    """A node pointing at a raw langgraph-store slot that doesn't live in
    the typed cache hierarchy \u2014 e.g. the StubRegistry / FileRegistry KV
    entries, which the pipeline writes via ``store.aput`` directly rather
    than through a ``WorkflowContext`` chain. Carries the slot explicitly
    instead of synthesizing a fake context just to smuggle it through."""
    label: str
    slot: StoreSlot
    value: CACHE_V | None = None
    children: list["CacheTreeNode[CACHE_V]"] = field(default_factory=list)


type CacheTreeNode[V] = CacheNode[V] | StoreNode[V] | OrgNode[V]


def node_slot[V](node: CacheTreeNode[V]) -> StoreSlot | None:
    """The ``(namespace, key)`` storage slot a node points at, if any.
    ``OrgNode`` returns ``None``; ``CacheNode`` projects from its
    ``WorkflowContext``; ``StoreNode`` returns its explicit slot."""
    match node:
        case OrgNode():
            return None
        case StoreNode(slot=slot):
            return slot
        case CacheNode(ctx=ctx):
            ns = ctx.cache_namespace
            if ns is None or len(ns) < 1:
                return None
            return (ns[:-1], ns[-1])


def node_memory_ns[V](node: CacheTreeNode[V]) -> str | None:
    """The memory namespace a node is associated with, if any. Only
    ``CacheNode`` carries one (via its ``WorkflowContext``)."""
    match node:
        case CacheNode(ctx=ctx):
            return ctx.memory_namespace
        case _:
            return None


def icon[V](node: CacheTreeNode[V]) -> str:
    match node:
        case OrgNode():
            return "\u25B7"
        case CacheNode() | StoreNode():
            return "\u2713" if node.value is not None else "\u25cb"


def node_label[V](node: CacheTreeNode[V]) -> str:
    return f"{icon(node)}  {node.label}"


# ---------------------------------------------------------------------------
# Tree-building helpers
#
# Workflow-specific explorers assemble their cache tree from these. The tree is
# built imperatively: ``node``/``section``/``node_for`` push a parent onto a
# ContextVar-backed stack so nested ``async with`` blocks attach to the right
# parent, while ``leaf``/``memory`` produce leaf nodes that an async generator
# yields out for ``collect_tree`` to drain into the current parent. (A generator
# that ``.set()``s a ContextVar leaks the change to its driver, which is exactly
# how the pushed parent reaches ``collect_tree``'s ``async for``.)
# ---------------------------------------------------------------------------

_node_context: ContextVar["CacheTreeNode | None"] = ContextVar("_node_context", default=None)


@contextmanager
def node(c: "CacheTreeNode") -> Iterator[None]:
    """Push ``c`` as the current parent for the block, attaching it to the
    enclosing parent (if any)."""
    prev = _node_context.get()
    if prev is not None:
        prev.children.append(c)
    tok = _node_context.set(c)
    try:
        yield
    finally:
        _node_context.reset(tok)


@contextmanager
def section(label: str) -> Iterator[None]:
    """A purely-organizational parent node with no backing cache slot."""
    with node(OrgNode(label)):
        yield


@asynccontextmanager
async def node_for[T: CacheTypes, S: CacheTypes](
    ctx: WorkflowContext[T],
    child: CacheKey[T, S],
    label: str,
    ty: type[S] | None = None,
) -> AsyncIterator[WorkflowContext[S]]:
    """A cache-backed parent node, yielding the child context for nesting. When
    ``ty`` is given the node's value is fetched from the cache for display."""
    child_ctx = ctx.child(child)
    value: S | None = await child_ctx.cache_get(ty) if ty is not None else None
    with node(CacheNode(label=label, ctx=child_ctx, value=value)):
        yield child_ctx


async def leaf[T: CacheTypes, S: BaseModel](
    ctx: WorkflowContext[T],
    child: CacheKey[T, S],
    label: str,
    ty: type[S],
) -> CacheNode[S]:
    """A cache-backed leaf node with its value fetched eagerly."""
    child_ctx = ctx.child(child)
    value: S | None = await child_ctx.cache_get(ty)
    return CacheNode[S](label=label, value=value, ctx=child_ctx)


def memory[T: CacheTypes, S: Marker](
    ctx: WorkflowContext[T], child: CacheKey[T, S], label: str,
) -> CacheNode[S]:
    """A node pointing at a memory marker (no cached value to fetch)."""
    return CacheNode[S](label=label, value=None, ctx=ctx.child(child))


async def collect_tree[V](
    root_label: str,
    root_ctx: WorkflowContext,
    nodes: AsyncIterator["CacheTreeNode[V]"],
) -> CacheNode[V]:
    """Drain an async generator of leaf nodes into a tree rooted at
    ``root_label``, honoring the ``node``/``section``/``node_for`` parent stack."""
    root: CacheNode[V] = CacheNode(label=root_label, ctx=root_ctx)
    with node(root):
        async for n in nodes:
            curr = _node_context.get()
            assert curr is not None
            curr.children.append(n)
    return root


# ---------------------------------------------------------------------------
# Memory browsing
# ---------------------------------------------------------------------------

def _get_memory_backend(memory_ns: str):
    """Get a MemoryBackend for the given namespace."""
    from composer.workflow.services import get_memory
    return get_memory(memory_ns)


@dataclass
class MemoryFile:
    """A file in the memory filesystem."""
    path: str
    name: str
    is_dir: bool
    children: list["MemoryFile"] = field(default_factory=list)


def _list_memory_tree(memory_ns: str, root: str = "/memories") -> list[MemoryFile]:
    """Recursively list the memory filesystem for a namespace."""
    backend = _get_memory_backend(memory_ns)
    results: list[MemoryFile] = []
    try:
        entries = list(backend.list_dir(root))
    except Exception:
        return results

    for name, is_dir in entries:
        path = f"{root}/{name}"
        node = MemoryFile(path=path, name=name, is_dir=is_dir)
        if is_dir:
            node.children = _list_memory_tree(memory_ns, path)
        results.append(node)
    return results


# ---------------------------------------------------------------------------
# TUI App
# ---------------------------------------------------------------------------

class CacheExplorerApp[V](App):
    TITLE = "Cache Explorer"

    CSS = """
    #tree-pane {
        width: 1fr;
        min-width: 30;
        border: solid $primary;
    }
    #detail-pane {
        width: 2fr;
        border: solid $primary;
        padding: 1;
    }
    #detail-content {
        width: 1fr;
    }
    #memory-pane {
        width: 2fr;
    }
    #memory-tree-pane {
        width: 1fr;
        border: solid $secondary;
    }
    #memory-file-pane {
        width: 2fr;
        border: solid $secondary;
        padding: 1;
    }
    #memory-editor {
        height: 1fr;
    }
    #status-line {
        dock: top;
        height: 1;
        background: $surface;
        padding: 0 1;
    }
    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("d", "delete_entry", "Delete", show=True),
        Binding("r", "refresh_tree", "Refresh", show=True),
        Binding("m", "toggle_tab", "Cache/Memory", show=True),
        Binding("e", "edit_memory", "Edit", show=True),
        Binding("ctrl+s", "save_memory", "Save", show=True),
        Binding("escape", "cancel_edit", "Cancel", show=False),
        Binding("q", "quit_app", "Quit", show=True),
    ]

    def __init__(
        self,
        build_tree: Callable[[], Awaitable[CacheNode[V]]],
        format_value: Callable[[V], list[str]],
        store: BaseStore,
        status: str,
    ):
        super().__init__()
        self._build_tree = build_tree
        self._format_value = format_value
        self._store = store
        self._status = status
        self._cache_root = build_tree()
        self._showing_memory = False
        self._selected_node: CacheNode[V] | StoreNode[V] | None = None
        self._editing = False
        self._editing_file: str | None = None
        self._editing_ns: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("", id="status-line")
        with Horizontal():
            with Vertical(id="tree-pane"):
                yield Tree("Cache", id="cache-tree")
            with VerticalScroll(id="detail-pane"):
                yield Static("Select a node to view details", id="detail-content", markup=False)
            with Vertical(id="memory-pane", classes="hidden"):
                with Horizontal():
                    with Vertical(id="memory-tree-pane"):
                        yield Tree("Memory", id="memory-tree")
                    with Vertical(id="memory-file-pane"):
                        yield Static("Select a memory file", id="memory-content")
                        yield TextArea(id="memory-editor", classes="hidden")
        yield Footer()

    async def on_mount(self) -> None:
        await self._build_tree_widget()
        self.query_one("#status-line", Label).update(self._status)

    async def _build_tree_widget(self) -> None:
        tree : Tree[CacheTreeNode[V]] = self.query_one("#cache-tree", Tree)
        tree.clear()
        tree.root.data = (root_data := await self._cache_root)
        tree.root.set_label(node_label(root_data))
        self._populate_children(tree.root, root_data)
        tree.root.expand()

    def _populate_children(self, tree_node: TreeNode[CacheTreeNode[V]], cache_node: CacheTreeNode[V]) -> None:
        for child in cache_node.children:
            as_node_data = child if isinstance(child, (CacheNode, StoreNode)) else None
            if child.children:
                branch = tree_node.add(node_label(child), data=as_node_data)
                self._populate_children(branch, child)
            else:
                tree_node.add_leaf(node_label(child), data=as_node_data)

    def _format_detail(self, node: CacheNode[V] | StoreNode[V]) -> str:
        lines = [f"[{node.label}]"]
        slot = node_slot(node)
        if slot is not None:
            ns, key = slot
            lines.append(f"Namespace: {ns}")
            lines.append(f"Key: {key}")
        else:
            lines.append("Namespace: (no backing slot)")
        mem_ns = node_memory_ns(node)
        if mem_ns:
            lines.append(f"Memory NS: {mem_ns}")
        lines.append("")

        val_raw = node.value
        val = val_raw
        if val is None:
            lines.append("No cached value")
            return "\n".join(lines)


        lines.append(f"Type: {type(val).__name__}")
        lines.extend(self._format_value(val))
        return "\n".join(lines)

    def _show_detail(self, node: CacheNode[V] | StoreNode[V]) -> None:
        self.query_one("#detail-content", Static).update(self._format_detail(node))

    def _show_memory(self, node: CacheNode[V] | StoreNode[V]) -> None:
        self._cancel_edit_mode()
        mem_ns = node_memory_ns(node)
        content_widget = self.query_one("#memory-content", Static)
        mem_tree = self.query_one("#memory-tree", Tree)
        mem_tree.clear()

        if not mem_ns:
            content_widget.update("Memory browsing disabled (no --memory-ns provided)")
            return

        mem_tree.root.set_label(f"[{mem_ns}]")
        entries = _list_memory_tree(mem_ns)
        if not entries:
            content_widget.update(f"No memory files for namespace: {mem_ns}")
            return

        self._populate_memory_tree(mem_tree.root, entries)
        mem_tree.root.expand()
        content_widget.update("Select a memory file to view")

    def _populate_memory_tree(self, tree_node: TreeNode, entries: list[MemoryFile]) -> None:
        for entry in entries:
            if entry.is_dir:
                branch = tree_node.add(f"\U0001f4c1 {entry.name}", data=entry)
                self._populate_memory_tree(branch, entry.children)
            else:
                tree_node.add_leaf(f"\U0001f4c4 {entry.name}", data=entry)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        tree_id = event.node.tree.id
        if tree_id == "cache-tree":
            node: CacheNode[V] | StoreNode[V] | None = event.node.data
            self._selected_node = node
            if node is None:
                return
            if self._showing_memory:
                self._show_memory(node)
            else:
                self._show_detail(node)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        tree_id = event.node.tree.id
        if tree_id == "memory-tree":
            self._on_memory_node_selected(event)

    def _on_memory_node_selected(self, event: Tree.NodeSelected) -> None:
        entry: MemoryFile | None = event.node.data
        if entry is None or entry.is_dir:
            return
        self._cancel_edit_mode()
        self._editing_file = entry.path
        if self._selected_node:
            self._editing_ns = node_memory_ns(self._selected_node)
        backend = _get_memory_backend(self._editing_ns) if self._editing_ns else None
        if backend is None:
            return
        content = backend.view(entry.path, None)
        self.query_one("#memory-content", Static).update(
            f"--- {entry.path} ---\n\n{content}"
        )

    def action_toggle_tab(self) -> None:
        self._showing_memory = not self._showing_memory
        detail = self.query_one("#detail-pane")
        memory = self.query_one("#memory-pane")

        if self._showing_memory:
            detail.add_class("hidden")
            memory.remove_class("hidden")
            if self._selected_node:
                self._show_memory(self._selected_node)
        else:
            memory.add_class("hidden")
            detail.remove_class("hidden")
            if self._selected_node:
                self._show_detail(self._selected_node)

    async def action_delete_entry(self) -> None:
        # In memory view, `d` deletes the currently-selected memory file.
        # In cache view, it deletes the selected cache slot. Same key, two
        # contexts — the visible tree disambiguates for the user.
        if self._showing_memory:
            self._delete_memory_file()
            return

        if self._selected_node is None:
            self.notify("No node selected", severity="warning")
            return
        node = self._selected_node
        if node.value is None:
            self.notify("No cached value to delete", severity="warning")
            return

        slot = node_slot(node)
        if slot is None:
            self.notify("Node has no backing slot to delete", severity="error")
            return
        ns, key = slot
        await self._store.adelete(ns, key)
        node.value = None
        self.notify(f"Deleted: {node.label}")

        tree = self.query_one("#cache-tree", Tree)
        self._update_tree_node_label(tree.root, node)
        self._show_detail(node)

    def _delete_memory_file(self) -> None:
        if not self._editing_file or not self._editing_ns:
            self.notify("Select a memory file first", severity="warning")
            return
        if self._editing:
            self._cancel_edit_mode()
        backend = _get_memory_backend(self._editing_ns)
        path = self._editing_file
        backend.delete(path)
        self.notify(f"Deleted memory file: {path}")
        # Refresh the memory pane so the deleted file disappears from the tree.
        self._editing_file = None
        if self._selected_node is not None:
            self._show_memory(self._selected_node)
        self.query_one("#memory-content", Static).update("Select a memory file")

    def _update_tree_node_label(self, tree_node: TreeNode, target: CacheNode[V] | StoreNode[V]) -> bool:
        if tree_node.data is target:
            tree_node.set_label(node_label(target))
            return True
        for child in tree_node.children:
            if self._update_tree_node_label(child, target):
                return True
        return False

    async def action_refresh_tree(self) -> None:
        self._cache_root = self._build_tree()
        self._selected_node = None
        await self._build_tree_widget()
        self.query_one("#detail-content", Static).update("Tree refreshed. Select a node.")
        self.notify("Tree refreshed")

    def _cancel_edit_mode(self) -> None:
        if not self._editing:
            return
        self._editing = False
        self._editing_file = None
        self._editing_ns = None
        editor = self.query_one("#memory-editor", TextArea)
        editor.add_class("hidden")
        self.query_one("#memory-content", Static).remove_class("hidden")

    def action_edit_memory(self) -> None:
        if not self._showing_memory:
            self.notify("Switch to memory view first (m)", severity="warning")
            return
        if not self._editing_file or not self._editing_ns:
            self.notify("Select a memory file first", severity="warning")
            return

        backend = _get_memory_backend(self._editing_ns)
        # ``backend.view`` decorates every line with a ``"   N: "`` gutter for
        # the LLM's benefit. Loading that into the editor and saving back
        # through ``backend.create`` would persist the gutter into the file
        # and re-gutter it on the next round-trip. Read raw bytes instead.
        content = backend.read_file(self._editing_file) or ""

        self._editing = True
        self.query_one("#memory-content", Static).add_class("hidden")
        editor = self.query_one("#memory-editor", TextArea)
        editor.load_text(content)
        editor.remove_class("hidden")
        editor.focus()
        self.notify(f"Editing: {self._editing_file}  (ctrl+s=save, escape=cancel)")

    def action_save_memory(self) -> None:
        if not self._editing or not self._editing_file or not self._editing_ns:
            self.notify("Not editing", severity="warning")
            return
        editor = self.query_one("#memory-editor", TextArea)
        content = editor.text
        backend = _get_memory_backend(self._editing_ns)
        backend.create(self._editing_file, content)
        self.notify(f"Saved: {self._editing_file}")
        self._cancel_edit_mode()
        if self._selected_node:
            self._show_memory(self._selected_node)

    def action_cancel_edit(self) -> None:
        if self._editing:
            self._cancel_edit_mode()
            self.notify("Edit cancelled")

    def action_quit_app(self) -> None:
        self.exit()
