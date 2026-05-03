"""Browse the conversation history of a CVL generation snapshot.

Compact by default: AI text shown inline, tool calls and results
collapsed to one-line summaries.  Expand any item to see full content.

Nested sub-agent threads are excluded — only the top-level conversation
for the snapshot's thread is shown.

Wired as ``snapshot-viewer`` in ``pyproject.toml``.

Usage::

    snapshot-viewer <mnemonic>
    snapshot-viewer --thread <thread_id>
"""

import argparse
import asyncio
import json
import pathlib
import sys

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Collapsible, Header, Footer
from textual.binding import Binding

from rich.text import Text
from rich.syntax import Syntax

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

from composer.diagnostics.handlers import normalize_content
from composer.workflow.services import checkpointer_context, store_context
from composer.spec.context import SNAPSHOT_NAMESPACE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

async def _load_thread_from_mnemonic(mnemonic: str) -> str:
    """Look up a snapshot by mnemonic and return its thread_id."""
    async with store_context() as store:
        item = await store.aget(SNAPSHOT_NAMESPACE, mnemonic)
        if item is None:
            print(f"No snapshot found for mnemonic: {mnemonic}", file=sys.stderr)
            sys.exit(1)
        return item.value["thread_id"]


async def _load_messages(
    thread_id: str,
) -> tuple[list, str | None, list[str | None]]:
    """Load messages from the latest checkpoint for a thread.

    Returns ``(messages, latest_checkpoint_id, per_message_checkpoint_id)``.
    The third value is a parallel list to ``messages``: index ``i`` holds
    the checkpoint id of the earliest checkpoint where the message with
    that id appears (``None`` for messages that never matched any
    historical checkpoint, e.g. their ``id`` was ``None``). This is what
    the per-turn copy-to-clipboard widget uses — clicking a turn copies
    the checkpoint id at which that turn was first persisted, not the
    snapshot's overall latest id.

    Robust to summarization (``RemoveMessage`` shrinks the channel and
    later turns repopulate it with new ids); the by-id match means the
    final-state turns get their *own* introducing checkpoints, not the
    pre-summary one.
    """
    async with checkpointer_context() as checkpointer:
        config = {"configurable": {"thread_id": thread_id}}
        latest = await checkpointer.aget_tuple(config)
        if latest is None:
            print(f"No checkpoint found for thread {thread_id}", file=sys.stderr)
            sys.exit(1)
        assert "configurable" in latest.config
        latest_checkpoint_id = latest.config["configurable"].get("checkpoint_id")
        messages = latest.checkpoint["channel_values"].get("messages", [])

        # Walk every checkpoint on the thread; record the message id sets at
        # each. ``alist`` is newest-first, so we reverse to walk
        # chronologically and stop updating ``seen_at[mid]`` once we've found
        # the introducing checkpoint.
        history: list[tuple[str, set[str]]] = []
        async for ct in checkpointer.alist(config):
            cid = ct.config["configurable"].get("checkpoint_id")
            if cid is None:
                continue
            ckpt_msgs = ct.checkpoint["channel_values"].get("messages", [])
            ids = {m.id for m in ckpt_msgs if getattr(m, "id", None) is not None}
            history.append((cid, ids))
        history.reverse()

        seen_at: dict[str, str] = {}
        for cid, ids in history:
            for mid in ids:
                seen_at.setdefault(mid, cid)
        per_message_checkpoint = [
            seen_at.get(m.id) if getattr(m, "id", None) is not None else None
            for m in messages
        ]
        return messages, latest_checkpoint_id, per_message_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_line(s: str, max_len: int = 100) -> str:
    """Return the first non-empty line, truncated."""
    for line in s.splitlines():
        stripped = line.strip()
        if stripped:
            if len(stripped) > max_len:
                return stripped[:max_len] + "..."
            return stripped
    return "(empty)"


def _compact_args(args: dict, max_len: int = 80) -> str:
    """One-line summary of tool call arguments."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            shown = v if len(v) <= 30 else v[:27] + "..."
            parts.append(f'{k}="{shown}"')
        elif isinstance(v, (int, float, bool)):
            parts.append(f"{k}={v}")
        elif isinstance(v, list):
            parts.append(f"{k}=[{len(v)} items]")
        elif isinstance(v, dict):
            parts.append(f"{k}={{...}}")
        else:
            parts.append(f"{k}=...")
    result = ", ".join(parts)
    if len(result) > max_len:
        return result[:max_len] + "..."
    return result


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

_CHECKPOINT_FILE = pathlib.Path("/tmp/checkpoint-id")


class TurnHeader(Static):
    """Clickable turn header. Click the row to write this turn's
    checkpoint id to ``/tmp/checkpoint-id``. We tried Textual's OSC 52
    ``copy_to_clipboard`` first; the file path is a more reliable
    fallback in terminals that swallow OSC 52 (VS Code's xterm.js
    among them, depending on settings)."""

    def __init__(self, content, checkpoint_id: str | None, **kwargs) -> None:
        super().__init__(content, **kwargs)
        self._checkpoint_id = checkpoint_id

    def on_click(self) -> None:
        if self._checkpoint_id is None:
            self.app.notify(
                "No checkpoint id available for this turn.",
                severity="warning",
            )
            return
        try:
            _CHECKPOINT_FILE.write_text(self._checkpoint_id)
        except OSError as exc:
            self.app.notify(
                f"Could not write {_CHECKPOINT_FILE}: {exc}",
                severity="error",
            )
            return
        self.app.notify(
            f"Wrote {self._checkpoint_id[:16]}... to {_CHECKPOINT_FILE}",
            severity="information",
        )


class SnapshotViewerApp(App):
    """Compact conversation viewer for CVL generation threads."""

    CSS = """
    #scroll { height: 1fr; padding: 0 2; }
    #scroll > * { margin-bottom: 1; }
    .turn-header { margin-top: 1; }
    .turn-header:hover { background: $accent 30%; }
    .tool-call { margin-left: 2; }
    .tool-result { margin-left: 2; }
    .ai-text { margin-left: 2; color: #6699cc; }
    Collapsible { background: transparent; border: none; padding: 0; }
    CollapsibleTitle { padding: 0 1; }
    Collapsible Contents { padding: 0 0 0 3; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    def __init__(
        self,
        thread_id: str,
        messages: list,
        checkpoint_id: str | None,
        per_message_checkpoint: list[str | None],
        mnemonic: str | None = None,
    ):
        super().__init__()
        self._lg_thread_id = thread_id
        self._messages = messages
        self._checkpoint_id = checkpoint_id
        self._per_message_checkpoint = per_message_checkpoint
        self._mnemonic = mnemonic

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="scroll")
        yield Footer()

    def on_mount(self) -> None:
        title_parts = [f"Thread: {self._lg_thread_id}"]
        if self._mnemonic:
            title_parts.insert(0, self._mnemonic)
        self.title = " | ".join(title_parts)

        sub_parts = [f"{len(self._messages)} messages"]
        if self._checkpoint_id:
            sub_parts.append(f"checkpoint: {self._checkpoint_id[:16]}...")
        self.sub_title = " | ".join(sub_parts)

        scroll = self.query_one("#scroll", VerticalScroll)
        widgets = self._render_all()
        scroll.mount_all(widgets)

    # ── Rendering ─────────────────────────────────────────────

    def _render_all(self) -> list[Static | Collapsible]:
        widgets: list[Static | Collapsible] = []

        # Index tool results by tool_call_id so we can pair them with calls
        tool_results: dict[str, ToolMessage] = {}
        for msg in self._messages:
            if isinstance(msg, ToolMessage):
                tool_results[msg.tool_call_id] = msg

        turn = 0
        for idx, msg in enumerate(self._messages):
            match msg:
                case SystemMessage():
                    widgets.append(self._render_system(msg, idx))
                case HumanMessage():
                    widgets.append(self._render_human(msg, idx))
                case AIMessage():
                    turn += 1
                    widgets.extend(self._render_turn(msg, idx, turn, tool_results))
                case ToolMessage():
                    pass  # rendered inline with their AI message
                case _:
                    widgets.append(Static(Text(f"[{idx}] {type(msg).__name__}", style="dim")))

        return widgets

    def _render_system(self, msg: SystemMessage, idx: int) -> Collapsible:
        content = msg.text()
        return Collapsible(
            Static(content),
            title=f"[{idx}] System ({len(content):,} chars)",
            collapsed=True,
        )

    def _render_human(self, msg: HumanMessage, idx: int) -> Collapsible:
        content = msg.text()
        tag = getattr(msg, "display_tag", None)
        tag_label = f" [{tag}]" if tag else ""
        preview = _first_line(content)
        return Collapsible(
            Static(content),
            title=f"[{idx}] Human{tag_label}: {preview}",
            collapsed=True,
        )

    def _render_turn(
        self,
        msg: AIMessage,
        idx: int,
        turn: int,
        tool_results: dict[str, ToolMessage],
    ) -> list[Static | Collapsible]:
        """Render an AI turn: header, text, tool calls with inline results."""
        widgets: list[Static | Collapsible] = []
        blocks = normalize_content(msg.content)
        n_tool_calls = len(msg.tool_calls) if msg.tool_calls else 0

        # Usage
        usage_str = ""
        if isinstance(msg.response_metadata, dict):
            u = msg.response_metadata.get("usage")
            if u:
                inp = u.get("input_tokens", 0)
                out = u.get("output_tokens", 0)
                cache_r = u.get("cache_read_input_tokens", 0)
                if inp or out:
                    parts = [f"in={inp:,}", f"out={out:,}"]
                    if cache_r:
                        parts.append(f"cached={cache_r:,}")
                    usage_str = f"  ({', '.join(parts)})"

        # Turn header — clickable; copies this turn's checkpoint id.
        header = Text()
        header.append(f"Turn {turn}", style="bold blue")
        header.append(f"  [{idx}]", style="dim")
        if n_tool_calls:
            header.append(f"  {n_tool_calls} tool call(s)", style="dim")
        header.append(usage_str, style="dim")
        header.append("  📋 click → /tmp/checkpoint-id", style="dim italic")
        ckpt_id = (
            self._per_message_checkpoint[idx]
            if idx < len(self._per_message_checkpoint)
            else None
        )
        widgets.append(TurnHeader(header, ckpt_id, classes="turn-header"))

        # Content blocks
        for block in blocks:
            match block["type"]:
                case "thinking":
                    text = block.get("thinking", "")
                    widgets.append(Collapsible(
                        Static(text),
                        title=f"  Thinking ({len(text):,} chars)",
                        collapsed=True,
                        classes="tool-call",
                    ))
                case "text":
                    text = block["text"]
                    if text.strip():
                        styled = Text(text, style="#6699cc")
                        widgets.append(Static(styled, classes="ai-text"))
                case "tool_use":
                    pass  # rendered below from tool_calls
                case other:
                    widgets.append(Static(f"  [{other}]"))

        # Tool calls + their results, paired
        for tc in msg.tool_calls or []:
            name = tc["name"]
            args = tc.get("args", {})
            tc_id = tc.get("id") or "?"
            summary = _compact_args(args)

            call_title = f"  > {name}({summary})"

            args_str = json.dumps(args, indent=2, default=str)
            widgets.append(Collapsible(
                Static(Syntax(args_str, "json", theme="monokai")),
                title=call_title,
                collapsed=True,
                classes="tool-call",
            ))

            # Inline result
            result_msg = tool_results.get(tc_id)
            if result_msg is not None:
                content = result_msg.text()
                status = getattr(result_msg, "status", "ok")
                preview = _first_line(content)

                result_title = Text()
                result_title = f"  < {name}"
                if status != "ok":
                    result_title += f" [{status}]"
                result_title += f": {preview}"

                widgets.append(Collapsible(
                    Static(content, markup=False),
                    title=result_title,
                    collapsed=True,
                    classes="tool-result",
                ))

        return widgets

    # ── Navigation ────────────────────────────────────────────

    def action_scroll_home(self) -> None:
        self.query_one("#scroll", VerticalScroll).scroll_home(animate=False)

    def action_scroll_end(self) -> None:
        self.query_one("#scroll", VerticalScroll).scroll_end(animate=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main():
    parser = argparse.ArgumentParser(
        description="Browse the conversation history of a CVL generation snapshot"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("mnemonic", nargs="?", help="Snapshot mnemonic ID")
    group.add_argument("--thread", help="Thread ID directly (skip snapshot lookup)")
    args = parser.parse_args()


    if args.thread:
        thread_id = args.thread
        mnemonic = None
    else:
        mnemonic = args.mnemonic
        print(f"Looking up snapshot: {mnemonic}...", file=sys.stderr)
        thread_id = await _load_thread_from_mnemonic(mnemonic)
        print(f"Thread ID: {thread_id}", file=sys.stderr)

    messages, checkpoint_id, per_message_checkpoint = await _load_messages(thread_id)
    if not messages:
        print("Thread has no messages.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(messages)} messages. Launching viewer...", file=sys.stderr)
    app = SnapshotViewerApp(
        thread_id, messages, checkpoint_id, per_message_checkpoint, mnemonic
    )
    await app.run_async()

def main() -> int:
    asyncio.run(_main())
    return 0

if __name__ == "__main__":
    main()
