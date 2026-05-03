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
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Collapsible, Header, Footer
from textual.binding import Binding

from rich.text import Text
from rich.syntax import Syntax

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import CheckpointTuple

from composer.diagnostics.handlers import normalize_content
from composer.workflow.services import checkpointer_context, store_context
from composer.spec.context import SNAPSHOT_NAMESPACE


# ---------------------------------------------------------------------------
# Timeline shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SummarizationMarker:
    """A point in the rendered timeline where summarization wiped the message
    channel and replaced it with a fresh system + initial-prompt + resume
    triple. We surface these so pre-summary turns stay visible — the
    checkpoint chain itself is intact, the latest-state ``messages`` view just
    loses everything dropped by the summarizer."""

    checkpoint_id: str


type TimelineItem = BaseMessage | SummarizationMarker


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
    checkpoint_id: str | None = None,
) -> tuple[list[TimelineItem], str | None, list[str | None]]:
    """Walk the checkpoint chain for ``thread_id``, return a timeline
    that includes pre-summary turns that the latest-state view has dropped.

    When ``checkpoint_id`` is supplied, the walk is truncated at that
    checkpoint — useful for inspecting the conversation as it stood at a
    specific point in time. Without it, the walk runs to the latest
    checkpoint.

    Returns ``(timeline, anchor_checkpoint_id, checkpoint_for_item)``:

    * ``timeline`` — chronological list of every ``BaseMessage`` that was
      checkpointed up to the anchor (deduped by message id), with
      ``SummarizationMarker`` entries inserted at the points where a
      summarization round wiped the channel.
    * ``anchor_checkpoint_id`` — id of the requested checkpoint, or the
      latest one when none was requested.
    * ``checkpoint_for_item`` — parallel to ``timeline``: for each
      message, the id of the checkpoint that first persisted it; for
      summarization markers, ``None``.

    Summarization detection: a checkpoint is a summarization point iff
    its message-id set is disjoint from the previous checkpoint's
    non-empty message-id set. The summarizer always RemoveMessages all
    prior messages and inserts three fresh ones (system + initial +
    resume); the disjoint-id signature is exact.
    """
    async with checkpointer_context() as checkpointer:
        anchor_config: dict = {"configurable": {"thread_id": thread_id}}
        if checkpoint_id is not None:
            anchor_config["configurable"]["checkpoint_id"] = checkpoint_id
        anchor = await checkpointer.aget_tuple(anchor_config)
        if anchor is None:
            if checkpoint_id is not None:
                print(
                    f"No checkpoint {checkpoint_id} found on thread {thread_id}",
                    file=sys.stderr,
                )
            else:
                print(f"No checkpoint found for thread {thread_id}", file=sys.stderr)
            sys.exit(1)
        assert "configurable" in anchor.config
        anchor_checkpoint_id = anchor.config["configurable"].get("checkpoint_id")

        # The thread's checkpoint table is a forest — every restart from a
        # non-tip checkpoint forks a new branch that shares the thread_id
        # with the original. ``alist`` returns the union, which is wrong:
        # it would surface messages from branches we abandoned. Walk the
        # parent chain from the anchor instead, which gives us exactly the
        # lineage that produced the anchor state.
        #
        # Pre-fetch the whole forest by id so the parent walk is in-memory
        # rather than one DB round trip per hop.
        list_config = {"configurable": {"thread_id": thread_id}}
        by_id: dict[str, CheckpointTuple] = {}
        async for ct in checkpointer.alist(list_config):
            cid = ct.config["configurable"].get("checkpoint_id")
            if cid is not None:
                by_id[cid] = ct

        history: list[tuple[str, list[BaseMessage]]] = []
        current_ct = anchor
        while current_ct is not None:
            cid = current_ct.config["configurable"].get("checkpoint_id")
            if cid is None:
                break
            ckpt_msgs = current_ct.checkpoint["channel_values"].get("messages", [])
            history.append((cid, ckpt_msgs))
            parent_cfg = current_ct.parent_config
            if parent_cfg is None:
                break
            parent_cid = parent_cfg.get("configurable", {}).get("checkpoint_id")
            if parent_cid is None:
                break
            current_ct = by_id.get(parent_cid)
        history.reverse()

        timeline: list[TimelineItem] = []
        checkpoint_for_item: list[str | None] = []
        seen_ids: set[str] = set()
        prev_ids: set[str] = set()

        for cid, msgs in history:
            curr_ids = {m.id for m in msgs if getattr(m, "id", None) is not None}

            # Summarization signature: prev was non-empty, curr is
            # non-empty, the two id-sets are disjoint. The intersection
            # check rules out normal checkpoint-to-checkpoint shrinkage
            # (e.g. RemoveMessage on a single id), which would still
            # carry over surviving messages.
            if prev_ids and curr_ids and prev_ids.isdisjoint(curr_ids):
                timeline.append(SummarizationMarker(checkpoint_id=cid))
                checkpoint_for_item.append(None)

            for m in msgs:
                mid = getattr(m, "id", None)
                if mid is not None and mid in seen_ids:
                    continue
                if mid is not None:
                    seen_ids.add(mid)
                timeline.append(m)
                checkpoint_for_item.append(cid)

            prev_ids = curr_ids

        return timeline, anchor_checkpoint_id, checkpoint_for_item


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
        Binding("r", "refresh", "Refresh", show=True),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    def __init__(
        self,
        thread_id: str,
        timeline: list[TimelineItem],
        checkpoint_id: str | None,
        checkpoint_for_item: list[str | None],
        mnemonic: str | None = None,
        pinned_checkpoint: str | None = None,
    ):
        super().__init__()
        self._lg_thread_id = thread_id
        self._timeline = timeline
        self._checkpoint_id = checkpoint_id
        self._checkpoint_for_item = checkpoint_for_item
        self._mnemonic = mnemonic
        # When set, refresh re-fetches at this exact checkpoint instead of
        # tracking the latest. Lets you re-render the same point-in-time
        # view after the harness has continued past it.
        self._pinned_checkpoint = pinned_checkpoint

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="scroll")
        yield Footer()

    def on_mount(self) -> None:
        title_parts = [f"Thread: {self._lg_thread_id}"]
        if self._mnemonic:
            title_parts.insert(0, self._mnemonic)
        self.title = " | ".join(title_parts)
        self._update_subtitle()

        scroll = self.query_one("#scroll", VerticalScroll)
        widgets = self._render_all()
        scroll.mount_all(widgets)

    def _update_subtitle(self) -> None:
        n_msgs = sum(
            1 for x in self._timeline if not isinstance(x, SummarizationMarker)
        )
        n_summaries = sum(
            1 for x in self._timeline if isinstance(x, SummarizationMarker)
        )
        sub_parts = [f"{n_msgs} messages"]
        if n_summaries:
            sub_parts.append(f"{n_summaries} summarization(s)")
        if self._checkpoint_id:
            label = "pinned" if self._pinned_checkpoint else "checkpoint"
            sub_parts.append(f"{label}: {self._checkpoint_id[:16]}...")
        self.sub_title = " | ".join(sub_parts)

    async def action_refresh(self) -> None:
        """Re-fetch the thread's full chain and replace the rendered timeline.
        Useful when watching a thread that's still being written to."""
        prev_count = len(self._timeline)
        timeline, checkpoint_id, ckpt_for_item = await _load_messages(
            self._lg_thread_id, checkpoint_id=self._pinned_checkpoint
        )
        self._timeline = timeline
        self._checkpoint_id = checkpoint_id
        self._checkpoint_for_item = ckpt_for_item
        self._update_subtitle()

        scroll = self.query_one("#scroll", VerticalScroll)
        await scroll.remove_children()
        await scroll.mount_all(self._render_all())

        delta = len(timeline) - prev_count
        if delta > 0:
            self.notify(f"Refreshed: +{delta} new item(s) ({len(timeline)} total)")
            scroll.scroll_end(animate=False)
        else:
            self.notify(f"Refreshed: no new items ({len(timeline)} total)")

    # ── Rendering ─────────────────────────────────────────────

    def _render_all(self) -> list[Static | Collapsible]:
        widgets: list[Static | Collapsible] = []

        # Index tool results by tool_call_id so we can pair them with calls.
        # Pairing across summary epochs is fine — tool_call_ids are unique.
        tool_results: dict[str, ToolMessage] = {}
        for item in self._timeline:
            if isinstance(item, ToolMessage):
                tool_results[item.tool_call_id] = item

        turn = 0
        for idx, item in enumerate(self._timeline):
            match item:
                case SummarizationMarker():
                    widgets.append(self._render_summarization(item, idx))
                case SystemMessage():
                    widgets.append(self._render_system(item, idx))
                case HumanMessage():
                    widgets.append(self._render_human(item, idx))
                case AIMessage():
                    turn += 1
                    widgets.extend(self._render_turn(item, idx, turn, tool_results))
                case ToolMessage():
                    pass  # rendered inline with their AI message
                case _:
                    widgets.append(Static(Text(f"[{idx}] {type(item).__name__}", style="dim")))

        return widgets

    def _render_summarization(
        self, marker: SummarizationMarker, idx: int
    ) -> Static:
        """Render a summarization boundary as a horizontal-rule separator
        with the checkpoint id of the post-summary state. Everything above
        belongs to a prior summary epoch."""
        line = Text()
        line.append("─" * 60 + "\n", style="yellow")
        line.append(f"[{idx}] Summarization", style="bold yellow")
        line.append(
            f"  (post-summary checkpoint {marker.checkpoint_id[:16]}...)",
            style="dim",
        )
        line.append("\n" + "─" * 60, style="yellow")
        return Static(line)

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
            self._checkpoint_for_item[idx]
            if idx < len(self._checkpoint_for_item)
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
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Pin the viewer to a specific checkpoint id; the timeline is "
            "rendered as it stood at that point. Without this flag, the "
            "latest checkpoint is used and refresh tracks the head."
        ),
    )
    args = parser.parse_args()


    if args.thread:
        thread_id = args.thread
        mnemonic = None
    else:
        mnemonic = args.mnemonic
        print(f"Looking up snapshot: {mnemonic}...", file=sys.stderr)
        thread_id = await _load_thread_from_mnemonic(mnemonic)
        print(f"Thread ID: {thread_id}", file=sys.stderr)

    timeline, checkpoint_id, checkpoint_for_item = await _load_messages(
        thread_id, checkpoint_id=args.checkpoint
    )
    if not timeline:
        print("Thread has no messages.", file=sys.stderr)
        sys.exit(1)

    n_msgs = sum(1 for x in timeline if not isinstance(x, SummarizationMarker))
    n_summaries = sum(1 for x in timeline if isinstance(x, SummarizationMarker))
    summary_note = f" across {n_summaries} summarization(s)" if n_summaries else ""
    pin_note = f" (pinned at {args.checkpoint[:16]}...)" if args.checkpoint else ""
    print(
        f"Loaded {n_msgs} messages{summary_note}{pin_note}. Launching viewer...",
        file=sys.stderr,
    )
    app = SnapshotViewerApp(
        thread_id,
        timeline,
        checkpoint_id,
        checkpoint_for_item,
        mnemonic,
        pinned_checkpoint=args.checkpoint,
    )
    await app.run_async()

def main() -> int:
    asyncio.run(_main())
    return 0

if __name__ == "__main__":
    main()
