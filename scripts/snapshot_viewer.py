"""Browse the conversation history of a CVL generation snapshot.

Compact by default: AI text shown inline, tool calls and results collapsed
to one-line summaries. Expand any item to see full content. Pre-summary
turns are recovered from the checkpoint chain and separated by visible
summarization markers.

Nested sub-agent threads are not surfaced here — use ``ap-trail view``
for the drill-down across a whole run.

Usage::

    python scripts/snapshot_viewer.py <mnemonic>
    python scripts/snapshot_viewer.py --thread <thread_id>
"""

import argparse
import asyncio
import sys

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from textual.binding import Binding

from composer.io.thread_timeline import SummarizationMarker, load_timeline
from composer.ui.thread_renderer import ThreadRenderer
from composer.workflow.services import checkpointer_context, store_context
from composer.spec.context import MNEMONIC_KEYS
from composer.io.mnemonic_store import _mnem_ns
from composer.core.user import user_data_ns


async def _resolve_thread_from_mnemonic(mnemonic: str) -> str:
    async with store_context() as store:
        item = await store.aget(_mnem_ns(user_data_ns() + MNEMONIC_KEYS), mnemonic)
        if item is None:
            print(f"No snapshot found for mnemonic: {mnemonic}", file=sys.stderr)
            sys.exit(1)
        return item.value["tid"]


class SnapshotViewerApp(App):
    """One-screen viewer: just a ThreadRenderer in a scroll pane."""

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    def __init__(
        self,
        thread_id: str,
        timeline: list,
        mnemonic: str | None,
    ) -> None:
        super().__init__()
        self._lg_thread_id = thread_id
        self._timeline = timeline
        self._mnemonic = mnemonic

    def compose(self) -> ComposeResult:
        yield Header()
        yield ThreadRenderer(self._timeline, id="renderer")
        yield Footer()

    def on_mount(self) -> None:
        title_parts = [f"Thread: {self._lg_thread_id}"]
        if self._mnemonic:
            title_parts.insert(0, self._mnemonic)
        self.title = " | ".join(title_parts)
        n_msgs = sum(1 for x, _ in self._timeline if not isinstance(x, SummarizationMarker))
        n_summaries = sum(1 for x, _ in self._timeline if isinstance(x, SummarizationMarker))
        sub = [f"{n_msgs} messages"]
        if n_summaries:
            sub.append(f"{n_summaries} summarization(s)")
        self.sub_title = " | ".join(sub)

    def action_scroll_home(self) -> None:
        self.query_one("#renderer", ThreadRenderer).scroll_home(animate=False)

    def action_scroll_end(self) -> None:
        self.query_one("#renderer", ThreadRenderer).scroll_end(animate=False)


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Browse the conversation history of a CVL generation snapshot"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("mnemonic", nargs="?", help="Snapshot mnemonic ID")
    group.add_argument("--thread", help="Thread ID directly (skip snapshot lookup)")
    args = parser.parse_args()

    if args.thread:
        thread_id = args.thread
        mnemonic: str | None = None
    else:
        mnemonic = args.mnemonic
        assert mnemonic is not None
        print(f"Looking up snapshot: {mnemonic}...", file=sys.stderr)
        thread_id = await _resolve_thread_from_mnemonic(mnemonic)
        print(f"Thread ID: {thread_id}", file=sys.stderr)

    async with checkpointer_context() as checkpointer:
        timeline = await load_timeline(checkpointer, thread_id)
    if not timeline:
        print("Thread has no messages.", file=sys.stderr)
        return 1

    n_msgs = sum(1 for x, _ in timeline if not isinstance(x, SummarizationMarker))
    n_summaries = sum(1 for x, _ in timeline if isinstance(x, SummarizationMarker))
    summary_note = f" across {n_summaries} summarization(s)" if n_summaries else ""
    print(f"Loaded {n_msgs} messages{summary_note}. Launching viewer...", file=sys.stderr)

    app = SnapshotViewerApp(thread_id, timeline, mnemonic)
    await app.run_async()
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
