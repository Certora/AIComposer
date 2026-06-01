"""Chain-walk timeline loader for thread debugging.

Walks a thread's checkpoint chain from an anchor backward, returning a
deduplicated list of messages with ``SummarizationMarker`` entries inserted
at points where the summarizer wiped the message channel.

Used by ``ap-trail view`` (one segment at a time, bounded by the parent
ThreadMeta's checkpoint range) and by ``snapshot-viewer`` (entire chain).
"""

from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple


@dataclass(frozen=True)
class SummarizationMarker:
    """A timeline divider for a checkpoint where summarization wiped the
    message channel. Pre-summary turns above the marker survived the walk
    because the checkpoint chain itself is intact — only the latest-state
    view of ``messages`` loses them.
    """

    checkpoint_id: str


type TimelineItem = BaseMessage | SummarizationMarker


async def load_timeline(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
    *,
    anchor_checkpoint_id: str | None = None,
    stop_at_checkpoint_id: str | None = None,
) -> list[tuple[TimelineItem, str | None]]:
    """Walk ``thread_id``'s checkpoint chain backward from ``anchor_checkpoint_id``
    (or the latest checkpoint when None), returning a chronological timeline.

    Each entry is paired with the id of the checkpoint that first persisted it
    (``None`` for ``SummarizationMarker``).

    ``stop_at_checkpoint_id``: when set, the walk halts at that checkpoint
    (inclusive), so the returned timeline covers exactly the segment that
    began there. Used by ``ap-trail view`` to render one ThreadMeta segment
    at a time when a thread was resumed/re-entered.

    Summarization detection: a checkpoint is a summarization point iff its
    message-id set is disjoint from the previous checkpoint's non-empty
    message-id set. The summarizer ``RemoveMessages`` everything and inserts
    a fresh system+initial+resume triple, so the disjoint-id signature is
    exact. Normal single-message removals (which keep most ids) don't trip it.
    """
    anchor_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    if anchor_checkpoint_id is not None:
        anchor_config["configurable"]["checkpoint_id"] = anchor_checkpoint_id
    anchor = await checkpointer.aget_tuple(anchor_config)
    if anchor is None:
        return []

    # The checkpoint table for a thread is a forest — restarts from a non-tip
    # checkpoint fork new branches that share the thread_id. ``alist`` returns
    # the union, which would surface messages from abandoned branches.
    # Pre-fetch everything by id, then walk the parent chain from the anchor:
    # the alternative (one ``aget_tuple`` per hop) costs an RTT per step.
    list_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    by_id: dict[str, CheckpointTuple] = {}
    async for ct in checkpointer.alist(list_config):
        if "configurable" not in ct.config:
            continue
        cid = ct.config["configurable"].get("checkpoint_id")
        if cid is not None:
            by_id[cid] = ct

    history: list[tuple[str, list[BaseMessage]]] = []
    current_ct: CheckpointTuple | None = anchor
    while current_ct is not None:
        cid = current_ct.config.get("configurable", {}).get("checkpoint_id")
        if cid is None:
            break
        ckpt_msgs = current_ct.checkpoint["channel_values"].get("messages", [])
        history.append((cid, ckpt_msgs))
        if cid == stop_at_checkpoint_id:
            break
        parent_cfg = current_ct.parent_config
        if parent_cfg is None:
            break
        parent_cid = parent_cfg.get("configurable", {}).get("checkpoint_id")
        if parent_cid is None:
            break
        # Common case: pre-loaded by ``alist``. Fallback: if ``alist`` was
        # bounded (large windows, partial TTL eviction, exotic checkpointer
        # implementation), pay a one-off RTT to fetch the parent directly.
        # Without this, the walk silently truncates at the alist boundary.
        current_ct = by_id.get(parent_cid)
        if current_ct is None:
            current_ct = await checkpointer.aget_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": parent_cid}}
            )
    history.reverse()

    timeline: list[tuple[TimelineItem, str | None]] = []
    seen_ids: set[str] = set()
    prev_ids: set[str] = set()

    for cid, msgs in history:
        curr_ids = {m_id for m in msgs if (m_id := getattr(m, "id", None)) is not None}

        # Disjoint id sets between non-empty checkpoints == summarization.
        # The intersection check rules out normal checkpoint-to-checkpoint
        # shrinkage (single RemoveMessage), which preserves overlap.
        if prev_ids and curr_ids and prev_ids.isdisjoint(curr_ids):
            timeline.append((SummarizationMarker(checkpoint_id=cid), None))

        for m in msgs:
            mid = getattr(m, "id", None)
            if mid is not None and mid in seen_ids:
                continue
            if mid is not None:
                seen_ids.add(mid)
            timeline.append((m, cid))

        prev_ids = curr_ids

    return timeline
