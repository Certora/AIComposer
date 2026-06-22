"""Store-side helpers and wire format for `ap-trail`.

Reads the per-user `RunMeta` / `ThreadMeta` records written by
``composer.io.thread_logging`` and packages them (with their per-thread
timelines) for either live drill-down or offline replay.
"""

import gzip
import os
from typing import Annotated, Literal, cast

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from composer.core.user import user_data_ns
from composer.io.thread_logging import (
    DEFAULT_META_NS,
    RunMeta,
    ThreadMeta,
    data_ns as _data_subns,
    runs_ns as _runs_subns,
    threads_ns as _threads_subns,
)
from composer.io.thread_timeline import SummarizationMarker, TimelineItem, load_timeline


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

def logging_ns(uid: str | None = None) -> tuple[str, ...]:
    """Conventional namespace for thread_logging records: ``user_data/<uid>/logging``."""
    return user_data_ns(uid) + DEFAULT_META_NS


def runs_ns(uid: str | None = None) -> tuple[str, ...]:
    return _runs_subns(logging_ns(uid))


def threads_ns(uid: str | None = None) -> tuple[str, ...]:
    return _threads_subns(logging_ns(uid))


def data_ns(run_id: str, uid: str | None = None) -> tuple[str, ...]:
    return _data_subns(logging_ns(uid), run_id)


# ---------------------------------------------------------------------------
# Store readers
# ---------------------------------------------------------------------------

async def list_runs(
    store: BaseStore,
    *,
    uid: str | None = None,
    limit: int = 50,
    since: str | None = None,
) -> list[tuple[str, RunMeta]]:
    """Return ``[(run_id, meta), ...]`` most-recent-first.

    ``since`` is an ISO-8601 string compared lexically against ``start_time``
    (which is also ISO-8601, so lex compare == chronological compare).
    """
    items = await store.asearch(runs_ns(uid), limit=limit)
    pairs: list[tuple[str, RunMeta]] = [(it.key, cast(RunMeta, it.value)) for it in items]
    if since is not None:
        pairs = [(rid, m) for rid, m in pairs if m["start_time"] >= since]
    pairs.sort(key=lambda kv: kv[1]["start_time"], reverse=True)
    return pairs


async def get_run(
    store: BaseStore, run_id: str, *, uid: str | None = None
) -> RunMeta | None:
    item = await store.aget(runs_ns(uid), run_id)
    if item is None:
        return None
    return cast(RunMeta, item.value)


async def list_threads_for_run(
    store: BaseStore, run_id: str, *, uid: str | None = None
) -> list[tuple[str, ThreadMeta]]:
    """Return ``[(thread_run_id, meta), ...]`` for one run, oldest-first."""
    items = await store.asearch(
        threads_ns(uid), filter={"run_id": run_id}, limit=1000
    )
    pairs: list[tuple[str, ThreadMeta]] = [
        (it.key, cast(ThreadMeta, it.value)) for it in items
    ]
    pairs.sort(key=lambda kv: kv[1]["start_time"])
    return pairs


async def list_run_data(
    store: BaseStore, run_id: str, *, uid: str | None = None
) -> list[tuple[str, dict]]:
    """Return ``[(key, metadata), ...]`` for one run's ``run_data`` records, key-sorted."""
    items = await store.asearch(data_ns(run_id, uid), limit=1000)
    pairs: list[tuple[str, dict]] = [(it.key, cast(dict, it.value)) for it in items]
    pairs.sort(key=lambda kv: kv[0])
    return pairs


async def get_run_data(
    store: BaseStore, run_id: str, key: str, *, uid: str | None = None
) -> dict | None:
    """Return one ``run_data`` metadata dict by key, or ``None`` if absent."""
    item = await store.aget(data_ns(run_id, uid), key)
    if item is None:
        return None
    return cast(dict, item.value)


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

WIRE_VERSION = 1


# Inner discrimination on BaseMessage's `type` Literal field. AIMessage.type ==
# "ai", HumanMessage.type == "human", etc. — pydantic uses these to dispatch to
# the right subclass when deserializing. Any future BaseMessage subclass we
# care about gets added here.
type ChatMessage = Annotated[
    AIMessage | HumanMessage | SystemMessage | ToolMessage,
    Field(discriminator="type"),
]


class ExportedMessage(BaseModel):
    kind: Literal["message"] = "message"
    data: ChatMessage
    checkpoint_id: str | None


class ExportedSummary(BaseModel):
    kind: Literal["summary"] = "summary"
    checkpoint_id: str


type ExportedTimelineItem = Annotated[
    ExportedMessage | ExportedSummary,
    Field(discriminator="kind"),
]


class ExportedThread(BaseModel):
    thread_run_id: str
    meta: ThreadMeta
    timeline: list[ExportedTimelineItem]


class ExportedRun(BaseModel):
    version: int
    run_id: str
    run: RunMeta
    threads: list[ExportedThread]


def _encode_timeline_item(item: TimelineItem, checkpoint_id: str | None) -> ExportedTimelineItem:
    if isinstance(item, SummarizationMarker):
        return ExportedSummary(checkpoint_id=item.checkpoint_id)
    return ExportedMessage(data=cast(ChatMessage, item), checkpoint_id=checkpoint_id)


def _decode_timeline_item(entry: ExportedTimelineItem) -> tuple[TimelineItem, str | None]:
    match entry:
        case ExportedSummary(checkpoint_id=cid):
            return SummarizationMarker(checkpoint_id=cid), None
        case ExportedMessage(data=msg, checkpoint_id=cid):
            return msg, cid


def decode_thread_timeline(exported: ExportedThread) -> list[tuple[TimelineItem, str | None]]:
    return [_decode_timeline_item(e) for e in exported.timeline]


async def build_export(
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,
    run_id: str,
    *,
    uid: str | None = None,
) -> ExportedRun:
    """Materialize a full ``ExportedRun`` from live DB state.

    For each thread segment, walks the checkpoint chain bounded by the
    ThreadMeta's ``start_checkpoint_id`` / ``end_checkpoint_id``.
    """
    run = await get_run(store, run_id, uid=uid)
    if run is None:
        raise KeyError(f"No such run: {run_id}")

    threads = await list_threads_for_run(store, run_id, uid=uid)
    exported_threads: list[ExportedThread] = []
    for thread_run_id, meta in threads:
        timeline = await load_timeline(
            checkpointer,
            meta["thread_id"],
            anchor_checkpoint_id=meta["end_checkpoint_id"],
            stop_at_checkpoint_id=meta["start_checkpoint_id"],
        )
        exported_threads.append(
            ExportedThread(
                thread_run_id=thread_run_id,
                meta=meta,
                timeline=[_encode_timeline_item(item, cid) for item, cid in timeline],
            )
        )

    return ExportedRun(
        version=WIRE_VERSION,
        run_id=run_id,
        run=run,
        threads=exported_threads,
    )


def write_export(exported: ExportedRun, path: str) -> int:
    """Gzip + JSON serialize an ``ExportedRun``. Returns bytes written."""
    payload = exported.model_dump_json().encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(payload)
    return os.path.getsize(path)


def read_export(path: str) -> ExportedRun:
    with gzip.open(path, "rb") as f:
        payload = f.read()
    return ExportedRun.model_validate_json(payload)
