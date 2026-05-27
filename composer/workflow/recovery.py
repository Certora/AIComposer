"""Crash-recovery snapshots for the codegen workflow.

When ``execute_ai_composer_workflow`` catches an exception, it asks
the checkpointer for the last live state and pulls the VFS (and any
in-flight working spec draft) out of it. That snapshot is stored under
``("crash_recovery",) / resume_key`` so a subsequent invocation passing
``resume_work_key=resume_key`` can rehydrate the file system without
re-running anything.

The snapshot is *just the user-visible state* — not the message
history, not tool-call traces, not any of the langgraph internals. The
checkpoint is still the source of truth if the user wants a full
resume; this is the "salvage what didn't get committed" path."""

import logging
import uuid
from typing import TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from composer.core.state import AIComposerState


logger = logging.getLogger(__name__)


RECOVERY_NS = ("crash_recovery",)


class VFSRecovery(TypedDict):
    vfs: dict[str, str]
    working_spec: str | None


async def recover_vfs(
    store: BaseStore,
    resume_key: str,
) -> VFSRecovery | None:
    """Fetch a crash snapshot by its resume key. Returns ``None`` if the
    key isn't in the store (stale link, already cleaned up, etc.)."""
    snapshot = await store.aget(RECOVERY_NS, resume_key)
    if snapshot is None:
        return None
    working_spec = snapshot.value.get("working_spec")
    return {
        "vfs": snapshot.value["vfs"],
        "working_spec": working_spec,
    }


async def recovery_from_thread(
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    thread_id: str,
) -> str | None:
    """Pull the latest checkpoint's VFS + working spec for ``thread_id``,
    mint a fresh resume key, and stash the snapshot under
    ``("crash_recovery",) / resume_key``. Returns the resume key (or
    ``None`` if there's no checkpoint to recover from)."""
    ct = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    if ct is None:
        return None
    channel_values = cast(AIComposerState, ct.checkpoint["channel_values"])
    vfs_snapshot = channel_values["vfs"]
    working_spec_snapshot = channel_values.get("working_spec", None)
    resume_key = f"crash_{thread_id}_{uuid.uuid4().hex[:8]}"
    to_store: VFSRecovery = {
        "vfs": vfs_snapshot,
        "working_spec": working_spec_snapshot,
    }
    await store.aput(RECOVERY_NS, resume_key, {**to_store})
    logger.info(f"Saved crash recovery snapshot: {resume_key}")
    return resume_key
