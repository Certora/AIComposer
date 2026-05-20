from typing import cast, TypedDict
from composer.core.state import AIComposerState
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver
import uuid

import logging

logger = logging.getLogger(__name__)

RECOVERY_NS = ("crash_recovery",)

class VFSRecovery(TypedDict):
    vfs: dict[str, str]
    working_spec: str | None

async def recover_vfs(
    store: BaseStore,
    resume_key: str
) -> VFSRecovery | None:
    snapshot = await store.aget(RECOVERY_NS, resume_key)
    if snapshot is None:
        return None
    working_spec = snapshot.value.get("working_spec")
    return {
        "vfs": snapshot.value["vfs"],
        "working_spec": working_spec
    }

async def recovery_from_thread(
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    thread_id: str
) -> str | None:
    ct = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    if ct is None:
        return None
    channel_values = cast(AIComposerState, ct.checkpoint["channel_values"])
    vfs_snapshot = channel_values["vfs"]
    working_spec_snapshot = channel_values.get("working_spec", None)
    resume_key = f"crash_{thread_id}_{uuid.uuid4().hex[:8]}"
    to_store : VFSRecovery = {
        "vfs": vfs_snapshot,
        "working_spec": working_spec_snapshot
    }
    await store.aput(RECOVERY_NS, resume_key, { **to_store })
    logger.info(f"Saved crash recovery snapshot: {resume_key}")
    return resume_key
