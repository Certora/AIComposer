from typing import TypedDict, Protocol, AsyncIterator, Any
from dataclasses import dataclass
from datetime import datetime, UTC
from uuid import uuid4
from contextlib import asynccontextmanager
from contextvars import ContextVar

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

class _WithTimings(TypedDict):
    start_time: str
    end_time: str | None

class RunMeta(_WithTimings):
    tags: dict[str, Any]

class ThreadMeta(_WithTimings):
    run_id: str
    thread_id: str
    description: str
    from_tool_id: str | None

    start_checkpoint_id: str | None
    end_checkpoint_id: str | None

DEFAULT_META_NS = ("logging",)


def runs_ns(parent_ns: tuple[str, ...]) -> tuple[str, ...]:
    """Sub-namespace under ``parent_ns`` where ``RunMeta`` records live."""
    return parent_ns + ("runs",)


def threads_ns(parent_ns: tuple[str, ...]) -> tuple[str, ...]:
    """Sub-namespace under ``parent_ns`` where ``ThreadMeta`` records live."""
    return parent_ns + ("threads",)

def _time_string() -> str:
    return datetime.now(UTC).isoformat()

class CheckpointLogger(Protocol):
    def last_checkpoint(self, last_checkpoint: str) -> None:
        ...

@dataclass
class _ThreadRunHandle:
    _store: BaseStore
    _partial: ThreadMeta
    _key: str
    _ns: tuple[str, ...]
    _last_checkpoint: str | None

    def last_checkpoint(self, last_checkpoint: str):
        self._last_checkpoint = last_checkpoint

    async def complete(
        self,
    ):
        end_time = _time_string()
        to_write : ThreadMeta = {
            **self._partial,
            "end_time": end_time,
            "end_checkpoint_id": self._last_checkpoint
        }
        await self._store.aput(self._ns, self._key, {**to_write})

class ThreadLogger:
    def __init__(self, store: BaseStore, run_id: str, ns: tuple[str, ...]):
        self.store = store
        self.ns = ns
        self.run_id = run_id

    async def start(
        self, thread_id: str, start_checkpoint: str | None, from_tool_id: str | None, description: str
    ) -> _ThreadRunHandle:
        start_time = _time_string()
        thread_run_id = uuid4().hex
        staged : ThreadMeta = {
            "start_time": start_time,
            "from_tool_id": from_tool_id,
            "end_checkpoint_id": None,
            "end_time": None,
            "run_id": self.run_id,
            "start_checkpoint_id": start_checkpoint,
            "thread_id": thread_id,
            "description": description
        }
        try:
            await self.store.aput(
                self.ns, thread_run_id, {**staged}
            )
        except Exception:
            pass # swallow
        return _ThreadRunHandle(
            _store=self.store,
            _partial=staged,
            _key=thread_run_id,
            _ns=self.ns,
            _last_checkpoint=None
        )

_logger : ContextVar[None | ThreadLogger] = ContextVar("_logger", default=None)

@asynccontextmanager
async def log_thread(
    description: str,
    runnable: RunnableConfig,
    within_tool: str | None
) -> AsyncIterator[CheckpointLogger]:
    cur = _logger.get()
    if not cur:
        class Dummy:
            def last_checkpoint(self, last_checkpoint: str):
                pass
        yield Dummy()
        return
    assert "configurable" in runnable

    thread_id = runnable["configurable"]["thread_id"]
    
    checkpoint_id = runnable["configurable"].get("checkpoint_id", None)

    handle = await cur.start(
        description=description,
        from_tool_id=within_tool,
        start_checkpoint=checkpoint_id,
        thread_id=thread_id
    )
    try:
        yield handle
    finally:
        try:
            await handle.complete()
        except Exception:
            pass # swallow

@asynccontextmanager
async def thread_logger(
    store: BaseStore,
    tags: dict[str, Any],
    ns: tuple[str, ...],
    *,
    run_id: str | None = None
) -> AsyncIterator[None]:
    run_id = uuid4().hex if run_id is None else run_id

    run_meta : RunMeta = {
        "start_time": _time_string(),
        "tags": tags,
        "end_time": None
    }
    run_ns = runs_ns(ns)
    await store.aput(run_ns, run_id, {**run_meta})
    tok = _logger.set(ThreadLogger(
        store, run_id, threads_ns(ns)
    ))
    try:
        yield
    finally:
        _logger.reset(tok)
        run_meta["end_time"] = _time_string()
        try:
            await store.aput(run_ns, run_id, {**run_meta})
        except Exception:
            pass
