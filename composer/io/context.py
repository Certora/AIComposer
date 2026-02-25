from contextvars import ContextVar
from contextlib import asynccontextmanager

import asyncio

from composer.io.protocol import IOHandler
from composer.io.stream import EventQueue
from composer.audit.sink import AuditSink

from typing import Any, cast

from composer.human.types import HumanInteractionType

from composer.io.events import (
    AllEvents, InnerEvent, Nested, NextCheckpoint,
    CustomUpdate, StateUpdate, Start, End,
)

from langgraph._internal._typing import StateLike
from langgraph.graph.state import CompiledStateGraph

from langchain_core.runnables import RunnableConfig

from composer.diagnostics.stream import PartialUpdates, SummarizationNotice
from composer.diagnostics.handlers import is_user_update, is_audit_update

from composer.io.graph_runner import SinkProtocol, run_graph as _run_graph


_io_handler : ContextVar[None | tuple[EventQueue, IOHandler, AuditSink | None]] = ContextVar("_io_handler", default=None)

# Tracks the current event sink and thread_id for nesting detection.
# Set by run_graph; if non-None when a new run_graph starts, the new
# call is nested and wraps the sink with Nested(...).
_current_sink : ContextVar[tuple[SinkProtocol, str] | None] = ContextVar("_current_sink", default=None)


def _unwrap(event: AllEvents) -> tuple[list[str], InnerEvent]:
    """Peel off Nested layers, collecting parent_ids into a path prefix."""
    path: list[str] = []
    while isinstance(event, Nested):
        path.append(event.parent_id)
        event = event.inner
    return (path, event)


async def _queue_drainer(
    q: EventQueue,
    h: IOHandler,
    audit: AuditSink | None
):
    async for e in q.stream_events():
        (parents, inner) = _unwrap(e)
        match inner:
            case Start():
                await h.log_start(path=parents + [inner.thread_id], tool_id=inner.tool_id)
            case End():
                await h.log_end(parents + [inner.thread_id])
            case NextCheckpoint():
                await h.log_checkpoint_id(path=parents + [inner.thread_id], checkpoint_id=inner.checkpoint_id)
            case CustomUpdate():
                full_path = parents + [inner.thread_id]
                d = cast(PartialUpdates, inner.payload)
                if d["type"] == "summarization_raw":
                    if audit is not None:
                        audit.on_summarization(checkpoint_id=inner.checkpoint_id, summary=d["summary"])
                    notice: SummarizationNotice = {"type": "summarization_notice", "summary": d["summary"]}
                    await h.progress_update(full_path, notice)
                elif is_audit_update(d) and audit is not None:
                        match d["type"]:
                            case "rule_result":
                                audit.on_rule_result(
                                    rule=d["rule"],
                                    status=d["status"],
                                    analysis=d["analysis"],
                                    tool_id=d["tool_id"]
                                )
                            case "manual_search":
                                audit.on_manual_search(
                                    tool_id=d["tool_id"],
                                    ref=d["ref"]
                                )
                elif is_user_update(d):
                    await h.progress_update(full_path, d)
            case StateUpdate():
                await h.log_state_update(parents + [inner.thread_id], inner.payload)

@asynccontextmanager
async def with_handler(
    h: IOHandler,
    audit: AuditSink | None = None
):
    ev_queue = EventQueue(
        asyncio.Event(),
        []
    )
    tok = _io_handler.set((ev_queue, h, audit))
    background_task = asyncio.create_task(
        _queue_drainer(ev_queue, h, audit)
    )
    try:
        yield
    finally:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass
        _io_handler.reset(tok)

async def run_graph[S: StateLike, C: StateLike | None, I: StateLike](
    graph: CompiledStateGraph[S, C, I, Any],
    ctxt: C,
    input: I,
    run_conf: RunnableConfig,
    within_tool: str | None = None,
) -> S:
    curr_io = _io_handler.get()
    if curr_io is None:
        raise ValueError("No IO handler installed")

    (ev, handle, _) = curr_io

    # Determine thread_id from config
    tid = run_conf.get("configurable", {}).get("thread_id")
    if tid is None:
        raise ValueError("thread_id required in run config")

    # Determine sink: top-level uses queue.push, nested wraps parent's sink
    parent = _current_sink.get()
    if parent is None:
        sink: SinkProtocol = ev.push
    else:
        (parent_sink, parent_tid) = parent
        sink = lambda event: parent_sink(Nested(event, parent_id=parent_tid))

    tok = _current_sink.set((sink, tid))

    async def handle_human(
        h: HumanInteractionType,
        st: S
    ) -> str:
        return await handle.human_interaction(h, lambda: None)

    try:
        return await _run_graph(
            event_sink=sink,
            graph=graph,
            ctxt=ctxt,
            input=input,
            run_conf=run_conf,
            human_handler=handle_human,
            within_tool=within_tool,
        )
    finally:
        _current_sink.reset(tok)
