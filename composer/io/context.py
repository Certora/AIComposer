from contextvars import ContextVar
from contextlib import asynccontextmanager

import asyncio

from composer.io.protocol import IOHandler
from composer.io.stream import EventQueue

from typing import Any, Protocol, Callable, Awaitable, cast
from contextvars import ContextVar
from contextlib import contextmanager

from composer.human.types import HumanInteractionType

from composer.io.events import AllEvents, NextCheckpoint, CustomUpdate, NestedEnd, NestedStart, StateUpdate

from langgraph._internal._typing import StateLike
from langgraph.runtime import get_runtime
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langchain_core.runnables import RunnableConfig

from composer.diagnostics.stream import PartialUpdates, AllUpdates

from composer.io.graph_runner import run_graph as _run_graph


_io_handler : ContextVar[None | tuple[EventQueue, IOHandler]] = ContextVar("_io_handler", default=None)


async def _queue_drainer(
    q: EventQueue,
    h: IOHandler
):
    async for e in q.stream_events():
        match e:
            case NextCheckpoint():
                await h.log_checkpoint_id(tid=e.thread_id, checkpoint_id=e.checkpoint_id)
            case CustomUpdate():
                d = cast(PartialUpdates, e.payload)
                new_payload : AllUpdates
                if d["type"] == "summarization_raw":
                    new_payload = {
                        "type": "summarization",
                        "summary": d["summary"],
                        "checkpoint_id": e.checkpoint_id
                    }
                else:
                    new_payload = d
                await h.progress_update(e.thread_id, new_payload) #type: ignore FIXME
            case NestedEnd():
                await h.log_end(parent=e.thread_id, child=e.child_thread_id)
            case NestedStart():
                await h.log_start(parent=e.thread_id, child=e.child_thread_id, tool_id=e.tool_context_id)
            case StateUpdate():
                await h.log_state_update(tid=e.thread_id, st=e.payload)

@asynccontextmanager
async def with_handler(
    h: IOHandler
):
    ev_queue = EventQueue(
        asyncio.Event(),
        []
    )
    tok = _io_handler.set((ev_queue, h))
    background_task = asyncio.create_task(
        _queue_drainer(ev_queue, h)
    )
    try:
        yield
    finally:
        background_task.cancel()
        try:
            await background_task
        except:
            pass
        _io_handler.reset(tok)

async def run_graph[S: StateLike, C: StateLike | None, I: StateLike](
    graph: CompiledStateGraph[S, C, I, Any],
    ctxt: C,
    input: I,
    run_conf: RunnableConfig,
    within_tool: str | None = None
) -> S:
    curr_io = _io_handler.get()
    if curr_io is None:
        raise ValueError("No IO handler installed")
    
    (ev, handle) = curr_io

    async def handle_human(
        h: HumanInteractionType,
        st: S
    ) -> str:
        return await handle.human_interaction(h, lambda: None)

    return await _run_graph(
        event_sink=ev.push,
        graph=graph,
        ctxt=ctxt,
        input=input,
        run_conf=run_conf,
        within_tool=within_tool,
        human_handler=handle_human
    )