"""
Context-scoped handler installation and graph execution.

This module is the glue between graph execution and event handling.
It provides two public entry points:

``with_handler(io_handler, event_handler)``
    Async context manager that installs a handler pair into a
    ``ContextVar``, creates an ``EventQueue``, and runs a
    background ``_queue_drainer`` task.  All ``run_graph()`` calls
    within the scope push events to this queue.

``run_graph(graph, ctxt, input, run_conf, description)``
    High-level wrapper that reads the installed handlers from the
    context, constructs an event sink (with automatic nesting
    support), and delegates to ``graph_runner.run_graph()``.  Also
    bridges HITL interrupts to ``IOHandler.human_interaction()``.

Nesting is automatic: if ``run_graph()`` is called while another
``run_graph()`` is already active (same ``with_handler`` scope),
the inner call's sink wraps events with ``Nested(event,
parent_id=outer_tid)`` before pushing to the queue.  The drainer
peels these layers to reconstruct the full execution path.
"""

from contextvars import ContextVar
from contextlib import asynccontextmanager

import asyncio

from composer.io.protocol import IOHandler
from composer.io.stream import EventQueue
from composer.io.event_handler import EventHandler

from typing import Any

from composer.io.events import (
    AllEvents, InnerEvent, Nested, NextCheckpoint,
    CustomUpdate, StateUpdate, Start, End,
)

from langgraph._internal._typing import StateLike
from langgraph.graph.state import CompiledStateGraph

from langchain_core.runnables import RunnableConfig

from composer.io.graph_runner import SinkProtocol, run_graph as _run_graph


_io_handler : ContextVar[None | tuple[EventQueue, IOHandler[Any], EventHandler]] = ContextVar("_io_handler", default=None)

_current_sink : ContextVar[tuple[SinkProtocol, str] | None] = ContextVar("_current_sink", default=None)
"""Tracks the active event sink and thread_id for nesting detection.

Set by ``run_graph()``; when non-None at the start of a new
``run_graph()`` call, the new call is nested and wraps the parent's
sink with ``Nested(...)``."""


def _unwrap(event: AllEvents) -> tuple[list[str], InnerEvent]:
    """Peel off Nested layers, collecting parent_ids into a path prefix."""
    path: list[str] = []
    while isinstance(event, Nested):
        path.append(event.parent_id)
        event = event.inner
    return (path, event)


async def _queue_drainer(
    q: EventQueue,
    h: IOHandler[Any],
    event_handler: EventHandler
):
    """Background task: consume events and dispatch to handlers.

    Structural events (``Start``, ``End``, ``StateUpdate``,
    ``NextCheckpoint``) go to the ``IOHandler``.  ``CustomUpdate``
    events go to the ``EventHandler``.  ``Nested`` wrappers are
    peeled off to reconstruct the execution path.
    """
    async for e in q.stream_events():
        (parents, inner) = _unwrap(e)
        match inner:
            case Start():
                await h.log_start(path=parents + [inner.thread_id], description=inner.description, tool_id=inner.tool_id)
            case End():
                await h.log_end(parents + [inner.thread_id])
            case NextCheckpoint():
                await h.log_checkpoint_id(path=parents + [inner.thread_id], checkpoint_id=inner.checkpoint_id)
            case CustomUpdate():
                full_path = parents + [inner.thread_id]
                await event_handler.handle_event(inner.payload, full_path, inner.checkpoint_id)
            case StateUpdate():
                await h.log_state_update(parents + [inner.thread_id], inner.payload)

@asynccontextmanager
async def with_handler(
    h: IOHandler[Any],
    event_handler: EventHandler
):
    """Install a handler pair and run a background drainer for this scope.

    All ``run_graph()`` calls within the scope push events to the
    same ``EventQueue``.  On exit, the drainer is cancelled and
    the context var is restored.
    """
    ev_queue = EventQueue(
        asyncio.Event(),
        []
    )
    tok = _io_handler.set((ev_queue, h, event_handler))
    background_task = asyncio.create_task(
        _queue_drainer(ev_queue, h, event_handler)
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
    description: str,
    within_tool: str | None = None,
) -> S:
    """Execute a graph within the current ``with_handler`` scope.

    Constructs an event sink that pushes to the scope's
    ``EventQueue``.  If another ``run_graph()`` is already active
    in the same scope, the sink wraps events with ``Nested`` so
    the drainer can reconstruct the execution path.

    HITL interrupts are bridged to ``IOHandler.human_interaction()``.
    """
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
        h: Any,
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
            description=description,
            human_handler=handle_human,
            within_tool=within_tool,
        )
    finally:
        _current_sink.reset(tok)
