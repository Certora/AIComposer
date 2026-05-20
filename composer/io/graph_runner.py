"""
Low-level graph execution with event emission.

Streams a compiled LangGraph graph and translates each stream item
into an event pushed to the caller-supplied sink.  Handles HITL
interrupts by delegating to a ``human_handler`` callback.

This module knows nothing about queues, handlers, or nesting — it
just writes events to the sink it is given.  The higher-level
``context.run_graph()`` sets up the sink (with nesting support)
and connects it to the ``EventQueue`` / drainer infrastructure.
"""

from typing import Any, Protocol, Callable, Awaitable, cast

from composer.io.events import GraphEvents, NextCheckpoint, CustomUpdate, Start, End, StateUpdate

from langgraph._internal._typing import StateLike
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langchain_core.runnables import RunnableConfig


def _normalize_updates_payload(payload: dict) -> dict:
    """Flatten langgraph's mixed-mode tools-node payload to ``dict[node, dict]``.

    When a ``ToolNode`` batch contains tools whose returns differ in shape
    (some return raw ``ToolMessage`` / ``str`` / dicts; some return
    ``Command(update=...)`` with extra state-channel deltas), langgraph's
    ``_combine_tool_outputs`` emits the node's value as a *list* of separate
    updates instead of a single merged dict — see
    ``langgraph/prebuilt/tool_node.py``. Every downstream ``log_state_update``
    handler in this codebase expects ``dict[node, dict_with_messages]`` and
    silently drops the list shape (either via ``isinstance(update, dict)``
    guards or ``"messages" in update`` membership checks, which return
    ``False`` against a list of dicts/Commands). The result is that any
    ToolMessage produced by a parallel batch with at least one
    ``Command``-returning tool never reaches the renderer — the tool widgets
    stay "pending" until something else forces a redraw.

    Merge here so handlers stay simple. ``messages`` lists are concatenated;
    other state keys retain last-write-wins semantics (matches what
    langgraph's reducers would do for non-message channels, which the
    handlers in this codebase don't introspect for ingest purposes).
    """
    out: dict[str, Any] = {}
    for node_name, value in payload.items():
        if isinstance(value, dict) or not isinstance(value, list):
            out[node_name] = value
            continue
        merged: dict[str, Any] = {}
        for item in value:
            d = item.update if isinstance(item, Command) and isinstance(item.update, dict) else item
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                if (
                    k == "messages"
                    and isinstance(merged.get(k), list)
                    and isinstance(v, list)
                ):
                    merged[k] = [*merged[k], *v]
                else:
                    merged[k] = v
        out[node_name] = merged
    return out


class SinkProtocol(Protocol):
    """Write-only event sink.  Synchronous — must not block."""
    def __call__(self, event: GraphEvents) -> None:
        ...

type HumanHandler[T, S] = Callable[[T, S], Awaitable[str]]


async def run_graph[H, S: StateLike, I: StateLike, C: StateLike | None](
    event_sink: SinkProtocol,
    graph: CompiledStateGraph[S, C, I, Any],
    ctxt: C,
    input: I | None,
    run_conf: RunnableConfig,
    description: str,
    human_handler: HumanHandler[H, S] | None = None,
    within_tool: str | None = None,
) -> S:
    """Stream a graph to completion, emitting events to *event_sink*.

    Emits ``Start`` on entry, ``End`` on exit (in ``finally``), and
    ``StateUpdate`` / ``NextCheckpoint`` / ``CustomUpdate`` as the
    graph produces output.

    When the graph raises an ``__interrupt__``, calls
    *human_handler* with the interrupt value and current state, then
    resumes with the returned string.

    ``input`` may be ``None`` to resume from the last checkpoint on
    ``thread_id`` without seeding any new state — the only safe way to
    reuse a thread_id, since LangGraph treats a non-None input on an
    existing thread as "merge into checkpoint state and replay from the
    entry node", which compounds append-style channels.
    """
    config = run_conf.get("configurable", None)
    if config is None or "thread_id" not in config:
        raise ValueError("`configurable` must be set in graph config with thread_id")
    tid : str = config["thread_id"]

    graph_input : I | Command | None = input

    if "checkpoint_id" in config:
        graph_input = None

    curr_config = run_conf.copy()
    curr_config["configurable"] = config.copy()

    curr_checkpoint : str
    event_sink(Start(tid, description=description, tool_id=within_tool))
    try:
        while True:
            curr_input = graph_input
            graph_input = None
            interrupted = False
            async for (ty, payload) in graph.astream(
                curr_input, config=curr_config, context=ctxt, stream_mode=["checkpoints", "updates", "custom"]
            ):
                assert isinstance(payload, dict)
                if ty == "checkpoints":
                    curr_checkpoint = payload["config"]["configurable"]["checkpoint_id"]
                    event_sink(
                        NextCheckpoint(tid, curr_checkpoint)
                    )
                elif ty == "custom":
                    event_sink(
                        CustomUpdate(payload, thread_id=tid, checkpoint_id=curr_checkpoint) # pyright: ignore[reportPossiblyUnboundVariable]
                    )
                else:
                    assert ty == "updates"
                    if "__interrupt__" in payload:
                        assert human_handler is not None
                        if "configurable" in curr_config and "checkpoint_id" in curr_config["configurable"]:
                            del curr_config["configurable"]["checkpoint_id"]
                        interrupt_data = cast(H, payload["__interrupt__"][0].value)
                        curr_state = cast(S, (await graph.aget_state({"configurable": {"thread_id": tid}})).values)
                        human_response = await human_handler(interrupt_data, curr_state)
                        graph_input = Command(resume=human_response)
                        interrupted = True
                        break
                    event_sink(
                        StateUpdate(
                            _normalize_updates_payload(payload), thread_id=tid
                        )
                    )
            if interrupted:
                continue

            result_state = (await graph.aget_state({"configurable": {"thread_id": tid}})).values
            return cast(S, result_state)
    finally:
        event_sink(End(tid))
