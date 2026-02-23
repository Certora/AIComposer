from typing import Any, Protocol, Callable, Awaitable, cast
from contextvars import ContextVar
from contextlib import contextmanager

from composer.io.events import AllEvents, NextCheckpoint, CustomUpdate, NestedEnd, NestedStart, StateUpdate

from langgraph._internal._typing import StateLike
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langchain_core.runnables import RunnableConfig


class SinkProtocol(Protocol):
    def __call__(self, event: AllEvents) -> None:
        ...
        
type HumanHandler[T, S] = Callable[[T, S], Awaitable[str]]

type _SinkContext = tuple[str, SinkProtocol]

_nested : ContextVar[_SinkContext | None] = ContextVar("_nested", default=None)

@contextmanager
def _start_sink(
    event_sink: SinkProtocol,
    tid: str,
    within_tool_id: str | None
):
    curr = _nested.get()
    if curr is not None:
        curr[1](NestedStart(thread_id=curr[0], child_thread_id=tid, tool_context_id=within_tool_id))
    tok = _nested.set((tid, event_sink))
    try:
        yield
    finally:
        _nested.reset(tok)
        if curr is not None:
            curr[1](NestedEnd(thread_id=curr[0], child_thread_id=tid))


async def run_graph[H, S: StateLike, I: StateLike, C: StateLike | None](
    event_sink: SinkProtocol,
    graph: CompiledStateGraph[S, C, I, Any],
    ctxt: C,
    input: I,
    run_conf: RunnableConfig,
    within_tool: str | None = None,
    human_handler: HumanHandler[H, S] | None = None,
) -> S:
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
    with _start_sink(event_sink, tid, within_tool):
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
                        curr_state = cast(S, graph.get_state({"configurable": {"thread_id": tid}}).values)
                        human_response = await human_handler(interrupt_data, curr_state)
                        graph_input = Command(resume=human_response)
                        interrupted = True
                        break
                    event_sink(
                        StateUpdate(
                            payload, thread_id=tid
                        )
                    )
            if interrupted:
                continue

            result_state = graph.get_state({"configurable": {"thread_id": tid}}).values
            return cast(S, result_state)
