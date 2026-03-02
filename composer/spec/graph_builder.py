"""
Convenience helpers for building agent sub-workflows.

- bind_standard: Extracts result type from state, adds result tool + summarizer
- run_to_completion: Thin wrapper around context.run_graph for sub-workflows
"""

from typing import Any, Callable, NotRequired, get_origin, get_args, cast

from pydantic import BaseModel

from langchain_core.runnables import RunnableConfig
from langgraph._internal._typing import StateLike
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph

from graphcore.graph import Builder, FlowInput
from graphcore.tools.results import ValidationResult, result_tool_generator

from composer.io.context import run_graph as _context_run_graph


def bind_standard[_S: MessagesState, _C: StateLike | None, _I: FlowInput | None, _R](
    builder: Builder[Any, _C, _I],
    state_type: type[_S],
    doc: str | None = None,
    validator: Callable[[_S, _R], str | None] | None = None
) -> Builder[_S, _C, _I]:
    """
    Bind a state type to the builder and generate a result tool based on the state's `result` annotation.

    Extracts the result type from the state's `result: NotRequired[T]` annotation and generates
    a result tool using `result_tool_generator`. The tool is then attached to the builder.

    Args:
        builder: The builder to modify
        state_type: The state type to bind, must have a `result: NotRequired[T]` annotation
        doc: Description for the result field. Required if the result type is not a BaseModel.
        validator: Optional validator function (state, result) -> error string or None

    Returns:
        Builder with state bound and result tool attached, preserving context and input types
    """
    annotations = getattr(state_type, '__annotations__', {})
    if 'result' not in annotations:
        raise ValueError(f"State type {state_type.__name__} must have a 'result' annotation")

    result_annotation = annotations['result']

    # Extract inner type from NotRequired[T]
    origin = get_origin(result_annotation)
    if origin is NotRequired:
        result_type = get_args(result_annotation)[0]
    else:
        result_type = result_annotation

    is_basemodel = isinstance(result_type, type) and issubclass(result_type, BaseModel)

    if not is_basemodel and doc is None:
        raise ValueError(f"doc parameter is required when result type {result_type} is not a BaseModel")

    tool_doc = "Used to indicate successful completion with result."

    valid: tuple[type[_S], Callable[[_S, Any, str], ValidationResult]] | None = None
    if validator:
        valid = (state_type, lambda s, r, _id: validator(s, cast(_R, r)))

    if is_basemodel:
        result_tool = result_tool_generator("result", result_type, tool_doc, valid)
    else:
        assert doc is not None
        result_tool = result_tool_generator("result", (result_type, doc), tool_doc, valid)

    return builder.with_state(state_type).with_tools([result_tool]).with_output_key("result").with_default_summarizer(
        max_messages=50
    )


async def run_to_completion[I: StateLike, S: StateLike](
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_id: str,
    *,
    checkpoint_id: str | None = None,
    recursion_limit: int = 100,
    description: str,
) -> S:
    """Run a compiled state graph to completion.

    Delegates to composer.io.context.run_graph, which handles event nesting
    automatically via context vars. Requires with_handler() to be active.
    """
    run_conf: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    if checkpoint_id is not None:
        run_conf["configurable"]["checkpoint_id"] = checkpoint_id

    return await _context_run_graph(
        graph=graph,
        ctxt=None,
        input=input,
        run_conf=run_conf,
        description=description,
    )
