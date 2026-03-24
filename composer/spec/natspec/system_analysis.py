from typing import NotRequired


from graphcore.graph import MessagesState, FlowInput

from composer.spec.context import (
    WorkflowContext, CacheKey,
    SystemDoc, PlainBuilder
)
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.system_model import Application


SOURCE_ANALYSIS_KEY = CacheKey[None, Application]("source-analysis")

DESCRIPTION = "Component analysis"


async def run_component_analysis(
    context: WorkflowContext[None],
    input: SystemDoc,
    builder: PlainBuilder
) -> Application | None:
    """Analyze application components from a system doc and optionally source code."""

    child_ctxt = context.child(SOURCE_ANALYSIS_KEY)
    if (cached := child_ctxt.cache_get(Application)) is not None:
        return cached

    memory = child_ctxt.get_memory_tool()

    class AnalysisState(MessagesState):
        result: NotRequired[Application]

    b = bind_standard(
        builder=builder,
        state_type=AnalysisState
    ).with_input(
        FlowInput
    ).with_sys_prompt_template(
        "application_analysis_system.j2"
    ).with_tools(
        [memory]
    )

    b = b.with_initial_prompt_template(
        "application_analysis_prompt.j2",
        has_source=False,
    )

    graph = b.compile_async(checkpointer=child_ctxt.checkpointer)

    flow_input = FlowInput(input=[
        "The system document is as follows",
        input.content
    ])

    res = await run_to_completion(
        graph,
        flow_input,
        thread_id=child_ctxt.thread_id,
        recursion_limit=100,
        description=DESCRIPTION,
    )
    assert "result" in res
    result: Application = res["result"]

    child_ctxt.cache_put(result)
    return result
