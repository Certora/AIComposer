"""
Property generation agent: extracts security properties from application components.

Parameterized by source availability via AnalysisInput tuple.
"""

from typing import NotRequired

from pydantic import BaseModel

from graphcore.graph import MessagesState, FlowInput

from composer.spec.context import WorkflowContext, CacheKey, ComponentGroup, SourceCode, AnalysisInput
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.prop import PropertyFormulation
from composer.spec.component import ComponentInst


class _BugAnalysisCache(BaseModel):
    items: list[PropertyFormulation]

BUG_ANALYSIS_KEY = CacheKey[ComponentGroup, _BugAnalysisCache]("bug_analysis")

DESCRIPTION = "Property extraction"


async def run_bug_analysis(
    ctx: WorkflowContext[ComponentGroup],
    component: ComponentInst,
    input: AnalysisInput,
) -> list[PropertyFormulation] | None:
    """Extract security properties for a component.

    The builder in the input tuple determines available tools:
    - (SourceCode, SourceBuilder): fs_tools available for source exploration
    - (SystemDoc, PlainBuilder): properties derived from design doc only
    """

    component_analysis = ctx.child(BUG_ANALYSIS_KEY)
    if (cached := component_analysis.cache_get(_BugAnalysisCache)) is not None:
        return cached.items

    doc, builder = input
    has_source = isinstance(doc, SourceCode)

    class ST(MessagesState):
        result: NotRequired[list[PropertyFormulation]]

    d = bind_standard(
        builder, ST, "The security properties you have extracted about the component"
    ).with_initial_prompt_template(
        "property_analysis_prompt.j2",
        context=component,
        has_source=has_source,
    ).with_sys_prompt(
        "You are an expert security and software analyst, with extensive knowledge of the types of issues and vulnerabilities found in DeFi protocols"
    ).compile_async(checkpointer=component_analysis.checkpointer)

    r = await run_to_completion(
        d,
        FlowInput(input=[]),
        thread_id=component_analysis.thread_id,
        description=DESCRIPTION,
    )
    assert "result" in r

    result: list[PropertyFormulation] = r["result"]

    component_analysis.cache_put(_BugAnalysisCache(items=result))
    return result
