from typing import NotRequired

from pydantic import BaseModel

from graphcore.graph import MessagesState


from composer.spec.trunner import run_to_completion_sync
from composer.spec.graph_builder import bind_standard
from composer.spec.context import WorkspaceContext, SourceBuilder
from graphcore.graph import FlowInput

from composer.spec.prop import PropertyFormulation
from composer.spec.component import ComponentInst

class _BugAnalysisCache(BaseModel):
    items: list[PropertyFormulation]

def run_bug_analysis(
    ctx: WorkspaceContext,
    component: ComponentInst,
    builder: SourceBuilder,
) -> list[PropertyFormulation] | None:
    # Check cache first
    component_analysis = ctx.child("bug_analysis")
    if (cached := component_analysis.cache_get(_BugAnalysisCache)) is not None:
        return cached.items

    class ST(MessagesState):
        result: NotRequired[list[PropertyFormulation]]

    d = bind_standard(
        builder, ST, "The security properties you have extracted about the component"
    ).with_initial_prompt_template(
        "property_analysis_prompt.j2",
        context=component
    ).with_sys_prompt(
        "You are an expert security and software analyst, with extensive knowledge of the types of issues and vulnerabilities found in DeFi protocols"
    ).build()[0].compile()

    r = run_to_completion_sync(
        d,
        FlowInput(input=[]),
        thread_id=component_analysis.thread_id
    )
    assert "result" in r

    result: list[PropertyFormulation] = r["result"]

    component_analysis.cache_put(_BugAnalysisCache(items=result))
    return result
