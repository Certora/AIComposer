"""
Property generation agent: extracts security properties from application components.

Parameterized by source availability via AnalysisInput tuple.
"""

from typing import NotRequired, Protocol

from pydantic import BaseModel

from langchain_core.tools import BaseTool

from graphcore.graph import MessagesState, FlowInput

from composer.spec.context import WorkflowContext, CacheKey, ComponentGroup
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.prop import PropertyFormulation
from composer.spec.system_model import ContractComponentInstance
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.spec.tool_env import BasicAgentTools


class _BugAnalysisCache(BaseModel):
    items: list[PropertyFormulation]

BUG_ANALYSIS_KEY = CacheKey[ComponentGroup, _BugAnalysisCache]("bug_analysis")

DESCRIPTION = "Property extraction"

class BugEnvironment(BasicAgentTools, Protocol):
    @property
    def bug_analysis_tools(self) -> tuple[BaseTool, ...]:
        ...

    @property
    def has_source(self) -> bool:
        ...

async def run_bug_analysis(
    ctx: WorkflowContext[ComponentGroup],
    env: BugEnvironment,
    component: ContractComponentInstance,
) -> list[PropertyFormulation] | None:
    """Extract security properties for a component.
    """

    component_analysis = ctx.child(BUG_ANALYSIS_KEY)
    if (cached := component_analysis.cache_get(_BugAnalysisCache)) is not None:
        return cached.items

    builder = env.builder

    class BugAnalysisInput(FlowInput, RoughDraftState):
        pass

    class ST(MessagesState, RoughDraftState):
        result: NotRequired[list[PropertyFormulation]]

    d = bind_standard(
        builder, ST, "The security properties you have extracted about the component"
    ).with_input(
        BugAnalysisInput
    ).with_initial_prompt_template(
        "property_analysis_prompt.j2",
        context=component,
        has_source=env.has_source
    ).with_tools(
        get_rough_draft_tools(ST)
    ).with_tools(
        env.bug_analysis_tools
    ).with_sys_prompt(
        "You are an expert security and software analyst, with extensive knowledge of the types of issues and vulnerabilities found in DeFi protocols"
    ).compile_async()

    r = await run_to_completion(
        d,
        BugAnalysisInput(input=[], memory=None, did_read=False),
        thread_id=component_analysis.thread_id,
        description=DESCRIPTION,
    )
    assert "result" in r

    result: list[PropertyFormulation] = r["result"]

    component_analysis.cache_put(_BugAnalysisCache(items=result))
    return result
