from dataclasses import dataclass
from typing import NotRequired

from pydantic import BaseModel, Field

from graphcore.graph import MessagesState, FlowInput

from composer.spec.context import (
    WorkflowContext, CacheKey,
    SourceCode, AnalysisInput,
)
from composer.spec.graph_builder import bind_standard, run_to_completion


class ComponentInteraction(BaseModel):
    """
    Describes an interaction between some component and another
    """
    other_component: str = Field(description="The name of the other component with which this component interacts")
    interaction_description: str = Field(description="Why the interaction occurs, and a brief description of what the interaction looks like")

class ExternalDependency(BaseModel):
    """
    A single external dependency for a component
    """
    name: str = Field(description="A succint name for the external dependency (e.g., 'Price Oracle', 'Off-chain oracle', 'ERC20 asset token', etc.)")
    requirements: list[str] = Field(description="A list of assumptions/requirements that this external dependency must satisfy (e.g., 'Honest validator', 'implements a standard erc20 interface', etc.)")

class ApplicationComponent(BaseModel):
    """
    A single component within the application
    """
    name: str = Field(description="The brief, concise name of the component (e.g., Price Tracking/Token Management/etc.)")
    description: str = Field(description="A longer description of *what* the component does, not *how* it does it.")
    requirements: list[str] = Field(description="A list of short, succint natural language requirements describing the component's *intended* behavior")
    external_entry_points: list[str] = Field(description="The signatures/names of any external methods that comprise this component")
    state_variables: list[str] = Field(description="State variables involved in the component")
    interactions: list[ComponentInteraction] = Field(description="A list of interactions with other components")
    dependencies: list[ExternalDependency] = Field(description="A list of external dependencies for this component")

class ApplicationSummary(BaseModel):
    """
    A summary of your analysis of the application
    """
    application_type: str = Field(description="A short, concise description of the type of application (AMM/Liquidity Provider/etc.)")
    components: list[ApplicationComponent] = Field(description="The list of components in the application")

@dataclass
class ComponentInst:
    summ: ApplicationSummary
    ind: int

    @property
    def component(self) -> ApplicationComponent:
        return self.summ.components[self.ind]

    @property
    def application_type(self) -> str:
        return self.summ.application_type


SOURCE_ANALYSIS_KEY = CacheKey[None, ApplicationSummary]("source-analysis")

DESCRIPTION = "Component analysis"


async def run_component_analysis(
    context: WorkflowContext[None],
    input: AnalysisInput,
) -> ApplicationSummary | None:
    """Analyze application components from a system doc and optionally source code."""

    child_ctxt = context.child(SOURCE_ANALYSIS_KEY)
    if (cached := child_ctxt.cache_get(ApplicationSummary)) is not None:
        return cached

    doc, builder = input
    if doc.content is None:
        return None

    memory = child_ctxt.get_memory_tool()

    class AnalysisState(MessagesState):
        result: NotRequired[ApplicationSummary]

    has_source = isinstance(doc, SourceCode)

    b = bind_standard(
        builder=builder,
        state_type=AnalysisState
    ).with_sys_prompt(
        "You are an expert software analyst, who is very methodical in their work."
    ).with_tools(
        [memory]
    )

    if has_source:
        b = b.with_initial_prompt_template(
            "source_summarization.j2",
            main_contract_name=doc.contract_name,
            relative_path=doc.relative_path,
            has_source=True,
        )
    else:
        b = b.with_initial_prompt_template(
            "source_summarization.j2",
            has_source=False,
        )

    graph = b.compile_async(checkpointer=child_ctxt.checkpointer)

    flow_input = FlowInput(input=[
        "The system document is as follows",
        doc.content
    ])

    res = await run_to_completion(
        graph,
        flow_input,
        thread_id=child_ctxt.thread_id,
        recursion_limit=50,
        description=DESCRIPTION,
    )
    assert "result" in res
    result: ApplicationSummary = res["result"]

    child_ctxt.cache_put(result)
    return result
