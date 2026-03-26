from typing import NotRequired, Protocol, Any

from langchain_core.tools import BaseTool

from graphcore.graph import MessagesState, FlowInput


from composer.spec.context import (
    WorkflowContext, CacheKey, SystemDoc
)
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.system_model import BaseApplication, ExplicitContract, ExternalActor, ExternalDependency
from composer.spec.tool_env import BasicAgentTools
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools

DESCRIPTION = "Component analysis"

class AnalysisEnv(BasicAgentTools, Protocol):
    @property
    def system_analysis_tools(self) -> tuple[BaseTool, ...]:
        ...

def _validate_connectivity(
    _: Any, app: BaseApplication
) -> str | None:
    known_components : dict[str, set[str]] = {}

    known_external : set[str] = set()

    for c in app.components:
        if isinstance(c, ExplicitContract):
            if c.name in known_components:
                return f"Duplicate contract names: {c.name}"
            known_components[c.name] = set()
            for sub_comp in c.components:
                if sub_comp.name in known_components[c.name]:
                    return f"Duplicate component names in {c.name}: {sub_comp.name}"
                known_components[c.name].add(sub_comp.name)
        else:
            assert isinstance(c, ExternalActor)
            known_external.add(c.name)
    
    for explicit in app.components:
        if not isinstance(explicit, ExplicitContract):
            continue
        for sub_comp in explicit.components:
            thing_interacts_with_str = f"Component {sub_comp.name} of {explicit.name} interacts with"
            for interaction in sub_comp.interactions:
                if isinstance(interaction, ExternalDependency):
                    if interaction.external_actor not in known_external:
                        return f"{thing_interacts_with_str} unknown external actor: {interaction.external_actor}"
                else:
                    if interaction.contract_name not in known_components:
                        return f"{thing_interacts_with_str} an unknown explicit contact: {interaction.contract_name}"
                    if interaction.component and interaction.component not in known_components[interaction.contract_name]:
                        return f"{thing_interacts_with_str} unknown component {interaction.component} of explicit contract {interaction.contract_name}"
    return None

async def run_component_analysis[T: BaseApplication](
    ty: type[T],
    child_ctxt: WorkflowContext[T],
    input: SystemDoc,
    env: AnalysisEnv,
    extra_input: list[str | dict]
) -> T | None:
    """Analyze application components from a system doc and optionally source code."""
    if (cached := child_ctxt.cache_get(ty)) is not None:
        return cached

    memory = child_ctxt.get_memory_tool()

    AnalysisState = type("AnalysisState", (MessagesState, RoughDraftState), {
        "__annotations__": {"result": NotRequired[ty]}
    })

    b = bind_standard(
        builder=env.builder,
        state_type=AnalysisState,
        validator=_validate_connectivity
    ).with_input(
        FlowInput
    ).with_sys_prompt_template(
        "application_analysis_system.j2",
        has_source=env.has_source
    ).with_tools(
        [memory, *get_rough_draft_tools(AnalysisState), *env.system_analysis_tools]
    ).with_initial_prompt_template(
        "application_analysis_prompt.j2",
        has_source=env.has_source
    )

    graph = b.compile_async()
    inputs : list[str | dict] = [
        "The system document is as follows",
        input.content,
        *extra_input
    ]

    flow_input = FlowInput(input=inputs)

    res = await run_to_completion(
        graph,
        flow_input,
        thread_id=child_ctxt.thread_id,
        recursion_limit=100,
        description=DESCRIPTION,
    )
    assert "result" in res
    result: T = res["result"] #type: ignore trust me bro

    child_ctxt.cache_put(result)
    return result
