from typing import NotRequired, Any

from graphcore.graph import MessagesState, FlowInput


from composer.spec.context import (
    WorkflowContext, SystemDoc
)
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.system_model import BaseApplication, ExplicitContract, ExternalActor, ExternalDependency, SolidityIdentifier
from composer.spec.service_host import ServiceHost
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools

DESCRIPTION = "Component analysis"

def _validate_connectivity(
    app: BaseApplication, expected_main_id: SolidityIdentifier | None
) -> str | None:
    errors: list[str] = []
    known_components: dict[str, set[str]] = {}
    known_external: set[str] = set()
    known_solidity_ids : set[str] = set()

    for c in app.components:
        if isinstance(c, ExplicitContract):
            if c.solidity_identifier in known_solidity_ids:
                errors.append(f"Duplicate solidity identifier: {c.solidity_identifier}")
            else:
                known_solidity_ids.add(c.solidity_identifier)
            if c.name in known_components:
                errors.append(f"Duplicate contract names: {c.name}")
            else:
                known_components[c.name] = set()
            for sub_comp in c.components:
                if sub_comp.name in known_components[c.name]:
                    errors.append(f"Duplicate component names in {c.name}: {sub_comp.name}")
                known_components[c.name].add(sub_comp.name)
        else:
            assert isinstance(c, ExternalActor)
            if c.name in known_external:
                errors.append(f"Duplicate external component name: {c.name}")
            known_external.add(c.name)
    
    if expected_main_id is not None and expected_main_id not in known_solidity_ids:
        errors.append(f"Expected an explicit contract instance with solidity identifier: {expected_main_id}")

    for explicit in app.components:
        if not isinstance(explicit, ExplicitContract):
            continue
        for sub_comp in explicit.components:
            thing_interacts_with_str = f"Component {sub_comp.name} of {explicit.name} interacts with"
            for interaction in sub_comp.interactions:
                if isinstance(interaction, ExternalDependency):
                    if interaction.external_actor not in known_external:
                        errors.append(f"{thing_interacts_with_str} unknown external actor: {interaction.external_actor}")
                else:
                    if interaction.contract_name not in known_components:
                        errors.append(f"{thing_interacts_with_str} an unknown explicit contact: {interaction.contract_name}")
                    elif interaction.component and interaction.component not in known_components[interaction.contract_name]:
                        errors.append(f"{thing_interacts_with_str} unknown component {interaction.component} of explicit contract {interaction.contract_name}")

    if not errors:
        return None

    def _fmt(items: set[str]) -> str:
        return ", ".join(sorted(items)) if items else "(none)"

    reference_lines = [
        f"- Declared contracts: {_fmt(set(known_components))}",
        f"- Declared external actors: {_fmt(known_external)}",
    ]
    for contract_name, subs in sorted(known_components.items()):
        reference_lines.append(f"- Components of {contract_name}: {_fmt(subs)}")
    reference = "\n\nFor reference, the names you declared in your submission:\n" + "\n".join(reference_lines)

    if len(errors) == 1:
        return errors[0] + reference
    return "Multiple validation errors found; fix all of them before resubmitting:\n" + "\n".join(f"- {e}" for e in errors) + reference

async def run_component_analysis[T: BaseApplication](
    ty: type[T],
    child_ctxt: WorkflowContext[T],
    input: SystemDoc,
    env: ServiceHost,
    extra_input: list[str | dict],
    expected_main_id: SolidityIdentifier | None = None,
) -> T | None:
    """Analyze application components from a system doc and optionally source code."""
    if (cached := await child_ctxt.cache_get(ty)) is not None:
        return cached

    memory = child_ctxt.get_memory_tool()

    class AnalysisInput(RoughDraftState, FlowInput):
        pass

    AnalysisState = type("AnalysisState", (MessagesState, RoughDraftState), {
        "__annotations__": {"result": NotRequired[ty]}
    })

    def _validation_wrapper(
        _: Any, app: BaseApplication
    ) -> str | None:
        return _validate_connectivity(app, expected_main_id)

    b = bind_standard(
        builder=env.builder,
        state_type=AnalysisState,
        validator=_validation_wrapper
    ).with_input(
        AnalysisInput
    ).with_sys_prompt_template(
        "application_analysis_system.j2",
        sort=env.sort,
    ).with_tools(
        [memory, *get_rough_draft_tools(AnalysisState), *env.analysis_tools]
    ).with_initial_prompt_template(
        "application_analysis_prompt.j2",
        sort=env.sort,
    )

    graph = b.compile_async()
    inputs : list[str | dict] = [
        "The system document is as follows",
        input.content.to_dict(),
        *extra_input
    ]

    flow_input = AnalysisInput(input=inputs, did_read=False, memory=None)

    res = await run_to_completion(
        graph,
        flow_input,
        thread_id=child_ctxt.thread_id,
        recursion_limit=child_ctxt.recursion_limit,
        description=DESCRIPTION,
    )
    assert "result" in res
    result: T = res["result"] #type: ignore trust me bro

    await child_ctxt.cache_put(result)
    return result
