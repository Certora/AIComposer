"""Formalize-properties phase for the known-properties pipeline.

Maps each *known* property (from a ``--properties`` YAML file) onto a discovered
component of the main contract plus the external functions it involves, producing
the per-component batches that feed CVL generation. Properties the agent cannot
confidently place are returned as ``unmatched`` so the pipeline can surface them
(warn + dump) and continue with whatever matched.

Mirrors the agent pattern of ``struct_invariant.get_invariant_formulation``:
``ctx.child(KEY)`` + cache, ``bind_standard`` with a validator that forces a retry
on a bad mapping, ``with_tools`` (memory + source tools), ``run_to_completion``.
"""

import logging
from typing import NotRequired

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import MessagesState

from graphcore.graph import FlowInput

from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.context import WorkflowContext, SourceCode, CacheKey, Properties
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.system_model import HarnessedApplication
from composer.spec.prop import PropertyFormulation, PropertyId
from composer.spec.gen_types import TypedTemplate
from composer.spec.util import string_hash
from composer.spec.source.known_properties import KnownProperties, KnownProperty
from composer.spec.source.common_pipeline import (
    _ComponentBatch, build_component_batch, _main_contract_index,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

class FormalizedProperty(BaseModel):
    property_id: PropertyId = Field(description="The exact property_id of a known property from the listing")
    component_name: str = Field(description="The exact name of a component of the main contract this property maps to")
    methods: list[str] = Field(description="The external entry points of that component involved in the property (may be empty for a pure state invariant)")


class UnmatchedProperty(BaseModel):
    property_id: PropertyId = Field(description="The exact property_id of a known property that could not be mapped")
    reason: str = Field(description="A concrete explanation of why this property could not be mapped to any component")


class PropertyMapping(BaseModel):
    """The result of mapping the known properties onto components."""
    matched: list[FormalizedProperty] = Field(description="The properties mapped to a component")
    unmatched: list[UnmatchedProperty] = Field(description="The properties that could not be mapped")


class FormalizeParams(TypedDict):
    context: HarnessedApplication
    contract_spec: SourceCode
    properties: list[KnownProperty]


_typed_formalize_prompt = TypedTemplate[FormalizeParams]("formalize_properties_prompt.j2")


def _formalize_cache_key(known: KnownProperties) -> CacheKey[None, PropertyMapping]:
    """Cache key derived from the YAML content so editing ``--properties``
    invalidates a cached mapping (mirrors ``_batch_cache_key``)."""
    return CacheKey(string_hash(known.model_dump_json()))


async def formalize_properties(
    ctx: WorkflowContext[None],
    source: SourceCode,
    env: SourceEnvironment,
    app: HarnessedApplication,
    known: KnownProperties,
    prop_context: WorkflowContext[Properties],
) -> tuple[list[_ComponentBatch], list[UnmatchedProperty]]:
    """Map known properties onto the main contract's components.

    Returns ``(component_batches, unmatched)``: the per-component batches for CVL
    generation, and the properties the agent could not place (logged prominently
    here and folded into ``uncovered_properties.json`` by the pipeline). The run
    continues with whatever matched.
    """
    main_idx = _main_contract_index(app, source.contract_name)
    main_components = app.contract_components[main_idx].components
    valid_names = {c.name: i for i, c in enumerate(main_components)}

    known_by_id = {p.property_id: p for p in known.properties}
    all_ids = set(known_by_id)

    fmt_ctx = ctx.child(_formalize_cache_key(known))
    cached = await fmt_ctx.cache_get(PropertyMapping)

    if cached is None:
        class ST(MessagesState):
            result: NotRequired[PropertyMapping]

        def _validate(s: ST, m: PropertyMapping) -> str | None:
            matched_ids: set[str] = set()
            for fp in m.matched:
                if fp.property_id not in all_ids:
                    return f"matched property_id {fp.property_id!r} is not one of the known properties"
                if fp.property_id in matched_ids:
                    return f"property_id {fp.property_id!r} is matched more than once"
                matched_ids.add(fp.property_id)
                if fp.component_name not in valid_names:
                    return (
                        f"component_name {fp.component_name!r} for property {fp.property_id!r} is not a "
                        f"component of {source.contract_name}. Valid names: {', '.join(valid_names)}"
                    )
            unmatched_ids: set[str] = set()
            for up in m.unmatched:
                if up.property_id not in all_ids:
                    return f"unmatched property_id {up.property_id!r} is not one of the known properties"
                if up.property_id in matched_ids:
                    return f"property_id {up.property_id!r} appears in both matched and unmatched"
                if up.property_id in unmatched_ids:
                    return f"property_id {up.property_id!r} is listed in unmatched more than once"
                unmatched_ids.add(up.property_id)
            missing = all_ids - (matched_ids | unmatched_ids)
            if missing:
                return f"these property_ids are neither matched nor unmatched: {', '.join(sorted(missing))}"
            return None

        bound_template = _typed_formalize_prompt.bind({
            "context": app,
            "contract_spec": source,
            "properties": known.properties,
        })

        graph = bind_standard(
            env.builder,
            ST,
            validator=_validate,
        ).with_sys_prompt_template(
            # Only source tools are bound here; suppress CVL researcher guidance.
            "source_cvl_system_prompt.j2", with_cvl_tools=False
        ).inject(
            lambda g: bound_template.render_to(g.with_initial_prompt_template)
        ).with_tools(
            [fmt_ctx.get_memory_tool(), *env.source_tools]
        ).with_input(
            FlowInput
        ).compile_async()

        st = await run_to_completion(
            graph=graph,
            input=FlowInput(input=[]),
            thread_id=fmt_ctx.thread_id,
            recursion_limit=fmt_ctx.recursion_limit,
            description="Formalize properties",
        )
        assert "result" in st
        cached = st["result"]
        await fmt_ctx.cache_put(cached)

    # Log unmatched prominently; the pipeline dumps them to uncovered_properties.json.
    for up in cached.unmatched:
        desc = known_by_id[up.property_id].property_desc
        _logger.warning(
            "Property %s could not be mapped to a component: %s (%s)",
            up.property_id, up.reason, desc,
        )

    # Group matched properties by component index, converting each to a
    # PropertyFormulation whose title==property_id and description==property_desc
    # (these carry into the spec provenance comments and the property->rules map).
    by_component: dict[int, list[PropertyFormulation]] = {}
    for fp in cached.matched:
        kp = known_by_id[fp.property_id]
        prop = PropertyFormulation(
            title=fp.property_id,
            sort=kp.sort,
            description=kp.property_desc,
            methods="invariant" if kp.sort == "invariant" else fp.methods,
        )
        by_component.setdefault(valid_names[fp.component_name], []).append(prop)

    batches: list[_ComponentBatch] = []
    for component_idx, props in by_component.items():
        batch = await build_component_batch(
            source_input=source,
            prop_context=prop_context,
            summary=app,
            component_idx=component_idx,
            props=props,
        )
        batches.append(batch)

    return batches, cached.unmatched
