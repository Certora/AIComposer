"""Formalize-properties phase for the known-properties pipeline.

Maps each *known* property (from a ``--properties`` YAML file) onto a discovered
component of the main contract plus the external functions it involves. Pure
mapping: it returns the matched known properties grouped by component index plus
the ``unmatched`` ones (logged here, dumped by the pipeline). The caller builds
the per-component batches and runs CVL generation.

Mirrors the agent pattern of ``struct_invariant.get_invariant_formulation``:
``ctx.child(KEY)`` + cache, ``bind_standard`` with a validator that forces a retry
on a bad mapping, ``with_tools`` (memory + source tools), ``run_to_completion``.
"""

import logging
from typing import NotRequired

from typing_extensions import TypedDict
from pydantic import BaseModel, ConfigDict, Field

from langgraph.graph import MessagesState

from graphcore.graph import FlowInput

from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.spec.context import WorkflowContext, SourceCode, CacheKey
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.system_model import HarnessedApplication
from composer.spec.prop import PropertyFormulation, PropertyId
from composer.spec.gen_types import TypedTemplate
from composer.spec.util import string_hash
from composer.spec.source.known_properties import KnownProperties, KnownProperty
from composer.spec.source.common_pipeline import _main_contract_index

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

class ComponentRef(BaseModel):
    """A reference to a component of the main contract: its index in the listing
    and its name, which must be consistent (the name at that index). Frozen so it
    can key the grouped formalize result."""
    model_config = ConfigDict(frozen=True)

    index: int = Field(description="The index of the component, as shown in square brackets in the listing")
    name: str = Field(description="The exact name of that component (must be the name shown at that index)")


class MatchedProperty(BaseModel):
    """The agent's mapping decision for one known property: which component it maps
    to and the entry points involved. ``sort``/``description`` are NOT here — they
    are authoritative on the ``KnownProperty`` (YAML) and assembled into a
    ``PropertyFormulation`` via ``KnownProperty.to_formulation``."""
    property_id: PropertyId = Field(description="The exact property_id of a known property from the listing")
    component: ComponentRef = Field(description="The component of the main contract this property maps to")
    methods: list[str] = Field(description="The external entry points of that component involved in the property (may be empty for a pure state invariant)")


class UnmatchedProperty(BaseModel):
    property_id: PropertyId = Field(description="The exact property_id of a known property that could not be mapped")
    reason: str = Field(description="A concrete explanation of why this property could not be mapped to any component")


class PropertyMapping(BaseModel):
    """The result of mapping the known properties onto components."""
    matched: list[MatchedProperty] = Field(description="The properties mapped to a component")
    unmatched: list[UnmatchedProperty] = Field(description="The properties that could not be mapped")


# Result of the formalize phase: the matched properties grouped per component
# (keyed by its validated index+name reference), plus the ones left unmatched. The
# caller resolves each ComponentRef to a ContractComponentInstance when building
# batches.
type FormalizeResult = tuple[dict[ComponentRef, list[PropertyFormulation]], list[UnmatchedProperty]]


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
) -> FormalizeResult:
    """Map known properties onto the main contract's components.

    Returns ``(matched, unmatched)``: the matched known properties grouped per
    component (as ``ComponentProperties``), and the properties the agent could not
    place (logged here, folded into ``uncovered_properties.json`` by the
    pipeline). The caller builds the per-component batches; this phase is pure
    mapping.
    """
    main_idx = _main_contract_index(app, source.contract_name)
    main_components = app.contract_components[main_idx].components

    known_by_id = {p.property_id: p for p in known.properties}
    all_ids = set(known_by_id)

    def _finalize(mapping: PropertyMapping) -> FormalizeResult:
        """Resolve a validated mapping to ``(matched, unmatched)``: log the
        unmatched, and group each matched property (converted to a
        ``PropertyFormulation`` whose title==property_id) under its component.
        Deterministic, so it runs identically on cache hit and miss."""
        for up in mapping.unmatched:
            _logger.warning(
                "Property %s could not be mapped to a component: %s (%s)",
                up.property_id, up.reason, known_by_id[up.property_id].property_desc,
            )
        grouped: dict[ComponentRef, list[PropertyFormulation]] = {}
        for fp in mapping.matched:
            grouped.setdefault(fp.component, []).append(
                known_by_id[fp.property_id].to_formulation(fp.methods)
            )
        return grouped, mapping.unmatched

    fmt_ctx = ctx.child(_formalize_cache_key(known))
    if (cached := await fmt_ctx.cache_get(PropertyMapping)) is not None:
        return _finalize(cached)

    # Cache miss: run the formalize agent to produce the mapping.
    class ST(MessagesState, RoughDraftState):
        result: NotRequired[PropertyMapping]

    class FormalizeInput(FlowInput, RoughDraftState):
        pass

    def _validate(s: ST, m: PropertyMapping) -> str | None:
        if not s.get("did_read"):
            return "You must read your rough draft before delivering the mapping"
        errors: list[str] = []
        matched_ids: set[str] = set()
        for fp in m.matched:
            if fp.property_id not in all_ids:
                errors.append(f"matched property_id {fp.property_id!r} is not one of the known properties")
            if fp.property_id in matched_ids:
                errors.append(f"property_id {fp.property_id!r} is matched more than once")
            matched_ids.add(fp.property_id)
            c = fp.component
            valid = ", ".join(f"[{i}] {sub.name}" for i, sub in enumerate(main_components))
            if not 0 <= c.index < len(main_components):
                errors.append(
                    f"component index {c.index} for property {fp.property_id!r} is out of range "
                    f"for {source.contract_name}. Valid components: {valid}"
                )
            elif main_components[c.index].name != c.name:
                errors.append(
                    f"component [{c.index}] of {source.contract_name} is {main_components[c.index].name!r}, "
                    f"not {c.name!r}, for property {fp.property_id!r}. Valid components: {valid}"
                )
        unmatched_ids: set[str] = set()
        for up in m.unmatched:
            if up.property_id not in all_ids:
                errors.append(f"unmatched property_id {up.property_id!r} is not one of the known properties")
            if up.property_id in matched_ids:
                errors.append(f"property_id {up.property_id!r} appears in both matched and unmatched")
            if up.property_id in unmatched_ids:
                errors.append(f"property_id {up.property_id!r} is listed in unmatched more than once")
            unmatched_ids.add(up.property_id)
        missing = all_ids - (matched_ids | unmatched_ids)
        if missing:
            errors.append(f"these property_ids are neither matched nor unmatched: {', '.join(sorted(missing))}")
        return "\n".join(errors) if errors else None

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
        [fmt_ctx.get_memory_tool(), *get_rough_draft_tools(ST), *env.source_tools]
    ).with_input(
        FormalizeInput
    ).compile_async()

    st = await run_to_completion(
        graph=graph,
        input=FormalizeInput(input=[], did_read=False, memory=None),
        thread_id=fmt_ctx.thread_id,
        recursion_limit=fmt_ctx.recursion_limit,
        description="Formalize properties",
    )
    assert "result" in st
    mapping = st["result"]
    await fmt_ctx.cache_put(mapping)
    return _finalize(mapping)
