"""
Direct property formulation.

Takes a user-provided plain-English list of properties, an optional threat model,
and the full set of Certora conf files for the verification effort, and produces a
mapping ``conf_path -> list[PropertyFormulation]`` covering every user property
across all confs in a single LLM call.

This is the Phase-1 counterpart of `composer.spec.bug.run_bug_analysis` for the
direct (skip_setup) pipeline: instead of mining properties from a component
description, the agent reformulates user-supplied prose into structured
`PropertyFormulation` entries and assigns each to the conf whose contracts it
belongs to.
"""

import json
from typing import NotRequired

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from graphcore.graph import FlowInput

from composer.spec.context import WorkflowContext, CacheKey, Properties
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.prop import PropertyFormulation
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.util import string_hash
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.templates.loader import load_jinja_template


DESCRIPTION = "Direct property formulation (all confs)"


class ConfPropertyMapping(BaseModel):
    """Per-conf bucket of property formulations produced by the single-shot prompt."""
    conf_path: str = Field(
        description="The .conf file path this bucket of properties belongs to. "
                    "Must be an exact string match to one of the conf paths provided in the prompt."
    )
    properties: list[PropertyFormulation] = Field(
        description="The list of properties scoped to the contracts of this conf. "
                    "May be empty if no user property applies to this conf."
    )


class _DirectPropResult(BaseModel):
    items: list[ConfPropertyMapping]


def direct_property_key(
    conf_paths: list[str],
    properties: str | dict,
    threat_model: str | dict | None,
) -> CacheKey[Properties, _DirectPropResult]:
    parts = list(conf_paths) + [str(properties)]
    if threat_model is not None:
        parts.append(str(threat_model))
    return CacheKey[Properties, _DirectPropResult](
        "direct_props-" + string_hash("|".join(parts))
    )


def main_contract_from_config(config: dict) -> str | None:
    """Best-effort extraction of the main contract name from a Certora conf."""
    parametric = config.get("parametric_contracts")
    if isinstance(parametric, str):
        return parametric
    if isinstance(parametric, list) and parametric:
        return parametric[0]
    verify = config.get("verify")
    if isinstance(verify, str) and verify:
        head = verify.split(":", 1)[0]
        if head.endswith(".sol"):
            return None
        return head
    return None


async def run_direct_property_formulation_all(
    ctx: WorkflowContext[Properties],
    env: SourceEnvironment,
    confs: list[tuple[str, dict]],
    properties: str | dict,
    threat_model: str | dict | None,
) -> dict[str, list[PropertyFormulation]]:
    """Reformulate user-supplied properties across all confs in a single prompt.

    ``confs`` is a list of ``(conf_path, parsed_config)`` tuples. The returned dict
    has one entry per conf path (in the same order as input), with the list of
    `PropertyFormulation` entries the agent assigned to that conf. Confs to which
    no property was assigned have an empty list.
    """
    conf_paths = [p for p, _ in confs]
    key_ctx = ctx.child(direct_property_key(conf_paths, properties, threat_model))

    cached = await key_ctx.cache_get(_DirectPropResult)
    if cached is not None:
        return _normalize_mapping(cached.items, conf_paths)

    builder = env.builder

    class FormState(MessagesState, RoughDraftState):
        result: NotRequired[list[ConfPropertyMapping]]

    class FormInput(FlowInput, RoughDraftState):
        pass

    confs_for_template = [
        {
            "conf_path": conf_path,
            "config_json": json.dumps(config, indent=2),
            "main_contract": main_contract_from_config(config),
        }
        for conf_path, config in confs
    ]

    initial_prompt = load_jinja_template(
        "direct_property_prompt.j2",
        confs=confs_for_template,
    )

    d = bind_standard(
        builder,
        FormState,
        "The per-conf mapping of security properties you formulated, with every user "
        "property assigned to exactly one conf based on its in-scope contracts.",
    ).with_input(
        FormInput
    ).with_initial_prompt(
        initial_prompt
    ).with_tools(
        get_rough_draft_tools(FormState)
    ).with_tools(
        env.bug_analysis_tools
    ).with_sys_prompt(
        "You are an expert security and software analyst, with extensive knowledge of "
        "the types of issues and vulnerabilities found in DeFi protocols."
    ).compile_async()

    extra_input: list[str | dict] = [
        "The following is the user-authored plain-English list of properties they want "
        "verified across the entire application. Treat this text as authoritative: "
        "preserve every label/name/identifier the user attached to a property, and "
        "preserve the user's wording verbatim in each `description`. Assign every "
        "property to exactly one of the conf files listed above (based on which conf's "
        "contracts the property is about). Do not duplicate a property across confs, "
        "and do not drop properties that fit at least one conf.",
        properties,
    ]

    if threat_model is not None:
        extra_input += [
            "In addition, a coworker has written a 'threat model' for this application, "
            "which may include vulnerabilities/issues that are common in this type of "
            "application. This threat model is written for the entire application (not "
            "just one conf) so some of the issues/vulnerabilities/attacks may not be "
            "relevant. Do *NOT* overfit to this threat model; only add "
            "PropertyFormulation entries motivated by it when clearly applicable to the "
            "in-scope contracts of some conf and clearly not already covered by the "
            "user's list. Mark any such entries with a `[from threat model]` prefix in "
            "the description and assign them to the conf whose contracts they relate to.",
            threat_model,
        ]

    r = await run_to_completion(
        d,
        FormInput(input=extra_input, memory=None, did_read=False),
        thread_id=key_ctx.thread_id,
        description=DESCRIPTION,
    )
    assert "result" in r
    items: list[ConfPropertyMapping] = r["result"]

    await key_ctx.cache_put(_DirectPropResult(items=items))
    return _normalize_mapping(items, conf_paths)


def _normalize_mapping(
    items: list[ConfPropertyMapping],
    conf_paths: list[str],
) -> dict[str, list[PropertyFormulation]]:
    """Collapse the agent's output to a dict keyed by every input conf path.

    Entries whose `conf_path` does not match any input path are dropped. Multiple
    entries for the same conf path are concatenated. Confs the agent omitted are
    materialized with an empty list so callers can iterate over a stable key set.
    """
    mapping: dict[str, list[PropertyFormulation]] = {p: [] for p in conf_paths}
    valid = set(conf_paths)
    for m in items:
        if m.conf_path in valid:
            mapping[m.conf_path].extend(m.properties)
    return mapping
