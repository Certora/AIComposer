"""
Direct property formulation.

Takes a user-provided plain-English list of properties, an optional threat model,
and a Certora conf file, and produces a `list[PropertyFormulation]` scoped to the
contracts that conf will verify.

This is the Phase-1 counterpart of `composer.spec.bug.run_bug_analysis` for the
direct (skip_setup) pipeline: instead of mining properties from a component
description, the agent reformulates user-supplied prose into structured
`PropertyFormulation` entries.
"""

import json
from typing import NotRequired

from pydantic import BaseModel
from langgraph.graph import MessagesState

from graphcore.graph import FlowInput

from composer.spec.context import WorkflowContext, CacheKey, ComponentGroup
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.prop import PropertyFormulation
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.util import string_hash
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.templates.loader import load_jinja_template


DESCRIPTION = "Direct property formulation"


class _DirectPropResult(BaseModel):
    items: list[PropertyFormulation]


def direct_property_key(
    conf_path: str,
    properties: str | dict,
    threat_model: str | dict | None,
) -> CacheKey[ComponentGroup, _DirectPropResult]:
    parts = [conf_path, str(properties)]
    if threat_model is not None:
        parts.append(str(threat_model))
    return CacheKey[ComponentGroup, _DirectPropResult](
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


async def run_direct_property_formulation(
    ctx: WorkflowContext[ComponentGroup],
    env: SourceEnvironment,
    config: dict,
    conf_path: str,
    properties: str | dict,
    threat_model: str | dict | None,
) -> list[PropertyFormulation]:
    """Reformulate user-supplied properties as `PropertyFormulation` for one conf."""
    key_ctx = ctx.child(direct_property_key(conf_path, properties, threat_model))
    if (cached := await key_ctx.cache_get(_DirectPropResult)) is not None:
        return cached.items

    builder = env.builder

    class FormState(MessagesState, RoughDraftState):
        result: NotRequired[list[PropertyFormulation]]

    class FormInput(FlowInput, RoughDraftState):
        pass

    initial_prompt = load_jinja_template(
        "direct_property_prompt.j2",
        config_path=conf_path,
        config_json=json.dumps(config, indent=2),
        main_contract=main_contract_from_config(config),
    )

    d = bind_standard(
        builder,
        FormState,
        "The security properties you formulated for this conf, "
        "scoped to the contracts listed in its `files` field.",
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
        "preserve the user's wording verbatim in each `description`. Some of these may "
        "not be in scope for the contracts of this conf; discard those, but do not "
        "rewrite or paraphrase the rest.",
        properties,
    ]

    if threat_model is not None:
        extra_input += [
            "In addition, a coworker has written a 'threat model' for this application, "
            "which may include vulnerabilities/issues that are common in this type of "
            "application. This threat model is written for the entire application (not "
            "just the contracts of this conf) so some of the issues/vulnerabilities/"
            "attacks may not be relevant. Do *NOT* overfit to this threat model; only "
            "add PropertyFormulation entries motivated by it when clearly applicable to "
            "the in-scope contracts and clearly not already covered by the user's list. "
            "Mark any such entries with a `[from threat model]` prefix in the description.",
            threat_model,
        ]

    r = await run_to_completion(
        d,
        FormInput(input=extra_input, memory=None, did_read=False),
        thread_id=key_ctx.thread_id,
        description=DESCRIPTION,
    )
    assert "result" in r
    items: list[PropertyFormulation] = r["result"]

    await key_ctx.cache_put(_DirectPropResult(items=items))
    return items
