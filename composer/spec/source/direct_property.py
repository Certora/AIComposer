"""Direct property formulation for the single-conf properties pipeline.

Takes a user-authored plain-English list of properties, the system document, and
a single Certora ``.conf`` (already produced by AutoSetup or supplied via
``--config-path``), and returns ``list[PropertyFormulation]`` covering every
input property for that conf.
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


DESCRIPTION = "Direct property formulation"


class _DirectPropResult(BaseModel):
    """Cached list of formulated properties for one conf."""
    properties: list[PropertyFormulation] = Field(
        description="Every user-written property formulated as a PropertyFormulation."
    )


def _direct_property_key(
    conf_path: str,
    config: dict,
    properties: str | dict,
    system_doc: str | dict,
) -> CacheKey[Properties, _DirectPropResult]:
    parts = [
        conf_path,
        json.dumps(config, sort_keys=True),
        str(properties),
        str(system_doc),
    ]
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


async def run_direct_property_formulation(
    ctx: WorkflowContext[Properties],
    env: SourceEnvironment,
    conf_path: str,
    config: dict,
    contract_name: str,
    properties: str | dict,
    system_doc: str | dict,
) -> list[PropertyFormulation]:
    """Formulate user-supplied properties for a single conf.

    Returns one ``PropertyFormulation`` per user-written property. Cached on
    ``(conf_path, config, properties, system_doc)``.
    """
    key_ctx = ctx.child(_direct_property_key(conf_path, config, properties, system_doc))

    cached = await key_ctx.cache_get(_DirectPropResult)
    if cached is not None:
        return cached.properties

    class FormState(MessagesState, RoughDraftState):
        result: NotRequired[list[PropertyFormulation]]

    class FormInput(FlowInput, RoughDraftState):
        pass

    initial_prompt = load_jinja_template(
        "direct_property_prompt.j2",
        conf_path=conf_path,
        config_json=json.dumps(config, indent=2),
        main_contract=contract_name,
    )

    d = bind_standard(
        env.builder,
        FormState,
        "The list of security properties you formulated from the user's input, one "
        "entry per user-written property.",
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
        "verified for this conf. Treat this text as authoritative: preserve every "
        "label/name/identifier the user attached to a property, and preserve the user's "
        "wording verbatim in each `description`. Do not drop any property; this is a "
        "classification task, not a judgment task.",
        properties,
        "In addition, a coworker has written a 'system document' for this application. ",
        system_doc,
    ]

    r = await run_to_completion(
        d,
        FormInput(input=extra_input, memory=None, did_read=False),
        thread_id=key_ctx.thread_id,
        recursion_limit=key_ctx.recursion_limit,
        description=DESCRIPTION,
    )
    assert "result" in r
    props: list[PropertyFormulation] = r["result"]

    await key_ctx.cache_put(_DirectPropResult(properties=props))
    return props
