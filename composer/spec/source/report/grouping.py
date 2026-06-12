"""LLM-driven grouping of CVL rules into high-level properties.

A single structured LLM call (langchain's `with_structured_output`) takes the
rule list and proposes high-level property groups; each group's status is then
rolled up from its member rules' verdicts. Groups are identified by the slug the
LLM assigns.

A `general`-bucket fallback (every rule in one group) is built by `build` when
the LLM call raises, when validation rejects the grouping, or when the grouping
covers no rules.
"""
import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from prover_output_utility.models import NodeStatus

from composer.templates.loader import load_jinja_template
from composer.spec.source.report.coverage import aggregate_status
from composer.spec.source.report.schema import (
    CVLRule, HighLevelProperty, HighLevelPropertyDraft, GroupingResult,
    InferredProperty, RuleForGrouping,
)

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

FALLBACK_SLUG = "general"
FALLBACK_TITLE = "General (fallback grouping)"


def build_rules_for_grouping(
    rules: list[CVLRule], properties: list[InferredProperty]
) -> list[RuleForGrouping]:
    """Project each `CVLRule` into the context row the grouping LLM sees,
    attaching the descriptions and sorts of the properties it implements."""
    by_ref = {(p.component, p.title): p for p in properties}
    out: list[RuleForGrouping] = []
    for r in rules:
        props = [by_ref[(ref.component, ref.title)] for ref in r.property_refs
                 if (ref.component, ref.title) in by_ref]
        out.append(RuleForGrouping(
            name=r.name,
            component=r.component,
            status=r.status,
            sorts=list(dict.fromkeys(p.sort for p in props)),
            property_descriptions=[p.description for p in props],
        ))
    return out


async def call_grouping_llm(
    *,
    llm: BaseChatModel,
    contract_name: str,
    rules: list[RuleForGrouping],
) -> GroupingResult:
    """One structured LLM call: the rule list in, a `GroupingResult` out, via
    langchain's `with_structured_output`. The model + token budget come from the
    passed `llm` (the run's configured model)."""
    system = load_jinja_template("autoprove_report_grouping_system.j2")
    user = load_jinja_template(
        "autoprove_report_grouping_prompt.j2",
        contract_name=contract_name,
        rules_json=json.dumps([r.model_dump() for r in rules], indent=2, ensure_ascii=False),
    )
    bound = llm.with_structured_output(GroupingResult)
    result = await bound.ainvoke([SystemMessage(system), HumanMessage(user)])
    assert isinstance(result, GroupingResult)
    return result


def validate_slugs(groups: list[HighLevelPropertyDraft]) -> list[str]:
    """Human-readable problems with the proposed slugs (kebab-case + unique)."""
    errors: list[str] = []
    seen: dict[str, int] = {}
    for i, g in enumerate(groups):
        if not _SLUG_RE.match(g.slug):
            errors.append(f"group #{i}: slug '{g.slug}' is not kebab-case ASCII")
        if g.slug in seen:
            errors.append(f"group #{i}: slug '{g.slug}' duplicates group #{seen[g.slug]}")
        else:
            seen[g.slug] = i
    return errors


def build_high_level(
    drafts: list[HighLevelPropertyDraft],
    rule_status: dict[str, NodeStatus],
) -> list[HighLevelProperty]:
    """Turn the LLM's drafts into final `HighLevelProperty`s, rolling each
    group's status up from its member rules' verdicts. Identity is the draft's
    own ``slug`` — there is no cross-run reconciliation."""
    return [
        HighLevelProperty(
            slug=d.slug,
            title=d.title,
            description=d.description,
            status=aggregate_status([rule_status[r] for r in d.rule_names if r in rule_status]),
            rule_names=list(d.rule_names),
        )
        for d in drafts
    ]


def build_fallback_grouping(rule_names: list[str], reason: str) -> GroupingResult:
    """A single 'general' bucket holding every rule, used when structured
    grouping is unavailable. `reason` is surfaced in the group description."""
    return GroupingResult(groups=[
        HighLevelPropertyDraft(
            slug=FALLBACK_SLUG,
            title=FALLBACK_TITLE,
            description=(
                "Fallback grouping: every rule was placed in a single bucket "
                "because structured grouping did not produce a usable result. "
                f"Reason: {reason}."
            ),
            rule_names=list(rule_names),
        )
    ])
