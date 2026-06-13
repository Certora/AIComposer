"""LLM-driven grouping of CVL rules into high-level properties.

A single structured LLM call (langchain's `with_structured_output`) takes the
rule list and proposes high-level property groups; each group's status is then
rolled up from its member rules' verdicts. Groups are identified by the slug the
LLM assigns — this is a per-run snapshot, with no cross-run reconciliation.

A `general`-bucket fallback (every rule in one group) is built by `build` when
the LLM call raises, when validation rejects the grouping, or when the grouping
covers no rules.
"""
from typing import Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from prover_output_utility.models import NodeStatus

from composer.templates.loader import load_jinja_template
from composer.spec.source.report.schema import (
    CVLRule, GroupStatus, ImplementedProperty, ImplementedPropertyDraft,
    GroupingResult, RuleForGrouping, RuleName,
)

FALLBACK_SLUG = "general"
FALLBACK_TITLE = "General (fallback grouping)"


def aggregate_status(statuses: Iterable[NodeStatus]) -> GroupStatus:
    """Roll up member-rule `NodeStatus`es into a `GroupStatus`:
      - any VIOLATED                            -> VIOLATED
      - all VERIFIED                            -> VERIFIED
      - some VERIFIED but not all (no VIOLATED) -> PARTIAL
      - none VERIFIED, none VIOLATED            -> NO_RESULTS
    """
    all_verified = True
    any_verified = False
    for s in statuses:
        if s == NodeStatus.VIOLATED:
            return GroupStatus.VIOLATED
        if s == NodeStatus.VERIFIED:
            any_verified = True
        else:
            all_verified = False
    if any_verified:
        return GroupStatus.VERIFIED if all_verified else GroupStatus.PARTIAL
    return GroupStatus.NO_RESULTS


def build_rules_for_grouping(rules: list[CVLRule]) -> list[RuleForGrouping]:
    """Flatten each `CVLRule` (with its embedded property formulations) into the
    context row the grouping LLM sees."""
    out: list[RuleForGrouping] = []
    for r in rules:
        out.append(RuleForGrouping(
            name=r.name,
            component=r.component,
            status=r.status,
            sorts=list(dict.fromkeys(p.sort for p in r.properties)),
            property_descriptions=[p.description for p in r.properties],
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
        rules=[r.model_dump(mode="json") for r in rules],
    )
    bound = llm.with_structured_output(GroupingResult)
    result = await bound.ainvoke([SystemMessage(system), HumanMessage(user)])
    assert isinstance(result, GroupingResult)
    return result


def build_implemented_properties(
    drafts: list[ImplementedPropertyDraft],
    rule_status: dict[RuleName, NodeStatus],
) -> list[ImplementedProperty]:
    """Turn the LLM's drafts into final `ImplementedProperty`s, rolling each
    group's status up from its member rules' verdicts."""
    return [
        ImplementedProperty(
            slug=d.slug,
            title=d.title,
            description=d.description,
            status=aggregate_status([rule_status[r] for r in d.rule_names if r in rule_status]),
            rule_names=list(d.rule_names),
        )
        for d in drafts
    ]


def build_fallback_grouping(rule_names: list[RuleName]) -> GroupingResult:
    """A single bucket holding every rule, used when structured grouping is
    unavailable. The reason is logged by the caller, not shown to the user."""
    return GroupingResult(groups=[
        ImplementedPropertyDraft(
            slug=FALLBACK_SLUG,
            title=FALLBACK_TITLE,
            description="All rules.",
            rule_names=list(rule_names),
        )
    ])
