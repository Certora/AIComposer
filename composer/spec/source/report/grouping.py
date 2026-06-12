"""LLM-driven grouping of CVL rules into high-level properties.

A single structured LLM call (langchain's `with_structured_output`) takes the
rule list and proposes high-level property groups; each group's status is then
rolled up from its member rules' verdicts. Groups are identified by the slug the
LLM assigns — this is a per-run snapshot, with no cross-run reconciliation.

A `general`-bucket fallback (every rule in one group) is built by `build` when
the LLM call raises, when validation rejects the grouping, or when the grouping
covers no rules.
"""
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from prover_output_utility.models import NodeStatus

from composer.templates.loader import load_jinja_template
from composer.spec.source.report.coverage import aggregate_status
from composer.spec.source.report.schema import (
    CVLRule, ImplementedProperty, ImplementedPropertyDraft, GroupingResult,
    RuleForGrouping,
)

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

FALLBACK_SLUG = "general"
FALLBACK_TITLE = "General (fallback grouping)"


def build_rules_for_grouping(rules: list[CVLRule]) -> list[RuleForGrouping]:
    """Project each `CVLRule` into the context row the grouping LLM sees. The
    rule already carries the property formulations it implements (embedded at
    collect time), so this is a straight read — no second join."""
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


def validate_slugs(groups: list[ImplementedPropertyDraft]) -> list[str]:
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


def build_implemented_properties(
    drafts: list[ImplementedPropertyDraft],
    rule_status: dict[str, NodeStatus],
) -> list[ImplementedProperty]:
    """Turn the LLM's drafts into final `ImplementedProperty`s, rolling each
    group's status up from its member rules' verdicts. Identity is the draft's
    own ``slug`` — there is no cross-run reconciliation."""
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


def build_fallback_grouping(rule_names: list[str], reason: str) -> GroupingResult:
    """A single 'general' bucket holding every rule, used when structured
    grouping is unavailable. `reason` is surfaced in the group description."""
    return GroupingResult(groups=[
        ImplementedPropertyDraft(
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
