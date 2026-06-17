"""LLM-driven grouping of inferred properties into high-level audit claims.

A single structured LLM call takes the `FormalizedProperty` list and partitions it into high-level
`PropertyGroup`s (the "P-NN" headings) — each property in exactly one group, while the rules those
properties are formalized by may surface under several groups. Each group's status is rolled up from
its members' rules' verdicts. Groups are identified by the slug the LLM assigns — a per-run snapshot.

A single ``general`` fallback group (every property in one group) is used by `build` when the LLM
call raises, validation rejects the grouping, or the grouping covers no properties.
"""
from typing import Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from prover_output_utility.models import NodeStatus
from pydantic import BaseModel, Field

from composer.templates.loader import load_jinja_template
from composer.spec.source.report.schema import (
    FormalizedProperty, GroupStatus, PropertyGroup, PropertyKey, RuleRef,
)

FALLBACK_SLUG = "general"
FALLBACK_TITLE = "General"


def aggregate_status(statuses: Iterable[NodeStatus]) -> GroupStatus:
    """Roll member-rule `NodeStatus`es up into a `GroupStatus`:
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


# ---------------------------------------------------------------------------
# LLM grouping I/O (structured-output shapes)
# ---------------------------------------------------------------------------

class PropertyGroupDraft(BaseModel):
    """One high-level property group proposed by the grouping LLM."""
    slug: str = Field(
        ..., min_length=1, max_length=64,
        description="kebab-case ASCII lower-case identifier for the grouping.",
    )
    title: str = Field(description="A 5-12 word human-readable headline for the high-level property.")
    description: str = Field(
        description="1-3 plain-English sentences summarising what the group establishes; "
        "do not name the CVL rules or the individual property titles."
    )
    members: list[PropertyKey] = Field(
        description="The [component, title] pairs in this group; every input property must appear "
        "in exactly one group."
    )


class GroupingResult(BaseModel):
    """The high-level property groups covering every input property exactly once."""
    groups: list[PropertyGroupDraft] = Field(
        description="The high-level property groups; collectively they cover every input property "
        "exactly once."
    )


async def call_grouping_llm(
    *,
    llm: BaseChatModel,
    contract_name: str,
    properties: list[FormalizedProperty],
) -> GroupingResult:
    """One structured LLM call: the property list in, a `GroupingResult` out, via langchain's
    `with_structured_output`. The model + token budget come from the passed `llm`."""
    system = load_jinja_template("autoprove_report_grouping_system.j2")
    user = load_jinja_template(
        "autoprove_report_grouping_prompt.j2",
        contract_name=contract_name,
        properties=properties,
    )
    bound = llm.with_structured_output(GroupingResult)
    result = await bound.ainvoke([SystemMessage(system), HumanMessage(user)])
    assert isinstance(result, GroupingResult)
    return result


def build_groups(
    drafts: list[PropertyGroupDraft],
    props_by_key: dict[PropertyKey, FormalizedProperty],
    rule_status: dict[RuleRef, NodeStatus],
) -> list[PropertyGroup]:
    """Turn the LLM's drafts into final `PropertyGroup`s, rolling each group's status up from the
    verdicts of the rules its member properties are formalized by."""
    out: list[PropertyGroup] = []
    for d in drafts:
        out.append(PropertyGroup(
            slug=d.slug,
            title=d.title,
            description=d.description,
            status=aggregate_status(
                rule_status.get(ref, NodeStatus.UNKNOWN)
                for k in d.members
                if (p := props_by_key.get(k)) is not None
                for ref in p.rule_refs
            ),
            members=d.members,
        ))
    return out


def build_fallback_grouping(properties: list[FormalizedProperty]) -> GroupingResult:
    """A single bucket holding every property, used when structured grouping is unavailable. The
    reason is logged by the caller, not shown to the user."""
    return GroupingResult(groups=[
        PropertyGroupDraft(
            slug=FALLBACK_SLUG,
            title=FALLBACK_TITLE,
            description="All properties.",
            members=[p.key for p in properties],
        )
    ])
