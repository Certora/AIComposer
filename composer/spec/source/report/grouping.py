"""LLM-driven grouping of CVL rules into high-level properties.

A single structured LLM call (AIComposer's `llm.with_structured_output`) takes
the rule list + the canonical map from prior runs and proposes high-level
property groups. Post-processing then:

1. reconciles each proposed group against the canonical map by Jaccard overlap
   of rule sets — J >= 0.6 reuses the canonical id/slug/title, 0.3 <= J < 0.6
   warns "merge candidate", J < 0.3 mints a fresh P-NN id;
2. aggregates each group's status from its member rules' verdicts.

A `general`-bucket fallback (every rule in one group) is built by `build` when
the LLM call raises, when validation rejects the grouping, or when the grouping
covers no rules.
"""
from __future__ import annotations

import json
import re
from typing import Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from composer.templates.loader import load_jinja_template
from composer.spec.source.report.coverage import aggregate_status
from composer.spec.source.report.schema import (
    CanonicalEntry, CanonicalMap, CVLRule, HighLevelProperty,
    HighLevelPropertyDraft, GroupingResult, InferredProperty, RuleForGrouping,
)

# Jaccard thresholds for canonical-map reconciliation.
_REUSE_THRESHOLD = 0.60   # >= this -> reuse the canonical id/slug/title verbatim
_REVIEW_THRESHOLD = 0.30  # >= this -> keep new id but flag as a merge candidate

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Constant slug for the fallback bucket, so the canonical map keeps a stable
# P-NN id for it across runs (exact-slug match in `reconcile_with_canonical`).
FALLBACK_SLUG = "general"
FALLBACK_TITLE = "General (fallback grouping)"


def build_rules_for_grouping(
    rules: list[CVLRule], properties: list[InferredProperty]
) -> list[RuleForGrouping]:
    """Project each `CVLRule` into the context row the grouping LLM sees,
    attaching the descriptions and sorts of the properties it implements."""
    by_ref = {(p.component, p.index): p for p in properties}
    out: list[RuleForGrouping] = []
    for r in rules:
        props = [by_ref[(ref.component, ref.index)] for ref in r.property_refs
                 if (ref.component, ref.index) in by_ref]
        sorts: list[str] = []
        for p in props:
            if p.sort not in sorts:
                sorts.append(p.sort)
        out.append(RuleForGrouping(
            name=r.name,
            component=r.component,
            status=r.status,
            sorts=sorts,  # type: ignore[arg-type]
            property_descriptions=[p.description for p in props],
        ))
    return out


async def call_grouping_llm(
    *,
    llm: BaseChatModel,
    contract_name: str,
    rules: list[RuleForGrouping],
    canonical: list[CanonicalEntry],
) -> GroupingResult:
    """One structured LLM call: rules (+ canonical map) in, `GroupingResult` out.

    Uses AIComposer's `with_structured_output` (no Autosetup, no tool loop). The
    model + token budget come from the passed `llm` (the run's configured model);
    thinking is disabled for this one-shot extraction."""
    system = load_jinja_template("autoprove_report_grouping_system.j2")
    user = load_jinja_template(
        "autoprove_report_grouping_prompt.j2",
        contract_name=contract_name,
        rules_json=json.dumps([r.model_dump() for r in rules], indent=2, ensure_ascii=False),
        canonical_json=(
            json.dumps([c.model_dump() for c in canonical], indent=2, ensure_ascii=False)
            if canonical else ""
        ),
    )
    bound = llm.copy(update={"thinking": None}).with_structured_output(GroupingResult)
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


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)


def _next_id(used_ids: set[str]) -> str:
    """Smallest unused ``P-NN`` id (>=2 digits; widens past P-99 naturally)."""
    n = 1
    while f"P-{n:02d}" in used_ids:
        n += 1
    return f"P-{n:02d}"


def reconcile_with_canonical(
    drafts: list[HighLevelPropertyDraft],
    canonical: CanonicalMap,
    rule_status: dict[str, "object"],
) -> tuple[list[HighLevelProperty], list[str], CanonicalMap]:
    """Produce final `HighLevelProperty`s, soft warnings, and an updated
    canonical map. The map grows monotonically: a group matching an existing
    entry (exact slug, or Jaccard >= reuse threshold) inherits its id/slug/
    title; otherwise it gets a fresh id and a new canonical entry. Existing
    entries are never renumbered."""
    warnings: list[str] = []
    used_ids = {c.id for c in canonical.entries}
    canonical_by_slug = {c.slug: c for c in canonical.entries}
    updated = CanonicalMap(schema_version=canonical.schema_version, entries=list(canonical.entries))

    finals: list[HighLevelProperty] = []
    for draft in drafts:
        canon = canonical_by_slug.get(draft.slug)
        if canon is None:
            best_j, best = 0.0, None
            for c in canonical.entries:
                j = _jaccard(draft.rule_names, c.anchor_rules)
                if j > best_j:
                    best_j, best = j, c
            if best is not None and best_j >= _REUSE_THRESHOLD:
                canon = best
                warnings.append(
                    f"group '{draft.slug}' force-matched to canonical '{canon.slug}' "
                    f"(Jaccard={best_j:.2f}); using id {canon.id}"
                )
            elif best is not None and best_j >= _REVIEW_THRESHOLD:
                warnings.append(
                    f"group '{draft.slug}' partially overlaps canonical '{best.slug}' "
                    f"(Jaccard={best_j:.2f}); kept as new id, consider merging"
                )

        if canon is not None:
            final_id, final_slug, final_title = canon.id, canon.slug, canon.title
        else:
            final_id = _next_id(used_ids)
            used_ids.add(final_id)
            final_slug, final_title = draft.slug, draft.title
            updated.entries.append(CanonicalEntry(
                id=final_id, slug=final_slug, title=final_title,
                anchor_rules=list(draft.rule_names),
            ))

        statuses = [rule_status[r] for r in draft.rule_names if r in rule_status]
        finals.append(HighLevelProperty(
            id=final_id, slug=final_slug, title=final_title,
            description=draft.description,
            status=aggregate_status(statuses),  # type: ignore[arg-type]
            rule_names=list(draft.rule_names),
        ))

    finals.sort(key=lambda h: int(h.id.split("-")[1]))
    return finals, warnings, updated


def build_fallback_grouping(rule_names: list[str], reason: str) -> GroupingResult:
    """A single 'general' bucket holding every rule. Routed through the normal
    `reconcile_with_canonical` path so the bucket gets a stable id from the
    canonical map. `reason` is surfaced in the group description."""
    return GroupingResult(groups=[
        HighLevelPropertyDraft(
            slug=FALLBACK_SLUG,
            title=FALLBACK_TITLE,
            description=(
                "Fallback grouping: every rule was placed in a single bucket "
                f"because structured grouping did not produce a usable result. "
                f"Reason: {reason}."
            ),
            rule_names=list(rule_names),
        )
    ])
