"""Coverage validation for the property-keyed report.

The partition is over **properties** (a claim has exactly one home), not rules — a rule whose
properties span several groups is expected (rules repeat) and reported as an informational stat, not
an error. Hard issues (a property in >1 group, or a group naming a property that doesn't exist) raise
`ValidationError`; `build` catches it and degrades to the single ``general`` bucket, whose shape is
trivially valid so a re-validate cannot raise. Soft issues go into `CoverageReport.warnings`.
"""
from collections import Counter, defaultdict

from composer.spec.source.report.schema import (
    CoverageReport, FormalizedProperty, GaveUpComponent, PropertyGroup, PropertyKey,
    RuleRef, RuleVerdict, SkippedClaim,
)


class ValidationError(RuntimeError):
    """Hard failure during validation — a bug in inputs or grouping."""


def validate(
    *,
    properties: list[FormalizedProperty],
    rules: list[RuleVerdict],
    groups: list[PropertyGroup],
    skipped: list[SkippedClaim],
    gave_up: list[GaveUpComponent],
    dropped_orphan_rules: int,
) -> CoverageReport:
    """Cross-check the grouping against the property set; produce a `CoverageReport`.

    Raises `ValidationError` on hard failures (a property in >1 group, or a group naming a property
    absent from the report). A property in no group is recorded (``properties_in_no_group``) but not
    fatal.
    """
    all_keys: set[PropertyKey] = {p.key for p in properties}
    props_by_key: dict[PropertyKey, FormalizedProperty] = {p.key: p for p in properties}

    appearance: Counter[PropertyKey] = Counter()
    for g in groups:
        for k in g.members:
            appearance[k] += 1

    multi = [k for k, c in appearance.items() if c > 1]
    if multi:
        raise ValidationError(f"properties appearing in multiple groups: {sorted(multi)}")

    excess = sorted(set(appearance) - all_keys)
    if excess:
        raise ValidationError(f"groups reference properties that don't exist: {excess}")

    missing = sorted(all_keys - set(appearance))

    # Informational: rules whose formalizing properties land in more than one group (expected).
    groups_of_rule: dict[RuleRef, set[str]] = defaultdict(set)
    for g in groups:
        for k in g.members:
            p = props_by_key.get(k)
            if p is not None:
                for ref in p.rule_refs:
                    groups_of_rule[ref].add(g.slug)
    spanning = sorted({ref[1] for ref, slugs in groups_of_rule.items() if len(slugs) > 1})

    sizes = [len(g.members) for g in groups]
    return CoverageReport(
        total_properties=len(properties),
        total_rules=len(rules),
        total_groups=len(groups),
        properties_per_group_min=min(sizes) if sizes else 0,
        properties_per_group_max=max(sizes) if sizes else 0,
        property_coverage_complete=not missing,
        properties_in_no_group=missing,
        rules_spanning_multiple_groups=spanning,
        skipped_count=len(skipped),
        gave_up_component_count=len(gave_up),
        dropped_orphan_rules=dropped_orphan_rules,
    )
