"""Coverage validation + status aggregation for the report.

Soft issues (a rule in no group, a status mismatch) go into
`CoverageReport.warnings`. Hard issues (a rule double-counted, or a group
referencing a non-existent rule) raise `ValidationError`; `build` catches it and
degrades to the single 'general' bucket so a report is always produced. The
fallback shape is trivially valid, so a re-validate after substitution cannot
raise.
"""
from collections import Counter

from prover_output_utility.models import NodeStatus

from composer.spec.source.report.grouping import aggregate_status
from composer.spec.source.report.schema import (
    CoverageReport, CVLRule, ImplementedProperty, RuleName,
)


class ValidationError(RuntimeError):
    """Hard failure during validation — a bug in inputs or grouping."""


def validate(
    *,
    rules: list[CVLRule],
    groups: list[ImplementedProperty],
    total_inferred: int,
) -> CoverageReport:
    """Cross-check the grouping against the rule set; produce a CoverageReport.

    Raises `ValidationError` on hard failures (a rule in >1 group, or a group
    naming a rule absent from the spec). A rule in no group is a soft warning.
    """
    warnings: list[str] = []
    rule_names_all = {r.name for r in rules}

    appearance: Counter[RuleName] = Counter()
    for g in groups:
        for n in g.rule_names:
            appearance[n] += 1

    multi = [n for n, c in appearance.items() if c > 1]
    if multi:
        raise ValidationError(f"rules appearing in multiple groups: {sorted(multi)}")

    excess = sorted(set(appearance) - rule_names_all)
    if excess:
        raise ValidationError(f"groups reference rules that don't exist: {excess}")

    missing = sorted(rule_names_all - set(appearance))

    rule_status: dict[RuleName, NodeStatus] = {r.name: r.status for r in rules}
    consistent = True
    for g in groups:
        expected = aggregate_status([rule_status[n] for n in g.rule_names if n in rule_status])
        if g.status != expected:
            consistent = False
            warnings.append(
                f"group '{g.slug}' status {g.status.value} doesn't match "
                f"aggregated child statuses ({expected.value})"
            )

    rules_per_group = [len(g.rule_names) for g in groups]
    return CoverageReport(
        total_property_formulations=total_inferred,
        total_rules=len(rules),
        total_groups=len(groups),
        rules_per_group_min=min(rules_per_group) if rules_per_group else 0,
        rules_per_group_max=max(rules_per_group) if rules_per_group else 0,
        rule_coverage_complete=not missing,
        rules_in_no_group=missing,
        status_aggregation_consistent=consistent,
        warnings=warnings,
    )
