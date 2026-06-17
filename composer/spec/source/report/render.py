"""Render an `AutoProverReport` (report.json) as a standalone HTML report.

Single self-contained page (inline CSS, no external assets): a header with outcome counts, one
section per high-level `PropertyGroup` (status badge + description + a rule table whose per-rule
descriptions are the in-group property claims that pull each rule in), a formalization-gaps section
(declined properties + components that gave up), and a coverage footer. The HTML is built by
``autoprove_report.html.j2``; this module only assembles the render context — no markup here. HTML is
opt-in — the pipeline writes report.json; render it on demand:

    autoprove-report-render certora/ap_report/report.json [--out report.html]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from prover_output_utility.models import NodeStatus

from composer.templates.loader import load_jinja_template
from composer.spec.source.report.schema import (
    AutoProverReport, FormalizedProperty, GroupStatus, PropertyGroup, PropertyKey, RuleRef, RuleVerdict,
)

# Status value -> CSS kind. Covers both the rule (NodeStatus) and group (GroupStatus) vocabularies.
_STATUS_KIND: dict[str, str] = {
    NodeStatus.VERIFIED.value: "ok",
    NodeStatus.VIOLATED.value: "bad",
    NodeStatus.TIMEOUT.value: "warn",
    NodeStatus.ERROR.value: "bad",
    NodeStatus.RUNNING.value: "info",
    NodeStatus.PENDING.value: "info",
    NodeStatus.UNKNOWN.value: "muted",
    GroupStatus.VERIFIED.value: "ok",
    GroupStatus.VIOLATED.value: "bad",
    GroupStatus.PARTIAL.value: "warn",
    GroupStatus.NO_RESULTS.value: "muted",
}

# Chip display order for the header outcome counts.
_RULE_ORDER = [s.value for s in (
    NodeStatus.VERIFIED, NodeStatus.VIOLATED, NodeStatus.TIMEOUT, NodeStatus.ERROR,
    NodeStatus.RUNNING, NodeStatus.PENDING, NodeStatus.UNKNOWN,
)]
_GROUP_ORDER = [s.value for s in (
    GroupStatus.VERIFIED, GroupStatus.VIOLATED, GroupStatus.PARTIAL, GroupStatus.NO_RESULTS,
)]


def _kind(status: str) -> str:
    return _STATUS_KIND.get(status, "muted")


def _is_url(link: str) -> bool:
    return link.startswith("http://") or link.startswith("https://")


def _link_view(link: str | None) -> dict:
    """How a prover link renders: a clickable URL, a plain 'local run' label, or an em-dash."""
    if link and _is_url(link):
        return {"href": link, "label": "prover run"}
    if link:
        return {"href": None, "label": "local run"}
    return {"href": None, "label": "—"}


def _counts(values: list[str], order: list[str]) -> list[dict]:
    """Per-status chip data, in display order, omitting statuses with no occurrences."""
    c = Counter(values)
    return [{"status": s, "kind": _kind(s), "n": c[s]} for s in order if c.get(s)]


def _group_view(
    group: PropertyGroup,
    props_by_key: dict[PropertyKey, FormalizedProperty],
    rules_by_ref: dict[RuleRef, RuleVerdict],
) -> dict:
    """Invert the group's members into rule rows: each rule the group's properties formalize, labelled
    with the descriptions of the in-group properties that pull it in (the edge labels). The same rule
    can label differently under another group, which is why this is computed per group, not stored."""
    descriptions: dict[RuleRef, list[str]] = {}
    order: list[RuleRef] = []
    for k in group.members:
        p = props_by_key.get(k)
        if p is None:
            continue
        for ref in p.rule_refs:
            if ref not in descriptions:
                descriptions[ref] = []
                order.append(ref)
            if p.description not in descriptions[ref]:
                descriptions[ref].append(p.description)

    rows = []
    for ref in order:
        rule = rules_by_ref.get(ref)
        status = rule.status.value if rule else NodeStatus.UNKNOWN.value
        rows.append({
            "name": ref[1],
            "status": status,
            "kind": _kind(status),
            "line": rule.line if rule else None,
            "link": _link_view(rule.prover_link if rule else None),
            "descriptions": descriptions[ref],
        })
    return {
        "slug": group.slug,
        "title": group.title,
        "description": group.description,
        "status": group.status.value,
        "kind": _kind(group.status.value),
        "rows": rows,
    }


def _build_context(report: AutoProverReport) -> dict:
    props_by_key = {p.key: p for p in report.properties}
    rules_by_ref = {r.ref: r for r in report.rules}
    return {
        "report": report,
        "coverage": report.coverage,
        "prover_runs": [
            {"slug": slug, "href": link if _is_url(link) else None}
            for slug, link in sorted(report.prover_links.items())
        ],
        "rule_counts": _counts([r.status.value for r in report.rules], _RULE_ORDER),
        "group_counts": _counts([g.status.value for g in report.groups], _GROUP_ORDER),
        "groups": [_group_view(g, props_by_key, rules_by_ref) for g in report.groups],
        "skipped": report.skipped,
        "gave_up": report.gave_up_components,
    }


def render_html(report: AutoProverReport) -> str:
    return load_jinja_template("autoprove_report.html.j2", **_build_context(report))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="autoprove-report-render",
        description="Render an autoprove report.json as a standalone HTML FV report.",
    )
    p.add_argument("input", type=Path, help="Path to a report.json produced by the autoprove report phase.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output HTML path (default: alongside the input as .html).")
    args = p.parse_args(argv)

    if not args.input.is_file():
        print(f"[autoprove-report-render] no such file: {args.input}", file=sys.stderr)
        return 1

    report = AutoProverReport.model_validate_json(args.input.read_text())
    out_path = args.out or args.input.with_suffix(".html")
    out_path.write_text(render_html(report))
    print(f"[autoprove-report-render] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
