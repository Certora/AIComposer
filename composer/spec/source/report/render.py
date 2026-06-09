"""Render an `AutoProverReport` (report.json) as a standalone HTML report.

Single self-contained page (inline CSS, no external assets): P-NN sections with
a status badge + high-level description + a rule table, plus a collapsible
appendix indexing every inferred property to the rules that implement it. HTML
is opt-in — the pipeline writes report.json; render it on demand:

    autoprove-report-render certora/ap_report/report.json [--out report.html]
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from prover_output_utility.models import NodeStatus

from composer.spec.source.report.schema import (
    AutoProverReport, CVLRule, GroupStatus, HighLevelProperty, InferredProperty,
)


_STATUS_KIND: dict[str, str] = {
    NodeStatus.VERIFIED.value: "ok",
    NodeStatus.VIOLATED.value: "bad",
    NodeStatus.TIMEOUT.value:  "warn",
    NodeStatus.ERROR.value:    "bad",
    NodeStatus.RUNNING.value:  "info",
    NodeStatus.PENDING.value:  "info",
    NodeStatus.UNKNOWN.value:  "muted",
    GroupStatus.VERIFIED.value:     "ok",
    GroupStatus.VIOLATED.value:     "bad",
    GroupStatus.PARTIAL.value:      "warn",
    GroupStatus.INCONCLUSIVE.value: "muted",
}


def _badge(status: str) -> str:
    kind = _STATUS_KIND.get(status, "muted")
    return f'<span class="badge badge-{kind}">{html.escape(status)}</span>'


_CSS = """\
:root {
    --bg: #ffffff;
    --fg: #1d2125;
    --muted-fg: #5b6770;
    --border: #e4e8ec;
    --bg-card: #fafbfc;
    --bg-pre: #f4f6f8;
    --link: #1a73e8;
    --ok-fg: #0a6b2a;
    --ok-bg: #e6f6ec;
    --bad-fg: #a8071a;
    --bad-bg: #fde7e9;
    --warn-fg: #8a5500;
    --warn-bg: #fff1d6;
    --info-fg: #054a8c;
    --info-bg: #e3effb;
    --muted-bg: #eef0f3;
}
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: var(--fg);
    background: var(--bg);
    margin: 0;
    padding: 0;
    line-height: 1.5;
    font-size: 14px;
}
.container {
    max-width: 1080px;
    margin: 0 auto;
    padding: 32px 28px 48px;
}
header.report-head {
    border-bottom: 2px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 28px;
}
header.report-head h1 {
    margin: 0 0 4px;
    font-size: 24px;
    font-weight: 600;
}
header.report-head .subtitle {
    color: var(--muted-fg);
    font-size: 13px;
}
.meta-grid {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 4px 18px;
    margin-top: 14px;
    font-size: 13px;
}
.meta-grid dt {
    color: var(--muted-fg);
    font-weight: 500;
    margin: 0;
}
.meta-grid dd {
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
    font-size: 12.5px;
    word-break: break-all;
}
.counts {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 14px;
}
.count-chip {
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12.5px;
    color: var(--muted-fg);
    background: var(--bg-card);
}
.count-chip strong {
    color: var(--fg);
    margin-right: 4px;
}
section.prop {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
section.prop h2 {
    margin: 0 0 6px;
    font-size: 16px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
section.prop h2 .id {
    color: var(--muted-fg);
    font-weight: 500;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
}
section.prop .desc {
    color: var(--fg);
    margin: 8px 0 12px;
}
section.prop .slug {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
    font-size: 11.5px;
    color: var(--muted-fg);
}
.badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    white-space: nowrap;
}
.badge-ok    { color: var(--ok-fg);    background: var(--ok-bg); }
.badge-bad   { color: var(--bad-fg);   background: var(--bad-bg); }
.badge-warn  { color: var(--warn-fg);  background: var(--warn-bg); }
.badge-info  { color: var(--info-fg);  background: var(--info-bg); }
.badge-muted { color: var(--muted-fg); background: var(--muted-bg); }
table.rules {
    width: 100%;
    border-collapse: collapse;
    margin-top: 4px;
    font-size: 13px;
}
table.rules th, table.rules td {
    text-align: left;
    border-bottom: 1px solid var(--border);
    padding: 8px 10px;
    vertical-align: top;
}
table.rules th {
    color: var(--muted-fg);
    font-weight: 500;
    background: var(--bg);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
table.rules td.col-name {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
    font-size: 12.5px;
    white-space: nowrap;
    width: 1%;
}
table.rules td.col-status {
    white-space: nowrap;
    width: 1%;
}
table.rules td.col-link {
    width: 1%;
    white-space: nowrap;
}
table.rules td.col-link a {
    color: var(--link);
    text-decoration: none;
    font-size: 12px;
}
table.rules td.col-link a:hover { text-decoration: underline; }
details.appendix {
    margin-top: 36px;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 20px;
    background: var(--bg-card);
}
details.appendix summary {
    cursor: pointer;
    font-weight: 600;
    font-size: 15px;
}
details.appendix table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 14px;
    font-size: 13px;
}
details.appendix table th,
details.appendix table td {
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}
details.appendix table th {
    color: var(--muted-fg);
    font-weight: 500;
    font-size: 12px;
    text-transform: uppercase;
}
details.appendix h3 {
    margin: 18px 0 6px;
    font-size: 14px;
    font-weight: 600;
}
.sort-tag {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 3px;
    background: var(--bg-pre);
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
    font-size: 11px;
    color: var(--muted-fg);
}
footer.report-foot {
    margin-top: 36px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--muted-fg);
}
.warnings {
    background: var(--warn-bg);
    color: var(--warn-fg);
    border-radius: 6px;
    padding: 10px 14px;
    margin: 14px 0;
    font-size: 13px;
}
.warnings ul { margin: 4px 0 0 18px; padding: 0; }
"""


def _is_url(link: str) -> bool:
    return link.startswith("http://") or link.startswith("https://")


def _link_cell(rule: CVLRule) -> str:
    """A clickable prover-run link when one is known and it's a URL; a plain
    label for local-path links; em-dash otherwise."""
    if rule.prover_link and _is_url(rule.prover_link):
        return (
            f'<a href="{html.escape(rule.prover_link)}" target="_blank" '
            f'rel="noopener">prover run</a>'
        )
    if rule.prover_link:
        return '<span style="color:var(--muted-fg)">local run</span>'
    return "—"


def _render_header(report: AutoProverReport) -> str:
    """Top-of-page summary card + per-status count chips."""
    status_counts: Counter[str] = Counter(r.status.value for r in report.rules)
    group_counts: Counter[str] = Counter(g.status.value for g in report.high_level_properties)

    chips = []
    for status in [NodeStatus.VERIFIED, NodeStatus.VIOLATED, NodeStatus.TIMEOUT,
                   NodeStatus.ERROR, NodeStatus.RUNNING, NodeStatus.PENDING,
                   NodeStatus.UNKNOWN]:
        n = status_counts.get(status.value, 0)
        if n:
            chips.append(f'<span class="count-chip">{_badge(status.value)} <strong>{n}</strong></span>')

    group_chips = []
    for status in [GroupStatus.VERIFIED, GroupStatus.VIOLATED,
                   GroupStatus.PARTIAL, GroupStatus.INCONCLUSIVE]:
        n = group_counts.get(status.value, 0)
        if n:
            group_chips.append(f'<span class="count-chip">{_badge(status.value)} <strong>{n}</strong></span>')

    if report.prover_links:
        runs = " ".join(
            (f'<a href="{html.escape(link)}" target="_blank" rel="noopener">{html.escape(slug)}</a>'
             if _is_url(link) else f'<code>{html.escape(slug)}</code>')
            for slug, link in sorted(report.prover_links.items())
        )
    else:
        runs = "—"

    meta = [
        ("Contract",       html.escape(report.contract_name)),
        ("Schema version", html.escape(report.schema_version)),
        ("Run timestamp",  html.escape(report.run_timestamp_utc or "—")),
        ("Prover runs",    runs),
    ]
    meta_rows = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in meta)

    return f"""\
<header class="report-head">
  <h1>Formal verification report — {html.escape(report.contract_name)}</h1>
  <div class="subtitle">
    {len(report.inferred_properties)} inferred properties &middot;
    {len(report.rules)} CVL rules &middot;
    {len(report.high_level_properties)} high-level properties
  </div>
  <dl class="meta-grid">{meta_rows}</dl>
  <div style="margin-top:14px;font-size:12px;color:var(--muted-fg);">Rule outcomes:</div>
  <div class="counts">{''.join(chips) or '—'}</div>
  <div style="margin-top:8px;font-size:12px;color:var(--muted-fg);">High-level property outcomes:</div>
  <div class="counts">{''.join(group_chips) or '—'}</div>
</header>
"""


def _render_warnings(report: AutoProverReport) -> str:
    if not report.coverage.warnings:
        return ""
    items = "".join(f"<li>{html.escape(w)}</li>" for w in report.coverage.warnings)
    return f'<div class="warnings"><strong>Coverage warnings:</strong><ul>{items}</ul></div>'


def _render_group(
    group: HighLevelProperty,
    rules_by_name: dict[str, CVLRule],
    desc_lookup: dict[str, str],
) -> str:
    """One P-NN section: heading, description, rule table."""
    rows = []
    for name in group.rule_names:
        rule = rules_by_name.get(name)
        if rule is None:
            rows.append(
                f'<tr><td class="col-name">{html.escape(name)}</td>'
                f'<td class="col-status">{_badge("UNKNOWN")}</td>'
                f'<td>(missing from rules[] — bug)</td>'
                f'<td class="col-link">—</td></tr>'
            )
            continue
        desc_text = desc_lookup.get(rule.name, "")
        desc_html = (
            html.escape(desc_text) if desc_text else
            '<span style="color:var(--muted-fg)">(no inferred description)</span>'
        )
        name_cell = html.escape(rule.name)
        if rule.line is not None:
            name_cell += f'<span style="color:var(--muted-fg)">:{rule.line}</span>'
        rows.append(
            f'<tr>'
            f'<td class="col-name">{name_cell}</td>'
            f'<td class="col-status">{_badge(rule.status.value)}</td>'
            f'<td>{desc_html}</td>'
            f'<td class="col-link">{_link_cell(rule)}</td>'
            f'</tr>'
        )

    return f"""\
<section class="prop" id="{html.escape(group.id)}">
  <h2><span class="id">{html.escape(group.id)}</span> {html.escape(group.title)} {_badge(group.status.value)}</h2>
  <div class="slug">slug: <code>{html.escape(group.slug)}</code></div>
  <p class="desc">{html.escape(group.description)}</p>
  <table class="rules">
    <thead><tr><th>Rule</th><th>Status</th><th>Description</th><th></th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _description_lookup(report: AutoProverReport) -> dict[str, str]:
    """{rule_name: first English description from its property_refs}."""
    by_ref = {(p.component, p.index): p.description for p in report.inferred_properties}
    out: dict[str, str] = {}
    for r in report.rules:
        text = ""
        for ref in r.property_refs:
            d = by_ref.get((ref.component, ref.index))
            if d:
                text = d
                break
        out[r.name] = text
    return out


def _render_appendix(report: AutoProverReport) -> str:
    """Collapsible appendix: every inferred property, by component, with the
    rule(s) that implement it."""
    rules_by_ref: dict[tuple[str, int], list[CVLRule]] = defaultdict(list)
    for r in report.rules:
        for ref in r.property_refs:
            rules_by_ref[(ref.component, ref.index)].append(r)

    by_component: dict[str, list[InferredProperty]] = defaultdict(list)
    for p in report.inferred_properties:
        by_component[p.component].append(p)

    parts = ['<details class="appendix"><summary>Inferred property index — '
             f'{len(report.inferred_properties)} properties across '
             f'{len(by_component)} components</summary>']
    for comp in sorted(by_component):
        props = sorted(by_component[comp], key=lambda p: p.index)
        parts.append(f'<h3>{html.escape(comp)} ({len(props)} properties)</h3>')
        parts.append('<table>')
        parts.append('<thead><tr><th>#</th><th>Sort</th><th>Description</th>'
                     '<th>Implemented by</th></tr></thead><tbody>')
        for p in props:
            impl = rules_by_ref.get((comp, p.index), [])
            impl_html = (
                ", ".join(f'<code>{html.escape(r.name)}</code>' for r in impl) if impl
                else '<span style="color:var(--muted-fg)">(no rule mapping)</span>'
            )
            # The agent-assigned snake_case title is the cross-reference key in
            # property_rules.json; show it under the P-number.
            num_cell = f'P{p.index}'
            if p.title:
                num_cell += f'<br><code style="font-size:11px">{html.escape(p.title)}</code>'
            parts.append(
                f'<tr>'
                f'<td>{num_cell}</td>'
                f'<td><span class="sort-tag">{html.escape(p.sort)}</span></td>'
                f'<td>{html.escape(p.description)}</td>'
                f'<td>{impl_html}</td>'
                f'</tr>'
            )
        parts.append('</tbody></table>')
    parts.append('</details>')
    return "\n".join(parts)


def _render_footer(report: AutoProverReport) -> str:
    cov = report.coverage
    return f"""\
<footer class="report-foot">
  Coverage check: {cov.total_rules} unique rules across {cov.total_groups} high-level
  properties ({cov.rules_per_group_min}–{cov.rules_per_group_max} rules each).
  Coverage complete: <strong>{cov.rule_coverage_complete}</strong>;
  status aggregation consistent: <strong>{cov.status_aggregation_consistent}</strong>.
</footer>
"""


def render_html(report: AutoProverReport) -> str:
    rules_by_name = {r.name: r for r in report.rules}
    desc_lookup = _description_lookup(report)

    body_parts = [_render_header(report), _render_warnings(report)]
    for g in report.high_level_properties:
        body_parts.append(_render_group(g, rules_by_name, desc_lookup))
    body_parts.append(_render_appendix(report))
    body_parts.append(_render_footer(report))

    body = "\n".join(body_parts)
    title = f"FV report — {html.escape(report.contract_name)}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
{body}
</div>
</body>
</html>
"""


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

    report = AutoProverReport.model_validate(json.loads(args.input.read_text()))
    out_path = args.out or args.input.with_suffix(".html")
    out_path.write_text(render_html(report))
    print(f"[autoprove-report-render] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
