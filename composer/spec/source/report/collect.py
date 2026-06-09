"""Collect the report's inputs from on-disk dumps + prover verdicts.

For each component (and the structural invariants) we read the property dumps
the pipeline already wrote under ``certora/properties/`` and fetch per-rule
verdicts from that component's prover run via ProverOutputUtility:

  - ``<stem>.properties.json``     -> ordered `PropertyFormulation`s
                                      (array order == 1-based property index)
  - ``<stem>.property_rules.json`` -> ``{property_title: [rule_names]}``
  - the component's prover link    -> POU `CheckResult`s (status + source line)

`<stem>` is ``autospec_<slugified_name>`` for components and ``invariants`` for
the structural invariants. No spec-text parsing and no
``.certora_recent_jobs.json``: rule line numbers and verdicts both come from POU.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from prover_output_utility import ProverOutputAPI
from prover_output_utility.models import CheckResult, NodeStatus

from composer.spec.gen_types import CERTORA_DIR, under_project
from composer.spec.source.report.schema import CVLRule, InferredProperty, RuleRef

_log = logging.getLogger(__name__)

PROPERTIES_SUBDIR = "properties"


@dataclass(frozen=True)
class ComponentInput:
    """One unit to collect: its human name, the property-dump stem, and the
    prover run link/path for its verdicts (``None`` if the run produced no
    link, e.g. a component that gave up before submitting)."""
    name: str
    stem: str
    prover_link: str | None


@dataclass(frozen=True)
class _Verdict:
    status: NodeStatus
    line: int | None
    duration_seconds: float | None


# Rollup priority when a rule_name has several per-method CheckResults: the most
# attention-worthy / terminal verdict wins as the rule-level status.
_STATUS_PRIORITY: dict[NodeStatus, int] = {
    NodeStatus.VIOLATED: 6,
    NodeStatus.ERROR:    5,
    NodeStatus.TIMEOUT:  4,
    NodeStatus.UNKNOWN:  3,
    NodeStatus.RUNNING:  2,
    NodeStatus.PENDING:  1,
    NodeStatus.VERIFIED: 0,
}


def _properties_dir(project_root: str) -> Path:
    return under_project(project_root, CERTORA_DIR) / PROPERTIES_SUBDIR


def _load_properties(pdir: Path, stem: str, component: str) -> list[InferredProperty]:
    path = pdir / f"{stem}.properties.json"
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    return [
        InferredProperty(component=component, index=i + 1, **prop)
        for i, prop in enumerate(raw)
    ]


def _load_rule_refs(
    pdir: Path, stem: str, component: str, properties: list[InferredProperty]
) -> dict[str, list[RuleRef]]:
    """Invert ``<stem>.property_rules.json`` ({title: [rules]}) into
    ``rule_name -> [RuleRef]``. Missing file -> empty (a component that gave up
    after extraction has properties but no rule mapping)."""
    path = pdir / f"{stem}.property_rules.json"
    if not path.is_file():
        return {}
    title_to_index = {p.title: p.index for p in properties if p.title}
    mapping: dict[str, list[str]] = json.loads(path.read_text())
    out: dict[str, list[RuleRef]] = {}
    for title, rule_names in mapping.items():
        idx = title_to_index.get(title)
        if idx is None:
            # A title in the mapping with no matching property entry means the
            # two files disagree; skip rather than fabricate an index.
            continue
        for rule_name in rule_names:
            out.setdefault(rule_name, []).append(RuleRef(component, idx))
    return out


def _fetch_verdicts(api: ProverOutputAPI, link: str) -> dict[str, _Verdict]:
    """rule_name -> rolled-up `_Verdict` for one prover run. Best-effort: any
    POU failure yields an empty map (rules fall back to UNKNOWN)."""
    try:
        checks: list[CheckResult] = api.get_all_checks(link)
    except Exception:
        _log.warning("autoprove report: POU get_all_checks failed for %s", link, exc_info=True)
        return {}

    verdicts: dict[str, _Verdict] = {}
    for c in checks:
        line = c.source_location.line if c.source_location else None
        cand = _Verdict(c.status, line, c.duration or None)
        prev = verdicts.get(c.rule_name)
        if prev is None or _STATUS_PRIORITY.get(c.status, 0) > _STATUS_PRIORITY.get(prev.status, 0):
            # Keep a line/duration even if a later (higher-priority) check lacks one.
            verdicts[c.rule_name] = _Verdict(
                cand.status,
                cand.line if cand.line is not None else (prev.line if prev else None),
                cand.duration_seconds if cand.duration_seconds is not None
                else (prev.duration_seconds if prev else None),
            )
    return verdicts


def collect(
    project_root: str,
    components: list[ComponentInput],
    *,
    api: ProverOutputAPI | None = None,
) -> tuple[list[InferredProperty], list[CVLRule]]:
    """Assemble the report's inferred properties and CVL rules.

    Components are processed in the given order with first-write-wins on rule
    name, so a rule re-stated across components (e.g. a structural invariant
    imported into a component spec) keeps its first definition; duplicates are
    logged. Pass components in the order you want that precedence (components
    before invariants).
    """
    pdir = _properties_dir(project_root)
    api = api or ProverOutputAPI()

    all_properties: list[InferredProperty] = []
    rules_by_name: dict[str, CVLRule] = {}

    for comp in components:
        props = _load_properties(pdir, comp.stem, comp.name)
        if not props and comp.prover_link is None:
            continue
        all_properties.extend(props)

        rule_refs = _load_rule_refs(pdir, comp.stem, comp.name, props)
        verdicts = _fetch_verdicts(api, comp.prover_link) if comp.prover_link else {}

        # Union of rules the agent mapped to properties and rules the prover
        # reported (the latter may include helper rules with no property ref).
        for rule_name in set(rule_refs) | set(verdicts):
            if rule_name in rules_by_name:
                _log.info(
                    "autoprove report: rule %r seen in multiple components; "
                    "keeping the first (%s)",
                    rule_name, rules_by_name[rule_name].component,
                )
                continue
            v = verdicts.get(rule_name)
            rules_by_name[rule_name] = CVLRule(
                name=rule_name,
                component=comp.name,
                property_refs=rule_refs.get(rule_name, []),
                status=v.status if v else NodeStatus.UNKNOWN,
                line=v.line if v else None,
                duration_seconds=v.duration_seconds if v else None,
                prover_link=comp.prover_link,
            )

    rules = sorted(rules_by_name.values(), key=lambda r: (r.component, r.name))
    return all_properties, rules
