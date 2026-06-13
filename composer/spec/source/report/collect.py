"""Collect the report's inputs from on-disk dumps + prover verdicts.

For each component (and the structural invariants) we read the property dumps
the pipeline already wrote under ``certora/properties/`` and fetch per-rule
verdicts from that component's prover run via ProverOutputUtility:

  - ``<stem>.properties.json``     -> ordered `PropertyFormulation`s
  - ``<stem>.property_rules.json`` -> ``{property_title: [rule_names]}``
  - the component's prover link    -> POU `CheckResult`s (status + source line)

`<stem>` is ``autospec_<slugified_name>`` for components and ``invariants`` for
the structural invariants. Properties are addressed by their unique ``title``;
rule line numbers and verdicts both come from POU (no spec-text parsing, no
``.certora_recent_jobs.json``).
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from prover_output_utility import ProverOutputAPI
from prover_output_utility.models import CheckResult, NodeStatus

from composer.spec.gen_types import PROPERTIES_DIR, under_project
from composer.spec.source.report.schema import (
    ComponentName, CVLRule, PropertyFormulationWithComponent, PropertyTitle, RuleName, StemName,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentInput:
    """One unit to collect: its human name, the property-dump stem, and the
    prover run link/path for its verdicts (``None`` if the run produced no
    link, e.g. a component that gave up before submitting). One prover run per
    component (and one for the structural invariants)."""
    name: ComponentName
    stem: StemName
    prover_link: str | None


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


@dataclass(frozen=True)
class _Verdict:
    status: NodeStatus
    line: int | None
    duration_seconds: float | None
    spec_file: str | None = None

    def merge(self, other: "_Verdict | None") -> "_Verdict":
        """Combine two CheckResults for the same rule within one prover run: the
        higher-priority (more terminal) status wins, and line/duration/spec_file
        are kept from whichever side has them."""
        if other is None:
            return self
        hi, lo = (
            (self, other)
            if _STATUS_PRIORITY.get(self.status, 0) >= _STATUS_PRIORITY.get(other.status, 0)
            else (other, self)
        )
        return _Verdict(
            hi.status,
            hi.line if hi.line is not None else lo.line,
            hi.duration_seconds if hi.duration_seconds is not None else lo.duration_seconds,
            hi.spec_file or lo.spec_file,
        )


def _properties_dir(project_root: str) -> Path:
    return under_project(project_root, PROPERTIES_DIR)


def _load_properties(
    pdir: Path, stem: StemName, component: ComponentName
) -> list[PropertyFormulationWithComponent]:
    """Load ``<stem>.properties.json`` and wrap each entry as a
    `PropertyFormulationWithComponent`, tagging it with the component that owns
    the spec. Missing file -> empty."""
    path = pdir / f"{stem}.properties.json"
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    return [PropertyFormulationWithComponent(component=component, **prop) for prop in raw]


def _load_rule_properties(
    pdir: Path, stem: StemName, properties: list[PropertyFormulationWithComponent]
) -> dict[RuleName, list[PropertyFormulationWithComponent]]:
    """Invert ``<stem>.property_rules.json`` into
    ``rule_name -> [the property formulations that rule implements]``, resolving
    each title to its property object so callers need no second join. Missing
    file -> empty (a component that gave up after extraction has properties but
    no rule mapping)."""
    path = pdir / f"{stem}.property_rules.json"
    if not path.is_file():
        return {}
    by_title = {p.title: p for p in properties if p.title}
    # property_rules.json maps each PropertyTitle to the RuleNames implementing it.
    mapping: dict[PropertyTitle, list[RuleName]] = json.loads(path.read_text())
    out: dict[RuleName, list[PropertyFormulationWithComponent]] = {}
    for title, rule_names in mapping.items():
        prop = by_title.get(title)
        if prop is None:
            # A title in the mapping with no matching property entry means the
            # two files disagree; skip rather than reference a phantom property.
            continue
        for rule_name in rule_names:
            out.setdefault(rule_name, []).append(prop)
    return out


def _fetch_verdicts(api: ProverOutputAPI, link: str) -> dict[RuleName, _Verdict]:
    """rule_name -> rolled-up `_Verdict` for one prover run. Best-effort: any
    POU failure yields an empty map (rules fall back to UNKNOWN)."""
    try:
        checks: list[CheckResult] = api.get_all_checks(link)
    except Exception:
        _log.warning("autoprove report: POU get_all_checks failed for %s", link, exc_info=True)
        return {}

    verdicts: dict[RuleName, _Verdict] = {}
    for c in checks:
        loc = c.source_location
        cand = _Verdict(
            c.status,
            loc.line if loc else None,
            c.duration or None,
            Path(loc.file).name if (loc and loc.file) else None,
        )
        verdicts[c.rule_name] = cand.merge(verdicts.get(c.rule_name))
    return verdicts


def collect(
    project_root: str,
    components: list[ComponentInput],
    *,
    api: ProverOutputAPI | None = None,
) -> tuple[list[PropertyFormulationWithComponent], list[CVLRule]]:
    """Assemble the report's inferred properties and CVL rules.

    Rules are identified by ``(spec_file, rule_name)``: a rule re-stated under
    the same name in a different spec stays distinct, while a single definition
    seen through several components (e.g. a structural invariant imported into
    component specs) collapses to one entry. The defining spec file comes from
    POU's source location; when absent it falls back to the component's own
    spec. Components are processed in the given order with first-write-wins on
    that key, so pass components before invariants for the precedence you want.
    """
    pdir = _properties_dir(project_root)
    api = api or ProverOutputAPI()

    all_properties: list[PropertyFormulationWithComponent] = []
    rules_by_key: dict[tuple[str, RuleName], CVLRule] = {}

    for comp in components:
        props = _load_properties(pdir, comp.stem, comp.name)
        if not props and comp.prover_link is None:
            continue
        all_properties.extend(props)

        rule_props = _load_rule_properties(pdir, comp.stem, props)
        verdicts = _fetch_verdicts(api, comp.prover_link) if comp.prover_link else {}

        comp_spec = f"{comp.stem}.spec"  # identity fallback when POU has no source location

        # Union of rules the agent mapped to properties and rules the prover
        # reported (the latter may include helper rules with no property).
        for rule_name in set(rule_props) | set(verdicts):
            v = verdicts.get(rule_name)
            if v is None:
                # No prover run reported this rule -> identify it by the
                # component's own spec.
                spec_file = comp_spec
            elif v.spec_file is None:
                # A proved rule must carry a source location; if POU doesn't give
                # one we can't determine the rule's defining spec — fail loudly
                # rather than mis-attribute it to the component stem.
                raise ValueError(
                    f"autoprove report: prover verdict for rule {rule_name!r} in component "
                    f"{comp.name!r} has no source location; cannot identify its defining spec."
                )
            else:
                spec_file = v.spec_file
            key = (spec_file, rule_name)
            if key in rules_by_key:
                _log.info(
                    "autoprove report: rule %r in %s already collected (first from %s); keeping the first",
                    rule_name, spec_file, rules_by_key[key].component,
                )
                continue
            rules_by_key[key] = CVLRule(
                name=rule_name,
                component=comp.name,
                spec_file=spec_file,
                properties=rule_props.get(rule_name, []),
                status=v.status if v else NodeStatus.UNKNOWN,
                line=v.line if v else None,
                duration_seconds=v.duration_seconds if v else None,
                prover_link=comp.prover_link,
            )

    rules = sorted(rules_by_key.values(), key=lambda r: (r.component, r.name))
    return all_properties, rules
