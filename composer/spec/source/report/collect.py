"""Collect the report's inputs from in-memory pipeline results + prover verdicts.

For each component (and the structural invariants) the report phase hands us the inferred
properties, the generation result (`GeneratedCVL` with its skip list + property->rules mapping, or
a give-up/crash), and the component's prover-run link. We split the properties into the ones a rule
formalizes (`FormalizedProperty`) and the formalization gaps (`SkippedClaim` / `GaveUpComponent`),
and fetch per-rule verdicts from each run via ProverOutputUtility. No on-disk dumps are read — the
data is already in memory.
"""
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from prover_output_utility import ProverOutputAPI
from prover_output_utility.models import CheckResult, NodeStatus

from composer.spec.prop import PropertyFormulation
from composer.spec.cvl_generation import GeneratedCVL
from composer.spec.source.author import GaveUp
from composer.spec.source.report.schema import (
    ComponentName, FormalizedProperty, GaveUpComponent, RuleName, RuleRef, RuleVerdict, SkippedClaim,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportComponentInput:
    """One unit to collect: a component or the structural invariants. ``spec_file`` is the basename
    of the spec the rules live in (``autospec_<slug>.spec`` / ``invariants.spec``) — the rule-identity
    fallback when a verdict carries no source location. ``result`` is the in-memory generation
    outcome; anything other than `GeneratedCVL` (give-up or crash) means no rules were formalized."""
    name: ComponentName
    spec_file: str
    props: list[PropertyFormulation]
    result: GeneratedCVL | GaveUp | BaseException
    prover_link: str | None


# Rollup priority when a rule has several per-method CheckResults: the most terminal verdict wins.
_STATUS_PRIORITY: dict[NodeStatus, int] = {
    NodeStatus.VIOLATED: 6, NodeStatus.ERROR: 5, NodeStatus.TIMEOUT: 4,
    NodeStatus.UNKNOWN: 3, NodeStatus.RUNNING: 2, NodeStatus.PENDING: 1, NodeStatus.VERIFIED: 0,
}


@dataclass(frozen=True)
class _Verdict:
    status: NodeStatus
    line: int | None
    duration_seconds: float | None
    spec_file: str | None = None

    def merge(self, other: "_Verdict | None") -> "_Verdict":
        """Combine two CheckResults for one rule within a run: higher-priority status wins,
        line/duration/spec_file kept from whichever side has them."""
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


def _fetch_verdicts(api: ProverOutputAPI, link: str) -> dict[RuleName, _Verdict]:
    """rule_name -> rolled-up `_Verdict` for one prover run. Best-effort: any POU failure -> {}."""
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


async def collect(
    inputs: list[ReportComponentInput],
    *,
    api: ProverOutputAPI | None = None,
) -> tuple[list[FormalizedProperty], list[RuleVerdict], list[SkippedClaim], list[GaveUpComponent], int]:
    """Assemble the report inputs.

    Returns ``(formalized_properties, rules, skipped, gave_up_components, dropped_orphan_count)``.
    Rules are identified by ``(spec_file, name)``: a single definition seen through several runs
    (e.g. a structural invariant imported into a component spec) collapses to one entry. Orphan
    rules — reported by the prover but referenced by no property — are dropped and counted. Verdicts
    are fetched concurrently (one blocking POU call per run, off the event loop).
    """
    api = api or ProverOutputAPI()

    async def _verdicts_for(inp: ReportComponentInput) -> dict[RuleName, _Verdict]:
        if isinstance(inp.result, GeneratedCVL) and inp.prover_link:
            return await asyncio.to_thread(_fetch_verdicts, api, inp.prover_link)
        return {}

    verdict_maps = await asyncio.gather(*[_verdicts_for(inp) for inp in inputs])

    properties: list[FormalizedProperty] = []
    skipped: list[SkippedClaim] = []
    gave_up: list[GaveUpComponent] = []
    rules_by_key: dict[RuleRef, RuleVerdict] = {}
    referenced: set[RuleRef] = set()

    for inp, verdicts in zip(inputs, verdict_maps):
        if not isinstance(inp.result, GeneratedCVL):
            # GaveUp or a crashed batch (BaseException): the whole component is a formalization gap.
            gave_up.append(GaveUpComponent(component=inp.name, properties=inp.props))
            continue
        res = inp.result
        skip_reasons = {s.property_title: s.reason for s in res.skipped}
        mapping = {m.property_title: m.rules for m in res.property_rules}

        def _ref(rule_name: str) -> RuleRef:
            v = verdicts.get(rule_name)
            return ((v.spec_file if v and v.spec_file else inp.spec_file), rule_name)

        for prop in inp.props:
            if prop.title in skip_reasons:
                skipped.append(SkippedClaim(
                    component=inp.name, reason=skip_reasons[prop.title], **prop.model_dump()
                ))
            elif prop.title in mapping:
                refs = [_ref(rn) for rn in mapping[prop.title] if rn.strip()]
                referenced.update(refs)
                properties.append(FormalizedProperty(
                    component=inp.name, rule_refs=refs, **prop.model_dump()
                ))
            else:
                # The completion validator guarantees skipped-or-mapped; a residue means the
                # property/skip/mapping disagree. Drop rather than invent a record.
                _log.warning(
                    "autoprove report: property %r in %s is neither skipped nor mapped; dropping",
                    prop.title, inp.name,
                )

        # Register every rule the prover reported (first run naming a (spec_file, rule) wins).
        for rule_name, v in verdicts.items():
            key = (v.spec_file or inp.spec_file, rule_name)
            if key not in rules_by_key:
                rules_by_key[key] = RuleVerdict(
                    name=rule_name, spec_file=key[0], status=v.status, line=v.line,
                    duration_seconds=v.duration_seconds, prover_link=inp.prover_link,
                )

    # A referenced rule with no prover verdict still needs an (UNKNOWN) entry to render.
    for ref in referenced:
        if ref not in rules_by_key:
            rules_by_key[ref] = RuleVerdict(name=ref[1], spec_file=ref[0])

    rules = sorted(
        (rv for key, rv in rules_by_key.items() if key in referenced),
        key=lambda r: r.ref,
    )
    dropped_orphans = sum(1 for key in rules_by_key if key not in referenced)
    return properties, rules, skipped, gave_up, dropped_orphans
