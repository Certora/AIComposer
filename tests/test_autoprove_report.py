"""Tests for the autoprove report package (composer.spec.source.report).

Property-keyed (schema 2.0). Covers the pure pieces — in-memory collect against a
fake POU, status aggregation, grouping + fallback, coverage's property-partition,
HTML render — plus the build orchestrator. No DB / no real LLM / no real prover:
POU is faked, the grouping LLM is a `BaseChatModel` stub whose structured output is
preset (so the real `call_grouping_llm` — templates + parsing — still runs), and
inputs are in-memory `GeneratedCVL` / `GaveUp` objects (collect no longer reads the
JSON dumps).
"""
from types import SimpleNamespace
from typing import cast

import pytest
from prover_output_utility.models import NodeStatus
from prover_output_utility import ProverOutputAPI
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from composer.spec.prop import PropertyFormulation, PropertyType
from composer.spec.cvl_generation import GeneratedCVL, PropertyRuleMapping, SkippedProperty
from composer.spec.source.author import GaveUp

from composer.spec.source.artifacts import ProverArtifactStore
from composer.spec.source.report import build
from composer.spec.source.report.collect import ReportComponentInput, collect
from composer.spec.source.report.coverage import ValidationError, validate
from composer.spec.source.report.grouping import (
    FALLBACK_SLUG, GroupingResult, PropertyGroupDraft, aggregate_status,
    build_fallback_grouping, build_groups,
)
from composer.spec.source.report.render import render_html
from composer.spec.source.report.schema import (
    AutoProverReport, CoverageReport, FormalizedProperty, GaveUpComponent, GroupStatus,
    PropertyGroup, RuleVerdict, SkippedClaim,
)


# ---------------------------------------------------------------------------
# Fakes / builders
# ---------------------------------------------------------------------------

def _fake_check(rule_name, status, line=None, duration=None, file: str | None = "autospec_Increment.spec"):
    """Stand-in CheckResult. ``file`` is the spec the rule is defined in (POU's source
    location); pass ``file=None`` to simulate POU not reporting one."""
    sl = SimpleNamespace(file=file, line=line)
    return SimpleNamespace(rule_name=rule_name, status=status, duration=duration, source_location=sl)


class _FakeAPI_Impl:
    """Stand-in for ProverOutputAPI: get_all_checks(link) -> list of checks."""
    def __init__(self, by_link: dict[str, list]):
        self.by_link = by_link

    def get_all_checks(self, link):
        return self.by_link.get(link, [])
    
def _FakeAPI(by_link: dict[str, list]) -> ProverOutputAPI:
    return cast(ProverOutputAPI, _FakeAPI_Impl(by_link))


def _prop(title, desc, *, sort: PropertyType = "safety_property", methods=None) -> PropertyFormulation:
    return PropertyFormulation(title=title, methods=methods or ["m"], sort=sort, description=desc)


def _gen(mapping: dict[str, list[str]] | None = None,
         skipped: dict[str, str] | None = None) -> GeneratedCVL:
    """A successful generation result: ``mapping`` is property_title -> [rule names];
    ``skipped`` is property_title -> reason."""
    return GeneratedCVL(
        commentary="", cvl="",
        property_rules=[PropertyRuleMapping(property_title=t, rules=rs)
                        for t, rs in (mapping or {}).items()],
        skipped=[SkippedProperty(property_title=t, reason=r)
                 for t, r in (skipped or {}).items()],
    )


def _input(name, spec_file, props, result, link: str | None = "L1") -> ReportComponentInput:
    return ReportComponentInput(name=name, spec_file=spec_file, props=props,
                                result=result, prover_link=link)


def _fp(component, title, refs, desc="d", sort: PropertyType = "safety_property") -> FormalizedProperty:
    return FormalizedProperty(component=component, title=title, methods=["m"],
                              sort=sort, description=desc, rule_refs=refs)


def _rv(spec, name, status=NodeStatus.VERIFIED) -> RuleVerdict:
    return RuleVerdict(name=name, spec_file=spec, status=status)


def _pg(slug, members, status=GroupStatus.VERIFIED) -> PropertyGroup:
    return PropertyGroup(slug=slug, title="T", description="d", status=status, members=members)


class _GroupingStubModel(BaseChatModel):
    """A `BaseChatModel` whose structured-output binding returns a preset `GroupingResult`.
    Lets the build tests drive the *real* `call_grouping_llm` — template rendering + the
    `isinstance` check — without a live model, only stubbing the model's output."""
    result: GroupingResult

    def with_structured_output(self, schema, **kwargs) -> Runnable:  # type: ignore[override]
        result = self.result
        return RunnableLambda(lambda _messages: result)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("stub is structured-output only")

    @property
    def _llm_type(self) -> str:
        return "grouping-stub"


# ---------------------------------------------------------------------------
# collect (async, in-memory)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collect_joins_properties_to_rules_and_verdicts():
    props = [_prop("count_increases", "count up by one"),
             _prop("count_eq_sum", "count == sum", sort="invariant", methods="invariant")]
    gen = _gen({"count_increases": ["increment_increases_count"], "count_eq_sum": ["countEqualsSum"]})
    api = _FakeAPI({"L1": [
        _fake_check("increment_increases_count", NodeStatus.VERIFIED, line=12, duration=1.5),
        _fake_check("countEqualsSum", NodeStatus.VIOLATED, line=40),
    ]})

    properties, rules, skipped, gave_up, dropped = await collect(
        [_input("Increment", "autospec_Increment.spec", props, gen)], api=api)

    assert [p.title for p in properties] == ["count_increases", "count_eq_sum"]
    assert properties[0].component == "Increment"
    assert properties[0].rule_refs == [("autospec_Increment.spec", "increment_increases_count")]
    by_ref = {r.ref: r for r in rules}
    r = by_ref[("autospec_Increment.spec", "increment_increases_count")]
    assert r.status == NodeStatus.VERIFIED and r.line == 12 and r.duration_seconds == 1.5
    assert r.prover_link == "L1"
    assert by_ref[("autospec_Increment.spec", "countEqualsSum")].status == NodeStatus.VIOLATED
    assert skipped == [] and gave_up == [] and dropped == 0


@pytest.mark.asyncio
async def test_collect_splits_skipped_property_into_gap():
    props = [_prop("p_done", "formalized"), _prop("p_skip", "cannot express in CVL")]
    gen = _gen({"p_done": ["r1"]}, skipped={"p_skip": "needs a ghost"})
    api = _FakeAPI({"L1": [_fake_check("r1", NodeStatus.VERIFIED)]})

    properties, _rules, skipped, gave_up, _dropped = await collect(
        [_input("C", "autospec_C.spec", props, gen)], api=api)

    assert [p.title for p in properties] == ["p_done"]
    assert [(s.component, s.title, s.reason) for s in skipped] == [("C", "p_skip", "needs a ghost")]
    assert gave_up == []


@pytest.mark.asyncio
async def test_collect_gave_up_or_crashed_component_is_a_gap():
    props = [_prop("p1", "d1")]
    for result in (GaveUp(reason="stuck"), RuntimeError("boom")):
        properties, rules, skipped, gave_up, dropped = await collect(
            [_input("C", "autospec_C.spec", props, result, link=None)], api=_FakeAPI({}))
        assert properties == [] and rules == [] and skipped == [] and dropped == 0
        assert [g.component for g in gave_up] == ["C"]
        assert [p.title for p in gave_up[0].properties] == ["p1"]


@pytest.mark.asyncio
async def test_collect_drops_and_counts_orphan_rules():
    """A rule the prover reported but no property maps to is dropped and counted."""
    gen = _gen({"p1": ["r1"]})
    api = _FakeAPI({"L1": [
        _fake_check("r1", NodeStatus.VERIFIED),
        _fake_check("sanity_helper", NodeStatus.VERIFIED),  # referenced by nothing
    ]})
    _props, rules, _skipped, _gave_up, dropped = await collect(
        [_input("C", "autospec_C.spec", [_prop("p1", "d1")], gen)], api=api)
    assert [r.name for r in rules] == ["r1"]
    assert dropped == 1


@pytest.mark.asyncio
async def test_collect_backfills_unknown_for_unproven_referenced_rule():
    gen = _gen({"p1": ["r1"]})
    api = _FakeAPI({"L1": []})  # prover reported no checks
    properties, rules, _s, _g, dropped = await collect(
        [_input("C", "autospec_C.spec", [_prop("p1", "d1")], gen)], api=api)
    assert [(r.name, r.status, r.spec_file) for r in rules] == [("r1", NodeStatus.UNKNOWN, "autospec_C.spec")]
    assert properties[0].rule_refs == [("autospec_C.spec", "r1")]
    assert dropped == 0


@pytest.mark.asyncio
async def test_collect_falls_back_to_input_spec_when_verdict_has_no_source():
    """A verdict without a source location is attributed to the component's own spec
    (no raise — the report is best-effort and every input carries a spec_file)."""
    gen = _gen({"p1": ["r1"]})
    api = _FakeAPI({"L1": [_fake_check("r1", NodeStatus.VERIFIED, file=None)]})
    properties, rules, *_ = await collect(
        [_input("C", "autospec_C.spec", [_prop("p1", "d1")], gen)], api=api)
    assert rules[0].ref == ("autospec_C.spec", "r1")
    assert properties[0].rule_refs == [("autospec_C.spec", "r1")]


@pytest.mark.asyncio
async def test_collect_shared_rule_dedupes_and_is_referenced_by_both():
    """An invariant imported into a component spec reports the same source file from
    both runs, so it collapses to one rule that both components' properties reference."""
    comp = _input("Increment", "autospec_Increment.spec", [_prop("c", "component view", sort="invariant")],
                  _gen({"c": ["countEqualsSum"]}), link="Lc")
    inv = _input("Structural Invariants", "invariants.spec", [_prop("i", "structural", sort="invariant")],
                 _gen({"i": ["countEqualsSum"]}), link="Li")
    api = _FakeAPI({
        "Lc": [_fake_check("countEqualsSum", NodeStatus.VERIFIED, file="invariants.spec")],
        "Li": [_fake_check("countEqualsSum", NodeStatus.VERIFIED, file="invariants.spec")],
    })
    properties, rules, *_ = await collect([comp, inv], api=api)
    ces = [r for r in rules if r.name == "countEqualsSum"]
    assert len(ces) == 1 and ces[0].spec_file == "invariants.spec"
    assert all(p.rule_refs == [("invariants.spec", "countEqualsSum")] for p in properties)


@pytest.mark.asyncio
async def test_collect_same_name_different_spec_stays_distinct():
    a = _input("A", "autospec_A.spec", [_prop("pa", "a")], _gen({"pa": ["transferIsSafe"]}), link="La")
    b = _input("B", "autospec_B.spec", [_prop("pb", "b")], _gen({"pb": ["transferIsSafe"]}), link="Lb")
    api = _FakeAPI({
        "La": [_fake_check("transferIsSafe", NodeStatus.VERIFIED, file="autospec_A.spec")],
        "Lb": [_fake_check("transferIsSafe", NodeStatus.VIOLATED, file="autospec_B.spec")],
    })
    _props, rules, *_ = await collect([a, b], api=api)
    safe = sorted((r for r in rules if r.name == "transferIsSafe"), key=lambda r: r.spec_file)
    assert [(r.spec_file, r.status) for r in safe] == [
        ("autospec_A.spec", NodeStatus.VERIFIED),
        ("autospec_B.spec", NodeStatus.VIOLATED),
    ]


# ---------------------------------------------------------------------------
# aggregate_status
# ---------------------------------------------------------------------------

def test_aggregate_status_table():
    assert aggregate_status([]) == GroupStatus.NO_RESULTS
    assert aggregate_status([NodeStatus.VERIFIED, NodeStatus.VERIFIED]) == GroupStatus.VERIFIED
    assert aggregate_status([NodeStatus.VERIFIED, NodeStatus.VIOLATED]) == GroupStatus.VIOLATED
    assert aggregate_status([NodeStatus.VERIFIED, NodeStatus.TIMEOUT]) == GroupStatus.PARTIAL
    assert aggregate_status([NodeStatus.TIMEOUT, NodeStatus.UNKNOWN]) == GroupStatus.NO_RESULTS


def test_aggregate_status_idempotent_under_duplicates():
    once = aggregate_status([NodeStatus.VERIFIED, NodeStatus.TIMEOUT])
    twice = aggregate_status([NodeStatus.VERIFIED, NodeStatus.VERIFIED, NodeStatus.TIMEOUT])
    assert once == twice == GroupStatus.PARTIAL


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def test_build_groups_rolls_up_status_over_member_rule_verdicts():
    p1 = _fp("C", "p1", [("s.spec", "a")])
    p2 = _fp("C", "p2", [("s.spec", "b")])
    props_by_key = {p.key: p for p in (p1, p2)}
    rule_status = {("s.spec", "a"): NodeStatus.VERIFIED, ("s.spec", "b"): NodeStatus.VIOLATED}
    draft = PropertyGroupDraft(slug="g", title="G", description="d", members=[("C", "p1"), ("C", "p2")])

    groups = build_groups([draft], props_by_key, rule_status)

    assert len(groups) == 1
    assert groups[0].status == GroupStatus.VIOLATED  # one member rule violated
    assert groups[0].members == [("C", "p1"), ("C", "p2")]


def test_build_fallback_grouping_covers_all_properties_once():
    out = build_fallback_grouping([_fp("C", "p1", [("s.spec", "a")]), _fp("D", "p2", [("s.spec", "b")])])
    assert len(out.groups) == 1
    g = out.groups[0]
    assert g.slug == FALLBACK_SLUG
    assert g.members == [("C", "p1"), ("D", "p2")]


# ---------------------------------------------------------------------------
# coverage (property partition; rule repetition is a stat, not an error)
# ---------------------------------------------------------------------------

def test_validate_property_in_two_groups_raises():
    props = [_fp("C", "p1", [("s.spec", "a")])]
    groups = [_pg("g1", [("C", "p1")]), _pg("g2", [("C", "p1")])]
    with pytest.raises(ValidationError, match="multiple groups"):
        validate(properties=props, rules=[_rv("s.spec", "a")], groups=groups,
                 skipped=[], gave_up=[], dropped_orphan_rules=0)


def test_validate_unknown_property_member_raises():
    props = [_fp("C", "p1", [("s.spec", "a")])]
    groups = [_pg("g", [("C", "ghost")])]
    with pytest.raises(ValidationError, match="don't exist"):
        validate(properties=props, rules=[_rv("s.spec", "a")], groups=groups,
                 skipped=[], gave_up=[], dropped_orphan_rules=0)


def test_validate_property_in_no_group_is_soft():
    props = [_fp("C", "p1", [("s.spec", "a")]), _fp("C", "p2", [("s.spec", "b")])]
    groups = [_pg("g", [("C", "p1")])]
    cov = validate(properties=props, rules=[_rv("s.spec", "a"), _rv("s.spec", "b")],
                   groups=groups, skipped=[], gave_up=[], dropped_orphan_rules=0)
    assert cov.property_coverage_complete is False
    assert cov.properties_in_no_group == [("C", "p2")]


def test_validate_reports_rules_spanning_groups_as_stat():
    """A rule formalizing properties that land in different groups is expected
    (rules repeat) — reported as an informational stat, not an error."""
    p1 = _fp("C", "p1", [("s.spec", "shared")])
    p2 = _fp("C", "p2", [("s.spec", "shared")])
    groups = [_pg("g1", [("C", "p1")]), _pg("g2", [("C", "p2")])]
    cov = validate(properties=[p1, p2], rules=[_rv("s.spec", "shared")], groups=groups,
                   skipped=[], gave_up=[], dropped_orphan_rules=2)
    assert cov.rules_spanning_multiple_groups == ["shared"]
    assert cov.dropped_orphan_rules == 2


def test_validate_carries_gap_counts():
    p1 = _fp("C", "p1", [("s.spec", "a")])
    sk = [SkippedClaim(component="C", title="s1", methods=["m"], sort="safety_property",
                       description="d", reason="r")]
    gu = [GaveUpComponent(component="D", properties=[_prop("x", "d")])]
    cov = validate(properties=[p1], rules=[_rv("s.spec", "a")], groups=[_pg("g", [("C", "p1")])],
                   skipped=sk, gave_up=gu, dropped_orphan_rules=3)
    assert (cov.skipped_count, cov.gave_up_component_count, cov.dropped_orphan_rules) == (1, 1, 3)
    assert cov.property_coverage_complete is True


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def _mini_report() -> AutoProverReport:
    # Two properties in one group share a single rule -> the rule row should carry
    # both in-group descriptions as a bullet list (the edge-label projection).
    p1 = _fp("C", "p_pay", [("c.spec", "revert_char")], desc="must accept ETH when value > 0")
    p2 = _fp("C", "p_open", [("c.spec", "revert_char")], desc="callable by any address")
    rules = [RuleVerdict(name="revert_char", spec_file="c.spec", status=NodeStatus.VERIFIED,
                         line=7, prover_link="https://prover.example/run/abc")]
    groups = [PropertyGroup(slug="deposit-openness", title="Deposit is open", description="d",
                            status=GroupStatus.VERIFIED, members=[("C", "p_pay"), ("C", "p_open")])]
    skipped = [SkippedClaim(component="C", title="atomic_on_revert", methods=["m"],
                            sort="safety_property", description="revert rolls back state",
                            reason="tautological under EVM semantics")]
    cov = CoverageReport(total_properties=2, total_rules=1, total_groups=1,
                         properties_per_group_min=2, properties_per_group_max=2,
                         property_coverage_complete=True)
    return AutoProverReport(contract_name="Counter",
                            prover_links={"C": "https://prover.example/run/abc"},
                            properties=[p1, p2], rules=rules, groups=groups,
                            skipped=skipped, coverage=cov)


def test_render_html_group_rows_and_edge_labels():
    h = render_html(_mini_report())
    assert "deposit-openness" in h and "Deposit is open" in h
    assert 'href="https://prover.example/run/abc"' in h
    # the shared rule row lists BOTH in-group property descriptions
    assert '<ul class="claims">' in h
    assert "must accept ETH" in h and "callable by any address" in h


def test_render_html_autoescapes_descriptions():
    h = render_html(_mini_report())
    assert "value &gt; 0" in h  # the ">" in the description is escaped, not raw


def test_render_html_gaps_section_and_footer_bool():
    h = render_html(_mini_report())
    assert "Formalization gaps" in h
    assert "revert rolls back state" in h and "tautological under EVM semantics" in h
    assert "Coverage complete: <strong>Yes</strong>" in h  # no raw Python bool


# ---------------------------------------------------------------------------
# build orchestrator (async)
# ---------------------------------------------------------------------------

def test_artifact_store_write_report_round_trips(tmp_path):
    report = _mini_report()
    ProverArtifactStore(str(tmp_path), "Counter").write_report(report)

    out = tmp_path / "certora" / "ap_report" / "report.json"
    assert out.is_file()
    reloaded = AutoProverReport.model_validate_json(out.read_text())
    assert reloaded.contract_name == "Counter"


@pytest.mark.asyncio
async def test_build_groups_properties(tmp_path):
    gen = _gen({"p1": ["r1"], "p2": ["r2"]})
    api = _FakeAPI({"L1": [_fake_check("r1", NodeStatus.VERIFIED), _fake_check("r2", NodeStatus.VERIFIED)]})
    llm = _GroupingStubModel(result=GroupingResult(groups=[PropertyGroupDraft(
        slug="g", title="G", description="d", members=[("C", "p1"), ("C", "p2")])]))

    report = await build.run_autoprove_report(
        contract_name="Counter",
        components=[_input("C", "autospec_C.spec", [_prop("p1", "d1"), _prop("p2", "d2")], gen)],
        llm=llm, api=api,
    )

    assert [g.slug for g in report.groups] == ["g"]
    assert {p.title for p in report.properties} == {"p1", "p2"}
    assert report.coverage.property_coverage_complete is True

@pytest.mark.asyncio
async def test_build_empty_grouping_falls_back(tmp_path):
    gen = _gen({"p1": ["r1"], "p2": ["r2"]})
    api = _FakeAPI({"L1": [_fake_check("r1", NodeStatus.VERIFIED), _fake_check("r2", NodeStatus.VIOLATED)]})
    llm = _GroupingStubModel(result=GroupingResult(groups=[]))  # empty grouping -> fallback

    report = await build.run_autoprove_report(
        contract_name="C",
        components=[_input("C", "autospec_C.spec", [_prop("p1", "d1"), _prop("p2", "d2")], gen)],
        llm=llm, api=api,
    )

    assert [g.slug for g in report.groups] == [FALLBACK_SLUG]
    g = report.groups[0]
    assert set(g.members) == {("C", "p1"), ("C", "p2")}
    assert g.status == GroupStatus.VIOLATED  # r2 violated
    assert any("FALLBACK GROUPING APPLIED" in w for w in report.coverage.warnings)

@pytest.mark.asyncio
async def test_build_surfaces_skipped_and_gave_up_gaps(tmp_path):
    gen = _gen({"p_ok": ["r1"]}, skipped={"p_skip": "needs a ghost"})
    api = _FakeAPI({"L1": [_fake_check("r1", NodeStatus.VERIFIED)]})
    llm = _GroupingStubModel(result=GroupingResult(groups=[PropertyGroupDraft(
        slug="g", title="G", description="d", members=[("C", "p_ok")])]))

    report = await build.run_autoprove_report(
        contract_name="C",
        components=[
            _input("C", "autospec_C.spec", [_prop("p_ok", "d"), _prop("p_skip", "d")], gen),
            _input("D", "autospec_D.spec", [_prop("q", "d")], GaveUp(reason="stuck"), link=None),
        ],
        llm=llm, api=api,
    )

    assert [(s.component, s.title) for s in report.skipped] == [("C", "p_skip")]
    assert [g.component for g in report.gave_up_components] == ["D"]
    assert report.coverage.skipped_count == 1 and report.coverage.gave_up_component_count == 1
