"""Tests for the autoprove report package (composer.spec.source.report).

Covers the pure pieces (collect from property dumps + a fake POU, coverage
aggregation, grouping + fallback, HTML render) and the build orchestrator's
fallback path. No DB / no real LLM / no real prover: POU is faked and the
grouping LLM is monkeypatched.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from prover_output_utility.models import NodeStatus

from composer.spec.source.report import build
from composer.spec.source.report.collect import ComponentInput, collect
from composer.spec.source.report.coverage import ValidationError, validate
from composer.spec.source.report.grouping import (
    FALLBACK_SLUG, aggregate_status, build_fallback_grouping,
    build_implemented_properties, build_rules_for_grouping,
)
from composer.spec.source.report.render import render_html
from composer.spec.source.report.schema import (
    AutoProverReport, CVLRule, CoverageReport, GroupStatus, ImplementedProperty,
    ImplementedPropertyDraft, PropertyFormulationWithComponent,
)


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------

def _fake_check(rule_name, status, line=None, duration=None, file: str | None = "autospec_Increment.spec"):
    """A stand-in CheckResult. ``file`` is the spec the rule is defined in (per
    POU's source location); pass ``file=None`` to simulate POU not reporting one."""
    sl = SimpleNamespace(file=file, line=line)
    return SimpleNamespace(
        rule_name=rule_name, status=status, duration=duration, source_location=sl,
    )


class _FakeAPI:
    """Stand-in for ProverOutputAPI: get_all_checks(link) -> list of checks."""
    def __init__(self, by_link: dict[str, list]):
        self.by_link = by_link

    def get_all_checks(self, link):
        return self.by_link.get(link, [])


def _prop(title, desc, *, sort="safety_property", methods=None):
    return {"title": title, "methods": methods or ["m"], "sort": sort, "description": desc}


def _write_props(certora: Path, stem: str, props: list[dict]):
    pdir = certora / "properties"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{stem}.properties.json").write_text(json.dumps(props))


def _write_rules(certora: Path, stem: str, mapping: dict[str, list[str]]):
    pdir = certora / "properties"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{stem}.property_rules.json").write_text(json.dumps(mapping))


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def test_collect_joins_properties_rules_and_pou_verdicts(tmp_path):
    certora = tmp_path / "certora"
    _write_props(certora, "autospec_Increment", [
        _prop("count_increases", "count up by one"),
        _prop("count_eq_sum", "count == sum", sort="invariant", methods="invariant"),
    ])
    _write_rules(certora, "autospec_Increment", {
        "count_increases": ["increment_increases_count"],
        "count_eq_sum": ["countEqualsSum"],
    })
    api = _FakeAPI({"L1": [
        _fake_check("increment_increases_count", NodeStatus.VERIFIED, line=12, duration=1.5),
        _fake_check("countEqualsSum", NodeStatus.VIOLATED, line=40),
    ]})
    comps = [ComponentInput("Increment", "autospec_Increment", "L1")]

    props, rules = collect(str(tmp_path), comps, api=api)

    # Properties are addressed by title, not a positional index.
    assert [p.title for p in props] == ["count_increases", "count_eq_sum"]
    by_name = {r.name: r for r in rules}
    r = by_name["increment_increases_count"]
    assert r.status == NodeStatus.VERIFIED and r.line == 12
    # The implemented property formulation is embedded on the rule (no second join).
    assert [(p.component, p.title) for p in r.properties] == [("Increment", "count_increases")]
    assert r.properties[0].description == "count up by one"
    assert r.spec_file == "autospec_Increment.spec"
    assert r.prover_link == "L1"
    assert by_name["countEqualsSum"].status == NodeStatus.VIOLATED


def test_collect_missing_rule_mapping_tolerated(tmp_path):
    certora = tmp_path / "certora"
    _write_props(certora, "autospec_Increment", [_prop("p1", "d1")])
    # no property_rules.json; POU still reports a verdict
    api = _FakeAPI({"L1": [_fake_check("helper", NodeStatus.VERIFIED)]})
    props, rules = collect(str(tmp_path), [ComponentInput("Increment", "autospec_Increment", "L1")], api=api)
    assert len(props) == 1
    assert [r.name for r in rules] == ["helper"]
    assert rules[0].properties == []


def test_collect_shared_invariant_dedupes_by_spec_file(tmp_path):
    """An invariant defined in invariants.spec but imported into a component
    spec reports the same source file from both runs, so it collapses to one
    rule (attributed to the first component seen)."""
    certora = tmp_path / "certora"
    _write_props(certora, "autospec_Increment", [_prop("c", "comp view", sort="invariant")])
    _write_rules(certora, "autospec_Increment", {"c": ["countEqualsSum"]})
    _write_props(certora, "invariants", [_prop("i", "structural", sort="invariant")])
    _write_rules(certora, "invariants", {"i": ["countEqualsSum"]})
    api = _FakeAPI({
        "Lc": [_fake_check("countEqualsSum", NodeStatus.VERIFIED, file="invariants.spec")],
        "Li": [_fake_check("countEqualsSum", NodeStatus.VERIFIED, file="invariants.spec")],
    })
    comps = [
        ComponentInput("Increment", "autospec_Increment", "Lc"),
        ComponentInput("Structural Invariants", "invariants", "Li"),
    ]
    _props, rules = collect(str(tmp_path), comps, api=api)
    ces = [r for r in rules if r.name == "countEqualsSum"]
    assert len(ces) == 1 and ces[0].component == "Increment"
    assert ces[0].spec_file == "invariants.spec"


def test_collect_same_name_different_spec_stays_distinct(tmp_path):
    """Two rules that happen to share a name but are defined in different spec
    files are kept as distinct rules (no silent first-write-wins drop)."""
    certora = tmp_path / "certora"
    _write_props(certora, "autospec_A", [_prop("pa", "a")])
    _write_props(certora, "autospec_B", [_prop("pb", "b")])
    api = _FakeAPI({
        "La": [_fake_check("transferIsSafe", NodeStatus.VERIFIED, file="autospec_A.spec")],
        "Lb": [_fake_check("transferIsSafe", NodeStatus.VIOLATED, file="autospec_B.spec")],
    })
    comps = [ComponentInput("A", "autospec_A", "La"), ComponentInput("B", "autospec_B", "Lb")]
    _props, rules = collect(str(tmp_path), comps, api=api)
    safe = sorted((r for r in rules if r.name == "transferIsSafe"), key=lambda r: r.spec_file)
    assert [(r.spec_file, r.status) for r in safe] == [
        ("autospec_A.spec", NodeStatus.VERIFIED),
        ("autospec_B.spec", NodeStatus.VIOLATED),
    ]


def test_collect_fails_when_verdict_has_no_source(tmp_path):
    """A proved rule must carry a source location; if POU reports none we cannot
    identify its defining spec, so collect raises rather than mis-attributing."""
    certora = tmp_path / "certora"
    _write_props(certora, "autospec_Increment", [_prop("p1", "d1")])
    api = _FakeAPI({"L1": [_fake_check("r1", NodeStatus.VERIFIED, file=None)]})
    with pytest.raises(ValueError, match="no source location"):
        collect(str(tmp_path), [ComponentInput("Increment", "autospec_Increment", "L1")], api=api)


def test_collect_tolerates_rule_without_verdict(tmp_path):
    """A rule mapped to a property but with no prover verdict (e.g. the run
    didn't report it) is kept, identified by the component's own spec."""
    certora = tmp_path / "certora"
    _write_props(certora, "autospec_Increment", [_prop("p1", "d1")])
    _write_rules(certora, "autospec_Increment", {"p1": ["r1"]})
    api = _FakeAPI({"L1": []})  # prover run produced no checks for r1
    _props, rules = collect(str(tmp_path), [ComponentInput("Increment", "autospec_Increment", "L1")], api=api)
    assert [(r.name, r.status, r.spec_file) for r in rules] == [
        ("r1", NodeStatus.UNKNOWN, "autospec_Increment.spec")
    ]


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_aggregate_status_table():
    assert aggregate_status([]) == GroupStatus.NO_RESULTS
    assert aggregate_status([NodeStatus.VERIFIED, NodeStatus.VERIFIED]) == GroupStatus.VERIFIED
    assert aggregate_status([NodeStatus.VERIFIED, NodeStatus.VIOLATED]) == GroupStatus.VIOLATED
    assert aggregate_status([NodeStatus.VERIFIED, NodeStatus.TIMEOUT]) == GroupStatus.PARTIAL
    assert aggregate_status([NodeStatus.TIMEOUT, NodeStatus.UNKNOWN]) == GroupStatus.NO_RESULTS


def _rule(name, status=NodeStatus.VERIFIED, component="C"):
    return CVLRule(name=name, component=component, status=status)


def _grp(slug, rule_names, status=GroupStatus.VERIFIED):
    return ImplementedProperty(slug=slug, title="T", description="d",
                               status=status, rule_names=rule_names)


def test_validate_rule_in_two_groups_raises():
    rules = [_rule("a"), _rule("b")]
    groups = [_grp("g1", ["a"]), _grp("g2", ["a", "b"])]
    with pytest.raises(ValidationError, match="multiple groups"):
        validate(rules=rules, groups=groups, total_inferred=2)


def test_validate_hallucinated_rule_raises():
    with pytest.raises(ValidationError, match="don't exist"):
        validate(rules=[_rule("a")], groups=[_grp("g", ["ghost"])], total_inferred=1)


def test_validate_missing_rule_is_soft():
    cov = validate(rules=[_rule("a"), _rule("b")], groups=[_grp("g", ["a"])], total_inferred=2)
    assert cov.rule_coverage_complete is False
    assert cov.rules_in_no_group == ["b"]


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def test_build_implemented_properties_rolls_up_status_and_keeps_slug():
    drafts = [ImplementedPropertyDraft(slug="grp-a", title="Group A", description="d", rule_names=["a", "b"])]
    finals = build_implemented_properties(drafts, {"a": NodeStatus.VERIFIED, "b": NodeStatus.VIOLATED})
    assert len(finals) == 1
    assert finals[0].slug == "grp-a" and finals[0].title == "Group A"
    assert finals[0].status == GroupStatus.VIOLATED


def test_fallback_grouping_covers_all_rules():
    out = build_fallback_grouping(["r1", "r2"])
    assert len(out.groups) == 1 and out.groups[0].slug == FALLBACK_SLUG
    assert out.groups[0].rule_names == ["r1", "r2"]
    assert out.groups[0].description == "All rules."


def test_build_rules_for_grouping_reads_embedded_properties():
    prop = PropertyFormulationWithComponent(component="C", title="t", methods=["m"],
                                            sort="invariant", description="the desc")
    rules = [CVLRule(name="r1", component="C", properties=[prop], status=NodeStatus.VERIFIED)]
    rows = build_rules_for_grouping(rules)
    assert rows[0].property_descriptions == ["the desc"] and rows[0].sorts == ["invariant"]


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def _mini_report() -> AutoProverReport:
    prop = PropertyFormulationWithComponent(component="Increment", title="count_increases",
                                            methods=["increment"], sort="safety_property",
                                            description="increment raises count")
    return AutoProverReport(
        contract_name="Counter",
        prover_links={"Increment": "https://prover.example/run/abc"},
        rules=[CVLRule(name="increment_increases_count", component="Increment",
                       spec_file="autospec_Increment.spec", properties=[prop],
                       status=NodeStatus.VERIFIED, line=10,
                       prover_link="https://prover.example/run/abc")],
        implemented_properties=[ImplementedProperty(slug="count-up", title="Count Increases",
                                                    description="d", status=GroupStatus.VERIFIED,
                                                    rule_names=["increment_increases_count"])],
        coverage=CoverageReport(total_property_formulations=1, total_rules=1, total_groups=1,
                                rules_per_group_min=1, rules_per_group_max=1, rule_coverage_complete=True),
    )


def test_render_html_contains_slug_links_and_appendix():
    h = render_html(_mini_report())
    assert "count-up" in h and "Count Increases" in h  # slug + title (no P-NN)
    assert 'href="https://prover.example/run/abc"' in h
    assert "increment_increases_count" in h  # appendix "implemented by"
    assert "count_increases" in h            # property title in appendix
    assert "VERIFIED" in h


def test_render_appendix_lists_unimplemented_properties():
    orphan = PropertyFormulationWithComponent(component="C", title="orphan_prop",
                                              methods=["m"], sort="safety_property",
                                              description="no rule implements this")
    rep = AutoProverReport(
        contract_name="C", rules=[], implemented_properties=[],
        unimplemented_properties=[orphan],
        coverage=CoverageReport(total_property_formulations=1, total_rules=0, total_groups=0,
                                rules_per_group_min=0, rules_per_group_max=0, rule_coverage_complete=True),
    )
    h = render_html(rep)
    assert "orphan_prop" in h and "(no rule mapping)" in h


# ---------------------------------------------------------------------------
# build orchestrator (async): empty grouping -> fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_empty_grouping_falls_back_and_persists(tmp_path, monkeypatch):
    from composer.spec.source.report.schema import GroupingResult

    certora = tmp_path / "certora"
    _write_props(certora, "autospec_Increment", [_prop("p1", "d1"), _prop("p2", "d2")])
    _write_rules(certora, "autospec_Increment", {"p1": ["r1"], "p2": ["r2"]})
    api = _FakeAPI({"L1": [
        _fake_check("r1", NodeStatus.VERIFIED), _fake_check("r2", NodeStatus.VIOLATED)]})

    async def _empty_llm(**kw):
        return GroupingResult(groups=[])
    monkeypatch.setattr(build, "call_grouping_llm", _empty_llm)

    comps = [ComponentInput("Increment", "autospec_Increment", "L1")]
    report = await build.run_autoprove_report(
        project_root=str(tmp_path), contract_name="Counter",
        components=comps, llm=object(), api=api,
    )

    # The empty grouping degraded to a single 'general' bucket covering all rules.
    assert len(report.implemented_properties) == 1
    g = report.implemented_properties[0]
    assert g.slug == FALLBACK_SLUG
    assert set(g.rule_names) == {"r1", "r2"}
    assert g.status == GroupStatus.VIOLATED  # r2 violated
    assert any("FALLBACK GROUPING APPLIED" in w for w in report.coverage.warnings)

    # Persisted as report.json only — no canonical_map.json anymore.
    report_json = tmp_path / "certora" / "ap_report" / "report.json"
    assert report_json.is_file()
    assert not (tmp_path / "certora" / "ap_report" / "canonical_map.json").exists()


@pytest.mark.asyncio
async def test_build_surfaces_unimplemented_properties(tmp_path, monkeypatch):
    """A property no rule implements lands in unimplemented_properties; mapped
    properties are not duplicated there (they're embedded in rules[].properties)."""
    from composer.spec.source.report.schema import GroupingResult, ImplementedPropertyDraft

    certora = tmp_path / "certora"
    _write_props(certora, "autospec_Increment", [_prop("p1", "d1"), _prop("p2", "d2")])
    _write_rules(certora, "autospec_Increment", {"p1": ["r1"]})  # p2 has no rule
    api = _FakeAPI({"L1": [_fake_check("r1", NodeStatus.VERIFIED)]})

    async def _llm(**kw):
        return GroupingResult(groups=[
            ImplementedPropertyDraft(slug="g", title="G", description="d", rule_names=["r1"])])
    monkeypatch.setattr(build, "call_grouping_llm", _llm)

    report = await build.run_autoprove_report(
        project_root=str(tmp_path), contract_name="C",
        components=[ComponentInput("Increment", "autospec_Increment", "L1")], llm=object(), api=api,
    )
    assert [(p.component, p.title) for p in report.unimplemented_properties] == [("Increment", "p2")]
    assert report.coverage.total_property_formulations == 2  # both still counted
