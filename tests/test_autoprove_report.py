"""Tests for the autoprove report package (composer.spec.source.report).

Covers the pure pieces (collect from property dumps + a fake POU, coverage
aggregation, grouping reconciliation/fallback, HTML render) and the build
orchestrator's fallback + canonical-map reuse. No DB / no real LLM / no real
prover: POU is faked and the grouping LLM is monkeypatched.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from prover_output_utility.models import NodeStatus

from composer.spec.source.report import build
from composer.spec.source.report.collect import ComponentInput, collect
from composer.spec.source.report.coverage import (
    ValidationError, aggregate_status, validate,
)
from composer.spec.source.report.grouping import (
    FALLBACK_SLUG, _jaccard, _next_id, build_fallback_grouping,
    build_rules_for_grouping, reconcile_with_canonical, validate_slugs,
)
from composer.spec.source.report.render import render_html
from composer.spec.source.report.schema import (
    AutoProverReport, CanonicalEntry, CanonicalMap, CVLRule, CoverageReport,
    GroupStatus, HighLevelProperty, HighLevelPropertyDraft, InferredProperty,
    RuleRef,
)


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------

def _fake_check(rule_name, status, line=None, duration=None):
    sl = SimpleNamespace(line=line) if line is not None else None
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

    assert [(p.index, p.title) for p in props] == [(1, "count_increases"), (2, "count_eq_sum")]
    by_name = {r.name: r for r in rules}
    assert by_name["increment_increases_count"].status == NodeStatus.VERIFIED
    assert by_name["increment_increases_count"].line == 12
    assert by_name["increment_increases_count"].property_refs == [RuleRef("Increment", 1)]
    assert by_name["countEqualsSum"].status == NodeStatus.VIOLATED
    assert by_name["countEqualsSum"].line == 40
    assert by_name["increment_increases_count"].prover_link == "L1"


def test_collect_missing_rule_mapping_tolerated(tmp_path):
    certora = tmp_path / "certora"
    _write_props(certora, "autospec_Increment", [_prop("p1", "d1")])
    # no property_rules.json; POU still reports a verdict
    api = _FakeAPI({"L1": [_fake_check("helper", NodeStatus.VERIFIED)]})
    props, rules = collect(str(tmp_path), [ComponentInput("Increment", "autospec_Increment", "L1")], api=api)
    assert len(props) == 1
    assert [r.name for r in rules] == ["helper"]
    assert rules[0].property_refs == []


def test_collect_rule_seen_in_two_components_keeps_first(tmp_path):
    certora = tmp_path / "certora"
    _write_props(certora, "autospec_Increment", [_prop("c", "comp view", sort="invariant")])
    _write_rules(certora, "autospec_Increment", {"c": ["countEqualsSum"]})
    _write_props(certora, "invariants", [_prop("i", "structural", sort="invariant")])
    _write_rules(certora, "invariants", {"i": ["countEqualsSum"]})
    api = _FakeAPI({})
    comps = [
        ComponentInput("Increment", "autospec_Increment", None),
        ComponentInput("Structural Invariants", "invariants", None),
    ]
    _props, rules = collect(str(tmp_path), comps, api=api)
    ces = [r for r in rules if r.name == "countEqualsSum"]
    assert len(ces) == 1 and ces[0].component == "Increment"


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_aggregate_status_table():
    assert aggregate_status([]) == GroupStatus.INCONCLUSIVE
    assert aggregate_status([NodeStatus.VERIFIED, NodeStatus.VERIFIED]) == GroupStatus.VERIFIED
    assert aggregate_status([NodeStatus.VERIFIED, NodeStatus.VIOLATED]) == GroupStatus.VIOLATED
    assert aggregate_status([NodeStatus.VERIFIED, NodeStatus.TIMEOUT]) == GroupStatus.PARTIAL
    assert aggregate_status([NodeStatus.TIMEOUT, NodeStatus.UNKNOWN]) == GroupStatus.INCONCLUSIVE


def _rule(name, status=NodeStatus.VERIFIED, component="C"):
    return CVLRule(name=name, component=component, status=status)


def _grp(slug, rule_names, status=GroupStatus.VERIFIED, id_="P-01"):
    return HighLevelProperty(id=id_, slug=slug, title="T", description="d",
                             status=status, rule_names=rule_names)


def test_validate_rule_in_two_groups_raises():
    rules = [_rule("a"), _rule("b")]
    groups = [_grp("g1", ["a"], id_="P-01"), _grp("g2", ["a", "b"], id_="P-02")]
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

def test_next_id_counter():
    assert _next_id(set()) == "P-01"
    assert _next_id({"P-01", "P-02"}) == "P-03"
    assert _next_id({f"P-{i:02d}" for i in range(1, 100)}) == "P-100"


def test_jaccard():
    assert _jaccard([], []) == 0.0
    assert _jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert _jaccard(["a", "b", "c"], ["a", "b", "d"]) == 0.5


def test_validate_slugs():
    assert validate_slugs([HighLevelPropertyDraft(slug="ok-slug", title="t", description="d", rule_names=["x"])]) == []
    errs = validate_slugs([HighLevelPropertyDraft(slug="Bad_Slug", title="t", description="d", rule_names=["x"])])
    assert errs and "kebab-case" in errs[0]


def test_reconcile_reuses_canonical_on_exact_slug():
    canonical = CanonicalMap(entries=[CanonicalEntry(id="P-01", slug="token", title="Canon", anchor_rules=["a"])])
    drafts = [HighLevelPropertyDraft(slug="token", title="LLM title", description="d", rule_names=["a"])]
    finals, warnings, updated = reconcile_with_canonical(drafts, canonical, {"a": NodeStatus.VERIFIED})
    assert finals[0].id == "P-01" and finals[0].title == "Canon"
    assert len(updated.entries) == 1 and warnings == []


def test_reconcile_assigns_fresh_id_for_new_group():
    canonical = CanonicalMap(entries=[CanonicalEntry(id="P-01", slug="token", title="C", anchor_rules=["a"])])
    drafts = [HighLevelPropertyDraft(slug="brand-new", title="New", description="d", rule_names=["x", "y"])]
    finals, _w, updated = reconcile_with_canonical(drafts, canonical, {"x": NodeStatus.VERIFIED, "y": NodeStatus.VERIFIED})
    assert finals[0].id == "P-02" and finals[0].slug == "brand-new"
    assert {e.id for e in updated.entries} == {"P-01", "P-02"}


def test_fallback_grouping_covers_all_rules():
    out = build_fallback_grouping(["r1", "r2"], "boom")
    assert len(out.groups) == 1 and out.groups[0].slug == FALLBACK_SLUG
    assert out.groups[0].rule_names == ["r1", "r2"]
    assert "boom" in out.groups[0].description


def test_build_rules_for_grouping_attaches_descriptions_and_sorts():
    props = [InferredProperty(component="C", index=1, title="t", methods=["m"],
                              sort="invariant", description="the desc")]
    rules = [CVLRule(name="r1", component="C", property_refs=[RuleRef("C", 1)], status=NodeStatus.VERIFIED)]
    rows = build_rules_for_grouping(rules, props)
    assert rows[0].property_descriptions == ["the desc"] and rows[0].sorts == ["invariant"]


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def _mini_report() -> AutoProverReport:
    return AutoProverReport(
        contract_name="Counter",
        prover_links={"Increment": "https://prover.example/run/abc"},
        inferred_properties=[InferredProperty(component="Increment", index=1, title="count_increases",
                                              methods=["increment"], sort="safety_property",
                                              description="increment raises count")],
        rules=[CVLRule(name="increment_increases_count", component="Increment",
                       property_refs=[RuleRef("Increment", 1)], status=NodeStatus.VERIFIED,
                       line=10, prover_link="https://prover.example/run/abc")],
        high_level_properties=[HighLevelProperty(id="P-01", slug="count-up", title="Count Increases",
                                                 description="d", status=GroupStatus.VERIFIED,
                                                 rule_names=["increment_increases_count"])],
        coverage=CoverageReport(total_inferred_properties=1, total_rules=1, total_groups=1,
                                rules_per_group_min=1, rules_per_group_max=1, rule_coverage_complete=True),
    )


def test_render_html_contains_ids_links_and_appendix():
    h = render_html(_mini_report())
    assert "P-01" in h and "Count Increases" in h
    assert 'href="https://prover.example/run/abc"' in h
    assert "increment_increases_count" in h  # appendix "implemented by"
    assert "count_increases" in h            # property title in appendix
    assert "VERIFIED" in h


# ---------------------------------------------------------------------------
# build orchestrator (async): empty grouping -> fallback; canonical reuse
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
    assert len(report.high_level_properties) == 1
    g = report.high_level_properties[0]
    assert g.slug == FALLBACK_SLUG
    assert set(g.rule_names) == {"r1", "r2"}
    assert g.status == GroupStatus.VIOLATED  # r2 violated
    assert any("FALLBACK GROUPING APPLIED" in w for w in report.coverage.warnings)

    # Persisted under certora/ap_report/.
    report_json = tmp_path / "certora" / "ap_report" / "report.json"
    canon_json = tmp_path / "certora" / "ap_report" / "canonical_map.json"
    assert report_json.is_file() and canon_json.is_file()
    first_id = g.id

    # A second run reuses the canonical 'general' id (stable across runs).
    report2 = await build.run_autoprove_report(
        project_root=str(tmp_path), contract_name="Counter",
        components=comps, llm=object(), api=api,
    )
    assert report2.high_level_properties[0].id == first_id
