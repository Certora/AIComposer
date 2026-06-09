"""Orchestrate the autoprove report: collect -> group -> reconcile -> validate
-> write ``certora/ap_report/report.json`` (+ ``canonical_map.json``).

`run_autoprove_report` is the entry point the pipeline's final phase calls. It
is structured so that any single failure (LLM, validation, an empty grouping)
degrades to a single 'general' bucket rather than producing no high-level
section; the caller additionally treats the whole phase as best-effort.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from prover_output_utility import ProverOutputAPI
from pydantic import ValidationError as PydanticValidationError

from composer.spec.gen_types import AP_REPORT_DIR, under_project
from composer.spec.source.report.collect import ComponentInput, collect
from composer.spec.source.report.coverage import ValidationError, validate
from composer.spec.source.report.grouping import (
    build_fallback_grouping, build_rules_for_grouping, call_grouping_llm,
    reconcile_with_canonical,
)
from composer.spec.source.report.schema import AutoProverReport, CanonicalMap

_log = logging.getLogger(__name__)

REPORT_JSON = "report.json"
CANONICAL_MAP_JSON = "canonical_map.json"


def load_canonical_map(path: Path) -> CanonicalMap:
    if not path.is_file():
        return CanonicalMap()
    try:
        return CanonicalMap.model_validate(json.loads(path.read_text()))
    except (json.JSONDecodeError, PydanticValidationError):
        # A corrupt map shouldn't sink the report; start fresh (and warn).
        _log.warning("autoprove report: canonical map at %s is malformed; ignoring", path)
        return CanonicalMap()


def save_canonical_map(path: Path, canonical: CanonicalMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical.model_dump_json(indent=2) + "\n")


async def run_autoprove_report(
    *,
    project_root: str,
    contract_name: str,
    components: list[ComponentInput],
    llm: BaseChatModel,
    api: ProverOutputAPI | None = None,
    timestamp_utc: str | None = None,
) -> AutoProverReport:
    """Build and persist the report. Returns the in-memory `AutoProverReport`."""
    properties, rules = collect(project_root, components, api=api)
    rule_status = {r.name: r.status for r in rules}

    report_dir = under_project(project_root, AP_REPORT_DIR)
    canonical_path = report_dir / CANONICAL_MAP_JSON
    canonical = load_canonical_map(canonical_path)

    rules_for_grouping = build_rules_for_grouping(rules, properties)

    # The grouping may fail three ways; each degrades to the 'general' bucket so
    # the report always has a high-level section: (a) the LLM call raises,
    # (b) validation rejects a structurally-invalid grouping, (c) the grouping
    # is valid but covers no rules.
    fallback_reason: str | None = None
    try:
        grouping = await call_grouping_llm(
            llm=llm, contract_name=contract_name,
            rules=rules_for_grouping, canonical=canonical.entries,
        )
    except Exception as e:  # noqa: BLE001 — best-effort; any LLM/transport error degrades
        fallback_reason = f"grouping LLM call failed: {e}"
        grouping = build_fallback_grouping([r.name for r in rules], str(e))

    high_level, warnings, updated = reconcile_with_canonical(grouping.groups, canonical, rule_status)

    try:
        coverage = validate(rules=rules, groups=high_level, total_inferred=len(properties))
    except ValidationError as e:
        fallback_reason = f"validation rejected the grouping: {e}"
        grouping = build_fallback_grouping([r.name for r in rules], str(e))
        high_level, warnings, updated = reconcile_with_canonical(grouping.groups, canonical, rule_status)
        coverage = validate(rules=rules, groups=high_level, total_inferred=len(properties))

    if fallback_reason is None and rules and not {n for g in high_level for n in g.rule_names}:
        fallback_reason = "grouping produced no high-level properties"
        grouping = build_fallback_grouping([r.name for r in rules], fallback_reason)
        high_level, warnings, updated = reconcile_with_canonical(grouping.groups, canonical, rule_status)
        coverage = validate(rules=rules, groups=high_level, total_inferred=len(properties))

    if fallback_reason is not None:
        coverage.warnings = [f"FALLBACK GROUPING APPLIED — {fallback_reason}"] + warnings + coverage.warnings
    else:
        coverage.warnings = warnings + coverage.warnings

    report = AutoProverReport(
        contract_name=contract_name,
        run_timestamp_utc=timestamp_utc or datetime.now(timezone.utc).isoformat(),
        prover_links={c.name: c.prover_link for c in components if c.prover_link},
        inferred_properties=properties,
        rules=rules,
        high_level_properties=high_level,
        coverage=coverage,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / REPORT_JSON).write_text(report.model_dump_json(indent=2) + "\n")
    save_canonical_map(canonical_path, updated)
    _log.info("autoprove report: wrote %s", report_dir / REPORT_JSON)
    return report
