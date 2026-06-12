"""Orchestrate the autoprove report: collect -> group -> validate -> write
``certora/ap_report/report.json``.

`run_autoprove_report` is the entry point the pipeline's final phase calls. It
is structured so that any single failure (LLM, validation, an empty grouping)
degrades to a single 'general' bucket rather than producing no high-level
section; the caller additionally treats the whole phase as best-effort.
"""
import logging
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from prover_output_utility import ProverOutputAPI

from composer.spec.gen_types import AP_REPORT_DIR, under_project
from composer.spec.source.report.collect import ComponentInput, collect
from composer.spec.source.report.coverage import ValidationError, validate
from composer.spec.source.report.grouping import (
    build_fallback_grouping, build_high_level, build_rules_for_grouping, call_grouping_llm,
)
from composer.spec.source.report.schema import AutoProverReport

_log = logging.getLogger(__name__)

REPORT_JSON = "report.json"


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
    rules_for_grouping = build_rules_for_grouping(rules, properties)

    # The grouping may fail three ways; each degrades to the single 'general'
    # bucket so the report always has a high-level section: (a) the LLM call
    # raises, (b) validation rejects a structurally-invalid grouping, (c) the
    # grouping is valid but covers no rules. The fallback bucket holds every
    # rule exactly once, so the re-validate below cannot raise.
    fallback_reason: str | None = None
    try:
        grouping = await call_grouping_llm(
            llm=llm, contract_name=contract_name, rules=rules_for_grouping,
        )
        high_level = build_high_level(grouping.groups, rule_status)
        coverage = validate(rules=rules, groups=high_level, total_inferred=len(properties))
        if rules and not {n for g in high_level for n in g.rule_names}:
            raise ValidationError("grouping produced no high-level properties")
    except Exception as e:  # noqa: BLE001 — any LLM/transport/validation error degrades
        fallback_reason = (
            f"validation rejected the grouping: {e}" if isinstance(e, ValidationError)
            else f"grouping failed: {e}"
        )
        high_level = build_high_level(
            build_fallback_grouping([r.name for r in rules], str(e)).groups, rule_status
        )
        coverage = validate(rules=rules, groups=high_level, total_inferred=len(properties))
        coverage.warnings = [f"FALLBACK GROUPING APPLIED — {fallback_reason}"] + coverage.warnings

    report = AutoProverReport(
        contract_name=contract_name,
        run_timestamp_utc=timestamp_utc or datetime.now(timezone.utc).isoformat(),
        prover_links={c.name: c.prover_link for c in components if c.prover_link},
        inferred_properties=properties,
        rules=rules,
        high_level_properties=high_level,
        coverage=coverage,
    )

    report_dir = under_project(project_root, AP_REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / REPORT_JSON).write_text(report.model_dump_json(indent=2) + "\n")
    _log.info("autoprove report: wrote %s", report_dir / REPORT_JSON)
    return report
