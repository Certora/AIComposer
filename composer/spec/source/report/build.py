"""Orchestrate the autoprove report: collect -> group -> validate -> write
``certora/ap_report/report.json``.

`run_autoprove_report` is the entry point the pipeline's final phase calls. It is structured so that
any single failure (LLM, validation, an empty grouping) degrades to a single ``general`` bucket
rather than producing no high-level section; the caller additionally treats the whole phase as
best-effort.
"""
import logging
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from prover_output_utility import ProverOutputAPI
from prover_output_utility.models import NodeStatus

from composer.spec.gen_types import AP_REPORT_DIR, under_project
from composer.spec.source.report.collect import ReportComponentInput, collect
from composer.spec.source.report.coverage import ValidationError, validate
from composer.spec.source.report.grouping import (
    build_fallback_grouping, build_groups, call_grouping_llm,
)
from composer.spec.source.report.schema import AutoProverReport, PropertyKey, RuleRef

_log = logging.getLogger(__name__)

REPORT_JSON = "report.json"


async def run_autoprove_report(
    *,
    project_root: str,
    contract_name: str,
    components: list[ReportComponentInput],
    llm: BaseChatModel,
    api: ProverOutputAPI | None = None,
) -> AutoProverReport:
    """Build and persist the report. Returns the in-memory `AutoProverReport`."""
    properties, rules, skipped, gave_up, dropped = await collect(components, api=api)
    rule_status: dict[RuleRef, NodeStatus] = {r.ref: r.status for r in rules}
    props_by_key = {p.key: p for p in properties}

    # The grouping may fail three ways; each degrades to the single 'general' bucket so the report
    # always has a high-level section: (a) the LLM call raises, (b) validation rejects a
    # structurally-invalid grouping, (c) the grouping is valid but covers no properties. The
    # fallback bucket holds every property exactly once, so the re-validate below cannot raise.
    fallback_reason: str | None = None
    try:
        grouping = await call_grouping_llm(
            llm=llm, contract_name=contract_name, properties=properties,
        )
        groups = build_groups(grouping.groups, props_by_key, rule_status)
        coverage = validate(
            properties=properties, rules=rules, groups=groups,
            skipped=skipped, gave_up=gave_up, dropped_orphan_rules=dropped,
        )
        grouped: set[PropertyKey] = {k for g in groups for k in g.members}
        if properties and not grouped:
            raise ValidationError("grouping produced no high-level properties")
    except Exception as e:  # noqa: BLE001 — any LLM/transport/validation error degrades
        fallback_reason = (
            f"validation rejected the grouping: {e}" if isinstance(e, ValidationError)
            else f"grouping failed: {e}"
        )
        _log.warning("autoprove report: %s; applying fallback grouping", fallback_reason)
        groups = build_groups(
            build_fallback_grouping(properties).groups, props_by_key, rule_status
        )
        coverage = validate(
            properties=properties, rules=rules, groups=groups,
            skipped=skipped, gave_up=gave_up, dropped_orphan_rules=dropped,
        )
        coverage.warnings = ["FALLBACK GROUPING APPLIED"] + coverage.warnings

    report = AutoProverReport(
        contract_name=contract_name,
        run_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        prover_links={c.name: c.prover_link for c in components if c.prover_link},
        properties=properties,
        rules=rules,
        groups=groups,
        skipped=skipped,
        gave_up_components=gave_up,
        coverage=coverage,
    )

    report_dir = under_project(project_root, AP_REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / REPORT_JSON).write_text(report.model_dump_json(indent=2) + "\n")
    _log.info("autoprove report: wrote %s", report_dir / REPORT_JSON)
    return report
