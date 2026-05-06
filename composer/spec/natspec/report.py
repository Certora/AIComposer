"""Per-contract property report for the natspec pipeline.

Two delivery channels:

* End-of-pipeline output. After ``run_natspec_pipeline`` completes,
  ``PipelineApp.on_pipeline_done`` builds a report from the in-memory
  ``PipelineResult`` and writes ``natspec_report.md`` alongside
  ``implementation_plan.json``.

* Standalone CLI (``natspec-report``). Re-derives the same report by
  walking the pipeline's cache namespace — same inputs as
  ``cache-natspec`` (input file + ``--cache-ns`` + ``--from-source``).

The report is intentionally narrow: per contract that produced at
least one spec, the contract's description, the components for which
a spec was generated (with their description), and per component the
formalized properties and the skipped ones (with reasons). No CVL, no
commentary, no source paths, no failures, no external actors — those
all live elsewhere (implementation plan, cache explorer).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from composer.spec.bug import (
    AGENT_RESULT_KEY,
    _AgentResult,
    _BugAnalysisCache,
    bug_analysis_key,
)
from composer.spec.context import (
    CacheKey,
    Contract,
    Properties,
    WorkflowContext,
)
from composer.spec.natspec.author import AuthorResult, GenerationSuccess
from composer.spec.natspec.pipeline import _batch_cache_key, _component_cache_key
from composer.spec.system_model import (
    Application,
    ExistingFromSource,
    FromSourceApplication,
    NatspecApplication,
)
from composer.spec.prop import PropertyFormulation
from composer.spec.util import string_hash

if TYPE_CHECKING:
    from composer.spec.natspec.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PropertyEntry:
    sort: str
    description: str

    @classmethod
    def from_formulation(cls, p: PropertyFormulation) -> "PropertyEntry":
        return cls(sort=p.sort, description=p.description)


@dataclass
class SkippedProperty:
    prop: PropertyEntry
    reason: str


@dataclass
class ComponentReport:
    """One component for which a spec was generated. ``formalized`` is
    the surviving (non-skipped) property list; ``skipped`` carries the
    properties dropped along the way with the agent's stated reason."""
    name: str
    description: str
    formalized: list[PropertyEntry]
    skipped: list[SkippedProperty]


@dataclass
class ContractReport:
    """A contract for which at least one component produced a spec.
    Components that produced no spec are not listed."""
    name: str
    description: str
    components: list[ComponentReport]


@dataclass
class NatspecReport:
    application_type: str
    application_description: str
    contracts: list[ContractReport]


# ---------------------------------------------------------------------------
# From in-memory PipelineResult
# ---------------------------------------------------------------------------


def report_from_pipeline(result: "PipelineResult") -> NatspecReport:
    """Build a ``NatspecReport`` from the in-memory pipeline result.

    Only contracts that produced at least one spec are included; only
    components with a spec are listed under each contract.
    """
    # Description lookup keyed by contract name. ``result.app`` carries
    # the canonical component descriptions; ``result.contracts`` (the
    # formulations) carries the per-component spec successes but NOT
    # rich descriptions on the explicit-contract objects.
    contract_descs: dict[str, str] = {
        c.name: c.description for c in result.app.contract_components
    }

    contract_reports: list[ContractReport] = []

    for formulation in result.contracts:
        if not formulation.spec_results.specs:
            continue

        components: list[ComponentReport] = []
        for s in formulation.spec_results.specs:
            comp = s.component.component
            components.append(ComponentReport(
                name=comp.name,
                description=comp.description,
                formalized=[
                    PropertyEntry.from_formulation(p)
                    for p in s.successful_properties
                ],
                skipped=[
                    SkippedProperty(
                        prop=PropertyEntry.from_formulation(sp.prop),
                        reason=sp.reason,
                    )
                    for sp in s.skipped_properties
                ],
            ))

        contract_reports.append(ContractReport(
            name=formulation.name,
            description=contract_descs.get(formulation.name, ""),
            components=components,
        ))

    return NatspecReport(
        application_type=result.app.application_type,
        application_description=result.app.description,
        contracts=contract_reports,
    )


# ---------------------------------------------------------------------------
# From cached pipeline state (standalone CLI)
# ---------------------------------------------------------------------------


# Cache-key literal, mirrors composer/cli/cache_natspec.py.
SOURCE_ANALYSIS_KEY = CacheKey[None, NatspecApplication]("source-analysis")


async def _component_props(
    feat_ctx: WorkflowContext,
) -> list[PropertyFormulation]:
    """Cumulative bug-analysis property list for a component context,
    or empty if the cache misses. Prefers ``_AgentResult.items`` and
    falls back to ``_BugAnalysisCache.items``."""
    bug_ctx = feat_ctx.child(bug_analysis_key(None))
    agent = await bug_ctx.child(AGENT_RESULT_KEY).cache_get(_AgentResult)
    if agent is not None:
        return list(agent.items)
    bug_cache = await bug_ctx.cache_get(_BugAnalysisCache)
    if bug_cache is not None:
        return list(bug_cache.items)
    return []


async def report_from_cache(
    root_ctx: WorkflowContext,
    *,
    from_source: bool,
) -> NatspecReport | None:
    """Walk the pipeline's cache hierarchy and rebuild a report. Returns
    ``None`` when the source-analysis result isn't cached.

    Same shape as ``report_from_pipeline``: only contracts with at
    least one cached ``GenerationSuccess`` survive into the output;
    only their successful components are listed.
    """
    sa_ctx = root_ctx.child(SOURCE_ANALYSIS_KEY)
    app_ty: type[NatspecApplication] = (
        FromSourceApplication if from_source else Application
    )
    summary: NatspecApplication | None = await sa_ctx.cache_get(app_ty)
    if summary is None:
        return None

    contracts: list[ContractReport] = []

    for c in summary.contract_components:
        if isinstance(c, ExistingFromSource):
            # Pre-existing source-side contract; pipeline doesn't
            # generate specs for it.
            continue

        contract_key = CacheKey[None, Contract](string_hash(c.model_dump_json()))
        contract_ctx = root_ctx.child(contract_key)
        prop_ctx = contract_ctx.child(
            CacheKey[Contract, Properties]("properties")
        )

        components: list[ComponentReport] = []

        for comp in c.components:
            comp_key = _component_cache_key(comp, summary.application_type)
            feat_ctx = prop_ctx.child(comp_key)

            props_full = await _component_props(feat_ctx)
            if not props_full:
                continue

            batch_ctx = feat_ctx.child(_batch_cache_key(props_full))
            author = await batch_ctx.cache_get(AuthorResult)
            if author is None:
                continue
            inner = author.result_wrapped
            if not isinstance(inner, GenerationSuccess):
                continue

            # ``GenerationSuccess.skipped`` is a list of
            # ``cvl_generation.SkippedProperty(property_index, reason)``
            # — 1-indexed into the batch's input property list (which
            # IS ``props_full``, since the batch cache key is
            # ``_batch_cache_key(props_full)``). The success doesn't
            # carry the surviving list, so we reconstruct it as
            # ``props_full`` minus the skipped indices.
            skipped_idxs: set[int] = {
                sp.property_index - 1 for sp in inner.skipped
            }
            formalized = [
                PropertyEntry.from_formulation(p)
                for i, p in enumerate(props_full)
                if i not in skipped_idxs
            ]
            skipped: list[SkippedProperty] = []
            for sp in inner.skipped:
                i = sp.property_index - 1
                if not (0 <= i < len(props_full)):
                    continue
                skipped.append(SkippedProperty(
                    prop=PropertyEntry.from_formulation(props_full[i]),
                    reason=sp.reason,
                ))

            components.append(ComponentReport(
                name=comp.name,
                description=comp.description,
                formalized=formalized,
                skipped=skipped,
            ))

        if components:
            contracts.append(ContractReport(
                name=c.name,
                description=c.description,
                components=components,
            ))

    return NatspecReport(
        application_type=summary.application_type,
        application_description=summary.description,
        contracts=contracts,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_property(p: PropertyEntry) -> str:
    return f"- *[{p.sort}]* {p.description}"


def report_to_markdown(report: NatspecReport) -> str:
    """Render a ``NatspecReport`` as a single markdown document."""
    lines: list[str] = []
    lines.append(f"# Natspec Report — {report.application_type}")
    lines.append("")
    if report.application_description.strip():
        lines.append(report.application_description.strip())
        lines.append("")

    if not report.contracts:
        lines.append("*No specs were generated.*")
        lines.append("")
        return "\n".join(lines)

    for contract in report.contracts:
        lines.append(f"## `{contract.name}`")
        if contract.description.strip():
            lines.append("")
            lines.append(contract.description.strip())
        lines.append("")

        for comp in contract.components:
            lines.append(f"### Component: `{comp.name}`")
            if comp.description.strip():
                lines.append("")
                lines.append(comp.description.strip())
            lines.append("")

            if comp.formalized:
                lines.append(f"**Formalized ({len(comp.formalized)}):**")
                lines.append("")
                for p in comp.formalized:
                    lines.append(_render_property(p))
                lines.append("")

            if comp.skipped:
                lines.append(f"**Skipped ({len(comp.skipped)}):**")
                lines.append("")
                for sp in comp.skipped:
                    lines.append(_render_property(sp.prop))
                    lines.append(f"  - reason: {sp.reason}")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Disk write
# ---------------------------------------------------------------------------


REPORT_FILENAME = "natspec_report.md"


def export_report(
    report: NatspecReport,
    output_root: pathlib.Path | str,
) -> pathlib.Path:
    """Write the rendered report to ``<output_root>/natspec_report.md``."""
    out = pathlib.Path(output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / REPORT_FILENAME
    path.write_text(report_to_markdown(report))
    return path
