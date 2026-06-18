
import asyncio
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Protocol

from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel

from composer.io.multi_job import (
    TaskInfo, HandlerFactory, run_task,
)
from composer.ui.autoprove_app import AutoProvePhase

from composer.input.files import Document
from composer.prover.core import ProverOptions
from composer.spec.context import (
    WorkflowContext, SourceCode, CacheKey, Properties, ComponentGroup, CVLGeneration,
)
from composer.spec.util import string_hash, ensure_dir
from composer.spec.prop_inference import run_property_inference
from composer.spec.prop import PropertyFormulation, PropertyId
from composer.spec.gen_types import (
    CVLResource, CERTORA_DIR, SPECS_DIR, AUTOPROVE_INTERNAL_DIR,
    certora_relative_to_project, under_project,
)
from composer.spec.service_host import ServiceHost
from composer.spec.system_model import (
    ContractComponentInstance, HarnessedApplication, ContractInstance,
    SourceApplication, SourceExplicitContract, HarnessedExplicitContract,
    SourceExternalActor, HarnessDefinition, SolidityIdentifier,
)
from composer.spec.cvl_generation import GeneratedCVL, PropertyRuleMapping
from composer.spec.source.author import batch_cvl_generation, GaveUp, BatchGeneratedCVLResult
from composer.spec.source.harness import (
    run_harness_creation, run_autosetup_phase, ContractSetup, SystemDescriptionHarnessed,
)
from composer.spec.source.system_analysis import run_component_analysis
from composer.spec.source.summarizer import setup_summaries
from composer.spec.source.struct_invariant import get_invariant_formulation
from composer.spec.source.prover import dump_final_conf, get_prover_tool
from composer.spec.source.task_ids import (
    bug_analysis_task_id, cvl_gen_task_id, INVARIANT_CVL_TASK_ID,
    SYSTEM_ANALYSIS_TASK_ID, HARNESS_TASK_ID, AUTOSETUP_TASK_ID,
    SUMMARIES_TASK_ID, INVARIANTS_TASK_ID,
)
from composer.diagnostics.timing import get_run_summary, RunSummary

_logger = logging.getLogger(__name__)

PROPERTIES_KEY = CacheKey[None, Properties]("properties")
INV_CVL_KEY = CacheKey[None, GeneratedCVL]("invariant-cvl")


def dump_properties(
    certora_dir: pathlib.Path,
    spec_stem: str,
    props: list[PropertyFormulation],
) -> None:
    """Write the analysis-phase properties (title, sort, methods, description) to
    ``properties/{spec_stem}.properties.json`` under ``certora_dir``, accompanying
    ``{spec_stem}.spec``. ``title`` is the cross-reference key used by
    ``{spec_stem}.property_rules.json``."""
    properties_dir = ensure_dir(certora_dir / "properties")
    properties_dump = [prop.model_dump() for prop in props]
    (properties_dir / f"{spec_stem}.properties.json").write_text(
        json.dumps(properties_dump, indent=2)
    )


def dump_property_rules(
    certora_dir: pathlib.Path,
    spec_stem: str,
    property_rules: list[PropertyRuleMapping],
) -> None:
    """Write the property->rules mapping ``{property title: [rule names]}`` to
    ``properties/{spec_stem}.property_rules.json`` under ``certora_dir``, accompanying
    ``{spec_stem}.spec``. Titles are unique (enforced at extraction) and validated against
    the batch at completion."""
    properties_dir = ensure_dir(certora_dir / "properties")
    mapping = {m.property_title: m.rules for m in property_rules}
    (properties_dir / f"{spec_stem}.property_rules.json").write_text(
        json.dumps(mapping, indent=2)
    )


def _output_link(link: str) -> str:
    """Rewrite a prover ``/jobStatus/`` URL to its ``/output/`` view. Local
    result-directory paths (which contain neither) pass through unchanged."""
    return link.replace("/jobStatus/", "/output/")


def dump_component_runs(
    project_root: str,
    component_runs: dict[str, str],
) -> None:
    """Write the ``{component: final prover run link}`` mapping to
    ``.certora_internal/autoProve/components_to_prover_runs.json`` under
    ``project_root``. Keys are each component's slug (the ``autospec_{slug}`` spec
    stem) plus ``"invariants"`` for the structural-invariant spec; the link is the
    URL (cloud) or local results directory (local) of that spec's last prover run."""
    out_dir = ensure_dir(under_project(project_root, AUTOPROVE_INTERNAL_DIR))
    (out_dir / "components_to_prover_runs.json").write_text(
        json.dumps(component_runs, indent=2)
    )


def dump_token_usage(
    project_root: str,
    summary: RunSummary,
) -> None:
    """Write the run's accumulated LLM token usage to
    ``.certora_internal/autoProve/token_usage.json`` under ``project_root``.

    Raw counts only (``input`` / ``output`` / ``cache_read`` / ``cache_write``),
    broken down ``by_model`` and ``by_phase`` plus run-wide ``totals``. Captures
    every call through the LLM factory — including out-of-graph prover/CEX-analysis
    side-calls — via the usage callback attached in ``create_llm_base``. The same
    breakdown is also persisted to the run's ``RunMeta.tags`` (see ``_entry_point``)."""
    payload = {"run_id": summary.run_id, **summary.token_usage_summary()}
    out_dir = ensure_dir(under_project(project_root, AUTOPROVE_INTERNAL_DIR))
    (out_dir / "token_usage.json").write_text(json.dumps(payload, indent=2))


def _component_cache_key(
    component: ContractComponentInstance,
) -> CacheKey[Properties, ComponentGroup]:
    combined = "|".join([component.app.model_dump_json(), str(component.ind), str(component._contract.ind)])
    return CacheKey(string_hash(combined))


def _batch_cache_key(props: list[PropertyFormulation]) -> CacheKey[ComponentGroup, GeneratedCVL]:
    combined = "|".join(p.model_dump_json() for p in props)
    return CacheKey(string_hash(combined))

@dataclass
class Unmatched:
    """The formalize phase could not map this property to any component."""
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass
class Skipped:
    """The CVL-generation agent explicitly skipped this property."""
    feat: ContractComponentInstance
    reason: str

    def __str__(self) -> str:
        return f"skipped: {self.reason}"


@dataclass
class BatchGaveUp:
    """The component's whole CVL batch gave up; this property went down with it."""
    feat: ContractComponentInstance
    reason: str

    def __str__(self) -> str:
        return f"batch gave up: {self.reason}"


@dataclass
class Errored:
    """CVL generation raised; the component produced no spec."""
    feat: ContractComponentInstance
    error: str

    def __str__(self) -> str:
        return f"exception: {self.error}"


# Why a property is uncovered, carrying only the fields relevant to that shape
# (the component is meaningless for an unmatched property, so it is absent there).
type UncoveredReason = Unmatched | Skipped | BatchGaveUp | Errored


@dataclass
class UncoveredProperty:
    """An input property that did not result in a verified rule. Surfaced to the
    user (warn + dump) so coverage is tracked per ``property_id`` end-to-end."""
    property_id: PropertyId
    reason: UncoveredReason


@dataclass
class AutoProveResult:
    n_components: int
    n_properties: int
    failures: list[str] = field(default_factory=list)
    uncovered: list[UncoveredProperty] = field(default_factory=list)   # default-empty: behaviour-compatible

@dataclass
class _ComponentBatch:
    feat: ContractComponentInstance
    props: list[PropertyFormulation]
    feat_ctx: WorkflowContext[ComponentGroup]

def _main_contract_index(summary: HarnessedApplication, name: str) -> int:
    """Index of the contract named *name* within *summary*'s contract components.
    Raises ``ValueError`` if not found."""
    for i, c in enumerate(summary.contract_components):
        if c.solidity_identifier == name:
            return i
    raise ValueError(f"Component not found: {name}")


async def build_component_batch(
    *,
    prop_context: WorkflowContext[Properties],
    feat: ContractComponentInstance,
    props: list[PropertyFormulation],
) -> _ComponentBatch:
    """Build a ``_ComponentBatch`` for *feat*, carrying *props*: derives the
    per-component ``feat_ctx`` (the cache scope CVL generation runs under) from
    *prop_context*. Callers that already know the component + properties (e.g. the
    formalize phase) use this instead of importing the private dataclass."""
    feat_ctx = await prop_context.child(
        _component_cache_key(feat),
        {"component": feat.component.model_dump()},
    )
    return _ComponentBatch(feat=feat, props=props, feat_ctx=feat_ctx)


async def extract_all_components(
    *,
    source_input: SourceCode,
    prop_context: WorkflowContext[Properties],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: ServiceHost,
    summary: HarnessedApplication,
    semaphore: asyncio.Semaphore,
    interactive: bool,
    threat_model: Document | None,
    max_bug_rounds: int = 3,
) -> list[_ComponentBatch]:
    """Phase 5 — per-component property extraction ("bug analysis").

    Runs ``run_property_inference`` for every component in parallel
    (semaphore-bounded) and dumps the analysis-phase properties. Returns the
    batches that yielded properties; an empty list means nothing was extracted
    (the caller decides how to react).
    """
    ind = _main_contract_index(summary, source_input.contract_name)

    contract_instance = ContractInstance(ind, app=summary)

    async def _analyze_component(component_idx: int) -> _ComponentBatch | None:
        feat = ContractComponentInstance(_contract=contract_instance, ind=component_idx)
        name = feat.component.name
        feat_ctx = await prop_context.child(
            _component_cache_key(feat),
            {
                "component": feat.component.model_dump(),
            },
        )

        props = await run_task(
            handler_factory,
            TaskInfo(bug_analysis_task_id(component_idx, feat.slugified_name), name, AutoProvePhase.BUG_ANALYSIS),
            lambda conv: run_property_inference(feat_ctx, env, feat, refinement=conv if interactive else None, threat_model=threat_model, max_rounds=max_bug_rounds),
            semaphore,
        )

        if not props:
            return None
        return _ComponentBatch(feat=feat, props=props, feat_ctx=feat_ctx)

    extraction_results = await asyncio.gather(*[
        _analyze_component(i) for i in range(len(contract_instance.contract.components))
    ])

    component_batches = [b for b in extraction_results if b is not None]
    if not component_batches:
        return []

    # Dump the analysis-phase properties for each component now that the
    # extraction phase is complete.
    certora_dir = under_project(source_input.project_root, CERTORA_DIR)
    for batch in component_batches:
        dump_properties(certora_dir, f"autospec_{batch.feat.slugified_name}", batch.props)

    return component_batches


async def generate_all_component_cvl(
    *,
    source_input: SourceCode,
    component_batches: list[_ComponentBatch],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: ServiceHost,
    prover_tool: BaseTool,
    prover_config: dict,
    resources: list[CVLResource],
    semaphore: asyncio.Semaphore,
) -> AutoProveResult:
    """Phase 6 — per-component CVL generation.

    Generates and writes a spec for each extracted batch in parallel
    (semaphore-bounded). ``resources`` is consumed read-only; callers that want
    the structural invariants assumed as preconditions must include
    ``invariants.spec`` in ``resources`` before calling.
    """
    async def _generate_batch(
        task_id: str,
        batch: _ComponentBatch,
    ) -> BatchGeneratedCVLResult:
        batch_child = await batch.feat_ctx.child(
            _batch_cache_key(batch.props),
            {"properties": [p.model_dump() for p in batch.props]},
        )
        if (cached := await batch_child.cache_get(GeneratedCVL)) is not None:
            return cached
        batch_ctx = batch_child.abstract(CVLGeneration)

        label = f"{batch.feat.component.name} ({len(batch.props)} properties)"
        res = await run_task(
            handler_factory,
            TaskInfo(task_id, label, AutoProvePhase.CVL_GEN),
            lambda: batch_cvl_generation(
                ctx=batch_ctx,
                init_config=prover_config,
                component=batch.feat,
                env=env,
                props=batch.props,
                prover_tool=prover_tool,
                resources=resources,
                description=label,
                source=source_input,
                spec_dir=SPECS_DIR,
            ),
            semaphore,
        )
        if isinstance(res, GeneratedCVL):
            await batch_child.cache_put(res)
        return res

    async def _generate_and_write_batch(
        batch: _ComponentBatch
    ) -> BatchGeneratedCVLResult:
        task_id = cvl_gen_task_id(batch.feat.ind, batch.feat.slugified_name)
        res = await _generate_batch(task_id=task_id, batch=batch)
        if isinstance(res, GaveUp):
            return res
        certora_dir = under_project(source_input.project_root, CERTORA_DIR)
        specs_dir = ensure_dir(certora_dir / "specs")  # absolute (project_root/certora/specs)
        properties_dir = ensure_dir(certora_dir / "properties")
        base = batch.feat.slugified_name
        spec_name = pathlib.Path(f"autospec_{base}.spec")
        (specs_dir / spec_name).write_text(res.cvl)
        # Canonical (project-root-relative) path of the persisted spec, used for
        # the conf's verify entry.
        spec_path = SPECS_DIR / spec_name
        (properties_dir / f"autospec_{base}.commentary.md").write_text(res.commentary)
        dump_property_rules(certora_dir, f"autospec_{base}", res.property_rules)
        dump_final_conf(
            project_root=source_input.project_root,
            main_contract=source_input.contract_name,
            task_id=task_id,
            spec_path=spec_path,
            conf=res.conf,
        )
        return res

    generation_results = await asyncio.gather(
        *[
            _generate_and_write_batch(batch)
            for batch in component_batches
        ],
        return_exceptions=True,
    )

    # Dump the final prover run link for each component (and the structural
    # invariant, which ran earlier in the staged pipeline) now that every CVL
    # generation task has completed — each link is recorded into the run summary
    # when its task's phase is finalized.
    link_by_task = {
        p.task_id: p.final_link for p in get_run_summary().phases if p.final_link
    }
    component_runs = {
        batch.feat.slugified_name: _output_link(link)
        for batch in component_batches
        if (link := link_by_task.get(cvl_gen_task_id(batch.feat.ind, batch.feat.slugified_name)))
    }
    if inv_link := link_by_task.get(INVARIANT_CVL_TASK_ID):
        component_runs["invariants"] = _output_link(inv_link)
    dump_component_runs(source_input.project_root, component_runs)

    failures: list[str] = []
    uncovered: list[UncoveredProperty] = []
    n_properties = 0
    for batch, result in zip(component_batches, generation_results):
        n_properties += len(batch.props)
        feat = batch.feat
        if isinstance(result, BaseException):
            failures.append(f"{feat.component.name}: {result}")
            # No spec produced: every property in this batch is uncovered.
            for prop in batch.props:
                uncovered.append(UncoveredProperty(prop.title, Errored(feat, str(result))))
        elif isinstance(result, GaveUp):
            failures.append(f"{feat.component.name}: GAVE_UP: {result.reason}")
            for prop in batch.props:
                uncovered.append(UncoveredProperty(prop.title, BatchGaveUp(feat, result.reason)))
        else:
            # GeneratedCVL: coverage of non-skipped properties is guaranteed by
            # validate_property_rules (cvl_generation.py); only explicit skips
            # leave a property uncovered.
            for s in result.skipped:
                uncovered.append(UncoveredProperty(s.property_title, Skipped(feat, s.reason)))

    return AutoProveResult(
        n_components=len(component_batches),
        n_properties=n_properties,
        failures=failures,
        uncovered=uncovered,
    )


async def run_generation_pipeline(
    source_input: SourceCode,
    prop_context: WorkflowContext[Properties],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: ServiceHost,
    summary: HarnessedApplication,
    semaphore: asyncio.Semaphore,
    resources: list[CVLResource],
    prover_tool: BaseTool,
    prover_config: dict,
    interactive: bool,
    threat_model: Document | None,
    max_bug_rounds: int = 3,
) -> AutoProveResult:
    """Property extraction followed by CVL generation for every component.

    Thin wrapper over ``extract_all_components`` + ``generate_all_component_cvl``
    for callers that run the two phases back-to-back (e.g. ``direct_pipeline``).
    The staged pipeline calls the two halves directly so it can interleave other
    phases (autosetup, invariant CVL) between them.
    """
    component_batches = await extract_all_components(
        source_input=source_input,
        prop_context=prop_context,
        handler_factory=handler_factory,
        env=env,
        summary=summary,
        semaphore=semaphore,
        interactive=interactive,
        threat_model=threat_model,
        max_bug_rounds=max_bug_rounds,
    )
    if not component_batches:
        raise ValueError("No properties extracted from any component.")
    return await generate_all_component_cvl(
        source_input=source_input,
        component_batches=component_batches,
        handler_factory=handler_factory,
        env=env,
        prover_tool=prover_tool,
        prover_config=prover_config,
        resources=resources,
        semaphore=semaphore,
    )


# ---------------------------------------------------------------------------
# Staged pipeline — shared orchestrator
#
# ``run_autoprove_pipeline`` and ``run_properties_pipeline`` share an identical
# skeleton: component analysis -> harness + harnessed_app -> prover tool -> a
# 3-branch parallel join (setup ‖ invariants ‖ property-source) -> invariant-CVL
# stage -> per-component-CVL stage. They diverge in exactly two pluggable parts,
# expressed below as variant/strategy types rather than boolean flags:
#   * ``SetupMode``       — autosetup+summaries vs. a user-supplied conf+summary.
#   * ``PropertySource``  — bug analysis vs. formalize-then-build-batches.
# Plus an optional ``CoverageReporter`` for the properties-only uncovered dump.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutosetupMode:
    """Default setup: run harness creation, then AutoSetup + custom summaries."""


@dataclass(frozen=True)
class UserSuppliedMode:
    """``--skip-setup``: skip harness creation AND AutoSetup, using a
    user-supplied ``.conf`` as the prover config and a single summary import."""
    conf_path: str
    summary_path: str


type SetupMode = AutosetupMode | UserSuppliedMode


@dataclass
class SetupArtifacts:
    """Normalized output of the setup branch: the prover config the CVL stages
    run against, plus the CVL resources (summaries) they import."""
    prover_config: dict
    resources: list[CVLResource]


class PropertySource(Protocol):
    """The divergent middle phase: produces the per-component batches to generate
    CVL for, plus any input properties that never reached a batch (empty for the
    inference source). Runs concurrently with setup + invariants."""
    async def __call__(
        self,
        *,
        ctx: WorkflowContext[None],
        source_input: SourceCode,
        prop_context: WorkflowContext[Properties],
        handler_factory: HandlerFactory[AutoProvePhase, None],
        env: ServiceHost,
        harnessed_app: HarnessedApplication,
        semaphore: asyncio.Semaphore,
    ) -> tuple[list["_ComponentBatch"], list[UncoveredProperty]]:
        ...


class CoverageReporter(Protocol):
    """Optional tail hook: dump a coverage report for the run's uncovered
    properties (properties pipeline only)."""
    def __call__(self, result: AutoProveResult, project_root: str) -> None:
        ...


def build_harnessed_app(
    s: SourceApplication,
    contract_to_harness: dict[SolidityIdentifier, list[HarnessDefinition]],
) -> HarnessedApplication:
    """Build a ``HarnessedApplication`` from a component analysis, attaching the
    harnesses for each explicit contract (empty dict ⇒ no harnesses). Harnesses
    are keyed by ``solidity_identifier`` — the same identifier
    ``HarnessDefinition.harness_of`` records — so lookup matches regardless of a
    contract's display ``name``."""
    comp: list[SourceExternalActor | HarnessedExplicitContract] = []
    for c in s.components:
        if not isinstance(c, SourceExplicitContract):
            comp.append(c)
            continue
        comp.append(HarnessedExplicitContract(
            sort=c.sort,
            name=c.name,
            solidity_identifier=c.solidity_identifier,
            components=c.components,
            description=c.description,
            path=c.path,
            harnesses=contract_to_harness.get(c.solidity_identifier, []),
        ))
    return HarnessedApplication(
        application_type=s.application_type,
        description=s.description,
        components=comp,
    )


def materialize_invariant_cvl(
    source_input: SourceCode,
    certora_dir: pathlib.Path,
    inv_cvl: GeneratedCVL,
) -> CVLResource:
    """Write the structural-invariant spec + its property->rules map + final conf
    to disk, and return the (optional) ``CVLResource`` the per-component CVLs may
    import as assumable preconditions. Called for both freshly generated and
    cache-loaded results so the on-disk artifacts always match the cache."""
    ensure_dir(certora_dir / "specs")
    inv_spec_path = SPECS_DIR / "invariants.spec"
    under_project(source_input.project_root, inv_spec_path).write_text(inv_cvl.cvl)
    dump_property_rules(certora_dir, "invariants", inv_cvl.property_rules)
    dump_final_conf(
        project_root=source_input.project_root,
        main_contract=source_input.contract_name,
        task_id=INVARIANT_CVL_TASK_ID,
        spec_path=inv_spec_path,
        conf=inv_cvl.conf,
    )
    return CVLResource(
        path=inv_spec_path,
        required=False,
        description="Structural invariants that may be assumed as preconditions",
        sort="import",
    )


async def _run_autosetup_branch(
    *,
    ctx: WorkflowContext[None],
    source_input: SourceCode,
    sys_desc: SystemDescriptionHarnessed,
    s: SourceApplication,
    harnessed_app: HarnessedApplication,
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: ServiceHost,
    prover_opts: ProverOptions,
) -> SetupArtifacts:
    """AutoSetup (+ custom summaries) → prover config + summary resources."""
    setup_config = await run_task(
        handler_factory,
        TaskInfo(AUTOSETUP_TASK_ID, "AutoSetup", AutoProvePhase.AUTOSETUP),
        lambda: run_autosetup_phase(ctx, source_input, sys_desc, s, prover_opts),
    )
    resources: list[CVLResource] = [
        CVLResource(
            path=certora_relative_to_project(setup_config.summaries_path),
            required=True,
            description="AutoSetup-generated summaries",
            sort="import",
        ),
    ]
    if sys_desc.erc20_contracts or sys_desc.external_interfaces:
        summary_resource = await run_task(
            handler_factory,
            TaskInfo(SUMMARIES_TASK_ID, "Custom Summaries", AutoProvePhase.SUMMARIES),
            lambda: setup_summaries(
                ctx=ctx,
                app=harnessed_app,
                config=ContractSetup(system_description=sys_desc, config=setup_config),
                env=env,
                source=source_input,
            ),
        )
        resources.append(summary_resource)
    return SetupArtifacts(prover_config=setup_config.prover_config, resources=resources)


def _user_supplied_setup(mode: UserSuppliedMode) -> SetupArtifacts:
    """``--skip-setup`` → user ``.conf`` as the prover config + a single summary
    import. No LLM work, so no AutoSetup/Summaries tasks are emitted."""
    with open(mode.conf_path, "r") as f:
        prover_config = json.load(f)
    return SetupArtifacts(
        prover_config=prover_config,
        resources=[
            CVLResource(
                path=certora_relative_to_project(mode.summary_path),
                required=True,
                description="User-supplied summaries",
                sort="import",
            ),
        ],
    )


async def run_staged_pipeline(
    llm: BaseChatModel,
    source_input: SourceCode,
    ctx: WorkflowContext[None],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: ServiceHost,
    *,
    prover_opts: ProverOptions,
    max_concurrent: int,
    setup_mode: SetupMode,
    property_source: PropertySource,
    coverage_reporter: CoverageReporter | None = None,
) -> AutoProveResult:
    """Shared auto-prove orchestrator. The two public pipelines are thin wrappers
    that pick a ``setup_mode`` + ``property_source`` (+ optional
    ``coverage_reporter``) and call this."""
    semaphore = asyncio.Semaphore(max_concurrent)

    # Phase 1: component analysis.
    s = await run_task(
        handler_factory,
        TaskInfo(SYSTEM_ANALYSIS_TASK_ID, "System Analysis", AutoProvePhase.COMPONENT_ANALYSIS),
        lambda: run_component_analysis(ctx, source_input, env=env),
    )
    if s is None:
        raise ValueError("System analysis failed")

    # Phase 2: harness + harnessed_app. AutoSetup mode runs harness creation
    # (required by AutoSetup, which reads harness files from disk); user-supplied
    # mode builds the harnessed app with empty harness lists.
    sys_desc: SystemDescriptionHarnessed | None = None
    if isinstance(setup_mode, AutosetupMode):
        sys_desc = await run_task(
            handler_factory,
            TaskInfo(HARNESS_TASK_ID, "Harness Creation", AutoProvePhase.HARNESS),
            lambda: run_harness_creation(ctx, source_input, env, s),
        )
        contract_to_harness: dict[SolidityIdentifier, list[HarnessDefinition]] = {}
        for c in sys_desc.transitive_closure:
            if not c.harness_definition:
                continue
            contract_to_harness.setdefault(c.harness_definition.harness_of, []).append(
                HarnessDefinition(name=c.solidity_identifier, path=c.path)
            )
        harnessed_app = build_harnessed_app(s, contract_to_harness)
    else:
        harnessed_app = build_harnessed_app(s, {})

    # Prover tool is stateless with respect to setup; build it once and share.
    prover_tool = get_prover_tool(
        llm, source_input.contract_name,
        source_input.project_root, prover_opts=prover_opts,
    )

    # Phase 3 (parallel branches, joined below):
    #   A) setup -> prover config + resources
    #   B) structural-invariant formulation
    #   C) property source (bug analysis or formalize) -> component batches
    async def stream_setup() -> SetupArtifacts:
        if isinstance(setup_mode, AutosetupMode):
            assert sys_desc is not None  # guaranteed by Phase 2
            return await _run_autosetup_branch(
                ctx=ctx, source_input=source_input, sys_desc=sys_desc, s=s,
                harnessed_app=harnessed_app, handler_factory=handler_factory,
                env=env, prover_opts=prover_opts,
            )
        return _user_supplied_setup(setup_mode)

    async def stream_invariants():
        return await run_task(
            handler_factory,
            TaskInfo(INVARIANTS_TASK_ID, "Structural Invariants", AutoProvePhase.INVARIANTS),
            lambda: get_invariant_formulation(ctx, source_input, env, harnessed_app),
        )

    async def stream_properties() -> tuple[list["_ComponentBatch"], list[UncoveredProperty]]:
        return await property_source(
            ctx=ctx,
            source_input=source_input,
            prop_context=ctx.child(PROPERTIES_KEY),
            handler_factory=handler_factory,
            env=env,
            harnessed_app=harnessed_app,
            semaphore=semaphore,
        )

    setup_artifacts, invariants, (component_batches, source_uncovered) = await asyncio.gather(
        stream_setup(),
        stream_invariants(),
        stream_properties(),
    )

    certora_dir = under_project(source_input.project_root, CERTORA_DIR)

    # If nothing reached a batch, there is no CVL to generate: report the
    # uncovered properties and stop here (instead of raising).
    if not component_batches:
        _logger.warning("No properties were mapped to a component; skipping CVL generation.")
        result = AutoProveResult(n_components=0, n_properties=0, uncovered=source_uncovered)
        if coverage_reporter is not None:
            coverage_reporter(result, source_input.project_root)
        return result

    resources = setup_artifacts.resources

    # Join, stage 1: structural-invariant CVL. Runs before the per-component CVL
    # so invariants.spec exists and can be imported as assumable preconditions.
    # A give-up is non-fatal: the invariants are an *optional* precondition
    # resource, so the run continues without them.
    if invariants.inv:
        inv_cvl_ctx = ctx.child(INV_CVL_KEY)

        inv_props = [
            PropertyFormulation(
                title=inv.name,
                methods="invariant",
                description=inv.description,
                sort="invariant",
            )
            for inv in invariants.inv
        ]
        dump_properties(certora_dir, "invariants", inv_props)

        cached_inv_cvl = await inv_cvl_ctx.cache_get(GeneratedCVL)
        if cached_inv_cvl is not None:
            resources.append(materialize_invariant_cvl(source_input, certora_dir, cached_inv_cvl))
        else:
            inv_cvl_result = await run_task(
                handler_factory,
                TaskInfo(INVARIANT_CVL_TASK_ID, "Invariant CVL", AutoProvePhase.CVL_GEN),
                lambda: batch_cvl_generation(
                    ctx=inv_cvl_ctx.abstract(CVLGeneration),
                    component=None,
                    props=inv_props,
                    env=env,
                    init_config=setup_artifacts.prover_config,
                    prover_tool=prover_tool,
                    resources=resources,
                    description="Structural invariant CVL",
                    source=source_input,
                    spec_dir=SPECS_DIR,
                ),
            )
            if isinstance(inv_cvl_result, GaveUp):
                _logger.warning(
                    "Structural invariant CVL generation gave up (%s); continuing "
                    "without assumable invariants.", inv_cvl_result.reason,
                )
            else:
                await inv_cvl_ctx.cache_put(inv_cvl_result)
                resources.append(materialize_invariant_cvl(source_input, certora_dir, inv_cvl_result))

    # Join, stage 2: per-component CVL (parallel, semaphore-bounded). Imports
    # invariants.spec (if any) as assumable preconditions.
    result = await generate_all_component_cvl(
        source_input=source_input,
        component_batches=component_batches,
        handler_factory=handler_factory,
        env=env,
        prover_tool=prover_tool,
        prover_config=setup_artifacts.prover_config,
        resources=resources,
        semaphore=semaphore,
    )

    # Consolidate coverage: merge the property-source's uncovered (empty for the
    # inference source) into the result, then run the optional reporter.
    result.uncovered.extend(source_uncovered)
    if coverage_reporter is not None:
        coverage_reporter(result, source_input.project_root)
    return result
