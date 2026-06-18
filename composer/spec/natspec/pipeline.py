"""
NatSpec multi-agent pipeline orchestration.

Replaces the monolithic natspec workflow with a multi-agent pipeline:
1. Component analysis (single agent)
2. Per-component property extraction (parallel)
3. Interface generation (single agent)
4. Initial stub generation (single agent)
5. Per-component batch CVL generation (parallel, semaphore-bounded) with merge

This is a plain asyncio orchestrator, not a LangGraph graph.

Every top-level agent invocation is wrapped in a per-task ``with_handler``
created by the caller-provided ``HandlerFactory``.  The TUI uses these to
populate a summary panel (collapsible by phase) with drill-down into
individual task event streams.
"""

import asyncio
import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, AsyncIterator, Iterable
from contextlib import asynccontextmanager
import tempfile

from langchain_core.tools import BaseTool

from langgraph.store.base import BaseStore
from graphcore.tools.vfs import FSBackend, Materializer

from composer.io.multi_job import (
    TaskInfo, HandlerFactory, run_task,
)

from composer.spec.context import (
    WorkflowContext,
    SystemDoc, CacheKey, Properties, ComponentGroup, CVLGeneration,
    Contract
)
from composer.spec.util import string_hash
from composer.spec.prop_inference import run_property_inference
from composer.spec.prop import PropertyFormulation
from composer.spec.natspec.interface_gen import generate_interface, DESCRIPTION as INTERFACE_GEN_DESC
from composer.spec.natspec.stub_gen import generate_stub
from composer.spec.natspec.models import InterfaceDeclModel, StubDeclarationModel
from composer.spec.natspec.registry import StubRegistry, FileRegistry
from composer.spec.natspec.typecheck import make_typechecker
from composer.spec.natspec.task_description import MentalModel, Assembler
from composer.spec.natspec.author import generate_cvl_batch, GaveUp, GenerationSuccess, AuthorResult
from composer.spec.natspec.system_analysis import run_component_analysis, DESCRIPTION as SYSTEM_DESC
from composer.spec.system_model import (
    ContractInstance, ContractComponentInstance, ContractComponent,
    ExplicitContract, NatspecApplication, ExistingFromSource, ContractName, SolidityIdentifier
)
from composer.spec.service_host import ServiceHost, PureServiceHost


# ---------------------------------------------------------------------------
# Generation gating
# ---------------------------------------------------------------------------

def _is_new(c: ExplicitContract) -> bool:
    """Whether this contract requires interface/stub/CVL generation.

    In greenfield, every contract is generated (plain ``ExplicitContract``).
    In from-source, only ``FreshFromSource`` contracts are generated;
    ``ExistingFromSource`` contracts (``unchanged``/``edited``) already have
    their source in the tree and are assumed correct for this task.
    """
    return not isinstance(c, ExistingFromSource)


# ---------------------------------------------------------------------------
# Phase type
# ---------------------------------------------------------------------------

class Phase(enum.Enum):
    """Pipeline phase tags carried on ``TaskInfo`` so the TUI can group
    tasks. Must be an ``Enum`` (not a ``Literal``) so that ``Phase``
    satisfies the ``HasName`` bound on ``HandlerFactory``: enum members
    expose ``.name``, bare string literals don't.
    """
    COMPONENT_ANALYSIS = "component_analysis"
    BUG_ANALYSIS = "bug_analysis"
    INTERFACE_GEN = "interface_gen"
    STUB_GEN = "stub_gen"
    CVL_GEN = "cvl_gen"


# ---------------------------------------------------------------------------
# Cache key helpers  (mirrors auto-prover's hash-based approach)
# ---------------------------------------------------------------------------

PROPERTIES_KEY = CacheKey[Contract, Properties]("properties")

def _component_cache_key(
    component: ContractComponent,
    app_type: str,
) -> CacheKey[Properties, ComponentGroup]:
    combined = "|".join([component.model_dump_json(), app_type])
    return CacheKey(string_hash(combined))


def _batch_cache_key(props: list[PropertyFormulation]) -> CacheKey[ComponentGroup, AuthorResult]:
    combined = "|".join(p.model_dump_json() for p in props)
    return CacheKey(string_hash(combined))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PropertyFailure:
    prop: PropertyFormulation
    reason: str

@dataclass
class ContractFormulation:
    interface: InterfaceDeclModel
    stub: StubDeclarationModel
    name: ContractName
    solidity_identifier: SolidityIdentifier
    spec_results: "ContractResult"

@dataclass
class PipelineResult:
    app: NatspecApplication
    contracts: list[ContractFormulation] = field(default_factory=list)



# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

MASTER_SPEC_NS = ("natspec_pipeline", "master_spec")
STUB_NS = ("natspec_pipeline", "stub")
FILES_NS = ("natspec_pipeline", "spec_files")

@dataclass
class ComponentGenerationSuccess():
    spec: str
    commentary: str
    suggested_path: str
    successful_properties: list[PropertyFormulation]
    component: ContractComponentInstance
    skipped_properties: list[PropertyFailure]

@dataclass
class ComponentGenerationFailure:
    component: ContractComponentInstance
    failed_properties: list[PropertyFormulation]
    reason: str

@dataclass
class ContractResult:
    specs: list[ComponentGenerationSuccess]
    failures: list[ComponentGenerationFailure] = field(default_factory=list)

@dataclass
class PipelineServices:
    sem: asyncio.Semaphore
    factory: HandlerFactory[Phase, None]
    env: ServiceHost
    mental_model: MentalModel
    file_registry: FileRegistry
    # When True, the bug-analysis step opens a per-component conversation
    # channel via the TUI's switcher so the user can refine the extracted
    # property list interactively. Parallel components each get their own
    # conversation_provider scoped to their own panel.
    #
    # Note for any future console_natspec driver: the multiplexing here
    # relies on the TUI switcher giving each task its own focusable panel.
    # A pure-console driver running this code path would need to serialize
    # interactive sessions via a conversation lock, the way
    # console_autoprove already does.
    interactive: bool = False
    max_bug_rounds: int = 3

async def analyze_single_contract(
    system_doc: SystemDoc,
    ctx: WorkflowContext[Contract],
    services: PipelineServices,
    solc_version: str,
    summary: ContractInstance,
    stub_registry: StubRegistry,
    stub: StubDeclarationModel,
    assembler: Assembler
) -> ContractResult:
    
    contract_name = summary.contract.name
    solidity_identifier = summary.contract.solidity_identifier
    handler_factory = services.factory
    semaphore = services.sem

    
    # ------------------------------------------------------------------
    # Shared artifacts for Phase 5
    # ------------------------------------------------------------------
    registry = stub_registry

    # ------------------------------------------------------------------
    # Phase 2 + 5:  Per-component extraction → per-component batch CVL gen
    # ------------------------------------------------------------------

    prop_context = ctx.child(PROPERTIES_KEY)

    failures: list[PropertyFailure] = []

    # Phase 2: per-component property extraction
    @dataclass
    class _ComponentBatch:
        feat: ContractComponentInstance
        props: list[PropertyFormulation]
        feat_ctx: WorkflowContext[ComponentGroup]

    async def _analyze_component(component_idx: int) -> _ComponentBatch | None:
        feat = ContractComponentInstance(_contract=summary, ind=component_idx)
        name = f"{feat.contract.name}: {feat.component.name}"
        feat_ctx = await prop_context.child(
            _component_cache_key(feat.component, summary.app.application_type),
            {
                "component": feat.component.model_dump(),
                "app_type": summary.app.application_type,
            },
        )

        interactive = services.interactive
        props = await run_task(
            handler_factory,
            TaskInfo(f"bug-{solidity_identifier}-{component_idx}", name, Phase.BUG_ANALYSIS),
            lambda conv: run_property_inference(
                feat_ctx, services.env, feat,
                refinement=conv if interactive else None,
                extra_input=[
                    "For reference, the system document describing the entire application is as follows.",
                    system_doc.content.to_dict(),
                ],
                max_rounds=services.max_bug_rounds,
            ),
            semaphore,
        )

        if not props:
            return None
        return _ComponentBatch(feat=feat, props=props, feat_ctx=feat_ctx)

    extraction_results = await asyncio.gather(*[
        _analyze_component(i) for i in range(len(summary.contract.components))
    ])

    component_batches = [b for b in extraction_results if b is not None]

    if not component_batches:
        raise ValueError("No properties extracted from any component.")

    # Phase 5: per-component batch CVL generation
    async def _generate_batch(
        batch_idx: int,
        batch: _ComponentBatch,
    ) -> GenerationSuccess | GaveUp:
        batch_ctx = await batch.feat_ctx.child(
            _batch_cache_key(batch.props),
            {"properties": [p.model_dump() for p in batch.props]},
        )
        batch_config_builder = services.mental_model.config_builder().with_solc(solc_version)

        stub_tools = registry.get_tools(solidity_identifier)
        file_tools = services.file_registry.get_tools(solidity_identifier)

        typechecker = make_typechecker(
            files=services.file_registry,
            assembler=assembler,
            config_builder=batch_config_builder,
            primary_contract=solidity_identifier,
        )

        label = f"{contract_name} {batch.feat.component.name} ({len(batch.props)} properties)"
        return await run_task(
            handler_factory,
            TaskInfo(f"cvl-{solidity_identifier}-{batch_idx}", label, Phase.CVL_GEN),
            lambda: generate_cvl_batch(
                stub_reader=lambda: stub_registry.read_stub(solidity_identifier),
                contract_name=contract_name,
                component=batch.feat,
                root_ctx=batch_ctx,
                env=services.env,
                props=batch.props,
                injected_tools=[*stub_tools, *file_tools],
                typechecker=typechecker,
                system_doc=system_doc,
                stub_path=stub.path
            ),
            semaphore,
        )

    generation_results = await asyncio.gather(
        *[
            _generate_batch(i, batch)
            for i, batch in enumerate(component_batches)
        ],
        return_exceptions=True,
    )

    succ : list[ComponentGenerationSuccess] = []
    fail : list[ComponentGenerationFailure] = []

    for batch, result in zip(component_batches, generation_results):
        match result:
            case BaseException():
                fail.append(
                    ComponentGenerationFailure(
                        component=batch.feat,
                        failed_properties=batch.props,
                        reason=str(result)
                    )
                )
            case GaveUp(reason=reason):
                fail.append(
                    ComponentGenerationFailure(
                        component=batch.feat,
                        failed_properties=batch.props,
                        reason=reason
                    )
                )
            case GenerationSuccess():
                props_by_title = {p.title: p for p in batch.props}
                skipped : set[str] = set()
                failures : list[PropertyFailure] = []
                for skip in result.skipped:
                    skipped_prop = props_by_title.get(skip.property_title)
                    if skipped_prop is not None:
                        failures.append(PropertyFailure(
                            prop=skipped_prop,
                            reason=f"Skipped: {skip.reason}",
                        ))
                    skipped.add(skip.property_title)
                succ_props = [
                    l for l in batch.props if l.title not in skipped 
                ]
                succ.append(ComponentGenerationSuccess(
                    commentary=result.commentary,
                    component=batch.feat,
                    skipped_properties=failures,
                    spec=result.spec,
                    successful_properties=succ_props,
                    suggested_path=result.suggested_path
                ))

    return ContractResult(
        specs=succ,
        failures=fail
    )

type ToolGenerator = Callable[[list[FSBackend]], tuple[list[BaseTool], Materializer]]

class MaterializerAssembler(Assembler):
    def __init__(self, mat: Materializer):
        self.mat = mat

    @asynccontextmanager
    async def project_directory(self) -> AsyncIterator[Path]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            await self.mat.dump_to(Path(tmp))
            yield Path(tmp)

class InMemoryBackend:
    def __init__(self, vfs: dict[str, str]):
        self._vfs = vfs

    def list(self) -> Iterable[str]:
        to_ret = []
        for i in self._vfs.keys():
            to_ret.append(i)
        return to_ret

    def get(self, path: str) -> str | None:
        return self._vfs.get(path, None)

    async def dump_to(
        self,
        target: Path,
        include_path: Callable[[str], bool] | None = None,
    ) -> None:
        for (k, v) in self._vfs.items():
            if include_path is not None and not include_path(k):
                continue
            tgt = (target / k)
            tgt.parent.mkdir(exist_ok=True, parents=True)
            tgt.write_text(v)

async def run_natspec_pipeline[A: NatspecApplication, I: InterfaceDeclModel, S: StubDeclarationModel](
    system_doc: SystemDoc,
    solc_version: str,
    start_env: PureServiceHost,
    ctx: WorkflowContext[None],
    store: BaseStore,
    handler_factory: HandlerFactory[Phase, None],
    mental_model: MentalModel[A, I, S],
    source_factory: ToolGenerator,
    *,
    max_concurrent: int = 4,
    interactive: bool = False,
    max_bug_rounds: int = 3,
) -> PipelineResult:
    """Run the full natspec multi-agent pipeline.

    Every agent invocation is wrapped in a per-task ``with_handler``
    obtained from ``handler_factory``.  The TUI can group tasks by
    ``TaskInfo.phase`` into collapsible sections.

    Cache hierarchy mirrors auto-prover::

        root [None]
          └── properties [Properties]
              └── <component-hash> [ComponentGroup]
                  ├── bug_analysis (internal to bug.py)
                  └── <batch-hash> [GeneratedCVL] → abstract(CVLGeneration)

    Args:
        system_doc: The design document for the application.
        contract_name: The expected contract name.
        solc_version: Solidity compiler version (e.g., "8.21").
        analysis_builder: Builder for component analysis, interface/stub gen, registry (no domain tools).
        cvl_authorship: Builder for CVL generation agents (has CVL manual tools).
        cvl_research: Builder for CVL research sub-agents and merge agent.
        ctx: Root workflow context.
        store: BaseStore for shared artifacts and caching.
        handler_factory: Creates per-task ``(IOHandler, EventHandler)``
            pairs.  Called once per top-level agent invocation.
        max_concurrent: Maximum concurrent LLM agents.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    init_tools, mat_ = source_factory([])

    mat = MaterializerAssembler(mat_)

    curr_env = start_env.bind_source_tools(init_tools)

    # ------------------------------------------------------------------
    # Phase 1: Component analysis
    # ------------------------------------------------------------------
    summary = await run_task(
        handler_factory,
        TaskInfo("component-analysis", SYSTEM_DESC, Phase.COMPONENT_ANALYSIS),
        lambda: run_component_analysis(ctx, system_doc, curr_env, mental_model),
    )
    if summary is None:
        raise ValueError("Component analysis produced no result — is the system doc empty?")

    new_contracts = [c for c in summary.contract_components if _is_new(c)]
    new_identifiers = {c.solidity_identifier for c in new_contracts}

    # ------------------------------------------------------------------
    # Phase 3: Interface generation (new contracts only)
    # ------------------------------------------------------------------
    interface = await run_task(
        handler_factory,
        TaskInfo("interface-gen", INTERFACE_GEN_DESC, Phase.INTERFACE_GEN),
        lambda: generate_interface(
            ctx, summary, curr_env, solc_version,
            description=mental_model.interface_desc,
            target_identifiers=new_identifiers, materializer=mat
        ),
    )

    intf_backend = InMemoryBackend(
        {
            v.path: v.content for (_, v) in interface.name_to_interface.items()
        }
    )

    with_intf_tools, mat_ = source_factory([intf_backend])

    curr_env = curr_env.bind_source_tools(with_intf_tools)

    mat = MaterializerAssembler(mat_)

    # ------------------------------------------------------------------
    # Phase 4: Initial stub generation (new contracts only)
    # ------------------------------------------------------------------
    async def gen_one_stub(
        contract_name: ContractName,
        solidity_identifier: SolidityIdentifier,
    ) -> tuple[SolidityIdentifier, StubDeclarationModel]:
        res = await run_task(
            handler_factory,
            TaskInfo(
                f"stub-gen-{solidity_identifier}",
                f"Stub: {contract_name}",
                Phase.STUB_GEN,
            ),
            lambda: generate_stub(
                ctx, interface, curr_env, contract_name, solidity_identifier, solc_version,
                materializer=mat,
                description=mental_model.stub_desc,
            ),
        )
        return (solidity_identifier, res)

    generated_stubs = await asyncio.gather(*[
        gen_one_stub(c.name, c.solidity_identifier) for c in new_contracts
    ])

    # ------------------------------------------------------------------
    # Shared artifacts for Phase 5
    # ------------------------------------------------------------------

    doc_digest = system_doc.content.to_digest()

    registry = await StubRegistry.acreate(
        store, STUB_NS + (doc_digest,), start_env.builder, interface,
        mat, dict(generated_stubs), solc_version,
        recursion_limit=ctx.recursion_limit,
    )

    name_to_stub = { nm: stub for (nm, stub) in generated_stubs }

    # Build the layered FS (with stubs + interfaces) BEFORE constructing the
    # FileRegistry so the registry can close over the same materializer the
    # agents will see — that's what backs its existence-check on register.
    with_stub_and_intf, mat_ = source_factory([
        registry, intf_backend
    ])

    curr_env = curr_env.bind_source_tools(with_stub_and_intf)

    mat = MaterializerAssembler(mat_)

    file_registry = await FileRegistry.acreate(
        store, FILES_NS + (doc_digest,), materializer=mat_,
    )

    for c in summary.contract_components:
        if not _is_new(c):
            continue
        stub = name_to_stub[c.solidity_identifier]
        # Stub validator enforces ``path.stem == c.solidity_identifier``, so
        # the bare path is sufficient — certora derives the identifier from
        # the stem and produces the same prover arg either way.
        await file_registry.register(
            contract_identifier=c.solidity_identifier,
            path=stub.path,
        )

    serv = PipelineServices(
        sem=semaphore,
        env=curr_env,
        factory=handler_factory,
        mental_model=mental_model,
        file_registry=file_registry,
        interactive=interactive,
        max_bug_rounds=max_bug_rounds,
    )

    tasks : list[Awaitable[ContractResult]] = []
    new_contracts_with_ind = [
        (ind, c) for ind, c in enumerate(summary.contract_components) if _is_new(c)
    ]
    for (ind, contract) in new_contracts_with_ind:
        contract_key = CacheKey[None, Contract](string_hash(contract.model_dump_json()))
        contract_ctx = await ctx.child(contract_key, contract.model_dump())
        cont = analyze_single_contract(
            system_doc=system_doc,
            ctx=contract_ctx,
            services=serv,
            solc_version=solc_version,
            stub_registry=registry,
            summary=ContractInstance(ind=ind, app=summary),
            stub=name_to_stub[contract.solidity_identifier],
            assembler=mat
        )

        tasks.append(cont)
    results = await asyncio.gather(*tasks)

    to_ret : list[ContractFormulation] = []
    for ((_, c), res) in zip(new_contracts_with_ind, results):
        to_ret.append(ContractFormulation(
            interface=interface.name_to_interface[c.solidity_identifier],
            stub=name_to_stub[c.solidity_identifier],
            name=c.name,
            solidity_identifier=c.solidity_identifier,
            spec_results=res
        ))

    return PipelineResult(
        app=summary,
        contracts=to_ret,
    )
