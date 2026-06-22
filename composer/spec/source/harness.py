"""
Harness analysis and prover setup.

Identifies external contracts, classifies them, generates harness files
for contracts needing multiple instances, and runs AutoSetup compilation
to produce a ``Configuration`` for downstream phases.

Two entry points:

``analyze_external_interactions``
    Library entry point: runs the classification agent and returns a
    ``HarnessSetup`` with classifications + generated VFS files.  Uses
    an in-memory checkpointer and no memory tool.

``setup_and_harness_agent``
    Pipeline entry point: runs the classification agent within a
    ``WorkflowContext``, writes harness files to disk, runs AutoSetup
    compilation, and returns a ``Configuration``.
"""

from typing import NotRequired, TypedDict
from pathlib import Path
import subprocess

from pydantic import Field, BaseModel

from langgraph.graph import MessagesState

from composer.prover.core import ProverOptions
from graphcore.graph import FlowInput
from graphcore.tools.vfs import VFSState, VFSToolConfig, vfs_tools
from graphcore.tools.results import result_tool_generator

from composer.diagnostics.timing import get_run_summary
from composer.spec.graph_builder import run_to_completion, bind_standard
from composer.spec.source.autosetup import run_autosetup, read_autosetup_usage, SetupFailure, SetupSuccess
from composer.spec.service_host import ServiceHost
from composer.spec.context import WorkflowContext, SourceCode, CacheKey
from composer.spec.util import string_hash
from composer.spec.gen_types import TypedTemplate, certora_relative_to_project, under_project
from composer.spec.system_model import SolidityIdentifier, SourceApplication, SourceExternalActor, SourceExplicitContract

def system_setup_key(s: SourceApplication) -> CacheKey["ContractSetup", "SystemDescriptionHarnessed"]:
    return CacheKey["ContractSetup", "SystemDescriptionHarnessed"](
        "system-setup-" + string_hash(s.model_dump_json())
    )

class LinkField(BaseModel):
    """
    Expressing a "linking" relationship
    """
    target : list[SolidityIdentifier] = Field(description=(
        "The Solidity identifier(s) of the contract(s) being linked to — must match "
        "the `solidity_identifier` of an entry in the application description's "
        "transitive closure."
    ))
    link_paths: list[str] = Field(description="The list of Solidity storage access paths linking to `target`")


class ClosureContractBase(BaseModel):
    """
    A contract in the transitive closure.
    """
    solidity_identifier: SolidityIdentifier = Field(description=(
        "The Solidity identifier of the contract — must match the `solidity_identifier` "
        "of the corresponding entry in the application description."
    ))
    link_fields: list[LinkField] = Field(description="The linking relationship with other contracts in the closure")

class ClosureContract(ClosureContractBase):
    """
    A contract in the transitive closure.
    """
    num_instances : int | None = Field(description="The number of instances of this contract needed to model a non-trivial state (None if N/A)")

class HarnessDef(BaseModel):
    harness_of: SolidityIdentifier
    harness_source: str

class HarnessedContract(ClosureContractBase):
    harness_definition : HarnessDef | None
    path: str

class ExternalInterface(BaseModel):
    """
    An external actor interacted through an interface which is NOT included in the transitive closure
    """
    name: str = Field(description="The name of the external actor (taken from the application description)")
    behavioral_spec: str = Field(description="A natural language description of the behavior of the interface expected" \
    " by the contracts in the closure.")

class SystemDescriptionBase[T: ClosureContractBase](BaseModel):
    non_trivial_state: str = Field(description="A semi-formal description of a `non-trivial state`.")
    transitive_closure: list[T] = Field(description="The list of contracts in the transitive closure that interact with the main contract")
    erc20_contracts: list[SolidityIdentifier] = Field(description=(
        "A list of the Solidity identifiers (matching `solidity_identifier` "
        "entries in the application description) of the contracts which are ERC20 tokens"
    ))
    external_interfaces: list[ExternalInterface] = Field(description="A list of the external contract actors interacted with by the closure")


class AgentSystemDescription(SystemDescriptionBase[ClosureContract]):
    """
    The result of your analysis
    """

    def needs_harnessing(self) -> bool:
        return any([
            c.num_instances for c in self.transitive_closure
        ])
    
class LocatedClosureContract(ClosureContract):
    path: str

class LocatedSystemDescription(SystemDescriptionBase[LocatedClosureContract]):
    pass

class SystemDescriptionHarnessed(SystemDescriptionBase[HarnessedContract]):
    pass

class HarnessAnalysisParams(TypedDict):
    contract_name: str
    relative_path: str
    context: SourceApplication


class ContractSetup(BaseModel):
    system_description: SystemDescriptionHarnessed
    config: SetupSuccess

HarnessAnalysis = TypedTemplate[HarnessAnalysisParams]("state_analysis.j2")

HARNESS_ANALYSIS_KEY = CacheKey[SystemDescriptionHarnessed, AgentSystemDescription]("harness-analysis")

async def classifier_agent(
    context: WorkflowContext[SystemDescriptionHarnessed],
    app: SourceApplication,
    source: SourceCode,
    env: ServiceHost,
) -> AgentSystemDescription:
    child = context.child(HARNESS_ANALYSIS_KEY)
    if (cached := await child.cache_get(AgentSystemDescription)) is not None:
        return cached
    class AnalysisState(MessagesState):
        result: NotRequired[AgentSystemDescription]

    bound = HarnessAnalysis.bind({
        "context": app,
        "contract_name": source.contract_name,
        "relative_path": source.relative_path
    })

    external_lkp = {
        c.name: c for c in app.components if isinstance(c, SourceExternalActor)
    }

    contract_lkp = {
        c.solidity_identifier: c for c in app.contract_components
    }

    def result_validator(
        s: AnalysisState,
        res: AgentSystemDescription
    ) -> str | None:
        for ext in res.external_interfaces:
            if ext.name not in external_lkp:
                return f"External interface {ext.name} does not appear in the system description"
            if external_lkp[ext.name].path is None:
                return f"External interface {ext.name} doesn't have a path, and can't be identified as an interface"
        for c in res.transitive_closure:
            if c.solidity_identifier not in contract_lkp:
                return f"Contract {c.solidity_identifier} in the interaction closure doesn't appear in the application description"
        return None

    d = bind_standard(
        builder=env.builder,
        state_type=AnalysisState,
        validator=result_validator
    ).with_input(
        FlowInput
    ).with_tools(
        [child.get_memory_tool(), *env.source_tools]
    ).inject(
        lambda g: bound.render_to(g.with_initial_prompt_template)
    ).with_sys_prompt_template(
        "state_analysis_system_prompt.j2"
    ).compile_async()

    res = await run_to_completion(
        graph=d,
        context=None,
        description="Harness Analysis",
        recursion_limit=child.recursion_limit,
        input=FlowInput(input=[]),
        thread_id=child.thread_id
    )

    assert "result" in res
    await child.cache_put(res["result"])
    return res["result"]

class GeneratedHarness(BaseModel):
    """A generated harness file that creates a uniquely-named contract extending an external contract."""
    path: str = Field(description="Path to the harness definition")
    harness_name: SolidityIdentifier = Field(description="The Solidity identifier of the contract defined in the harness file")

class GeneratedHarnessSource(GeneratedHarness):
    source: str

class HarnessAgentResult(BaseModel):
    """
    The results of your harness generation
    """
    identifier_to_source: dict[SolidityIdentifier, list[GeneratedHarness]] = Field(description=(
        "A map from each target contract's `solidity_identifier` (exactly as given in "
        "the input list) to the harnesses chosen for it."
    ))
    solidity_compiler: str = Field(description=f"The solidity compiler to use for compiling these harnesses.")

class HarnessResult(BaseModel):
    identifier_to_source: dict[SolidityIdentifier, list[GeneratedHarnessSource]]

class HarnessInput(BaseModel):
    path: str
    n_harnesses: int
    solidity_identifier: SolidityIdentifier

class HarnessGenParams(TypedDict):
    to_harness: list[HarnessInput]

_HarnessGenerationPrompt = TypedTemplate[HarnessGenParams]("harness_generation_prompt.j2")

def harness_generation_key(
    instructions: AgentSystemDescription
) -> CacheKey[SystemDescriptionHarnessed, HarnessResult]:
    return CacheKey[SystemDescriptionHarnessed, HarnessResult](string_hash(instructions.model_dump_json()))

async def generate_harnesses(
    context: WorkflowContext[SystemDescriptionHarnessed],
    env: ServiceHost,
    source: SourceCode,
    application: SourceApplication,
    instructions: AgentSystemDescription
) -> HarnessResult:
    child = await context.child(harness_generation_key(instructions), instructions.model_dump())
    if (cached := await child.cache_get(HarnessResult)) is not None:
        return cached

    tool_conf = VFSToolConfig(
        fs_layer=source.project_root,
        immutable=False,
        put_doc_extra="You may only write into the `certora/harnesses` directory",
        forbidden_write="^(?!certora/harnesses)",
        forbidden_read=source.forbidden_read
    )

    class GenerationState(MessagesState, VFSState):
        result: NotRequired[HarnessAgentResult]
    
    class GenerationInput(FlowInput, VFSState):
        pass

    v_tools, mat = vfs_tools(tool_conf, GenerationState)

    contract_paths = {
        c.solidity_identifier: c.path for c in application.contract_components
    }

    harness_inputs = [
        HarnessInput(
            solidity_identifier=c.solidity_identifier,
            n_harnesses=c.num_instances,
            path=contract_paths[c.solidity_identifier]
        )
        for c in instructions.transitive_closure if c.num_instances is not None
    ]

    bound_template = _HarnessGenerationPrompt.bind({
        "to_harness": harness_inputs
    })

    expected = {
        c.solidity_identifier: c.n_harnesses for c in harness_inputs
    }


    def result_validator(
        s: GenerationState,
        res: HarnessAgentResult,
        tid: str
    ) -> str | None:
        check_copy = expected.copy()
        all_files = [
            
        ]
        for (nm, r) in res.identifier_to_source.items():
            if nm not in check_copy:
                return f"Delivered result for contract {nm}, but no instructions were given to harness it"
            if len(r) != check_copy[nm]:
                return f"Delivered {len(r)} harnesses for {nm}, but {check_copy[nm]} were required"
            for res_c in r:
                if mat.get(s, res_c.path) is None:
                    return f"Delivered harness {res_c.harness_name} at {res_c.path} for {nm}, but it doesn't exist on the VFS"
                all_files.append(res_c.path)
            del check_copy[nm]
        if len(check_copy) != 0:
            error = ", ".join(
                [ f"contract {k} ({n} copies)" for (k,n) in check_copy.items() ]
            )
            return f"Missing harnesses in results: {error}"
        if False: # this doesn't work
            with mat.materialize(s) as temp_dir:
                compile_result = subprocess.run(
                    [res.solidity_compiler] + all_files,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True
                )
                if compile_result.returncode != 0 and False:
                    return f"Harness compilation failed:\nstdout:\n{compile_result.stdout}\nstderr:\n{compile_result.stderr}"
        return None

    result_tool = result_tool_generator(
        "result",
        HarnessAgentResult,
        "Signal the completion of your workflow",
        validator=(GenerationState, result_validator)
    )

    g = env.builder.with_input(
        GenerationInput
    ).with_state(
        GenerationState
    ).with_output_key(
        "result"
    ).inject(
        lambda g: bound_template.render_to(g.with_initial_prompt_template)
    ).with_sys_prompt_template(
        "harness_generation_system_prompt.j2"
    ).with_tools(
        v_tools + [result_tool]
    ).with_default_summarizer().compile_async()

    res_state = await run_to_completion(
        graph=g,
        input=GenerationInput(input=[], vfs={}),
        context=None,
        description="Harness Implementation Generation",
        recursion_limit=child.recursion_limit,
        thread_id=child.thread_id
    )

    assert "result" in res_state

    res_dict : dict[SolidityIdentifier, list[GeneratedHarnessSource]] = {}
    for (nm, r) in res_state["result"].identifier_to_source.items():
        generated_source : list[GeneratedHarnessSource] = []
        for gen in r:
            source_code = mat.get(res_state, gen.path)
            assert source_code is not None, gen.path
            generated_source.append(GeneratedHarnessSource(
                path=gen.path,
                harness_name=gen.harness_name,
                source=source_code.decode("utf-8")
            ))
        res_dict[nm] = generated_source
    to_ret = HarnessResult(
        identifier_to_source=res_dict
    )
    await child.cache_put(to_ret)
    return to_ret

def _multi_replace(
    s: list[SolidityIdentifier],
    patch: dict[SolidityIdentifier, list[SolidityIdentifier]]
) -> list[SolidityIdentifier]:
    to_ret = []
    for i in s:
        if i in patch:
            to_ret.extend(patch[i])
        else:
            to_ret.append(i)
    return to_ret

def _patch_links(
    s: list[LinkField],
    patch: dict[SolidityIdentifier, list[SolidityIdentifier]]
) -> list[LinkField]:
    return [
        LinkField(
            link_paths=f.link_paths,
            target=_multi_replace(f.target, patch)
        ) for f in s
    ]

def apply_harness_result(
    s: LocatedSystemDescription,
    harness_result: HarnessResult
) -> SystemDescriptionHarnessed:
    new_contracts : list[HarnessedContract] = []
    forward_link = {
        k: [ h.harness_name for h in v ] for (k, v) in harness_result.identifier_to_source.items()
    }
    for c in s.transitive_closure:
        if not c.num_instances:
            new_contracts.append(HarnessedContract(
                solidity_identifier=c.solidity_identifier,
                link_fields=_patch_links(c.link_fields, forward_link),
                harness_definition=None,
                path=c.path
            ))
            continue
        patched_links = _patch_links(c.link_fields, forward_link)
        for gen in harness_result.identifier_to_source[c.solidity_identifier]:
            new_contracts.append(HarnessedContract(
                harness_definition=HarnessDef(
                    harness_of=c.solidity_identifier,
                    harness_source=gen.source,
                ),
                solidity_identifier=gen.harness_name,
                link_fields=patched_links,
                path=gen.path
            ))
    return SystemDescriptionHarnessed(
        erc20_contracts=s.erc20_contracts,
        external_interfaces=s.external_interfaces,
        non_trivial_state=s.non_trivial_state,
        transitive_closure=new_contracts
    )



async def run_setup_part1(
    context: WorkflowContext[ContractSetup],
    source: SourceCode,
    env: ServiceHost,
    application_desc: SourceApplication
) -> SystemDescriptionHarnessed:
    setup_ctx = await context.child(system_setup_key(application_desc), application_desc.model_dump())
    if (cached := await setup_ctx.cache_get(SystemDescriptionHarnessed)):
        return cached

    analysis_results = await classifier_agent(
        context=setup_ctx,
        app=application_desc,
        env=env,
        source=source
    )

    name_to_path = {
        c.solidity_identifier: c.path for c in application_desc.contract_components
    }

    located_desc = LocatedSystemDescription(
        non_trivial_state=analysis_results.non_trivial_state,
        erc20_contracts=analysis_results.erc20_contracts,
        external_interfaces=analysis_results.external_interfaces,
        transitive_closure=[
            LocatedClosureContract(
                link_fields=c.link_fields,
                solidity_identifier=c.solidity_identifier,
                num_instances=c.num_instances,
                path=name_to_path[c.solidity_identifier]
            ) for c in analysis_results.transitive_closure
        ]
    )

    harnessed_system : SystemDescriptionHarnessed

    if analysis_results.needs_harnessing():
        harness_result = await generate_harnesses(
            application=application_desc,
            context=setup_ctx,
            env=env,
            instructions=analysis_results,
            source=source
        )

        harnessed_system = apply_harness_result(
            located_desc,
            harness_result
        )
    else:
        harnessed_system = SystemDescriptionHarnessed(
            non_trivial_state=analysis_results.non_trivial_state,
            erc20_contracts=analysis_results.erc20_contracts,
            external_interfaces=analysis_results.external_interfaces,
            transitive_closure=[
                HarnessedContract(
                    link_fields=c.link_fields,
                    solidity_identifier=c.solidity_identifier,
                    harness_definition=None,
                    path=c.path
                ) for c in located_desc.transitive_closure
            ]
        )
    await setup_ctx.cache_put(harnessed_system)
    return harnessed_system

async def run_and_apply_part1(
    context: WorkflowContext[ContractSetup],
    source: SourceCode,
    env: ServiceHost,
    application_desc: SourceApplication
) -> SystemDescriptionHarnessed:
    res = await run_setup_part1(context, source, env, application_desc)
    for c in res.transitive_closure:
        if c.harness_definition is not None:
            tgt = Path(source.project_root) / c.path
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(c.harness_definition.harness_source)
    return res

config_key = CacheKey[None, ContractSetup]("config")

from logging import getLogger
_logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Split phases.
#
# Harness creation and AutoSetup are exposed as two separate, independently
# cached steps so the pipeline can run AutoSetup in parallel with invariant/bug
# analysis. They share the ``config_key`` parent context, so existing
# harness-creation caches (keyed by ``system_setup_key``) still hit; the
# AutoSetup result is cached under its own key.
# ---------------------------------------------------------------------------

async def run_harness_creation(
    context: WorkflowContext[None],
    source: SourceCode,
    env: ServiceHost,
    application_desc: SourceApplication,
) -> SystemDescriptionHarnessed:
    """Classify external contracts, generate harness files, and write them to
    disk. ``run_and_apply_part1`` re-writes the harness files on every call
    (idempotent), so they are guaranteed present for the AutoSetup phase even on
    a cache hit."""
    config_ctxt = context.child(config_key)
    return await run_and_apply_part1(config_ctxt, source, env, application_desc)


def autosetup_key(
    app: SourceApplication,
    prover_opts: ProverOptions,
) -> CacheKey[ContractSetup, SetupSuccess]:
    """Cache key for the AutoSetup phase. Includes ``prover_opts`` so cloud and
    local configurations never collide (the old composite ``config_key`` omitted
    them, which could reuse a stale config across modes)."""
    return CacheKey[ContractSetup, SetupSuccess](
        "autosetup-" + string_hash(
            app.model_dump_json() + "\x00" + "\x00".join(prover_opts.extra_args)
        )
    )


async def run_autosetup_phase(
    context: WorkflowContext[None],
    source: SourceCode,
    sys_desc: SystemDescriptionHarnessed,
    application_desc: SourceApplication,
    prover_opts: ProverOptions,
) -> SetupSuccess:
    """Run AutoSetup compilation against the (already written) harness files and
    return the compilation config + summaries. Depends on harness creation
    having run first: it reads the transitive-closure file paths from disk.

    Cache hits are guarded by the on-disk existence of ``summaries_path``."""
    config_ctxt = context.child(config_key)
    cache = await config_ctxt.child(
        autosetup_key(application_desc, prover_opts),
        application_desc.model_dump(),
    )
    if (cached := await cache.cache_get(SetupSuccess)) is not None:
        if under_project(source.project_root, certora_relative_to_project(cached.summaries_path)).exists():
            return cached

    extra_files = [
        c.path for c in sys_desc.transitive_closure if c.solidity_identifier != source.contract_name
    ]

    setup_result = await run_autosetup(
        Path(source.project_root),
        source.relative_path,
        source.contract_name,
        prover_opts,
        *extra_files,
    )

    if isinstance(setup_result, SetupFailure):
        raise RuntimeError(f"Auto setup failed: {setup_result.error}\nProc stderr:\n{setup_result.stderr}")

    # AutoSetup runs as a subprocess; its LLM token usage never reaches composer's
    # UsageCallback. Fold the counts it wrote to disk into the run summary so they
    # land in token_usage.json, the run tag, and the end-of-run table. No task_id:
    # the active task is already AUTOSETUP_TASK_ID, so this attributes to the
    # autosetup phase. Guarded — read_autosetup_usage returns [] if absent. This is
    # only reached on a cache miss (cache hits return above), so usage spent in this
    # process's autosetup run is counted exactly once.
    summary = get_run_summary()
    for usage in read_autosetup_usage(Path(source.project_root)):
        summary.record_token_usage(usage)

    await cache.cache_put(setup_result)
    return setup_result

