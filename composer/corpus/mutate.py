import asyncio
import json
from dataclasses import dataclass, field
import difflib
from datetime import datetime

import pathlib
from typing import NotRequired, override, Callable, AsyncContextManager

from pydantic import BaseModel, Field

from composer.corpus.models import PipelineState, CorpusEntry
from composer.corpus.pipeline import StateCache, ensure_report

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from langgraph.types import Command
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import InMemorySaver

from composer.corpus.preprocess import pdf_key
from composer.corpus.analysis import lib_dir_is_packages
from composer.prover.core import run_prover, ProverOptions, CloudConfig, CexHandler, ProverCallbacks, RawReport
from composer.prover.ptypes import RuleResult
from graphcore.tools.vfs import vfs_tools, VFSState, VFSAccessor, fs_tools
from graphcore.tools.schemas import WithAsyncDependencies, WithInjectedState, WithInjectedId
from graphcore.graph import MessagesState, tool_state_update, tool_return, Builder, FlowInput
from composer.templates.loader import load_jinja_template
from composer.spec.code_explorer import indexed_code_explorer_tool
from composer.spec.agent_index import AgentIndex
from composer.io.context import run_graph
from composer.spec.util import uniq_thread_id
import hashlib

from pathlib import Path

def url_to_ns(u: str):
    return hashlib.sha256(u.encode()).hexdigest()

type ProverSemaphore = AsyncContextManager[None]

class RuleValidationCache:
    def __init__(self, store: BaseStore, base_ns: tuple[str, ...]):
        self.store = store
        self.base_ns = base_ns

    async def get_validated(self, url: str, rule: str) -> bool | None:
        r = await self.store.aget(self.base_ns + (url_to_ns(url),), rule)
        if r is None:
            return None
        return r.value["validated"]

    async def save_validated(self, url: str, rule: str, validated: bool):
        await self.store.aput(self.base_ns + (url_to_ns(url),), rule, {"validated": validated})

class RuleValidationWrapper:
    def __init__(self, parent: RuleValidationCache, url: str):
        self.parent = parent
        self.url = url

    async def get_validated(self, rule: str) -> bool | None:
        return await self.parent.get_validated(self.url, rule)
    
    async def save_validated(self, rule: str, validated: bool):
        await self.parent.save_validated(self.url, rule, validated)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SourceEnv:
    builder: Builder[None, None, None]
    base_source_tools: tuple[BaseTool, ...]
    index: AgentIndex

    @property
    def has_source(self) -> bool:
        return True

class MutantDesc(BaseModel):
    what_i_did: str
    killed: bool

class Mutant(MutantDesc):
    mutated_vfs: dict[str, str]
    diff: str

    what_i_did: str
    killed: bool

class MutationState(VFSState, MessagesState):
    mutant: NotRequired[MutantDesc]

class MutationInput(FlowInput, VFSState):
    pass

@dataclass
class ProverDeps:
    mat: VFSAccessor[MutationState]
    rule_name: str
    prover_sem: ProverSemaphore
    agent_desc: str

class NullCexHandler(CexHandler):
    @override
    async def analyze_cex(self, rule: RuleResult) -> str | None:
        return None
    
class ProverDebug(ProverCallbacks):
    def __init__(self, desc: str):
        self.desc = desc

    async def on_prover_run(self, args: list[str]) -> None:
        print(f"{self.desc} Prover start -> {args}")
    
    async def on_prover_result(self, results: dict[str, RuleResult]) -> None:
        print(f"{self.desc} Prover done")

class ProverRunner(WithAsyncDependencies[Command, ProverDeps], WithInjectedId, WithInjectedState):
    """
    Try to run the prover against your mutation
    """
    what_i_did: str = Field(description="A description of the mutation you are running")
    @override
    async def run(self) -> Command:
        with self.tool_deps() as deps, deps.mat.materialize(self.state) as where:
            async with deps.prover_sem:
                res = await run_prover(
                    pathlib.Path(where),
                    ["./run.conf", "--rule", deps.rule_name, "--msg", deps.agent_desc], ProverOptions(cloud=CloudConfig()),
                    ProverDebug(deps.agent_desc),
                    NullCexHandler()
                )
                if isinstance(res, str):
                    return tool_return(self.tool_call_id, f"Prover run failed: {res}")
                assert isinstance(res, RawReport)
                assert deps.rule_name in res.rule_status
                return tool_state_update(tool_call_id=self.tool_call_id, content="Done", mutant=MutantDesc(
                    what_i_did=self.what_i_did,
                    killed=not res.rule_status[deps.rule_name]
                ))


# ---------------------------------------------------------------------------
# Per-rule and per-project result containers
# ---------------------------------------------------------------------------

@dataclass
class PropAndMutants:
    ent: CorpusEntry
    mutants: list[Mutant]


@dataclass
class ProjectResult:
    protocol_name: str
    rules: list[PropAndMutants]


def _run_id() -> str:
    return f"run_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"


@dataclass
class IncrementalProjectWriter:
    project_dir: Path
    protocol_name: str
    _completed_rules: list[PropAndMutants] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def write_mutant(self, rule_name: str, index: int, mutant: Mutant) -> None:
        rule_dir = self.project_dir / _safe_dirname(rule_name)
        mutant_dir = rule_dir / f"Mutant{index}"
        _write_mutant_dir(mutant_dir, mutant)

    def finish_rule(self, prop: PropAndMutants) -> None:
        rule_dir = self.project_dir / _safe_dirname(prop.ent.rule_name)
        rule_dir.mkdir(parents=True, exist_ok=True)
        _write_rule_summary(rule_dir, prop)
        self._completed_rules.append(prop)

    def finish_project(self) -> None:
        result = ProjectResult(
            protocol_name=self.protocol_name,
            rules=self._completed_rules,
        )
        _write_project_summary(self.project_dir, result)


# ---------------------------------------------------------------------------
# Mutator — builds and runs the mutation agent for a single source tree
# ---------------------------------------------------------------------------

class Mutator():
    def __init__(
        self,
        project_state: PipelineState,
        llm: BaseChatModel,
        source_path: Path,
        index: AgentIndex,
        sem: asyncio.Semaphore,
        prover_sem: ProverSemaphore,
        validation_cache: RuleValidationWrapper
    ):
        self.state = project_state
        self.llm = llm
        self.source_path = source_path
        path_layer = str(source_path.resolve().absolute())

        forbid_re = r".+(?<!\.sol)$|^node_modules/.+|^certora/.+|^\.certora_.+"
        if lib_dir_is_packages(str(source_path)):
            forbid_re += "|^lib/.+"
        v_tools, mat = vfs_tools({
            "immutable": False,
            "forbidden_read": forbid_re,
            "forbidden_write": forbid_re,
            "fs_layer": path_layer
        }, MutationState)
        self.llm_sem = sem
        self.prover_sem = prover_sem
        self.rule_cache = validation_cache

        self.fs_tools = fs_tools(fs_layer=path_layer, forbidden_read=forbid_re)

        base_builder = Builder().with_llm(llm).with_checkpointer(InMemorySaver()).with_loader(load_jinja_template)

        explorer_tool = indexed_code_explorer_tool(
            SourceEnv(base_builder, tuple(self.fs_tools), index)
        )

        self.mat = mat

        self.sanity_locks : dict[str, asyncio.Lock] = {}

        self.builder = base_builder.with_input(MutationInput).with_output_key(
            "mutant"
        ).with_state(MutationState).with_default_summarizer(max_messages=50).with_tools(
            [*v_tools, explorer_tool]
        ).with_sys_prompt_template(
            "corpus_mutation_system.j2"
        )

    async def _sanity_check(
        self, rule: str
    ) -> bool:
        if rule not in self.sanity_locks:
            self.sanity_locks[rule] = asyncio.Lock()
        async with self.sanity_locks[rule]:
            res = await self.rule_cache.get_validated(rule)
            if res is not None:
                return res
            async with self.prover_sem:
                r = await run_prover(
                    self.source_path, ["./run.conf", "--rule", rule, "--msg", f"validation: {rule}"], ProverOptions(CloudConfig()), ProverCallbacks(), NullCexHandler()
                )
            to_ret = isinstance(r, RawReport) and r.rule_status.get(rule, False)
            await self.rule_cache.save_validated(rule, to_ret)
            return to_ret

    async def mutate(
        self, entry: CorpusEntry, ind: int
    ) -> Mutant | None:
        agent_desc = f"Mutation Agent: {entry.rule_name} {ind}"
        runner = self.builder.with_initial_prompt_template(
            "corpus_mutation_prompt.j2",
            protocol_description=self.state.triage.protocol_description if self.state.triage else "(No description provided)",
            property_description=entry.extracted_property_description,
            rule_name=entry.rule_name
        ).with_tools(
            [ProverRunner.bind(ProverDeps(mat=self.mat, rule_name=entry.rule_name, prover_sem=self.prover_sem, agent_desc=agent_desc)).as_tool("prover_runner")]
        ).compile_async()
        async with self.llm_sem:
            if not await self._sanity_check(entry.rule_name):
                return None
            try:
                r = await run_graph(
                    graph=runner, ctxt=None,
                    input=MutationInput(input=[], vfs={}),
                    description=agent_desc,
                    within_tool=None,
                    run_conf={
                        "configurable": {
                            "thread_id": uniq_thread_id("mutation")
                        },
                        "recursion_limit": 100
                    }
                )
            except GraphRecursionError:
                return None
        assert "mutant" in r
        diff_str = []

        for (k,v) in r["vfs"].items():
            orig_file = self.source_path / k
            orig_lines = orig_file.read_text().splitlines()
            new_lines = v.splitlines()
            diff_lines = difflib.unified_diff(
                a=orig_lines,
                b=new_lines,
                fromfile=f"a/{k}",
                tofile=f"b/{k}"
            )
            diff_str.extend(diff_lines)

        return Mutant(
            mutated_vfs=r["vfs"], diff="\n".join(diff_str), killed=r["mutant"].killed, what_i_did=r["mutant"].what_i_did
        )


type MutatorFactory = Callable[[pathlib.Path, str, PipelineState], Mutator]

def mutator_factory(
    llm: BaseChatModel,
    sem: asyncio.Semaphore,
    cache_ns: tuple[str, ...],
    store_index: BaseStore,
    prover_sem: ProverSemaphore,
    rule_cache: RuleValidationCache,
) -> MutatorFactory:
    return lambda path, url, pipeline: Mutator(pipeline, llm, path, AgentIndex(store_index, cache_ns+(url_to_ns(url),)), sem, prover_sem, RuleValidationWrapper(rule_cache, url))


# ---------------------------------------------------------------------------
# Mutation generation
# ---------------------------------------------------------------------------

async def generate_property_mutant(
    prop: CorpusEntry,
    mut: Mutator,
    writer: IncrementalProjectWriter,
) -> PropAndMutants | None:    
    non_null = []
    # we do this sequentially because it's incredibly likely that each agent will ask similar
    # questions of the code explorer; by staggering their runs, hopefully the later ones work better
    for ind in range(0, 1):
        result = await mut.mutate(prop, ind)
        if result is not None:
            writer.write_mutant(prop.rule_name, ind + 1, result)
            non_null.append(result)

    if not non_null:
        return None
    pam = PropAndMutants(prop, non_null)
    writer.finish_rule(pam)
    return pam


async def generate_tree_mutants(
    work_dir: Path,
    state: PipelineState,
    url: str,
    props: list[CorpusEntry],
    report_sem: asyncio.Semaphore,
    gen: MutatorFactory,
    writer: IncrementalProjectWriter,
) -> list[PropAndMutants] | None:
    report_path = await ensure_report(work_dir, state, url, report_sem)
    if report_path is None or not (report_path / "run.conf").exists():
        return None
    mutator = gen(report_path, url, state)
    mutants = await asyncio.gather(*[
        generate_property_mutant(
            mut=mutator, prop=prop, writer=writer,
        ) for prop in props
    ])
    return [m for m in mutants if m is not None]


async def mutate_project(
    work_dir: Path,
    state: PipelineState,
    mutator_fact: MutatorFactory,
    report_sem: asyncio.Semaphore,
    writer: IncrementalProjectWriter,
) -> None:
    tree_jobs = []
    for (k, v) in state.analyzed_trees.items():
        tree_jobs.append(
            generate_tree_mutants(
                work_dir=work_dir, props=v, url=k, gen=mutator_fact,
                state=state, report_sem=report_sem, writer=writer,
            )
        )
    await asyncio.gather(*tree_jobs)


# ---------------------------------------------------------------------------
# Output reporting
# ---------------------------------------------------------------------------

def _safe_dirname(name: str) -> str:
    """Convert a human-readable name to a filesystem-safe directory name."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _write_mutant_dir(mutant_dir: Path, mutant: Mutant) -> None:
    """Write a single mutant's artifacts to its directory."""
    mutant_dir.mkdir(parents=True, exist_ok=True)

    (mutant_dir / "mutant_desc.md").write_text(mutant.what_i_did)
    (mutant_dir / "mutant.patch").write_text(mutant.diff)

    for rel_path, content in mutant.mutated_vfs.items():
        out_file = mutant_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(content)


def _rule_stats(prop: PropAndMutants) -> dict:
    killed = sum(1 for m in prop.mutants if m.killed)
    return {
        "rule_name": prop.ent.rule_name,
        "property_title": prop.ent.property_title,
        "property_description": prop.ent.property_description,
        "rule_description": prop.ent.rule_description,
        "total_mutants": len(prop.mutants),
        "killed": killed,
        "survived": len(prop.mutants) - killed,
    }


def _write_rule_summary(rule_dir: Path, prop: PropAndMutants) -> None:
    """Write the per-rule executive summary."""
    stats = _rule_stats(prop)
    md = load_jinja_template(
        "mutation_rule_summary.md.j2",
        **stats,
        mutants=[
            {"desc": m.what_i_did.replace("\n", " ").strip(), "killed": m.killed}
            for m in prop.mutants
        ],
    )
    (rule_dir / "summary.md").write_text(md)


def _write_project_summary(project_dir: Path, result: ProjectResult) -> None:
    """Write the project-level executive summary (markdown + JSON)."""
    rules = [_rule_stats(pm) for pm in result.rules]
    total_mutants = sum(r["total_mutants"] for r in rules)
    total_killed = sum(r["killed"] for r in rules)

    summary_vars = {
        "protocol_name": result.protocol_name,
        "total_rules": len(rules),
        "rules_all_killed": sum(1 for r in rules if r["survived"] == 0),
        "rules_with_gaps": sum(1 for r in rules if r["survived"] > 0),
        "total_mutants": total_mutants,
        "total_killed": total_killed,
        "total_survived": total_mutants - total_killed,
        "rules": rules,
    }

    (project_dir / "summary.md").write_text(
        load_jinja_template("mutation_project_summary.md.j2", **summary_vars)
    )
    (project_dir / "summary.json").write_text(json.dumps(summary_vars, indent=2))


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

async def try_mutate(
    work_dir: Path,
    pdf_dir: Path,
    output_dir: Path,
    mutation_llm: BaseChatModel,
    state_cache: StateCache,
    ind_store: BaseStore,
    store_ns: tuple[str, ...],
    sem: asyncio.Semaphore,
    rule_cache: RuleValidationCache
) -> None:
    report_sem = asyncio.Semaphore(10)
    prover_sem = asyncio.Semaphore(5)
    factory = mutator_factory(mutation_llm, sem, store_ns, ind_store, prover_sem, rule_cache=rule_cache)

    run_dir = output_dir / _run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    async def _process_pdf(pdf_path: Path) -> None:
        key = pdf_key(pdf_path)
        st = await state_cache.load(key)
        if st is None:
            return
        if st.triage is None:
            return
        if not st.analyzed_trees:
            return

        protocol_name = st.triage.protocol_name
        writer = IncrementalProjectWriter(
            run_dir / _safe_dirname(protocol_name), protocol_name,
        )

        await mutate_project(
            work_dir=work_dir,
            state=st,
            mutator_fact=factory,
            report_sem=report_sem,
            writer=writer,
        )
        writer.finish_project()
        print(f"  Project complete: {writer.project_dir}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found.")
        return

    total = len(pdfs)
    print(f"Found {total} PDFs for mutation testing.")

    jobs = []
    for i, pdf_path in enumerate(pdfs):
        print(f"[{i + 1}/{total}] Queuing {pdf_path.name}...")
        jobs.append(_process_pdf(pdf_path))

    await asyncio.gather(*jobs)

    print(f"\nDone. Results in {run_dir}")
