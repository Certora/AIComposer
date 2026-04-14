import asyncio
from dataclasses import dataclass
import difflib

import pathlib
from typing import NotRequired, override, Callable

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
from composer.spec.code_explorer import indexed_code_explorer_tool, AgentIndex
from composer.io.context import run_graph
from composer.spec.util import uniq_thread_id

from pathlib import Path

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

class NullProverCB:
    pass

@dataclass
class PropAndMutants:
    ent: CorpusEntry
    mutants: list[Mutant]

class NullCexHandler(CexHandler):
    @override
    async def analyze_cex(self, rule: RuleResult) -> str | None:
        return None

class ProverRunner(WithAsyncDependencies[Command, ProverDeps], WithInjectedId, WithInjectedState):
    """
    Try to run the prover against your mutation
    """
    what_i_did: str = Field(description="A description of the mutation you are running")
    @override
    async def run(self) -> Command:
        with self.tool_deps() as deps, deps.mat.materialize(self.state) as where:
            res = await run_prover(
                pathlib.Path(where),
                ["--rule", deps.rule_name], ProverOptions(cloud=CloudConfig()),
                ProverCallbacks(),
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

class Mutator():
    def __init__(self, llm: BaseChatModel, source_path: Path, index: AgentIndex, sem: asyncio.Semaphore):
        self.llm = llm
        self.source_path = source_path
        path_layer = str(source_path.resolve().absolute())

        forbid_re = r"(?<!\.sol)$|^node_modules/|^certora/|^.certora_"
        if lib_dir_is_packages(str(source_path)):
            forbid_re += "|^lib/"
        v_tools, mat = vfs_tools({
            "immutable": False,
            "forbidden_read": forbid_re,
            "forbidden_write": forbid_re,
            "fs_layer": path_layer
        }, MutationState)
        self.sem = sem

        self.fs_tools = fs_tools(fs_layer=path_layer, forbidden_read=forbid_re)

        base_builder = Builder().with_llm(llm).with_checkpointer(InMemorySaver()).with_loader(load_jinja_template)

        explorer_tool = indexed_code_explorer_tool(
            SourceEnv(base_builder, tuple(self.fs_tools), index)
        )

        self.mat = mat

        self.builder = base_builder.with_input(MutationInput).with_output_key(
            "mutant"
        ).with_state(MutationState).with_default_summarizer(max_messages=50).with_tools(
            [*v_tools, explorer_tool]
        ).with_sys_prompt_template(
            "...."
        )

    async def sanity_check(
        self, rule: str
    ) -> bool:
        r = await run_prover(
            self.source_path, ["--rule", rule], ProverOptions(CloudConfig()), ProverCallbacks(), NullCexHandler()
        )
        return isinstance(r, RawReport) and r.rule_status.get(rule, False)

    async def mutate(
        self, property: str, rule: str
    ) -> Mutant | None:
        runner = self.builder.with_initial_prompt_template(
            "..."
        ).with_tools(
            [ProverRunner.bind(ProverDeps(mat=self.mat, rule_name=rule)).as_tool("prover_runner")]
        ).compile_async()
        async with self.sem:
            try:
                r = await run_graph(
                    graph=runner, ctxt=None,
                    input=MutationInput(input=[], vfs={}),
                    description=f"Mutation Agent: {property}",
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

type MutatorFactory = Callable[[pathlib.Path], Mutator]

def mutator_factory(llm: BaseChatModel, index: AgentIndex, sem: asyncio.Semaphore) -> MutatorFactory:
    return lambda path: Mutator(llm, path, index, sem)

async def generate_property_mutant(
    prop: CorpusEntry,
    mut: Mutator
) -> PropAndMutants | None:
    if not await mut.sanity_check(prop.rule_name):
        return None
    to_res = await asyncio.gather(*[
        mut.mutate(prop.property_description, prop.rule_name)
        for _ in range(0, 10)
    ])
    non_null = [ r for r in to_res if r ]
    if not non_null:
        return None
    return PropAndMutants(
        prop, non_null
    )
    

async def generate_tree_mutants(
    work_dir: Path,
    state: PipelineState,
    url: str,
    props: list[CorpusEntry],
    report_sem: asyncio.Semaphore,
    gen: MutatorFactory
) -> list[PropAndMutants] | None:
    report_path = await ensure_report(work_dir, state, url, report_sem)
    if report_path is None:
        return None
    mutator = gen(report_path)
    mutants = await asyncio.gather(*[
        generate_property_mutant(
            mut=mutator, prop=prop
        ) for prop in props
    ])
    return [ m for m in mutants if m is not None ]


async def mutate_project(
    work_dir: Path,
    state: PipelineState,
    mutator_fact: MutatorFactory,
    report_sem: asyncio.Semaphore
):
    for (k, v) in state.analyzed_trees.items():
        await generate_tree_mutants(work_dir=work_dir, props=v, url=k, gen=mutator_fact, state=state, report_sem=report_sem)

async def try_mutate(
    work_dir: Path,
    pdf_dir: Path,
    mutation_llm: BaseChatModel,
    state: StateCache,
    agent_store: BaseStore,
    store_ns: tuple[str, ...],
    sem: asyncio.Semaphore
):
    report_sem = asyncio.Semaphore(10)
    jobs = []
    for i in pdf_dir.glob("*.pdf"):
        key = pdf_key(i)
        st = await state.load(key)
        if st is None:
            continue
        
        jobs.append(mutate_project(work_dir, st, sem, mutation_llm))
