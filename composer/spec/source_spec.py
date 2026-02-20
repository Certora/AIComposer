"""
Source-based spec generation.

Generate CVL specs for existing smart contracts using PreAudit for
compilation analysis and verification.
"""

import argparse
import asyncio
import hashlib

import composer.certora as _

from pathlib import Path
from typing import Annotated, TypeVar, Protocol, Optional

from langchain_core.tools import BaseTool

from composer.input.types import ModelOptions, RAGDBOptions, LangraphOptions, OptionalArg
from composer.input.parsing import add_protocol_args
from composer.rag.db import PostgreSQLRAGDatabase
from graphcore.graph import Builder, LLM
from graphcore.tools.vfs import VFSState, VFSToolConfig, fs_tools, VFSAccessor

import uuid
from typing import cast


from composer.spec.context import WorkspaceContext, JobSpec, Builders, SourceBuilder, CVLBuilder, CacheKey, Properties, ComponentGroup, CVLGeneration
from composer.spec.harness import setup_and_harness_agent
from composer.spec.struct_invariant import structural_invariants_flow

from composer.tools.search import cvl_manual_tools
from composer.workflow.services import create_llm, get_store, get_indexed_store, get_checkpointer
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.templates.loader import load_jinja_template
from graphcore.graph import FlowInput
from graphcore.tools.vfs import vfs_tools

from composer.kb.knowledge_base import DefaultEmbedder, kb_tools

from composer.spec.prop import PropertyFormulation
from composer.spec.component import ApplicationComponent, ComponentInst
from composer.spec.bug import run_bug_analysis
from composer.spec.component import run_component_analysis
from composer.spec.cvl_generation import generate_property_cvl, CVLResource, ProverContext, GeneratedCVL
from composer.spec.prover import get_prover_tool, WithCVL
from composer.spec.summarizer import setup_summaries
from composer.spec.trunner import buffer_collection, fresh_buffer
from composer.spec.job_manager import JobManagerApp

T = TypeVar('T')

R = TypeVar('R')

class StateOptions(Protocol):
    memory_ns: Annotated[Optional[str], OptionalArg(
        help="The namespace to use for memory (default: thread id)"
    )]
    cache_ns: Annotated[Optional[str], OptionalArg(
        help="The namespace to use for caching (default: no caching)"
    )]

class SourceSpecArgs(ModelOptions, RAGDBOptions, LangraphOptions, StateOptions):
    """Arguments for source-based spec generation."""
    project_root: str
    main_contract: str
    system_doc: str
    ignore_existing_config: bool
    cloud: bool
    max_parallel: int

def _hash_file(path: Path) -> str:
    """Return SHA256 hash of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root_cache(js: JobSpec) -> str:
    """Generate a cache key for source analysis based on inputs."""
    components = [
        js.project_root,
        _hash_file(Path(js.system_doc)),
        js.relative_path,
        js.contract_name,
    ]
    combined = "|".join(str(c) for c in components)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]

def _string_hash(
    to_hash: str
) -> str:
    return hashlib.sha256(to_hash.encode()).hexdigest()[:16]

def _cache_key_bug_analysis(
    component: "ApplicationComponent",
    summ: str
) -> CacheKey[Properties, ComponentGroup]:
    """Generate a cache key for bug analysis based on inputs."""
    components = [
        component.model_dump_json(),
        summ,
    ]
    combined = "|".join(str(c) for c in components)
    return CacheKey(_string_hash(combined))

# Common forbidden read pattern for source analysis
FS_FORBIDDEN_READ = "(^lib/.*)|(^\\.certora_internal.*)|(^\\.git.*)|(^test/.*)|(^emv-.*)|(.*\\.json$)"

PROPERTIES_KEY = CacheKey[None, Properties]("properties")

type PropertyFormalization = tuple[PropertyFormulation, str, str]

def _property_cache_key(prop: PropertyFormulation) -> CacheKey[ComponentGroup, GeneratedCVL]:
    return CacheKey(_string_hash(prop.model_dump_json()))

async def _analyze_component(
    ctx: WorkspaceContext[Properties],
    conf: ProverContext,
    feat: ComponentInst,
    builders: Builders,
    workflow_sem: asyncio.Semaphore,
) -> None | list[tuple[PropertyFormulation, str, str]]:
    name = feat.component.name
    feat_ctx = ctx.child(
        _cache_key_bug_analysis(feat.component, feat.summ.application_type),
        {
            "component": feat.component.model_dump(),
            "app_type": feat.summ.application_type
        }
    )

    async with workflow_sem:
        with fresh_buffer(name=f"{name}: analysis"):
            res = await run_bug_analysis(feat_ctx, feat, builders.source)
    if res is None:
        print("Didn't work")
        return None

    async def _verify_property(prop: PropertyFormulation) -> tuple[PropertyFormulation, str, str]:
        prop_ctx = feat_ctx.child(_property_cache_key(prop), prop.model_dump())
        m = prop_ctx.cache_get(GeneratedCVL)
        if m is not None:
            r = m.commentary
            cvl = m.cvl
        else:
            prop_label = prop.description[:50]
            async with workflow_sem:
                with fresh_buffer(name=f"{name}: {prop_label}"):
                    d = await generate_property_cvl(
                        ctx=prop_ctx.abstract(CVLGeneration), builders=builders, prover_setup=conf, feat=feat, prop=prop, with_memory=False
                    )
            prop_ctx.cache_put(d)
            cvl = d.cvl
            r = d.commentary
        print(cvl)
        print(r)
        return (prop, r, cvl)

    results = await asyncio.gather(*[_verify_property(prop) for prop in res])
    return list(results)


async def execute(args: SourceSpecArgs) -> int:
    """Execute source-based spec generation workflow."""

    thread_id = args.thread_id if args.thread_id else f"source_spec_{uuid.uuid4().hex}"
    print(f"Thread ID: {thread_id}")

    project_root = Path(args.project_root)

    main_contract_path, main_contract_name = args.main_contract.split(":", 1)

    full_contract_path = Path(main_contract_path).resolve()

    if not full_contract_path.is_relative_to(project_root.resolve()):
        print(f"Invalid path: {full_contract_path} doesn't appear in project root {project_root}")
        return 1

    relativized_main = full_contract_path.relative_to(project_root.resolve())

    model = get_model()

    indexed_store = get_indexed_store(DefaultEmbedder(model))

    prover_sem = asyncio.Semaphore(args.max_parallel if args.cloud else 1)
    _checkpointer = get_checkpointer()

    class SVCHost():
        def kb_tools(self, read_only: bool) -> list[BaseTool]:
            return kb_tools(
                store=indexed_store,
                kb_ns=("cvl",),
                read_only=read_only
            )

        def fs_tools(self) -> list[BaseTool]:
            return fs_tools(args.project_root, forbidden_read=FS_FORBIDDEN_READ)

        def vfs_tools[S: VFSState](
                self,
                ty: type[S],
                forbidden_write: str | None = None,
                put_doc_extra: str | None = None) -> tuple[list[BaseTool], VFSAccessor[S]]:
            tool_conf : VFSToolConfig = VFSToolConfig(
                fs_layer=args.project_root,
                forbidden_read=FS_FORBIDDEN_READ,
                immutable=False
            )
            if forbidden_write:
                tool_conf["forbidden_write"] = forbidden_write
            if put_doc_extra:
                tool_conf["put_doc_extra"] = put_doc_extra
            return vfs_tools(tool_conf, ty)

        def llm(self) -> LLM:
            return llm

        @property
        def checkpointer(self):
            return _checkpointer

        def prover_tool[T: WithCVL](self, ty: type[T], config: dict) -> BaseTool:
            return get_prover_tool(
                llm, ty, config, main_contract_name, args.project_root,
                cloud=args.cloud, semaphore=prover_sem,
            )

    host = SVCHost()

    spec = JobSpec(
        project_root=args.project_root,
        system_doc=args.system_doc,
        relative_path=str(relativized_main),
        contract_name=main_contract_name,
    )

    store = get_store()

    cache_ns : tuple[str, ...] | None = None
    if (ns := args.cache_ns) is not None:
        import time
        cache_ns = (ns, _root_cache(spec))
        cache_key = (ns, _root_cache(spec), uuid.uuid4().hex)
        print(f"Job cache: {cache_key}")
        store.put(cache_key, "job_info", {
            "root": args.project_root,
            "relative_path": spec.relative_path,
            "system_doc": spec.system_doc,
            "main_contract": spec.contract_name,
            "ts": time.time()
        })

    ctx : WorkspaceContext[None] = WorkspaceContext.create(
        services=host,
        js=spec,
        thread_id=thread_id,
        store=store,
        memory_namespace=args.memory_ns,
        cache_namespace=cache_ns,
    )

    llm = create_llm(args)

    basic_builder: Builder[None, None, None] = Builder().with_llm(llm).with_loader(load_jinja_template)

    d = await setup_and_harness_agent(
        ctx, basic_builder, ignore_existing_config=args.ignore_existing_config
    )

    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db,
        model=get_model(),
        skip_test=True
    )

    resources : list[CVLResource] = [
        CVLResource(
            required=True,
            import_path=d.summaries_path,
            description="Summaries to simplify the formal verification.",
            sort="import"
        )
    ]

    cvl_manual = cvl_manual_tools(rag_db)

    cvl_builder: CVLBuilder = basic_builder.with_tools(
        [*cvl_manual, *host.fs_tools()]
    ).with_input(FlowInput)

    source_builder: SourceBuilder = basic_builder.with_tools(
        host.fs_tools()
    ).with_input(FlowInput)

    builders = Builders(
        source=source_builder,
        cvl=cvl_builder,
        cvl_only=basic_builder.with_tools(cvl_manual).with_input(FlowInput),
    )

    custom_summaries = await setup_summaries(
        ctx, d, cvl_builder
    )

    resources.append(custom_summaries)

    invariants = await structural_invariants_flow(
        ctx, ProverContext(d.config, resources), basic_builder, builders
    )

    resources.extend(invariants)

    prover_context = ProverContext(d.config, resources)

    analysis = await run_component_analysis(ctx, source_builder)

    if analysis is None:
        print("It didn't work :(")
        return 1

    prop_context = ctx.child(PROPERTIES_KEY)

    workflow_sem = asyncio.Semaphore(10)

    # -- parallel execution with job manager TUI ----------------------------
    with buffer_collection() as buffers:
        app = JobManagerApp(buffers=buffers)

        async def _run_all_jobs() -> None:
            await asyncio.gather(*[
                _analyze_component(
                    prop_context, prover_context,
                    ComponentInst(analysis, i), builders,
                    workflow_sem
                )
                for i in range(len(analysis.components))
            ])
            # All jobs done — give user a moment then exit TUI
            app.set_timer(5, lambda: app.exit())

        # Run TUI and jobs concurrently
        job_task = asyncio.create_task(_run_all_jobs())
        await app.run_async()
        # If TUI exits before jobs finish (user quit), cancel them
        if not job_task.done():
            job_task.cancel()
        elif (exc := job_task.exception()) is not None:
            raise exc

    return 0

def auto_prover() -> int:
    parser = argparse.ArgumentParser()
    add_protocol_args(parser, ModelOptions)
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, LangraphOptions)
    add_protocol_args(parser, StateOptions)
    parser.add_argument("project_root")
    parser.add_argument("main_contract")
    parser.add_argument("system_doc")
    parser.add_argument("--ignore-existing-config", action="store_true", dest="ignore_existing_config",
                        help="Proceed even if certora/ directory already exists in project root")
    parser.add_argument("--cloud", action="store_true", default=False,
                        help="Run verification in the Certora cloud")
    parser.add_argument("--max-parallel", type=int, default=4, dest="max_parallel",
                        help="Max parallel cloud verification jobs (default: 4)")

    res = cast(SourceSpecArgs, parser.parse_args())

    return asyncio.run(execute(res))
