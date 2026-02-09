"""
Source-based spec generation.

Generate CVL specs for existing smart contracts using PreAudit for
compilation analysis and verification.
"""

import argparse
import hashlib
import sys

import composer.certora as _

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, NotRequired, TypeVar, Protocol, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel

from composer.input.types import ModelOptions, RAGDBOptions, LangraphOptions, OptionalArg
from composer.input.parsing import add_protocol_args
from composer.rag.db import PostgreSQLRAGDatabase
from graphcore.graph import MessagesState, Builder
from graphcore.tools.vfs import VFSState, VFSToolConfig, fs_tools, VFSAccessor

import uuid
from typing import cast

from langchain_core.runnables import RunnableConfig

from composer.spec.context import WorkspaceContext, JobSpec
from composer.spec.harness import setup_and_harness_agent
from composer.spec.struct_invariant import structural_invariants_flow

from composer.spec.preaudit_setup import run_preaudit_setup, SetupFailure
from composer.spec.cvl_tools import put_cvl_raw, put_cvl
from composer.tools.search import cvl_manual_search
from composer.human.handlers import handle_human_interrupt
from graphcore.tools.results import result_tool_generator
from composer.workflow.services import create_llm, get_checkpointer, get_memory, get_store, get_indexed_store
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.templates.loader import load_jinja_template
from graphcore.graph import build_workflow, FlowInput
from graphcore.tools.vfs import vfs_tools
from graphcore.tools.memory import memory_tool
from graphcore.summary import SummaryConfig

from composer.kb.knowledge_base import DefaultEmbedder, kb_tools

from composer.spec.prop import PropertyFormulation
from composer.spec.component import ApplicationComponent, ComponentInst
from composer.spec.bug import run_bug_analysis

from composer.spec.cvl_generation import generate_property_cvl

T = TypeVar('T')

R = TypeVar('R')

@dataclass
class SourceSpecContext:
    """Context for source-based spec generation."""
    project_root: Path
    main_contract: str
    main_contract_path: str
    compilation_config: dict
    summaries_import: str | None
    rag_db: PostgreSQLRAGDatabase
    unbound_llm: BaseChatModel

class SourceSpecInput(FlowInput):
    vfs: dict
    curr_spec: None

class SourceSpecState(MessagesState, VFSState):
    """State for source-based spec generation workflow."""
    curr_spec: str | None
    result: NotRequired[dict]


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

class GetCVLSchema(BaseModel):
    """
    View the (pretty-printed) version of the CVL file.
    """
    state: Annotated[SourceSpecState, InjectedState]

@tool(args_schema=GetCVLSchema)
def get_cvl(
    state: Annotated[SourceSpecState, InjectedState]
) -> str:
    if state["curr_spec"] is None:
        return "No spec file on VFS"
    return state["curr_spec"]


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
) -> str:
    """Generate a cache key for bug analysis based on inputs."""
    components = [
        component.model_dump_json(),
        summ,
    ]
    combined = "|".join(str(c) for c in components)
    return _string_hash(combined)

# Common forbidden read pattern for source analysis
FS_FORBIDDEN_READ = "(^lib/.*)|(^\\.certora_internal.*)|(^\\.git.*)|(^test/.*)|(^emv-.*)|(.*\\.json$)"

def format_container(d: dict) -> str:
    c = d.get("containingContract", None)
    if c is None:
        return "at the top level"
    else:
        return f"in contract {c}"

def format_type(s: dict) -> str | None:
    kind = s.get("typeCategory", None)
    if not kind:
        return None
    where_def = format_container(s)
    ty_name = s.get("typeName", None)
    if not ty_name:
        return None
    qual_name = s.get("qualifiedName", None)
    match kind:
        case "UserDefinedStruct":
            return f"A struct {ty_name} {where_def}: use `{qual_name}`"
        case "UserDefinedEnum":
            return f"An enum {ty_name} {where_def}: use `{qual_name}`"
        case "UserDefinedValueType":
            base = s.get("baseType", None)
            if not base:
                return None
            return f"An alias for {base} called {ty_name} {where_def}: use `{qual_name}`"
        case _:
            return None

def format_types(udts: list[dict]) -> str:
    to_format: list[str] = []
    for ty in udts:
        r = format_type(ty)
        if not r:
            continue
        to_format.append(r)
    return "\n".join(to_format)

type PropertyFormalization = tuple[PropertyFormulation, str, str]

def _analyze_component(
    ctx: WorkspaceContext,
    feat: ComponentInst,
    b: Builder[None, None, FlowInput],
    cvl_builder: Builder[None, None, FlowInput],
) -> None | list[tuple[PropertyFormulation, str, str]]:
    cache_key = _cache_key_bug_analysis(feat.component, feat.summ.application_type)
    feat_ctx = ctx.child(
        cache_key,
        {
            "component": feat.component.model_dump(),
            "app_type": feat.summ.application_type
        }
    )
    res = run_bug_analysis(feat_ctx, feat, b)
    if res is None:
        print("Didn't work")
        return None
    work : list[tuple[PropertyFormulation, str, str]] = []
    for prop in res:
        print(prop)
        prop_model = prop.model_dump()
        prop_key = _string_hash(prop.model_dump_json())
        prop_ctx = feat_ctx.child(prop_key, prop_model)
        r, cvl = generate_property_cvl(
            prop_ctx, feat, prop, cvl_builder
        )
        print(cvl)
        print(r)
        work.append((prop, r, cvl))
    return work


def execute(args: SourceSpecArgs) -> int:
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
            "relative_path": spec.project_root,
            "system_doc": spec.system_doc,
            "main_contract": spec.contract_name,
            "ts": time.time()
        })

    ctx = WorkspaceContext.create(
        services=host,
        js=spec,
        thread_id=thread_id,
        store=store,
        memory_namespace=args.memory_ns,
        cache_namespace=cache_ns,
    )

    llm = create_llm(args)

    basic_builder = Builder().with_llm(llm).with_loader(load_jinja_template)

    d = setup_and_harness_agent(
        ctx, basic_builder, ignore_existing_config=args.ignore_existing_config
    )

    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db,
        model=get_model(),
        skip_test=True
    )

    cvl_builder = basic_builder.with_tools(
        [cvl_manual_search(rag_db), *fs_tools(args.project_root, forbidden_read=FS_FORBIDDEN_READ)]
    )

    structural_invariants_flow(
        llm, ctx, d, basic_builder, cvl_builder
    )

    sys.exit(1)

    b : Builder[None, None, FlowInput] = Builder().with_llm(
        llm
    ).with_input(
        FlowInput
    ).with_tools(
        fs_tools(args.project_root, forbidden_read=FS_FORBIDDEN_READ)
    )
    
    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db,
        model=get_model(),
        skip_test=True
    )

    analysis = run_source_analysis(args, thread_id, spec, store, b)

    cvl_builder = b.with_tools(
        [cvl_manual_search(rag_db)]
    )

    if analysis is None:
        print("Oh well")
        return 1


    work : list[tuple[ComponentInst, None | list[PropertyFormalization]]] = []
    for feature_idx in range(0, len(analysis.components)):
        def thunk() -> tuple[ComponentInst, None | list[PropertyFormalization]]:
            feat = ComponentInst(
                contract_name=spec.contract_name,
                relative_path=spec.relative_path,
                summ=analysis,
                ind=feature_idx
            )
            l = _analyze_component(
                args, feat, b, cvl_builder, store
            )
            return (feat, l)
        work.append(thunk())



    sys.exit(1)

    # Step 1: Run PreAudit setup
    print("Running PreAudit compilation analysis...")
    setup_result = run_preaudit_setup(
        project_root=Path(args.project_root),
        main_contract=main_contract_name,
        relative_path=str(relativized_main)
    )
    match setup_result:
        case SetupFailure(error=e):
            print(f"Auto setup failed: {e}")
            return 1
        case _:
            pass

    # Step 2: Create LLM and thread
    llm = create_llm(args)

    # Step 3: Build context
    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db,
        model=get_model(),
        skip_test=True
    )

    summaries_import = f'import "{setup_result.summaries_path}";'

    context = SourceSpecContext(
        project_root=Path(args.project_root),
        main_contract=main_contract_name,
        main_contract_path=main_contract_path,
        compilation_config=setup_result.config,
        summaries_import=summaries_import,
        rag_db=rag_db,
        unbound_llm=llm
    )

    # Step 4: Build workflow
    manual = cvl_manual_search(SourceSpecContext)
    checkpointer = get_checkpointer()

    memory = memory_tool(get_memory(f"memory-{thread_id}"))
    generation_complete = result_tool_generator(
        outkey="result",
        result_schema=(dict, "Final result"),
        doc="Used to indicate successful result of your analysis."
    )

    v_tools, _ = vfs_tools(
        conf=VFSToolConfig(immutable=True, fs_layer=args.project_root, forbidden_read="(^lib/.*)|(^\\.certora_internal.*)|(^\\.git.*)|(^test/.*)|(^emv-.*)"),
        ty=SourceSpecState
    )

    graph = build_workflow(
        context_schema=SourceSpecContext,
        initial_prompt=load_jinja_template(
            "source_spec_prompt.j2",
            main_contract=args.main_contract,
            summaries_import=summaries_import
        ),
        sys_prompt=load_jinja_template("cvl_system_prompt.j2"),
        input_type=SourceSpecInput,
        output_key="result",
        state_class=SourceSpecState,
    
        unbound_llm=llm,
        summary_config=SummaryConfig(max_messages=50),
        tools_list=[
            # VFS tools for source code access
            *v_tools,

            # CVL spec tools
            put_cvl,
            put_cvl_raw,
            get_cvl,

            # Verification
            verify_spec,

            # Support tools
            manual,
            memory,
            generation_complete
        ]
    )[0].compile(checkpointer=checkpointer, store=get_store())

    # Step 5: Run workflow
    def fresh_config() -> RunnableConfig:
        return {
            "recursion_limit": args.recursion_limit,
            "configurable": {
                "thread_id": thread_id
            }
        }

    runnable_conf = fresh_config()

    sys_doc = get_system_doc(Path(args.system_doc))
    assert sys_doc is not None

    if args.checkpoint_id is not None:
        runnable_conf["configurable"]["checkpoint_id"] = args.checkpoint_id #type: ignore

    formatted_types = format_types(setup_result.user_types)

    graph_input = SourceSpecInput(input=[
        "The major components of the software is as follows",
        format_summary_xml(analysis),
        "The system document is as follows",
        sys_doc,
        "User defined types can be referenced in your specification, according to the following guidance",
        formatted_types
        ], curr_spec=None, vfs={}) if args.checkpoint_id is None else None

    while True:
        t = graph_input
        graph_input = None
        for (tag, payload) in graph.stream(
            input=t,
            config=runnable_conf,
            context=context,
            stream_mode=["updates", "checkpoints"]
        ):
            assert isinstance(payload, dict)
            if tag == "checkpoints":
                print("current checkpoint: " + payload["config"]["configurable"]["checkpoint_id"])
                continue
            if "__interrupt__" in payload:
                if "configurable" in runnable_conf and "checkpoint_id" in runnable_conf["configurable"]:
                    del runnable_conf["configurable"]["checkpoint_id"]
                interrupt_data = cast(dict, payload["__interrupt__"][0].value)
                def debug_thunk() -> None:
                    pass  # TODO: implement debug console if needed
                human_response = handle_human_interrupt(interrupt_data, debug_thunk)
                graph_input = Command(resume=human_response)
                break
            else:
                print(payload)
        if graph_input is None:
            break

    final_state = cast(SourceSpecState, graph.get_state(fresh_config()).values)
    if "result" not in final_state:
        return 1

    print("Spec file generation complete")
    print(final_state["result"])

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

    res = cast(SourceSpecArgs, parser.parse_args())

    return execute(res)
