from typing import Optional, cast, Any
import logging
import uuid
from dataclasses import dataclass
import pathlib

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph.state import CompiledStateGraph
from langgraph._internal._typing import StateLike
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore

from graphcore.graph import Builder
from graphcore.tools.memory import async_memory_tool
from graphcore.tools.vfs import VFSState

from composer.input.types import WorkflowOptions, InputData, ResumeFSData, ResumeIdData, ResumeInput, NativeFS
from composer.kb.knowledge_base import DefaultEmbedder, kb_tools as make_kb_tools, DEFAULT_KB_NS
from composer.workflow.factories import get_cryptostate_builder, get_vfs_tools, get_memory_ns
from composer.workflow.types import PromptParams, WorkflowSuccess, WorkflowFailure, WorkflowCrash, WorkflowResult
from composer.workflow.meta import create_resume_commentary
from composer.core.context import AIComposerContext, ProverOptions
from composer.prover.core import CloudConfig
from composer.core.validation import ValidationType, prover, reqs as req_type
from composer.rag.db import PostgreSQLRAGDatabase, rag_context, ComposerRAGDB
from composer.rag.models import get_model as get_rag_model
from composer.audit.store import AuditStore, AuditStoreSink, ResumeArtifact
from composer.audit.types import InputFileLike
from composer.natreq.extractor import get_requirements
from composer.natreq.judge import get_judge_tool
from composer.spec.cvl_research import CVL_RESEARCH_BASE_DOC, _build_research_tool
from composer.tools.relaxation import requirements_relaxation
from composer.tools.search import cvl_manual_tools
from composer.templates.loader import load_jinja_template
from composer.io.protocol import CodeGenIOHandler, WorkflowPurpose
from composer.io.context import with_handler, run_graph
from composer.ui.codegen_events import CodeGenEventHandler
from composer.core.state import AIComposerInput, AIComposerExtra
from composer.workflow.services import checkpointer_context, store_context, indexed_store_context, memory_backend_context, MemoryBackendGenerator


_KB_NS = DEFAULT_KB_NS

def get_reference_input(input_data: InputData, debug_prompt: Optional[str]) -> str:
    return load_jinja_template(
        "workflow_info.j2",
        spec_filename=input_data.spec.basename,
        interface_filename=input_data.intf.basename,
        system_doc_filename=input_data.system_doc.basename,
        debug_prompt=debug_prompt)

def _get_empty_extra() -> AIComposerExtra:
    return AIComposerExtra(
        validation={}, skipped_reqs=set(), working_spec=None
    )


def get_fresh_input(input: InputData, workflow_options: WorkflowOptions) -> AIComposerInput:
    return AIComposerInput(input=[
                input.intf.to_document_dict(),
                input.spec.to_document_dict(),
                input.system_doc.to_document_dict(),
                {
                    "type": "text",
                    "text": get_reference_input(input_data=input, debug_prompt=workflow_options.debug_prompt_override)
                }
            ], vfs={"rules.spec": input.spec.read()}, **_get_empty_extra())

@dataclass
class InputChangeDesc:
    orig_text: str
    updated_text: str

    single_form: str
    plural: str

    vfs_note: Optional[str]

def get_resume_prompt_common(
        art: ResumeArtifact,
        res: ResumeInput,
        updated_spec: str,
        other_changes: list[InputChangeDesc] | None = None
        ) -> list[str | dict]:
    changes = []
    if other_changes is not None:
        changes.extend(other_changes)

    if res.new_system is not None:
        changes.append(InputChangeDesc(
            orig_text=art.system_doc,
            updated_text=res.new_system.string_contents,
            plural="system documents",
            single_form="system document",
            vfs_note=None
        ))

    return [load_jinja_template(
        "resume_prompt.j2",
        commentary=art.commentary,
        spec_change_commentary=res.comments,
        orig_spec=art.spec_file,
        new_spec=updated_spec,
        other_changes=changes
    )]

def get_resume_id_input(input: ResumeIdData, resume_art: ResumeArtifact, workflow_options: WorkflowOptions) -> AIComposerInput:

    input_messages : list[str | dict] = get_resume_prompt_common(
        art=resume_art,
        res=input,
        updated_spec=input.new_spec.string_contents
    )
    if workflow_options.debug_prompt_override is not None:
        input_messages.append(workflow_options.debug_prompt_override)

    vfs_materialize = resume_art.vfs.to_dict()
    new_vfs = { k: v.decode("utf-8") for (k, v) in vfs_materialize.items() }
    new_vfs["rules.spec"] = input.new_spec.string_contents
    return AIComposerInput(
        input=input_messages,
        vfs=new_vfs,
        **_get_empty_extra()
    )

def get_resume_fs_input(input: ResumeFSData, resume_art: ResumeArtifact, workflow_options: WorkflowOptions) -> tuple[AIComposerInput, InputFileLike, InputFileLike]:
    path = pathlib.Path(input.file_path)

    spec_p = path / "rules.spec"
    if not spec_p.is_file():
        raise RuntimeError("Specification file is apparently missing")
    new_spec = spec_p.read_text()

    intf_p = path / resume_art.interface_path
    if not intf_p.is_file():
        raise RuntimeError("Interface file was moved or deleted")
    changes = []
    if (intf_text := intf_p.read_text()) != resume_art.interface_file:
        changes.append(InputChangeDesc(
            orig_text=resume_art.interface_file,
            updated_text=intf_text,
            single_form="interface",
            plural="interfaces",
            vfs_note=resume_art.interface_path
        ))
    input_messages = get_resume_prompt_common(
        art=resume_art,
        res=input,
        other_changes=changes,
        updated_spec=new_spec
    )
    input_messages.append("In addition to the explicit changes mentioned above, the contents of the VFS may have been arbitrarily changed since your last work. " \
    "Some of these changes may cause the current implementation to no longer compile. Thus, analyze the current implementation and consider what changes are necessary to " \
    "fix any compilation errors.")

    if workflow_options.debug_prompt_override is not None:
        input_messages.append(workflow_options.debug_prompt_override)

    return (AIComposerInput(input=input_messages, vfs={}, **_get_empty_extra()), NativeFS(intf_p), NativeFS(spec_p))


async def _execute_ai_composer_workflow(
    handler: CodeGenIOHandler,
    llm: BaseChatModel,
    input: InputData | ResumeFSData | ResumeIdData,
    workflow_options: WorkflowOptions,
    memory_namespace: str | None,
    resume_work_key: str | None,

    checkpointer: AsyncPostgresSaver,
    store: BaseStore,
    indexed_store: BaseStore,
    rag: ComposerRAGDB,
    mem: MemoryBackendGenerator
) -> WorkflowResult:
    logger = logging.getLogger(__name__)

    thread_id = workflow_options.thread_id

    if thread_id is None:
        thread_id = "crypto_session_" + str(uuid.uuid1())
        await handler.log_workflow_thread(WorkflowPurpose.CODEGEN, thread_id)
        logger.info(f"Selected thread id: {thread_id}")

    mem_root = memory_namespace or thread_id

    prompt_params: PromptParams
    fs_layer: str | None = None
    flow_input: AIComposerInput

    system_doc: InputFileLike
    interface_file: InputFileLike
    spec_file: InputFileLike
    resume_art : None | ResumeArtifact = None

    audit_store = AuditStore(store)

    match input:
        case InputData():
            prompt_params = PromptParams(is_resume=False)
            flow_input = get_fresh_input(input, workflow_options)
            system_doc = input.system_doc
            interface_file = input.intf
            spec_file = input.spec

        case ResumeIdData() | ResumeFSData():
            prompt_params = PromptParams(is_resume=True)

            resume_art = await audit_store.get_resume_artifact(thread_id=input.thread_id)
            if input.new_system is None:
                system_doc = resume_art.system_vfs_handle
            else:
                system_doc = input.new_system
            match input:
                case ResumeFSData():
                    (flow_input, interface_file, spec_file) = get_resume_fs_input(input, resume_art, workflow_options)
                    fs_layer = input.file_path
                case ResumeIdData():
                    interface_file = resume_art.intf_vfs_handle
                    flow_input = get_resume_id_input(input, resume_art, workflow_options)
                    spec_file = input.new_spec

    req_memories = mem(
        get_memory_ns(mem_root, "natreq")
    )

    extra_reqs = await store.aget((thread_id,), "requirements")
    reqs_list : list[str] | None
    if extra_reqs is None:
        if workflow_options.skip_reqs:
            reqs_list = None
        elif workflow_options.set_reqs is not None:
            if workflow_options.set_reqs.startswith("@"):
                other_reqs = await store.aget((workflow_options.set_reqs[1:],), "requirements")
                assert other_reqs is not None
                reqs_list = other_reqs.value["reqs"]
            else:
                reqs_list = [ v for l in pathlib.Path(workflow_options.set_reqs).read_text().splitlines() if (v := l.strip()) ]
        else:
            print("Analyzing requirements...")
            extraction = await get_requirements(
                handler,
                workflow_options,
                llm,
                system_doc,
                spec_file,
                req_memories,
                resume_art,
                workflow_options.requirements_oracle
            )
            reqs_list = extraction.reqs
            await handler.log_workflow_thread(WorkflowPurpose.NATREQ, extraction.thread_id)
        await store.aput((thread_id,), "requirements", {"reqs": reqs_list})
    else:
        print("Read requirements from store")
        reqs_list = extra_reqs.value["reqs"]
    extra_tools = []

    if reqs_list is not None:
        judge_tool = get_judge_tool(
            reqs=reqs_list,
            mem=req_memories,
            unbound=llm,
            vfs_tools=get_vfs_tools(
                fs_layer=fs_layer, immutable=True
            )[0]
        )
        extra_tools.append(judge_tool)
        extra_tools.append(requirements_relaxation)

    if "context-management-2025-06-27" in getattr(llm, "betas"):
        memory = async_memory_tool(mem(get_memory_ns(mem_root, "composer")))
        extra_tools.append(memory)

    # CVL research sub-agent — KB needs indexed store for semantic search
    cvl_builder = Builder().with_llm(
        llm
    ).with_loader(
        load_jinja_template
    ).with_tools(
        cvl_manual_tools(rag)
    ).with_checkpointer(
        checkpointer
    ).with_tools(
        make_kb_tools(indexed_store, _KB_NS, read_only=True)
    )

    research_doc = CVL_RESEARCH_BASE_DOC + " Do NOT use this for source code questions — use the VFS tools for that."
    async def runner[S: StateLike, I: StateLike](
        graph: CompiledStateGraph[S, Any, I, Any],
        i: I,
    ) -> S:
        return await run_graph(
            ctxt=None,
            description="CVL Researcher",
            graph=graph,
            input=i,
            run_conf={
                "recursion_limit": 100,
                "configurable": {
                    "thread_id": "research-" + uuid.uuid4().hex[:16]
                }
            }
        )
    extra_tools.append(_build_research_tool(cvl_builder, runner, research_doc))

    (workflow_builder, materializer) = get_cryptostate_builder(
        llm=llm,
        fs_layer=fs_layer,
        summarization_threshold=workflow_options.summarization_threshold,
    )

    workflow_graph = workflow_builder.with_tools(extra_tools).with_sys_prompt_template(
        "system_prompt.j2"
    ).with_initial_prompt_template(
        "synthesis_prompt.j2", **prompt_params
    ).build_async()[0]

    await audit_store.register_run(
        thread_id=thread_id,
        system_doc=system_doc,
        interface_file=interface_file,
        spec_file=spec_file,
        vfs_init=materializer.iterate(flow_input),
        reqs=reqs_list
    )

    workflow_exec = workflow_graph.compile(checkpointer=checkpointer, store=store)
    if reqs_list is not None:
        flow_input["input"].append(f"""
    Additionally, the implementation MUST satisfy the following requirements:
    {"\n".join(f"{i}. {r}" for (i, r) in enumerate(reqs_list, start = 1))}
    """)

    if resume_work_key is not None:
        snapshot = await store.aget(("crash_recovery",), resume_work_key)
        if snapshot is not None:
            vfs_files = list(snapshot.value["vfs"].items())
            recovery_msg = load_jinja_template("crash_recovery_context.j2", vfs_files=vfs_files)
            flow_input["input"].insert(0, recovery_msg)

    try:
        import grandalf # type: ignore
        layout = workflow_exec.get_graph().draw_ascii()
        logger.debug(f"\n{layout}")
    except ModuleNotFoundError:
        pass

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    config["recursion_limit"] = workflow_options.recursion_limit

    if workflow_options.checkpoint_id is not None:
        config["configurable"]["checkpoint_id"] = workflow_options.checkpoint_id

    prover_opts: ProverOptions = ProverOptions(
        capture_output=workflow_options.prover_capture_output,
        keep_folder=workflow_options.prover_keep_folders,
        cloud=None if workflow_options.local_prover else CloudConfig(),
    )

    required_validations : list[ValidationType] = [prover]
    if reqs_list is not None:
        required_validations.append(req_type)

    work_context = AIComposerContext(llm=llm, rag_db=rag, prover_opts=prover_opts, vfs_materializer=materializer, required_validations=required_validations)

    audit_sink = AuditStoreSink(audit_store, thread_id)

    try:
        async with with_handler(handler, CodeGenEventHandler(handler, audit_sink)):
            final_state = await run_graph(workflow_exec, work_context, flow_input, config, description="Code generation")

        result = final_state.get("generated_code", None)
        if result is None:
            return WorkflowFailure()

        res_commentary = await create_resume_commentary(final_state, llm=llm)
        await audit_store.register_complete(
            thread_id, materializer.iterate(final_state), res_commentary.interface_path, res_commentary.commentary
        )

        await handler.output(result, materializer, final_state)
        return WorkflowSuccess()
    except Exception as exc:
        await handler.show_error(exc)
        # Attempt to capture VFS from last checkpoint
        resume_key: str | None = None
        try:
            ct = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
            if ct is not None:
                channel_values = cast(VFSState, ct.checkpoint["channel_values"])
                vfs_snapshot = {path: content.decode("utf-8") for path, content in materializer.iterate(channel_values)}
                resume_key = f"crash_{thread_id}_{uuid.uuid4().hex[:8]}"
                await store.aput(("crash_recovery",), resume_key, {"vfs": vfs_snapshot})
                logger.info(f"Saved crash recovery snapshot: {resume_key}")
        except Exception as snapshot_exc:
            logger.warning(f"Failed to capture crash snapshot: {snapshot_exc}")
        return WorkflowCrash(resume_work_key=resume_key, error=exc)


async def execute_ai_composer_workflow(
    handler: CodeGenIOHandler,
    llm: BaseChatModel,
    input: InputData | ResumeFSData | ResumeIdData,
    workflow_options: WorkflowOptions,
    memory_namespace: str | None = None,
    resume_work_key: str | None = None,
) -> WorkflowResult:
    """Execute the AI Composer workflow with interrupt handling."""
    model = get_rag_model()
    async with checkpointer_context() as checkpointer, \
        store_context() as store, \
        indexed_store_context(DefaultEmbedder(model)) as indexed_store, \
        rag_context(workflow_options.rag_db, model) as rag_db, \
        memory_backend_context() as mem:
        return await _execute_ai_composer_workflow(
            handler, llm, input, workflow_options, memory_namespace, resume_work_key, checkpointer,
            store, indexed_store, rag_db, mem
        )
