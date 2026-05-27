from typing import Optional, Protocol, Any
import logging
import shlex
import uuid
from dataclasses import dataclass
import pathlib

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from langgraph.graph.state import CompiledStateGraph
from langgraph._internal._typing import StateLike

from graphcore.graph import Builder

from composer.input.types import (
    WorkflowOptions, ModelOptionsBase, InputData, ResumeFSData, ResumeIdData,
    ResumeInput, NativeFS, TextNativeFS
)
from composer.input.files import TextUploadable, Uploadable
from composer.input.files import Document, TextDocument
from composer.kb.knowledge_base import DefaultEmbedder, kb_tools as make_kb_tools
from composer.workflow.factories import get_vfs_tools, get_memory_ns
from composer.workflow.recovery import recover_vfs, recovery_from_thread
from composer.workflow.services import IndexedConnections, standard_connections
from composer.workflow.summarization import SummaryGeneration
from composer.workflow.types import (
    PromptParams, WorkflowSuccess, WorkflowFailure, WorkflowCrash, WorkflowResult,
)
from composer.workflow.meta import create_resume_commentary
from composer.core.context import AIComposerContext, ProverOptions
from composer.prover.core import make_prover_options
from composer.core.state import AIComposerInput, AIComposerExtra, AIComposerState
from composer.core.validation import ValidationType, prover, reqs as req_type
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model as get_rag_model
from composer.audit.store import AuditStore, AuditStoreSink, ResumeArtifact
from composer.natreq.extractor import get_requirements
from composer.natreq.judge import get_judge_tool
from composer.spec.cvl_research import CVL_RESEARCH_BASE_DOC, _build_research_tool
from composer.tools.relaxation import requirements_relaxation
from composer.tools.search import cvl_manual_search, cvl_manual_tools
from composer.templates.loader import load_jinja_template
from composer.io.protocol import CodeGenIOHandler, WorkflowPurpose
from composer.io.context import with_handler, run_graph
from composer.ui.codegen_events import CodeGenEventHandler
from composer.ui.tool_display import async_tool_context


_KB_NS = ("cvl",)


# Codegen's historical convention: the spec lives at ``rules.spec``
# in the VFS. Used by ``get_fresh_input`` (which seeds the VFS) and
# ``register_run`` (which records the spec's vfs_path). On resume,
# the audit-stored ``ResumeSpecEntry.vfs_path`` is the source of
# truth instead.
_SPEC_VFS_PATH = "rules.spec"


class _ExecutorOptions(WorkflowOptions, ModelOptionsBase, Protocol):
    """Combined runtime options consumed by the executor: workflow
    config + model identification. Callers already pass dataclasses
    that satisfy both protocols."""


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
                input.intf.to_dict(),
                input.spec.to_dict(),
                input.system_doc.to_dict(),
                {
                    "type": "text",
                    "text": get_reference_input(input_data=input, debug_prompt=workflow_options.debug_prompt_override)
                }
            ], vfs={_SPEC_VFS_PATH: input.spec.string_contents}, **_get_empty_extra())


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
    """Build the resume-prompt content blocks. Works against the
    audit-restored ``Uploadable`` handles directly — nothing is
    rehydrated through the uploader here, since the prompt only needs
    text bodies (``string_contents``) and a basename for the binary
    fallback."""
    changes = []
    if other_changes is not None:
        changes.extend(other_changes)

    if res.new_system is not None:
        prior_system = art.system_doc
        new_system_text = res.new_system.string_contents
        changes.append(InputChangeDesc(
            orig_text=prior_system if prior_system is not None else f"[prior {art.system_vfs_handle.basename}]",
            updated_text=new_system_text if new_system_text is not None else f"[updated {res.new_system.basename}]",
            plural="system documents",
            single_form="system document",
            vfs_note=None,
        ))

    return [load_jinja_template(
        "resume_prompt.j2",
        commentary=art.commentary,
        spec_change_commentary=res.comments,
        orig_spec=art.spec.contents,
        new_spec=updated_spec,
        other_changes=changes
    )]


def get_resume_id_input(input: ResumeIdData, resume_art: ResumeArtifact, workflow_options: WorkflowOptions) -> AIComposerInput:
    new_spec_text = input.new_spec.string_contents

    input_messages : list[str | dict] = get_resume_prompt_common(
        art=resume_art,
        res=input,
        updated_spec=new_spec_text,
    )
    if workflow_options.debug_prompt_override is not None:
        input_messages.append(workflow_options.debug_prompt_override)

    vfs_materialize = resume_art.vfs.to_dict()
    new_vfs = { k: v.decode("utf-8") for (k, v) in vfs_materialize.items() }
    new_vfs[resume_art.spec.vfs_path] = new_spec_text
    return AIComposerInput(
        input=input_messages,
        vfs=new_vfs,
        **_get_empty_extra()
    )


def get_resume_fs_input(input: ResumeFSData, resume_art: ResumeArtifact, workflow_options: WorkflowOptions) -> tuple[AIComposerInput, TextUploadable, TextUploadable]:
    path = pathlib.Path(input.file_path)

    spec_p = path / resume_art.spec.vfs_path
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

    return (AIComposerInput(input=input_messages, vfs={}, **_get_empty_extra()), TextNativeFS(intf_p), TextNativeFS(spec_p))


# ---------------------------------------------------------------------------
# Inner executor — runs against pre-opened resources
# ---------------------------------------------------------------------------


async def _execute_ai_composer_workflow(
    handler: CodeGenIOHandler,
    input: InputData | ResumeFSData | ResumeIdData,
    workflow_options: _ExecutorOptions,
    memory_namespace: str | None,
    resume_work_key: str | None,
    resources: IndexedConnections,
    rag_db: PostgreSQLRAGDatabase,
) -> WorkflowResult:
    """Body of the codegen workflow against already-opened resources.

    All persistence (checkpointer, store, indexed store, memory factory,
    file uploader, LLM, provider) comes off ``resources``. The
    ``rag_db`` handle is also passed in by the outer; the executor
    closes over both for everything it needs."""
    logger = logging.getLogger(__name__)

    llm = resources.llm
    provider = resources.provider
    store = resources.store
    checkpointer = resources.checkpointer
    indexed_store = resources.indexed_store

    audit_store = AuditStore(store)

    thread_id = workflow_options.thread_id

    if thread_id is None:
        thread_id = "crypto_session_" + str(uuid.uuid1())
        await handler.log_workflow_thread(WorkflowPurpose.CODEGEN, thread_id)
        logger.info(f"Selected thread id: {thread_id}")

    mem_root = memory_namespace or thread_id

    prompt_params: PromptParams
    fs_layer: str | None = None
    flow_input: AIComposerInput

    system_doc: Document
    interface_file: TextDocument
    spec_file: TextDocument
    resume_art : None | ResumeArtifact = None

    match input:
        case InputData():
            prompt_params = PromptParams(is_resume=False)
            flow_input = get_fresh_input(input, workflow_options)
            system_doc = input.system_doc
            interface_file = input.intf
            spec_file = input.spec

        case ResumeIdData() | ResumeFSData():
            prompt_params = PromptParams(is_resume=True)

            resume_art = await audit_store.get_resume_artifact(
                thread_id=input.thread_id
            )

            # The audit store hands back ``Uploadable`` handles (raw
            # bytes/text, no provider-aware rendering). Rehydrate each
            # through the uploader so we get real Document / TextDocument
            # instances tagged for the active provider. Resume-prompt
            # construction (above) works against the unhydrated forms
            # directly — it only needs text bodies for the diff blocks.
            system_uploadable: Uploadable = (
                input.new_system if input.new_system is not None
                else resume_art.system_vfs_handle
            )
            system_doc = await resources.uploader.upload_if_needed(system_uploadable)

            interface_uploadable: TextUploadable
            spec_uploadable: TextUploadable
            match input:
                case ResumeFSData():
                    (flow_input, interface_uploadable, spec_uploadable) = get_resume_fs_input(input, resume_art, workflow_options)
                    fs_layer = input.file_path
                case ResumeIdData():
                    interface_uploadable = resume_art.intf_vfs_handle
                    flow_input = get_resume_id_input(input, resume_art, workflow_options)
                    spec_uploadable = input.new_spec  # type: ignore[assignment]

            interface_file = await resources.uploader.upload_text_if_needed(interface_uploadable)
            spec_file = await resources.uploader.upload_text_if_needed(spec_uploadable)

    # Memory tool for the natreq sub-workflow. Both natreq and the
    # judge share the same namespace ("natreq") so they see each
    # other's notes.
    req_mem_tool = resources.memory(get_memory_ns(mem_root, "natreq"))

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
                req_mem_tool,
                provider,
                resume_art,
                workflow_options.requirements_oracle,
            )
            reqs_list = extraction.reqs
            await handler.log_workflow_thread(WorkflowPurpose.NATREQ, extraction.thread_id)
        await store.aput((thread_id,), "requirements", {"reqs": reqs_list})
    else:
        print("Read requirements from store")
        reqs_list = extra_reqs.value["reqs"]
    extra_tools: list[BaseTool] = []

    if reqs_list is not None:
        judge_tool = get_judge_tool(
            reqs=reqs_list,
            mem_tool=req_mem_tool,
            unbound=llm,
            vfs_tools=get_vfs_tools(
                fs_layer=fs_layer, immutable=True
            )[0]
        )
        extra_tools.append(judge_tool)
        extra_tools.append(requirements_relaxation)

    if workflow_options.memory_tool:
        # Second memory tool, namespaced for the codegen author itself
        # (separate from natreq's namespace).
        extra_tools.append(resources.memory(get_memory_ns(mem_root, "composer")))

    # ------------------------------------------------------------------
    # CVL research sub-agent — KB needs indexed store for semantic search.
    # Build the basic builder inline (was previously hidden inside
    # ``get_cryptostate_builder`` + ``_CodegenResearchContext``).
    # ------------------------------------------------------------------
    basic_builder = (
        Builder()
        .with_llm(llm)
        .with_loader(load_jinja_template)
        .with_checkpointer(checkpointer)
    )

    cvl_builder = basic_builder.with_tools(
        cvl_manual_tools(rag_db, provider)
    ).with_tools(
        make_kb_tools(indexed_store, _KB_NS, read_only=True)
    )

    research_doc = CVL_RESEARCH_BASE_DOC + " Do NOT use this for source code questions — use the VFS tools for that."
    async def runner[S: StateLike, I: StateLike](
        graph: CompiledStateGraph[S, Any, I, Any],
        i: I,
        tool_id: str | None,
    ) -> S:
        return await run_graph(
            ctxt=None,
            description="CVL Researcher",
            graph=graph,
            input=i,
            run_conf={
                "recursion_limit": workflow_options.recursion_limit,
                "configurable": {
                    "thread_id": "research-" + uuid.uuid4().hex[:16]
                }
            },
            within_tool=tool_id
        )
    extra_tools.append(_build_research_tool(cvl_builder, runner, research_doc))

    # ------------------------------------------------------------------
    # Codegen author graph.
    # ------------------------------------------------------------------
    from composer.tools.prover import certora_prover
    from composer.tools.proposal import propose_spec_change
    from composer.tools.question import human_in_the_loop
    from composer.tools.result import code_result
    from composer.tools.working_spec import CommitWorkingSpec, ReadWorkingSpec, WriteWorkingSpec

    (vfs_tooling, materializer) = get_vfs_tools(fs_layer=fs_layer, immutable=False)

    # The codegen author gets the bare ``cvl_manual_search`` directly;
    # the full research sub-agent (``cvl_manual_tools`` + KB + indexed
    # researcher) is exposed via ``extra_tools`` above.
    crypto_tools: list[BaseTool] = [
        certora_prover,
        propose_spec_change,
        human_in_the_loop,
        code_result,
        cvl_manual_search(rag_db, provider),
        *vfs_tooling,
        ReadWorkingSpec.as_tool("read_working_spec"),
        WriteWorkingSpec.as_tool("write_working_spec"),
        CommitWorkingSpec.as_tool("commit_working_spec"),
    ]

    summary_conf: SummaryGeneration = SummaryGeneration()

    author_builder = (
        basic_builder
        .with_context(AIComposerContext)
        .with_input(AIComposerInput)
        .with_state(AIComposerState)
        .with_output_key("generated_code")
        .with_tools(crypto_tools)
        .with_tools(extra_tools)
        .with_sys_prompt_template("system_prompt.j2")
        .with_initial_prompt_template("synthesis_prompt.j2", **prompt_params)
    )
    if summary_conf is not None:
        author_builder = author_builder.with_summary_config(summary_conf)

    workflow_graph = author_builder.build_async()[0]

    spec_vfs_path = (
        resume_art.spec.vfs_path if resume_art is not None else _SPEC_VFS_PATH
    )
    await audit_store.register_run(
        thread_id=thread_id,
        spec_vfs_path=spec_vfs_path,
        spec_file=spec_file,
        interface_file=interface_file,
        system_doc=system_doc,
        vfs_init=materializer.iterate(flow_input),
        reqs=reqs_list,
    )

    workflow_exec = workflow_graph.compile(checkpointer=checkpointer, store=store)
    if reqs_list is not None:
        flow_input["input"].append(f"""
    Additionally, the implementation MUST satisfy the following requirements:
    {"\n".join(f"{i}. {r}" for (i, r) in enumerate(reqs_list, start = 1))}
    """)

    if resume_work_key is not None:
        snapshot = await recover_vfs(store, resume_work_key)
        if snapshot is not None:
            vfs_files = list(snapshot["vfs"].items())
            recovery_msg = load_jinja_template("crash_recovery_context.j2", vfs_files=vfs_files)
            flow_input["input"].insert(0, recovery_msg)
            if snapshot["working_spec"] is not None:
                flow_input["working_spec"] = snapshot["working_spec"]

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

    resolved = make_prover_options(
        cloud=not workflow_options.local_prover,
        user_extra_args=shlex.split(workflow_options.prover_extra_args) if workflow_options.prover_extra_args else [],
    )
    prover_opts: ProverOptions = ProverOptions(
        capture_output=workflow_options.prover_capture_output,
        keep_folder=workflow_options.prover_keep_folders,
        extra_args=resolved.extra_args,
    )

    required_validations : list[ValidationType] = [prover]
    if reqs_list is not None:
        required_validations.append(req_type)

    work_context = AIComposerContext(
        llm=llm, rag_db=rag_db, prover_opts=prover_opts,
        vfs_materializer=materializer, required_validations=required_validations,
    )

    audit_sink = AuditStoreSink(audit_store, thread_id)

    try:
        async with with_handler(handler, CodeGenEventHandler(handler, audit_sink)):
            final_state = await run_graph(workflow_exec, work_context, flow_input, config, description="Code generation")

        result = final_state.get("generated_code", None)
        if result is None:
            return WorkflowFailure()

        res_commentary = await create_resume_commentary(final_state, llm=llm)
        await audit_store.register_complete(
            thread_id, materializer.iterate(final_state),
            res_commentary.interface_path, res_commentary.commentary,
        )

        await handler.output(result, materializer, final_state)
        return WorkflowSuccess()
    except Exception as exc:
        await handler.show_error(exc)
        # Attempt to capture VFS from last checkpoint
        resume_key: str | None = None
        try:
            resume_key = await recovery_from_thread(
                checkpointer=checkpointer, store=store, thread_id=thread_id,
            )
        except Exception as snapshot_exc:
            logger.warning(f"Failed to capture crash snapshot: {snapshot_exc}")
        return WorkflowCrash(resume_work_key=resume_key, error=exc)


# ---------------------------------------------------------------------------
# Outer executor — opens resources, delegates to inner.
# ---------------------------------------------------------------------------


async def execute_ai_composer_workflow(
    handler: CodeGenIOHandler,
    input: InputData | ResumeFSData | ResumeIdData,
    workflow_options: _ExecutorOptions,
    memory_namespace: str | None = None,
    resume_work_key: str | None = None,
) -> WorkflowResult:
    """Execute the AI Composer workflow.

    Opens every per-workflow resource (checkpointer, store, indexed
    store, memory factory, file uploader, LLM, provider) via
    ``standard_connections`` once at the top, plus the RAG-DB handle
    and the tool-display context, then runs the inner executor against
    them. Nothing here reaches for a sync global getter; everything
    flows through the bundle."""
    model = get_rag_model()
    async with (
        async_tool_context(),
        standard_connections(args=workflow_options, embedder=DefaultEmbedder(model)) as conn,
        PostgreSQLRAGDatabase.rag_context(model, workflow_options.rag_db) as rag_db,
    ):
        return await _execute_ai_composer_workflow(
            handler=handler,
            input=input,
            workflow_options=workflow_options,
            memory_namespace=memory_namespace,
            resume_work_key=resume_work_key,
            resources=conn,
            rag_db=rag_db,
        )
