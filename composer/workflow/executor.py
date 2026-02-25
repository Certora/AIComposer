from typing import Optional
import logging
import uuid
from dataclasses import dataclass
import pathlib
import psycopg

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel

from graphcore.tools.memory import memory_tool

from composer.input.types import WorkflowOptions, InputData, ResumeFSData, ResumeIdData, ResumeInput, NativeFS
from composer.workflow.factories import get_checkpointer, get_cryptostate_builder, get_store, get_memory, get_vfs_tools, get_memory_ns
from composer.workflow.types import Input, PromptParams
from composer.workflow.meta import create_resume_commentary
from composer.core.state import AIComposerState  # used by VFSAccessor type param
from composer.core.context import AIComposerContext, ProverOptions
from composer.core.validation import ValidationType, prover, reqs as req_type
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model as get_rag_model
from composer.audit.db import AuditDB, AuditDBSink, ResumeArtifact, InputFileLike
from composer.natreq.extractor import get_requirements
from composer.natreq.judge import get_judge_tool
from composer.tools.relaxation import requirements_relaxation
from composer.templates.loader import load_jinja_template
from composer.io.protocol import CodeGenIOHandler
from composer.io.context import with_handler, run_graph
from composer.io.codegen_events import CodeGenEventHandler


def get_reference_input(input_data: InputData, debug_prompt: Optional[str]) -> str:
    return load_jinja_template(
        "workflow_info.j2",
        spec_filename=input_data.spec.basename,
        interface_filename=input_data.intf.basename,
        system_doc_filename=input_data.system_doc.basename,
        debug_prompt=debug_prompt)


def get_fresh_input(input: InputData, workflow_options: WorkflowOptions) -> Input:
    return Input(input=[
                input.intf.to_document_dict(),
                input.spec.to_document_dict(),
                input.system_doc.to_document_dict(),
                {
                    "type": "text",
                    "text": get_reference_input(input_data=input, debug_prompt=workflow_options.debug_prompt_override)
                }
            ], vfs={"rules.spec": input.spec.read()})

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

def get_resume_id_input(input: ResumeIdData, resume_art: ResumeArtifact, workflow_options: WorkflowOptions) -> Input:

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
    return Input(
        input=input_messages,
        vfs=new_vfs
    )

def get_resume_fs_input(input: ResumeFSData, resume_art: ResumeArtifact, workflow_options: WorkflowOptions) -> tuple[Input, InputFileLike, InputFileLike]:
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

    return (Input(input=input_messages, vfs={}), NativeFS(intf_p), NativeFS(spec_p))


async def execute_ai_composer_workflow(
    handler: CodeGenIOHandler,
    llm: BaseChatModel,
    input: InputData | ResumeFSData | ResumeIdData,
    workflow_options: WorkflowOptions
) -> int:
    """Execute the AI Composer workflow with interrupt handling."""
    logger = logging.getLogger(__name__)

    checkpointer = get_checkpointer()

    audit_conn = psycopg.connect(workflow_options.audit_db)
    audit_db = AuditDB(audit_conn)

    thread_id = workflow_options.thread_id

    if thread_id is None:
        thread_id = "crypto_session_" + str(uuid.uuid1())
        await handler.log_thread_id(thread_id, chosen=True)
        logger.info(f"Selected thread id: {thread_id}")

    prompt_params: PromptParams
    fs_layer: str | None = None
    flow_input: Input

    system_doc: InputFileLike
    interface_file: InputFileLike
    spec_file: InputFileLike
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

            resume_art = audit_db.get_resume_artifact(thread_id=input.thread_id)
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

    store = get_store()

    from_previous_ns : str | None = None
    match input:
        case ResumeFSData(thread_id=src_id) | ResumeIdData(thread_id=src_id):
            from_previous_ns = get_memory_ns(src_id, "natreq")
        case InputData():
            # here for completeness of matching...
            pass

    req_memories = get_memory(
        get_memory_ns(thread_id, "natreq"),
        from_previous_ns
    )

    extra_reqs = store.get((thread_id,), "requirements")
    reqs_list : list[str] | None
    if extra_reqs is None:
        if workflow_options.skip_reqs:
            reqs_list = None
        elif workflow_options.set_reqs is not None:
            if workflow_options.set_reqs.startswith("@"):
                other_reqs = store.get((workflow_options.set_reqs[1:],), "requirements")
                assert other_reqs is not None
                reqs_list = other_reqs.value["reqs"]
            else:
                reqs_list = [ v for l in pathlib.Path(workflow_options.set_reqs).read_text().splitlines() if (v := l.strip()) ]
        else:
            print("Analyzing requirements...")
            reqs = await get_requirements(
                handler,
                workflow_options,
                llm,
                system_doc,
                spec_file,
                req_memories,
                resume_art,
                workflow_options.requirements_oracle
            )
            reqs_list = reqs
        store.put((thread_id,), "requirements", {"reqs": reqs_list})
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
        memory = memory_tool(get_memory(thread_id, "composer"))
        extra_tools.append(memory)


    (workflow_builder, bound_llm, materializer) = get_cryptostate_builder(
        llm=llm,
        fs_layer=fs_layer,
        prompt_params=prompt_params,
        summarization_threshold=workflow_options.summarization_threshold,
        extra_tools=extra_tools
    )

    audit_db.register_run(
        thread_id=thread_id,
        system_doc=system_doc,
        interface_file=interface_file,
        spec_file=spec_file,
        vfs_init=materializer.iterate(flow_input),
        reqs=reqs_list
    )

    workflow_exec = workflow_builder.compile(checkpointer=checkpointer, store=store)
    if reqs_list is not None:
        flow_input["input"].append(f"""
    Additionally, the implementation MUST satisfy the following requirements:
    {"\n".join(f"{i}. {r}" for (i, r) in enumerate(reqs_list, start = 1))}
    """)

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

    rag_connection = workflow_options.rag_db

    prover_opts: ProverOptions = ProverOptions(
        capture_output=workflow_options.prover_capture_output,
        keep_folder=workflow_options.prover_keep_folders
    )

    rag_db = PostgreSQLRAGDatabase(rag_connection, get_rag_model(), skip_test=True)
    required_validations : list[ValidationType] = [prover]
    if reqs_list is not None:
        required_validations.append(req_type)

    work_context = AIComposerContext(llm=bound_llm, rag_db=rag_db, prover_opts=prover_opts, vfs_materializer=materializer, required_validations=required_validations)

    audit_sink = AuditDBSink(audit_db, thread_id)

    async with with_handler(handler, CodeGenEventHandler(handler, audit_sink)):
        final_state = await run_graph(workflow_exec, work_context, flow_input, config)

    result = final_state.get("generated_code", None)
    if result is None:
        return 1

    res_commentary = create_resume_commentary(final_state, llm=llm)
    audit_db.register_complete(
        thread_id, materializer.iterate(final_state), res_commentary.interface_path, res_commentary.commentary
    )

    await handler.output(result, materializer, final_state)
    return 0
