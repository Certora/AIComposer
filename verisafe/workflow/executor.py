from typing import Optional, Literal, cast
import logging
import uuid
import logging
from dataclasses import dataclass
import pathlib
import psycopg

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command

from verisafe.input.types import WorkflowOptions, InputData, ResumeFSData, ResumeIdData, ResumeInput, NativeFS
from verisafe.workflow.factories import get_checkpointer, get_cryptostate_builder, get_store
from verisafe.workflow.types import Input, PromptParams
from verisafe.workflow.meta import create_resume_commentary
from verisafe.core.state import ResultStateSchema, CryptoStateGen
from verisafe.core.context import CryptoContext, ProverOptions
from verisafe.rag.db import PostgreSQLRAGDatabase
from verisafe.rag.models import get_model as get_rag_model
from verisafe.audit.db import AuditDB, ResumeArtifact, InputFileLike
from verisafe.diagnostics.stream import AllUpdates, PartialUpdates, Summarization
from verisafe.diagnostics.handlers import summarize_update, handle_custom_update
from verisafe.human.handlers import handle_human_interrupt
from verisafe.templates.loader import load_jinja_template

StreamEvents = Literal["checkpoints", "custom", "updates"]


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


def execute_cryptosafe_workflow(
    llm: BaseChatModel,
    input: InputData | ResumeFSData | ResumeIdData,
    workflow_options: WorkflowOptions
) -> int:
    """Execute the CryptoSafe workflow with interrupt handling."""
    logger = logging.getLogger(__name__)

    checkpointer = get_checkpointer()

    audit_db: Optional[AuditDB] = None
    if workflow_options.audit_db is not None:
        conn = psycopg.connect(workflow_options.audit_db)
        audit_db = AuditDB(conn)

    thread_id = workflow_options.thread_id

    if thread_id is None:
        thread_id = "crypto_session_" + str(uuid.uuid1())
        print(f"Selected thread id: {thread_id}")
        logger.info(f"Selected thread id: {thread_id}")

    prompt_params: PromptParams
    fs_layer: str | None = None
    flow_input: Input

    system_doc: InputFileLike
    interface_file: InputFileLike
    spec_file: InputFileLike

    match input:
        case InputData():
            prompt_params = PromptParams(is_resume=False)
            flow_input = get_fresh_input(input, workflow_options)
            system_doc = input.system_doc
            interface_file = input.intf
            spec_file = input.spec

        case ResumeIdData() | ResumeFSData():
            prompt_params = PromptParams(is_resume=True)

            if audit_db is None:
                raise RuntimeError("Cannot do resume workflows without audit db")

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

    (workflow_builder, bound_llm, materializer) = get_cryptostate_builder(
        llm=llm,
        fs_layer=fs_layer,
        prompt_params=prompt_params,
        summarization_threshold=workflow_options.summarization_threshold
    )

    if audit_db is not None:
        audit_db.register_run(
            thread_id=thread_id,
            system_doc=system_doc,
            interface_file=spec_file,
            spec_file=interface_file,
            vfs_init=materializer.iterate(cast(CryptoStateGen, flow_input)) #hack
        )

    store = get_store()

    workflow_exec = workflow_builder.compile(checkpointer=checkpointer, store=store)

    try:
        import grandalf # type: ignore
        layout = workflow_exec.get_graph().draw_ascii()
        logger.debug(f"\n{layout}")
    except ModuleNotFoundError:
        pass

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    config["recursion_limit"] = workflow_options.recursion_limit

    current_input: Optional[Input | Command] = flow_input

    if workflow_options.checkpoint_id is not None:
        config["configurable"]["checkpoint_id"] = workflow_options.checkpoint_id
        current_input = None

    rag_connection = workflow_options.rag_db

    prover_opts: ProverOptions = ProverOptions(
        capture_output=workflow_options.prover_capture_output,
        keep_folder=workflow_options.prover_keep_folders
    )   

    rag_db = PostgreSQLRAGDatabase(rag_connection, get_rag_model(), skip_test=True)
    work_context = CryptoContext(llm=bound_llm, rag_db=rag_db, prover_opts=prover_opts, vfs_materializer=materializer)

    curr_state_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    while True:
        interrupted = False
        r = current_input
        current_input = None
        for (event_ty_raw, payload) in workflow_exec.stream(input=r, config=config, context=work_context, stream_mode=["custom", "updates", "checkpoints"]):
            event_ty = cast(StreamEvents, event_ty_raw)
            assert isinstance(payload, dict)
            match event_ty:
                case "checkpoints":
                    print("current checkpoint: " + payload["config"]["configurable"]["checkpoint_id"])
                    logger.info("current checkpoint: " + payload["config"]["configurable"]["checkpoint_id"])
                case "updates":
                    if "__interrupt__" in payload:
                        if "configurable" in config and "checkpoint_id" in config["configurable"]:
                            del config["configurable"]["checkpoint_id"]
                        interrupt_data = cast(dict, payload["__interrupt__"][0].value)
                        human_response = handle_human_interrupt(interrupt_data)
                        current_input = Command(resume=human_response)
                        interrupted = True
                        break
                    summarize_update(payload)
                case "custom":
                    p = cast(PartialUpdates, payload)
                    full_update: AllUpdates
                    if p["type"] == "summarization_raw":
                        curr_checkpoint = workflow_exec.get_state(curr_state_config).config.get("configurable", {}).get("checkpoint_id", None)
                        if curr_checkpoint is None:
                            raise RuntimeError("Have summarization before ever hitting a checkpoint; this is sus")
                        full_update = Summarization(
                            type="summarization",
                            checkpoint_id=curr_checkpoint,
                            summary=p["summary"]
                        )
                    else:
                        full_update = p
                    handle_custom_update(full_update, thread_id, audit_db)

        if interrupted:
            continue
        state = workflow_exec.get_state(curr_state_config)
        final_state = cast(CryptoStateGen, state.values)
        result = final_state.get("generated_code", None)
        if result is None:
            return 1
        if audit_db is not None:
            res_commentary = create_resume_commentary(final_state, llm=llm)
            audit_db.register_complete(
                thread_id, materializer.iterate(final_state), res_commentary.interface_path, res_commentary.commentary
            )

        assert isinstance(result, ResultStateSchema)
        print("\n" + "=" * 80)
        print("CODE GENERATION COMPLETED")
        print("=" * 80)
        print("Generated Source Files:")
        for path in result.source:
            print(f"\n--- {path} ---")
            file_contents = materializer.get(final_state, path)
            assert file_contents is not None
            content = file_contents.decode("utf-8")
            print(content)

        print(f"\nComments: {result.comments}")
        return 0
