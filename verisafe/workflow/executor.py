from typing import Optional, Literal, cast
import uuid
import sqlite3
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command

from verisafe.input.types import WorkflowOptions, InputData
from verisafe.workflow.factories import get_checkpointer, get_cryptostate_builder
from verisafe.workflow.types import Input
from verisafe.core.state import ResultStateSchema, CryptoStateGen
from verisafe.core.context import CryptoContext, ProverOptions
from verisafe.rag.types import DatabaseConfig
from verisafe.rag.db import PostgreSQLRAGDatabase
from verisafe.rag.models import get_model as get_rag_model
from verisafe.audit.db import AuditDB, InputFileLike
from verisafe.diagnostics.stream import AllUpdates
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

def execute_cryptosafe_workflow(
    llm: BaseChatModel,
    input: InputData,
    workflow_options: WorkflowOptions
) -> int:
    """Execute the CryptoSafe workflow with interrupt handling."""
    logger = logging.getLogger(__name__)

    checkpointer = get_checkpointer()

    audit_db: Optional[AuditDB] = None
    if workflow_options.audit_db is not None:
        conn = sqlite3.connect(workflow_options.audit_db)
        audit_db = AuditDB(conn)

    thread_id = workflow_options.thread_id

    if thread_id is None:
        thread_id = "crypto_session_" + str(uuid.uuid1())
        print(f"Selected thread id: {thread_id}")
        logger.info(f"Selected thread id: {thread_id}")

    fs_layer: str | None = None
    flow_input: Input

    system_doc: InputFileLike
    interface_file: InputFileLike
    spec_file: InputFileLike
    flow_input = get_fresh_input(input, workflow_options)
    system_doc = input.system_doc
    interface_file = input.intf
    spec_file = input.spec

    if audit_db is not None:
        audit_db.register_run(
            thread_id=thread_id,
            system_doc=system_doc,
            interface_file=spec_file,
            spec_file=interface_file
        )

    (workflow_builder, bound_llm, materializer) = get_cryptostate_builder(llm, fs_layer=fs_layer, summarization_threshold=workflow_options.summarization_threshold)

    workflow_exec = workflow_builder.compile(checkpointer=checkpointer)

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

    rag_connection: DatabaseConfig = {
        "host": workflow_options.db_host,
        "port": workflow_options.db_port,
        "database": workflow_options.db_name,
        "user": workflow_options.db_user,
        "password": workflow_options.db_password
    }

    prover_opts: ProverOptions = ProverOptions(
        capture_output=workflow_options.prover_capture_output,
        keep_folder=workflow_options.prover_keep_folders
    )   

    rag_db = PostgreSQLRAGDatabase(rag_connection, get_rag_model(), skip_test=True)
    work_context = CryptoContext(llm=bound_llm, rag_db=rag_db, prover_opts=prover_opts, vfs_materializer=materializer)

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
                    p = cast(AllUpdates, payload)
                    handle_custom_update(p, thread_id, audit_db)

        if interrupted:
            continue
        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        state = workflow_exec.get_state(result_config)
        final_state = cast(CryptoStateGen, state.values)
        result = final_state.get("generated_code", None)
        if result is None:
            return 1
        if audit_db is not None:
            audit_db.register_complete(
                thread_id, materializer.iterate(final_state)
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
