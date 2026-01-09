import argparse
import tempfile
import subprocess
import os
import hashlib
import sqlite3
import uuid
import pathlib

from dataclasses import dataclass
from typing import cast, Annotated, Literal, NotRequired, Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from langchain_core.tools import tool, InjectedToolCallId, BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.runtime import get_runtime
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from composer.input.types import ModelOptions, RAGDBOptions, LangraphOptions
from composer.input.parsing import add_protocol_args
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.cvl.schema import CVLFile
from composer.cvl.pretty_print import pretty_print
from composer.spec.types import NatSpecState, Result, NatSpecInput
from composer.tools.search import cvl_manual_search
from composer.workflow.services import create_llm, get_checkpointer, get_memory
from composer.tools.human_tool import human_interaction_tool
from composer.human.handlers import prompt_input
from composer.templates.loader import load_jinja_template

from graphcore.tools.memory import memory_tool, SqliteMemoryBackend
from graphcore.tools.results import result_tool_generator, ValidationResult
from graphcore.graph import build_workflow, FlowInput, MessagesState, tool_state_update
from graphcore.summary import SummaryConfig

guidelines_validation = Literal["guidelines"]

all_validations : list[guidelines_validation] = ["guidelines"]

class NatSpecArgs(ModelOptions, RAGDBOptions, LangraphOptions):
    input_file: str

@dataclass
class NatSpecContext:
    orig_doc: str
    rag_db: PostgreSQLRAGDatabase

class GuidelineJudgeSchema(BaseModel):
    """
    Invoke an oracle to determine if the proposed spec file meets all of the guidelines for CVL authorship.

    The generation workflow is not complete until the Guidelines judge returns no critical feedback.
    """
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[NatSpecState, InjectedState]


class FeedbackItem(BaseModel):
    """
    A single point of feedback on the CVL file.

    Each piece of feedback is assigned a severity from (lowest to highest): Informational < Advice < Critical.

    "Critical" severity should be used for any feedback for which there is overwhelming evidence that the guidelines have
    been violated.

    "Advice" should be used for any feedback which cannot be considered critical, but should still *likely* be changed due
    to potentially contravening the guidelines.

    "Information" is for any other feedback which may, or may not, be actionable.
    """
    severity: Literal["Informational", "Advice", "Critical"] = Field(description="The severity of the feedback.")
    description: str = Field(description="The natural language description of the feedback.")
    code_reference: str = Field(description="The code snippet pulled verbatim from the CVL file which is the subject of this feedback")
    guideline_ref: str = Field(description="The text of the guideline which is violated.")

class Feedback(BaseModel):
    items: list[FeedbackItem]

class FeedbackState(MessagesState):
    output: NotRequired[Feedback]

def format_feedback(feedback: Feedback) -> str:
    buff = []
    for f in feedback.items:
        buff.append(
f"""
<feedback>
    <severity>{f.severity}</severity>
    <guideline>{f.guideline_ref}</guideline>
    <description>{f.description}</description>
    <code_ref>{f.code_reference}</code_ref>
</feedback>
""")
    return "\n".join(buff)

def compute_spec_digest(s: NatSpecState) -> str:
    h = hashlib.md5()
    assert s["curr_spec"] is not None
    h.update(s["curr_spec"].encode("utf-8"))
    return h.hexdigest()

def get_judge_tool(
    llm: BaseChatModel,
) -> BaseTool:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    mem = memory_tool(SqliteMemoryBackend("none", conn))
    search_tool = cvl_manual_search(NatSpecContext)
    result = result_tool_generator("output", Feedback, """
Used to communicate the result of your analysis of the CVL spec against the guidelines.
""")
    work = build_workflow(
        state_class=FeedbackState,
        input_type=FlowInput,
        tools_list=[mem, search_tool, result],
        sys_prompt=load_jinja_template("cvl_system_prompt.j2"),
        initial_prompt=load_jinja_template("guidelines_judge_prompt.j2"),
        output_key="output",
        context_schema=NatSpecContext,
        unbound_llm=llm
    )[0].compile()

    @tool(args_schema=GuidelineJudgeSchema)
    def feedback_tool(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[NatSpecState, InjectedState]
    ) -> Command | str:
        if state["curr_intf"] is None or state["curr_spec"] is None:
            return "Both the interface and curr_spec must be put on the VFS before calling this tool"
        curr_context = get_runtime(NatSpecContext).context
        r = work.invoke(FlowInput(input=[
            state["curr_spec"],
            state["curr_intf"]
        ]), context=curr_context)
        assert "output" in r
        feedback = r["output"]
        assert isinstance(feedback, Feedback)
        validated = all([
            t.severity != "Critical" for t in feedback.items
        ])
        formatted_feedback = format_feedback(feedback)
        if not validated:
            return formatted_feedback
        return tool_state_update(
            tool_call_id=tool_call_id,
            content=formatted_feedback,
            validations = {
                "guidelines": compute_spec_digest(state)
            }
        )
    return feedback_tool

@tool
def get_document() -> str:
    """
    Retrieves the original design document if necessary
    """
    return get_runtime(NatSpecContext).context.orig_doc

put_cvl_description = """
Put a new version of the proposed spec file onto the VFS. The tool schema constrains
you to putting only syntactically valid CVL. However, a pretty printed version of this syntax
is ultimately what is saved on the VFS.

This pretty printed file is then run through the official CVL parser. If the code fails to parse,
this tool will reject the update, with the reported errors.
"""

class PutCVLSchemaModel(BaseModel):
    cvl_file: CVLFile = Field(description="The CVL AST to put in the VFS")

class PutCVLSchemaLG(BaseModel):
    cvl_file: dict = Field(description="The CVL AST to put in the VFS")
    tool_call_id: Annotated[str, InjectedToolCallId]

PutCVLSchemaLG.__doc__ = put_cvl_description

def _maybe_update_cvl(
    tool_call_id: str,
    pp: str
) -> str | Command:
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".spec") as f:
            f.write(pp)
            certora_dir = os.environ["CERTORA"]
            emv_jar = os.path.join(certora_dir, "emv.jar")
            res = subprocess.run(
                ["java", "-classpath", emv_jar, "spec.ParseCheckerKt", f.name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if res.returncode != 0:
                return f"""
    Update rejected, the syntax checker exited with non-zero status

    stdout:
    {res.stdout}

    stderr:
    {res.stderr}
    """
    except:
        return "Syntax checker failed"
    return tool_state_update(
        tool_call_id=tool_call_id,
        content="Accepted",
        curr_spec=pp
    )

# for deeply cursed reasons, we need to have two versions of this schema
@tool(args_schema=PutCVLSchemaLG)
def put_cvl(
    cvl_file: dict,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command | str:
    pp: str
    try:
        pp = pretty_print(CVLFile.model_validate(cvl_file))
    except:
        import traceback
        traceback.print_exc()
        return "Failed to pretty print the AST"
    return _maybe_update_cvl(tool_call_id, pp)
    

class PutCVLRaw(BaseModel):
    """
    A version of put CVL which accepts the surface syntax of CVL. You should only use
    this if you have extremely high confidence that the CVL representation you are passing in
    is correct.

    If `cvl_file` is determined to have a syntax error, this update is rejected.
    """
    cvl_file: str = Field(description="The raw, surface syntax of the CVL file.")
    tool_call_id: Annotated[str, InjectedToolCallId]

@tool(args_schema=PutCVLRaw)
def put_cvl_raw(
    tool_call_id: Annotated[str, InjectedToolCallId],
    cvl_file: str
) -> str | Command:
    return _maybe_update_cvl(tool_call_id, cvl_file)

class GetCVLSchema(BaseModel):
    """
    View the (pretty-printed) version of the CVL file.
    """
    state: Annotated[NatSpecState, InjectedState]

@tool(args_schema=GetCVLSchema)
def get_cvl(
    state: Annotated[NatSpecState, InjectedState]
) -> str:
    if state["curr_spec"] is None:
        return "No spec file on VFS"
    return state["curr_spec"]

class PutInterfaceSchema(BaseModel):
    """
    Put the proposed interface file for the system entry point on the VFS.
    If the interface file is not syntactically correct (according to the solidity compiler)
    this update is rejected.
    """
    contents: str = Field(description="The contents of the interface file")
    tool_call_id: Annotated[str, InjectedToolCallId]
    solc: str = Field(description="The solidity compiler version to use for checking the syntax of the interface. Expected to be"
    "the string X.Y, where 0.X.Y is the official version of a Solidity release. For example, 8.21 refers to version 0.8.21")

@tool(args_schema=PutInterfaceSchema)
def put_interface(
    contents: str,
    solc: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command | str:
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sol") as f:
            f.write(contents)
            proc = subprocess.run(
                [f"solc{solc}", f.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if proc.returncode != 0:
                return f"""
Update rejected: Solidity compiler exited with non-zero status.

stdout:
{proc.stdout}

stderr:
{proc.stderr}
"""
    except FileNotFoundError:
        return f"Solidity compiler {solc} doesn't exist on this system"
    return tool_state_update(
        tool_call_id=tool_call_id,
        content="Accepted",
        curr_intf=contents
    )

def validate_spec_completion(
    st: NatSpecState,
    m: Result,
    id: str
) -> ValidationResult:
    if st["curr_intf"] is None:
        return "No interface file has been placed yet."
    if st["curr_spec"] is None:
        return "No spec file has been generated yet."
    digest = compute_spec_digest(st)
    for req in all_validations:
        hash = st["validations"].get(req, None)
        if hash is None:
            return f"The required validation {req} has not been performed"
        if hash != digest:
            return f"The required validation {req} is out of date; re-run it on the most recent version of the spec"
    return None

generation_complete = result_tool_generator(
    outkey="result",
    result_schema=Result,
    doc="Used to indicate the successful generation of a spec + interface for the natural language input.",
    validator=(NatSpecState, validate_spec_completion)
)

class HumanQuestionSchema(BaseModel):
    """
    Use to pose a question to the user. You should *not* assume the user is necessarily familiar with
    CVL. The primary usage of this tool should be to clarify intent over ambiguities in the natural language
    specification.
    """
    question: Annotated[str, "The question to pose to the user"]
    context: Annotated[str, "Any additional context to the question, e.g. a citation from the natural language spec."]

human_question_tool = human_interaction_tool(
    HumanQuestionSchema,
    NatSpecState,
    "human_in_the_loop"
)

def execute(args: NatSpecArgs) -> int:
    llm = create_llm(args)

    judge = get_judge_tool(
        llm
    )

    thread_id : str
    if args.thread_id is None:
        thread_id = "natspec_session_" + uuid.uuid4().hex
        print(f"Selected {thread_id}")
    else:
        thread_id = args.thread_id
    
    thread_memories = get_memory(f"natspec-{thread_id}")

    mem_tool = memory_tool(thread_memories)

    manual = cvl_manual_search(NatSpecContext)

    checkpointer = get_checkpointer()

    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db, model=get_model(), skip_test=True
    )

    document = pathlib.Path(args.input_file).read_text()

    ctxt = NatSpecContext(orig_doc=document, rag_db=rag_db)

    graph = build_workflow(
        context_schema=NatSpecContext,
        initial_prompt=load_jinja_template("cvl_generation_prompt.j2"),
        sys_prompt=load_jinja_template("cvl_system_prompt.j2"),
        input_type=NatSpecInput,
        output_key="result",
        state_class=NatSpecState,
        unbound_llm=llm,
        summary_config=SummaryConfig(),
        tools_list=[
            manual,
            human_question_tool,
            generation_complete,
            put_interface,
            get_document,
            get_cvl,
            put_cvl_raw,
            judge,
            mem_tool,
            ({"name": put_cvl.name, "description": put_cvl_description, "input_schema": PutCVLSchemaModel.model_json_schema()}, put_cvl)
        ]
    )[0].compile(checkpointer=checkpointer)

    def fresh_config() -> RunnableConfig:
        return {
            "recursion_limit": args.recursion_limit,
            "configurable": {
                "thread_id": thread_id
            }
        }


    runnable_conf : RunnableConfig = fresh_config()


    if args.checkpoint_id is not None:
        assert "configurable" in runnable_conf
        runnable_conf["configurable"]["checkpoint_id"] = args.checkpoint_id

    graph_input : Command | NatSpecInput | None = NatSpecInput(input=[
        "The system/design document is as follows",
        document
    ], curr_intf=None, curr_spec=None, validations={})

    if args.checkpoint_id is not None:
        graph_input = None
    
    while True:
        t = graph_input
        graph_input = None
        for (tag, payload) in graph.stream(
            input=t,
            config=runnable_conf,
            context=ctxt,
            stream_mode=["updates", "checkpoints"]
        ):
            assert isinstance(payload, dict)
            if tag == "checkpoints":
                assert isinstance(payload, dict)
                print("current checkpoint: " + payload["config"]["configurable"]["checkpoint_id"])
                continue
            if "__interrupt__" in payload:
                runnable_conf = fresh_config()
                data = payload["__interrupt__"][0].value
                assert isinstance(data, HumanQuestionSchema)
                print("=" * 80)
                print("HUMAN ASSISTANCE REQUESTED")
                print("=" * 80)
                print(f"Question:\n{data.question}")
                print(f"Context:\n{data.context}")
                res = prompt_input("Enter your response", lambda: None)
                graph_input = Command(
                    resume=res
                )
                break
            else:
                print(payload)
        if graph_input is None:
            break

    final_state = cast(NatSpecState, graph.get_state(fresh_config()).values)
    if "result" not in final_state:
        return 1
    
    print("Spec file generation complete")
    print(final_state["curr_spec"])
    print(final_state["curr_intf"])
    print(final_state["result"])

    return 0

def main() -> int:
    parser = argparse.ArgumentParser(usage="Generate a CVL from a natural language design doc.")
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, ModelOptions)
    add_protocol_args(parser, LangraphOptions)
    parser.add_argument("input_file", help="The input file to use for the spec generation")

    args = cast(NatSpecArgs, parser.parse_args())
    return execute(args)