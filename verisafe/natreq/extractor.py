from typing import NotRequired, cast
from dataclasses import dataclass
import uuid

from pydantic import BaseModel, Field

from graphcore.graph import FlowInput, build_workflow
from graphcore.tools.results import result_tool_generator
from graphcore.tools.memory import memory_tool, MemoryBackend

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import MessagesState
from langgraph.types import interrupt, Command

from verisafe.audit.types import InputFileLike
from verisafe.input.types import RAGDBOptions
from verisafe.rag.db import PostgreSQLRAGDatabase
from verisafe.rag.models import get_model
from verisafe.workflow.factories import get_checkpointer
from verisafe.tools.search import cvl_manual_search
from verisafe.templates.loader import load_jinja_template


class ExtractionState(MessagesState):
    reqs: NotRequired[list[str]]

@dataclass
class ExtractionContext:
    rag_db: PostgreSQLRAGDatabase

class HumanClarificationArgs(BaseModel):
    """
    Ask a question to the user to help extract the natural language specifications. A *non-exhaustive* list of topics
    appropriate for discussion are:
    1. Ambiguities in the system document
    2. Clarifying multiple potential interpretations of the natural language text of the system doc
    3. Clarifying the intention behind the various rules in the specification
    4. Resolving apparently conflicts between the system document and the specification
    5. Clarifying whether passages in the system doc exposition vs. code requirements

    The above are just guidelines, you should use this tool to resolve any potential confusion or uncertainty you may have.
    """
    question: str = Field(description="The specific question to ask the user.")

    context: str = Field(description="Context or explanation surrounding the question. Use this to explain your thinking, cite " \
    "specific portions of the spec/system doc, or any other salient information to help ground the question.")

@tool(args_schema=HumanClarificationArgs)
def human_in_the_loop(
    question: str,
    context: str
) -> str:
    response = interrupt({
        "question": question,
        "context": context
    })
    return response

results_tool = result_tool_generator(
    "reqs",
    (list[str], "The list of natural language requirements you extracted during this process."),
    """
Tool used to indicate your analysis is complete and communicate the generated requirements back to the user.

REMINDER: You should call this tool only AFTER you have updated your memories.
"""
)


system_prompt = load_jinja_template("req_role_prompt.j2")

initial_prompt = load_jinja_template("req_extraction_prompt.j2")

def get_requirements(
    options: RAGDBOptions,
    llm: BaseChatModel,
    sys_doc: InputFileLike,
    spec_file: InputFileLike,
    mem_backend: MemoryBackend
) -> list[str]:
    tools = [
        memory_tool(mem_backend),
        results_tool,
        human_in_the_loop,
        cvl_manual_search
    ]
    built = build_workflow(
        state_class=ExtractionState,
        context_schema=ExtractionContext,
        input_type=FlowInput,
        output_key="reqs",
        tools_list=tools,
        unbound_llm=llm,
        summary_config=None,
        sys_prompt=system_prompt,
        initial_prompt=initial_prompt
    )[0].compile(checkpointer=get_checkpointer())
    
    thread_id = uuid.uuid1().hex

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    db = PostgreSQLRAGDatabase(
        conn_string=options.rag_db,
        model=get_model(),
    )

    graph_input : Command | FlowInput | None = FlowInput(
        input=[
            sys_doc.string_contents,
            spec_file.string_contents
        ]
    )
    while graph_input is not None:
        to_send = graph_input
        graph_input = None
        for payload in built.stream(input = to_send, context=ExtractionContext(rag_db=db), config=config):
            if "__interrupt__" in payload:
                interrupt_data = cast(dict, payload["__interrupt__"][0].value)
                context = interrupt_data["context"]
                question = interrupt_data["question"]
                print("=" * 80)
                print(" HUMAN ASSISTANCE REQUESTED")
                print("=" * 80)
                print(f"Context:\n{context}")
                print(f"Question: {question}")
                human_response = input("Enter your reponse: ")
                graph_input = Command(resume=human_response)
                break
    return built.get_state(config).values["reqs"]