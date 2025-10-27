from typing import NotRequired, cast
from dataclasses import dataclass
import uuid

from pydantic import BaseModel, Field

from graphcore.graph import FlowInput, build_workflow
from graphcore.tools.results import result_tool_generator

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
"""
)


system_prompt = """
You are an expert software systems architect. As such, you have years of experience
reading design documents and analyzing requirements. You also know how to translate, high-level
natural language descriptions of complex systems into individual requirements which can be
independently evaluated. For example, you know to extract from a *description* like "the servers will exchange
credentials" the following requirements:
* Server A must maintain a database of encrypted and salted passwords
* Server B must implement an HTTP endpoint secured with pub/priv encryption to receive credential requests
etc.

Most of your experience is in the design and implementation of blockchain protocols, particularly DeFi.
You thus have an understanding of the mathematics behinds securities, asset exchange, and so on.
You have a good grasp of the economics of on-chain financial products, and how those are translated into
"smart contract" implementations.
"""

initial_prompt = """
You have been provided with a system document for a DeFi protocol. This document is a natural
language description of a protocol, with varying levels of formalism/rigor. This system document
may describe the interactions of one or more components, along with the behavior of the individual
components.

In addition, you have been provided with a *formal* specification of the behavior of one of the components
of this protocol. This specification is written in the Certora Verification Language (CVL), a DSL for
writing properties of smart contracts.

Analyze both the system document and the specification to identify any implementation
requirements/invariants/properties implied by the system document which are *NOT* covered by the provided specification. In other words,
identify key "gaps" in the specification. Focus *only* on properties/invariants/requirements which
can be stated in terms of the component that is the focus of the specification. Do *NOT* consider
interactions between components, as they are out of scope.

After your analysis, formulate these extracted requirements in natural language directives. Each such
directive should be similar to "The implementation must ensure that ...", 
"When X happens, the implementation must ...", etc.

IMPORTANT: If you are unclear as to what component the specification is targeting, as the user for help.

IMPORTANT: You *MUST* consult the user if you are uncertain about specific requirements or the meaning
behind parts of the system document or specification. You MUST use the human_in_the_loop tool to clarify any
uncertainties you may have or ambiguities in the specifications.
"""

def get_requirements(
    options: RAGDBOptions,
    llm: BaseChatModel,
    sys_doc: InputFileLike,
    spec_file: InputFileLike,
) -> list[str]:
    tools = [
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