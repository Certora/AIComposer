from typing import NotRequired, Callable, Any
from dataclasses import dataclass
import uuid
import pathlib

from pydantic import BaseModel, Field

from graphcore.graph import FlowInput, build_async_workflow
from graphcore.tools.results import result_tool_generator
from graphcore.tools.memory import memory_tool, MemoryBackend

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from composer.audit.types import InputFileLike
from composer.audit.db import ResumeArtifact
from composer.input.types import RAGDBOptions
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.workflow.factories import get_checkpointer
from composer.tools.search import cvl_manual_search
from composer.tools.thinking import explicit_thinking, RoughDraftState, get_rough_draft_tools
from composer.templates.loader import load_jinja_template
from composer.natreq.automation import requirements_oracle
from composer.human.types import HumanInteractionType
from composer.io.protocol import IOHandler
from composer.io.context import with_handler, run_graph
from composer.io.event_handler import NullEventHandler


@dataclass
class ExtractionResult:
    """Result of requirements extraction, including the thread_id for post-mortem introspection."""
    reqs: list[str]
    thread_id: str


class ExtractionState(MessagesState, RoughDraftState):
    reqs: NotRequired[list[str]]

class ExtractionInput(FlowInput, RoughDraftState):
    pass

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
    4. Resolving apparent conflicts between the system document and the specification
    5. Clarifying whether passages in the system doc are exposition vs. code requirements

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
        "type": "extraction_question",
        "question": question,
        "context": context
    })
    return response

def _extraction_res_checker(
    st: ExtractionState,
    _r: list[str],
    _id: str
) -> str | None:
    if "memory" in st and not st.get("did_read", False):
        return "Completion REJECTED: You must read your rough draft before submitting. Call read_rough_draft first."
    return None

results_tool = result_tool_generator(
    "reqs",
    (list[str], "The list of natural language requirements you extracted during this process."),
    """
Tool used to indicate your analysis is complete and communicate the generated requirements back to the user.

REMINDER: You should call this tool only AFTER you have updated your memories.
""",
    validator=(ExtractionState, _extraction_res_checker)
)


system_prompt = load_jinja_template("req_role_prompt.j2")

initial_prompt = load_jinja_template("req_extraction_prompt.j2")


class OracleHandler:
    """Delegation-based wrapper around any IOHandler.

    Forwards all methods except human_interaction, where extraction_question
    interrupts are routed to the oracle (if available).
    """

    def __init__(self, inner: IOHandler, oracle: Callable[[tuple[str, str]], str] | None):
        self._inner = inner
        self._oracle = oracle

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    async def human_interaction(self, ty: HumanInteractionType, debug_thunk: Callable[[], None]) -> str:
        if ty["type"] == "extraction_question" and self._oracle is not None:
            print(f"Calling oracle...\nQuestion: {ty['question']}\nContext: {ty['context']}")
            resp = self._oracle((ty["context"], ty["question"]))
            print(f"Oracle response: {resp}")
            return resp
        return await self._inner.human_interaction(ty, debug_thunk)


async def get_requirements(
    io: IOHandler,
    options: RAGDBOptions,
    llm: BaseChatModel,
    sys_doc: InputFileLike,
    spec_file: InputFileLike,
    mem_backend: MemoryBackend,
    resume_artifact: ResumeArtifact | None,
    oracle: list[str]
) -> ExtractionResult:
    tools = [
        memory_tool(mem_backend),
        results_tool,
        human_in_the_loop,
        cvl_manual_search(ExtractionContext),
        explicit_thinking,
        *get_rough_draft_tools(ExtractionState),
    ]
    built : CompiledStateGraph[ExtractionState, ExtractionContext, ExtractionInput, Any] = build_async_workflow(
        state_class=ExtractionState,
        context_schema=ExtractionContext,
        input_type=ExtractionInput,
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

    input_text : list[str | dict] = [
        "The system document is as follows:",
        sys_doc.string_contents,
        "The spec file is as follows:",
        spec_file.string_contents
    ]

    if resume_artifact is not None:
        input_text.append("""
You have previously performed this analysis on a prior version of the spec file. You have access to the
memories you generated during that prior analysis. Be sure to consult those memories to inform your analysis
of the system document. In addition, be sure to analyze the difference between the two specification files,
being sure to determine which natural language requirements are no longer needed (as they are now covered by the
spec).
""")
        input_text.append("The OLD spec file is as follows:")
        input_text.append(
            resume_artifact.spec_file
        )

    req_oracle : Callable[[tuple[str, str]], str] | None = None
    if oracle is not None and len(oracle):
        req_oracle = requirements_oracle(
            llm,
            [ pathlib.Path(p) for p in oracle ]
        )

    graph_input = ExtractionInput(input=input_text, memory=None, did_read=False)

    handler = OracleHandler(io, req_oracle)
    async with with_handler(handler, NullEventHandler()):  # type: ignore[arg-type]
        final_state = await run_graph(built, ExtractionContext(rag_db=db), graph_input, config, description="Requirements extraction")
    assert "reqs" in final_state
    return ExtractionResult(reqs=final_state["reqs"], thread_id=thread_id)
