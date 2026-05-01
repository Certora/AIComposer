from typing import NotRequired, Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from graphcore.graph import Builder, FlowInput
from graphcore.tools.results import result_tool_generator
from graphcore.tools.memory import async_memory_tool, AsyncMemoryBackend

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from langgraph.graph import MessagesState
from langgraph.types import interrupt

from composer.input.types import InputFileLike, TextInputFile
from composer.audit.store import ResumeArtifact
from composer.spec.cvl_research import CVL_RESEARCH_BASE_DOC
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.templates.loader import load_jinja_template
from composer.human.types import HumanInteractionType
from composer.io.protocol import IOHandler
from composer.io.context import with_handler, run_graph
from composer.io.event_handler import NullEventHandler
from composer.ui.tool_display import tool_display
from composer.spec.util import uniq_thread_id


@dataclass
class ExtractionResult:
    """Result of requirements extraction, including the thread_id for post-mortem introspection."""
    reqs: list[str]
    thread_id: str


class ExtractionState(MessagesState, RoughDraftState):
    reqs: NotRequired[list[str]]

class ExtractionInput(FlowInput, RoughDraftState):
    pass

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

@tool_display(
    lambda p: (
        f"Asking for input: {p['question']}"
        if p.get("question") else "Asking for input"
    ),
    None,
)
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


async def get_requirements(
    io: IOHandler,
    cvl_builder: Builder[None, None, None],

    sys_doc: InputFileLike,
    specs: list[tuple[str, TextInputFile]],
    mem_backend: AsyncMemoryBackend,
    resume_artifact: ResumeArtifact | None,
) -> ExtractionResult:
    """Extract natural-language requirements that the spec leaves implicit.

    ``rag_db`` / ``indexed_store`` / ``checkpointer`` are passed in by the
    executor (which already owns those connections for the main codegen
    graph) instead of being opened privately here — saves a duplicate set
    of pool connections per run and lets the CVL research sub-agent share
    the parent's checkpointer for its own thread state."""

    tools = [
        async_memory_tool(mem_backend),
        results_tool,
        human_in_the_loop,
        *get_rough_draft_tools(ExtractionState),
    ]

    built = (
        cvl_builder
        .with_state(ExtractionState)
        .with_input(ExtractionInput)
        .with_tools(tools)
        .with_output_key("reqs")
        .with_initial_prompt_template("req_extraction_prompt.j2")
        .with_sys_prompt_template("req_role_prompt.j2")
    ).compile_async()

    thread_id = uniq_thread_id("extraction")

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # ``sys_doc`` is ``InputFileLike`` (may be PDF). Use the
    # universal document-block path so the LLM ingests text and
    # binary system docs identically.
    input_text : list[str | dict] = [
        "The system document is as follows:",
        sys_doc.to_document_dict(),
    ]
    if len(specs) == 1:
        input_text.append("The spec file is as follows:")
        input_text.append(specs[0][1].string_contents)
    else:
        input_text.append(
            f"The contract has {len(specs)} spec files describing its behavior. "
            f"Consider them collectively as the formal specification; requirements "
            f"you extract should cover the contract as a whole."
        )
        for (vfs_path, f) in specs:
            input_text.append(f"Spec file at `{vfs_path}`:")
            input_text.append(f.string_contents)

    if resume_artifact is not None:
        input_text.append("""
You have previously performed this analysis on a prior version of the spec file(s). You have access to the
memories you generated during that prior analysis. Be sure to consult those memories to inform your analysis
of the system document. In addition, be sure to analyze the difference between the old and new versions of
each spec file, being sure to determine which natural language requirements are no longer needed (as they
are now covered by the spec).
""")
        if len(resume_artifact.specs) == 1:
            input_text.append("The OLD spec file is as follows:")
            input_text.append(resume_artifact.specs[0].string_contents)
        else:
            for entry in resume_artifact.specs:
                input_text.append(f"OLD spec file at `{entry.vfs_path}`:")
                input_text.append(entry.string_contents)

    graph_input = ExtractionInput(input=input_text, memory=None, did_read=False)

    async with with_handler(io, NullEventHandler()):  # type: ignore[arg-type]
        final_state = await run_graph(built, None, graph_input, config, description="Requirements extraction")
    assert "reqs" in final_state
    return ExtractionResult(reqs=final_state["reqs"], thread_id=thread_id)
