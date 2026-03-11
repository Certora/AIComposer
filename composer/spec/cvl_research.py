"""
CVL research sub-agent: answers questions about CVL by searching the manual and knowledge base.
"""

from typing import Callable, Awaitable, NotRequired, Protocol

from pydantic import Field

from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState
from langgraph.types import Checkpointer

from graphcore.graph import FlowInput
from graphcore.tools.schemas import WithAsyncImplementation

from composer.spec.context import ThreadProvider, CVLOnlyBuilder
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.tools.thinking import get_rough_draft_tools, RoughDraftState

type ResearchTool = Callable[[str], Awaitable[str]]

CVL_RESEARCH_BASE_DOC = (
    "Delegate a question about CVL syntax, patterns, or techniques to a research sub-agent. "
    "The sub-agent searches the CVL manual and knowledge base, then delivers a synthesized answer.\n\n"
    "Use this when you need to understand how to express something in CVL, what patterns to "
    "use, or how a specific CVL feature works. "
    "Do not use this tool to ask questions about how to use other tools available to you; it only understands "
    "questions related to CVL authorship."
)


class ResearchContext(ThreadProvider, Protocol):
    """Narrow protocol for what cvl_researcher needs — satisfied by WorkflowContext."""
    def kb_tools(self, read_only: bool) -> list[BaseTool]: ...
    @property
    def checkpointer(self) -> Checkpointer: ...


CVL_RESEARCH_INITIAL_PROMPT = """\
Answer the following question about the Certora Verification Language (CVL).

Follow these steps:

## Step 1: Search
Search the CVL manual and knowledge base for relevant information. Use multiple searches
with different queries to ensure thorough coverage. Prefer semantic search for conceptual
questions and keyword search for syntax questions.

## Step 2: Draft
Using the write_rough_draft tool, organize your findings into a structured draft answer.
Include:
- Direct quotes or paraphrases from the manual with section references
- Relevant knowledge base articles, if any
- Concrete CVL code examples where appropriate

## Step 3: Review
Read back your rough draft. Verify that:
- Every claim is backed by manual/KB evidence (not speculation)
- Code examples follow CVL guidelines (mathint defaults, envfree rules, etc.)
- The answer directly addresses the question asked

## Step 4: Deliver
Output your final answer using the result tool. Be concise and actionable — the caller
needs a dense, precise answer they can immediately apply.
"""


def cvl_researcher(
    ctx: ResearchContext,
    builder: CVLOnlyBuilder,
) -> ResearchTool:

    class CVLResearchInput(FlowInput, RoughDraftState):
        pass

    class ST(MessagesState, RoughDraftState):
        result: NotRequired[str]

    def did_read_draft(s: ST, _) -> str | None:
        if s.get("did_read", None) is None:
            return "You must read your rough draft before delivering your answer"
        return None

    rough_draft_tools = get_rough_draft_tools(ST)

    workflow = bind_standard(
        builder, ST, "Your research findings", validator=did_read_draft
    ).with_input(
        CVLResearchInput
    ).with_tools(
        [*ctx.kb_tools(read_only=True), *rough_draft_tools]
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_initial_prompt(
        CVL_RESEARCH_INITIAL_PROMPT
    ).compile_async(
        checkpointer=ctx.checkpointer
    )

    async def research(question: str) -> str:
        res = await run_to_completion(
            workflow,
            CVLResearchInput(input=[question], did_read=False, memory=None),
            thread_id=ctx.uniq_thread_id(),
            description="CVL research",
        )
        assert "result" in res
        return res["result"]

    return research


def cvl_research_tool(
    ctx: ResearchContext,
    builder: CVLOnlyBuilder,
    doc: str,
) -> BaseTool:
    """Create a CVL research BaseTool with a caller-provided docstring."""
    researcher = cvl_researcher(ctx, builder)

    class CVLResearchSchema(WithAsyncImplementation[str]):
        __doc__ = doc
        question: str = Field(
            description="A specific question about CVL. "
            "Good: 'How do I use ghost state to track cumulative token transfers?' "
            "Good: 'What is the correct syntax for a preserved block with require statements?' "
            "Bad: 'How does the withdraw function work?' (not a CVL question)"
        )
        async def run(self) -> str:
            return await researcher(self.question)

    return CVLResearchSchema.as_tool("cvl_research")
