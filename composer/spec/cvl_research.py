"""
CVL research sub-agent: answers questions about CVL by searching the manual and knowledge base.
"""

import uuid
from typing import Any, Callable, Awaitable, NotRequired, Protocol, override, cast

from pydantic import Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from graphcore.graph import Builder, FlowInput
from graphcore.tools.schemas import WithAsyncImplementation

from composer.spec.context import ThreadProvider, CVLOnlyBuilder
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.tools.thinking import get_rough_draft_tools, RoughDraftState
from composer.templates.loader import load_jinja_template

CVL_RESEARCH_BASE_DOC = (
    "Delegate a question about CVL syntax, patterns, or techniques to a research sub-agent. "
    "The sub-agent searches the CVL manual and knowledge base, then delivers a synthesized answer.\n\n"
    "Use this when you need to understand how to express something in CVL, what patterns to "
    "use, or how a specific CVL feature works. "
    "Do not use this tool to ask questions about how to use other tools available to you; it only understands "
    "questions related to CVL authorship."
)


class ResearchContext(ThreadProvider, Protocol):
    """Narrow protocol for what the context-based research tool needs."""
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


# ---------------------------------------------------------------------------
# Shared core
# ---------------------------------------------------------------------------

class _CVLResearchInput(FlowInput, RoughDraftState):
    pass


class _CVLResearchST(MessagesState, RoughDraftState):
    result: NotRequired[str]


_CompiledResearchGraph = CompiledStateGraph[_CVLResearchST, None, _CVLResearchInput, Any]

type GraphRunner = Callable[
    [_CompiledResearchGraph, _CVLResearchInput],
    Awaitable[_CVLResearchST],
]


def _did_read_draft(s: _CVLResearchST, _: Any) -> str | None:
    if s.get("did_read", None) is None:
        return "You must read your rough draft before delivering your answer"
    return None


def _build_research_tool(
    builder: Builder,
    checkpointer: Checkpointer,
    runner: GraphRunner,
    doc: str,
) -> BaseTool:
    """Build a CVL research BaseTool.

    Args:
        builder: Builder with LLM and all external tools (CVL manual, KB, etc.)
            already bound.
        checkpointer: Checkpointer for the sub-agent graph.
        runner: How to invoke the compiled graph. Thread ID management is
            the runner's responsibility.
        doc: Docstring for the tool schema.
    """
    rough_draft_tools = get_rough_draft_tools(_CVLResearchST)

    graph = bind_standard(
        builder, _CVLResearchST, "Your research findings", validator=_did_read_draft
    ).with_input(
        _CVLResearchInput
    ).with_tools(
        rough_draft_tools
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_initial_prompt(
        CVL_RESEARCH_INITIAL_PROMPT
    ).compile_async(
        checkpointer=checkpointer
    )

    class CVLResearchSchema(WithAsyncImplementation[str]):
        __doc__ = doc
        question: str = Field(
            description="A specific question about CVL. "
            "Good: 'How do I use ghost state to track cumulative token transfers?' "
            "Good: 'What is the correct syntax for a preserved block with require statements?' "
            "Bad: 'How does the withdraw function work?' (not a CVL question)"
        )

        @override
        async def run(self) -> str:
            st = await runner(
                graph,
                _CVLResearchInput(input=[self.question], did_read=False, memory=None),
            )
            assert "result" in st
            return st["result"]

    return CVLResearchSchema.as_tool("cvl_research")


# ---------------------------------------------------------------------------
# Public API — context-based (existing callers)
# ---------------------------------------------------------------------------

def cvl_research_tool(
    ctx: ResearchContext,
    builder: CVLOnlyBuilder,
    doc: str,
) -> BaseTool:
    """Create a CVL research BaseTool using a WorkflowContext."""
    enriched = builder.with_tools(ctx.kb_tools(read_only=True))

    async def runner(
        graph: _CompiledResearchGraph, inp: _CVLResearchInput,
    ) -> _CVLResearchST:
        return await run_to_completion(
            graph, inp,
            thread_id=ctx.uniq_thread_id(),
            description="CVL research",
        )

    return _build_research_tool(enriched, ctx.checkpointer, runner, doc)


# ---------------------------------------------------------------------------
# Public API — standalone (no IO handlers needed)
# ---------------------------------------------------------------------------

def standalone_cvl_research_tool(
    llm: BaseChatModel,
    cvl_tools: list[BaseTool],
) -> BaseTool:
    """Create a CVL research tool that works without IO handlers.

    Args:
        llm: Language model for the research sub-agent.
        cvl_tools: CVL manual search tools (from ``cvl_manual_tools``).

    Returns:
        A BaseTool named ``cvl_research``.
    """
    builder = Builder().with_llm(llm).with_tools(cvl_tools).with_loader(load_jinja_template)

    async def runner(
        graph: _CompiledResearchGraph, inp: _CVLResearchInput,
    ) -> _CVLResearchST:
        res = await graph.ainvoke(
            inp,
            config={
                "configurable": {"thread_id": uuid.uuid4().hex},
                "recursion_limit": 100,
            },
        )
        assert isinstance(res, dict)
        return cast(_CVLResearchST, res)

    return _build_research_tool(builder, InMemorySaver(), runner, CVL_RESEARCH_BASE_DOC)
