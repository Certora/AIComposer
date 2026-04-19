"""
CVL research sub-agent: answers questions about CVL by searching the manual and knowledge base.
"""

from typing import Any, Callable, Awaitable, NotRequired, Protocol, override, TypedDict

from pydantic import Field, BaseModel

from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph

from graphcore.graph import Builder, FlowInput
from graphcore.tools.schemas import WithAsyncImplementation

from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.tools.thinking import get_rough_draft_tools, RoughDraftState
from composer.spec.tool_env import BaseRAGTools, BasicAgentTools
from composer.spec.util import uniq_thread_id
from composer.spec.agent_index import AgentIndex, IndexedTool
from composer.spec.gen_types import TypedTemplate
from composer.ui.tool_display import tool_display_of, CommonTools

DEFAULT_CVL_AGENT_INDEX_NS: tuple[str, ...] = ("cvl_research", "cached")

CVL_RESEARCH_BASE_DOC = (
    "Delegate a question about CVL syntax, patterns, or techniques to a research sub-agent. "
    "The sub-agent searches the CVL manual and knowledge base, then delivers a synthesized answer.\n\n"
    "Use this when you need to understand how to express something in CVL, what patterns to "
    "use, or how a specific CVL feature works. "
    "Do not use this tool to ask questions about how to use other tools available to you; it only understands "
    "questions related to CVL authorship."
)

class CVLResearchEnv(BaseRAGTools, BasicAgentTools, Protocol):
    pass

class CVLResearchSysParams(TypedDict):
    context_instructions: str | None

_ResearchSys = TypedTemplate[CVLResearchSysParams]("cvl_research_system_prompt.j2")

class IndexedCVLResearcherEnv(CVLResearchEnv, Protocol):
    @property
    def agent_index(self) -> AgentIndex:
        ...

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

def _build_research_graph(
    builder: Builder,
    with_index: bool
) -> _CompiledResearchGraph:
    rough_draft_tools = get_rough_draft_tools(_CVLResearchST)

    sys_templ = _ResearchSys.bind({
        "context_instructions": AgentIndex.WITH_INDEX_SYS_COMMON if with_index else None
    })

    graph = bind_standard(
        builder, _CVLResearchST, "Your research findings", validator=_did_read_draft
    ).with_input(
        _CVLResearchInput
    ).with_tools(
        rough_draft_tools
    ).inject(
        lambda g: sys_templ.render_to(g.with_sys_prompt_template)
    ).with_initial_prompt(
        "Answer the following question"
    ).compile_async()
    return graph

class CVLResearchSchemaBase(BaseModel):
    question: str = Field(
        description="A specific question about CVL. "
        "Good: 'How do I use ghost state to track cumulative token transfers?' "
        "Good: 'What is the correct syntax for a preserved block with require statements?' "
        "Bad: 'How does the withdraw function work?' (not a CVL question)"
    )


def _build_research_tool(
    builder: Builder,
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
    graph = _build_research_graph(builder, False)

    @tool_display_of(CommonTools.cvl_research)
    class CVLResearchSchema(CVLResearchSchemaBase, WithAsyncImplementation[str]):
        __doc__ = doc

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
    env: CVLResearchEnv,
    doc: str,
) -> BaseTool:
    """Create a CVL research BaseTool using a WorkflowContext."""
    enriched = env.builder.with_tools(env.base_rag_tools)

    async def runner(
        graph: _CompiledResearchGraph, inp: _CVLResearchInput,
    ) -> _CVLResearchST:
        return await run_to_completion(
            graph, inp,
            thread_id=uniq_thread_id("cvl-research"),
            description="CVL research",
        )

    return _build_research_tool(enriched, runner, doc)

def indexed_cvl_research_tool(
    env: IndexedCVLResearcherEnv,
    doc: str
) -> BaseTool:
    graph = _build_research_graph(
        env.builder.with_tools(env.base_rag_tools),
        with_index=True
    )
    @tool_display_of(CommonTools.cvl_research)
    class CVLResearcher(CVLResearchSchemaBase, IndexedTool[AgentIndex]):
        __doc__ = doc
        
        @override
        def get_question(self) -> str:
            return self.question
        
        @override
        async def answer_question(self, context: list[str]) -> str:
            res = await run_to_completion(
                graph = graph,
                context=None,
                description="CVL Researcher",
                thread_id=uniq_thread_id("cvl-research"),
                input=_CVLResearchInput(input=[
                    self.question,
                    *context
                ], did_read=False, memory=None)
            )
            assert "result" in res
            return res["result"]

    return CVLResearcher.bind(env.agent_index).as_tool("cvl_research")

