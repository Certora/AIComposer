from typing import Callable, Awaitable, NotRequired

from langgraph.graph import MessagesState

from graphcore.graph import FlowInput

from composer.spec.context import WorkspaceContext, CVLOnlyBuilder, CVLGeneration
from composer.spec.graph_builder import bind_standard
from composer.spec.trunner import run_to_completion
from composer.spec.draft import get_rough_draft_tools
from composer.workflow.services import get_checkpointer

type ResearchTool = Callable[[str], Awaitable[str]]

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
    ctx: WorkspaceContext[CVLGeneration],
    builder: CVLOnlyBuilder,
) -> ResearchTool:

    class ST(MessagesState):
        memory: NotRequired[str]
        result: NotRequired[str]
        did_read: NotRequired[bool]

    def did_read_draft(s: ST, _) -> str | None:
        if s.get("did_read", None) is None:
            return "You must read your rough draft before delivering your answer"
        return None

    rough_draft_tools = get_rough_draft_tools(ST)

    workflow = bind_standard(
        builder, ST, "Your research findings", validator=did_read_draft
    ).with_input(
        FlowInput
    ).with_tools(
        [*ctx.kb_tools(read_only=True), *rough_draft_tools]
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_initial_prompt(
        CVL_RESEARCH_INITIAL_PROMPT
    ).compile_async(
        checkpointer=get_checkpointer()
    )

    async def research(question: str) -> str:
        res = await run_to_completion(
            workflow,
            FlowInput(input=[question]),
            thread_id=ctx.uniq_thread_id(),
        )
        assert "result" in res
        return res["result"]

    return research
