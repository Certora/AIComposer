"""
Reusable code exploration sub-agent tool.

Creates a BaseTool that delegates focused source code questions to a
sub-agent with file system tools (list_files, get_file, grep_files).
"""

import uuid
from typing import NotRequired, override

from pydantic import Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from graphcore.graph import Builder, FlowInput, MessagesState
from graphcore.tools.schemas import WithAsyncImplementation
from graphcore.tools.vfs import fs_tools

from composer.spec.graph_builder import bind_standard
from composer.templates.loader import load_jinja_template


CODE_EXPLORER_SYS_PROMPT = """\
You are a code exploration assistant analyzing smart contract source code.
You have access to file tools (list_files, get_file, grep_files) to explore the project.

Your job is to answer a specific question about the codebase thoroughly and precisely.

Guidelines:
- Ground every claim in what you find in the source code.
- Include relevant function signatures, state variable declarations, or code snippets in your answer.
- If the question asks about behavior, trace through the actual implementation rather than speculating.
- Be concise: the caller needs a dense, actionable answer, not a walkthrough of your exploration process.
"""


class _ExplorerST(MessagesState):
    result: NotRequired[str]


def code_explorer_tool_from_builder(builder: Builder) -> BaseTool:
    """Create a code exploration sub-agent tool from a pre-configured builder.

    Args:
        builder: Builder with LLM and file tools already bound.

    Returns:
        A BaseTool named ``explore_code``.
    """
    graph = bind_standard(
        builder, _ExplorerST, "Your findings about the source code"
    ).with_input(
        FlowInput
    ).with_sys_prompt(
        CODE_EXPLORER_SYS_PROMPT
    ).with_initial_prompt(
        "Answer the following question about the source code"
    ).compile_async(
        checkpointer=InMemorySaver()
    )

    class ExploreCodeSchema(WithAsyncImplementation[str]):
        """
        Delegate a focused question about the source code to a code exploration sub-agent.
        The sub-agent has its own conversation thread with file tools (list_files, get_file,
        grep_files) and will return a synthesized answer. Use this instead of reading files
        directly when you need to understand a specific aspect of the codebase.
        """
        question: str = Field(
            description="A specific, focused question about the source code. "
            "Good: 'What state variables does withdraw() modify and how?' "
            "Bad: 'Tell me about the contract' "
            "Bad: 'What is the definition of function X?' (read the source directly)"
        )

        @override
        async def run(self) -> str:
            st = await graph.ainvoke(
                FlowInput(input=[self.question]),
                config={
                    "configurable": {"thread_id": uuid.uuid4().hex},
                    "recursion_limit": 100,
                },
            )
            assert "result" in st
            return st["result"]

    return ExploreCodeSchema.as_tool("explore_code")


def code_explorer_tool(
    llm: BaseChatModel,
    project_path: str,
    forbidden_read: str | None = None,
) -> BaseTool:
    """Create a code exploration sub-agent tool from a local filesystem path.

    Convenience wrapper that builds the underlying Builder with fs_tools
    and delegates to ``code_explorer_tool_from_builder``.

    Args:
        llm: Language model for the sub-agent.
        project_path: Path to the project directory to explore.
        forbidden_read: Optional regex pattern for paths that should not be read.

    Returns:
        A BaseTool named ``explore_code``.
    """
    builder = Builder().with_llm(llm).with_tools(
        fs_tools(project_path, forbidden_read=forbidden_read)
    ).with_loader(load_jinja_template)

    return code_explorer_tool_from_builder(builder)
