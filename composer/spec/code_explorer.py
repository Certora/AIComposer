"""
Reusable code exploration sub-agent tool.

Creates a BaseTool that delegates focused source code questions to a
sub-agent with file system tools (list_files, get_file, grep_files).
"""

import uuid
from typing import NotRequired, override, Protocol

from pydantic import Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from graphcore.graph import Builder, FlowInput, MessagesState
from graphcore.tools.schemas import WithAsyncImplementation
from graphcore.tools.vfs import fs_tools

from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.templates.loader import load_jinja_template
from composer.spec.tool_env import BaseSourceTools, BasicAgentTools
from composer.spec.util import uniq_thread_id


CODE_EXPLORER_SYS_PROMPT = """\
You are a code exploration assistant analyzing smart contract source code.
You have access to file tools (list_files, get_file, grep_files) to explore the project.

Your job is to answer a specific question about the codebase thoroughly and precisely.

Guidelines:
- Ground every claim in what you find in the source code.
- Include relevant function signatures, state variable declarations, or code snippets in your answer.
- If the question asks about behavior, trace through the actual implementation rather than speculating.
- Be concise: the caller needs a dense, actionable answer, not a walkthrough of your exploration process.

When complete, deliver your answer via the `result` tool.
"""


class _ExplorerST(MessagesState):
    result: NotRequired[str]

class CodeExplorerEnv(BaseSourceTools, BasicAgentTools, Protocol):
    pass

def code_explorer_tool(env: CodeExplorerEnv) -> BaseTool:
    """Create a code exploration sub-agent tool from a pre-configured builder.

    Args:
        builder: Builder with LLM and file tools already bound.

    Returns:
        A BaseTool named ``explore_code``.
    """
    graph = bind_standard(
        env.builder, _ExplorerST, "Your findings about the source code"
    ).with_input(
        FlowInput
    ).with_tools(
        env.base_source_tools
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
            st = await run_to_completion(
                graph=graph,
                context=None,
                description=f"Code Explorer: {self.question}",
                input=FlowInput(
                    input=[self.question]
                ),
                recursion_limit=100,
                thread_id=uniq_thread_id("code_explorer")
            )
            assert "result" in st
            return st["result"]

    return ExploreCodeSchema.as_tool("explore_code")
