from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from graphcore.graph import Builder, FlowInput

from composer.assistant.types import (
    ConversationTurn,
    OrchestratorContext,
    OrchestratorState,
)
from composer.assistant.tools import build_tools
from composer.templates.loader import load_jinja_template


def _conversation_node(state: OrchestratorState) -> dict[str, list[BaseMessage]]:
    last = state["messages"][-1]
    assert isinstance(last, AIMessage)
    response = interrupt(ConversationTurn(message=last))
    return {"messages": [HumanMessage(content=response)]}


def build_orchestrator(
    workspace: Path,
    llm: BaseChatModel,
) -> CompiledStateGraph[OrchestratorState, OrchestratorContext, FlowInput, Any]:
    """Build and compile the orchestrator agent graph."""
    tools = build_tools(workspace)

    sys_prompt = load_jinja_template("orchestrator_system.j2")
    initial_prompt = load_jinja_template(
        "orchestrator_prompt.j2",
        workspace=str(workspace),
    )

    graph = (
        Builder()
        .with_state(OrchestratorState)
        .with_input(FlowInput)
        .with_context(OrchestratorContext)
        .with_tools(tools)
        .with_sys_prompt(sys_prompt)
        .with_initial_prompt(initial_prompt)
        .with_output_key("result")
        .with_llm(llm)
        .with_conversation(fn=_conversation_node)
        .compile(checkpointer=MemorySaver())
    )

    return graph
