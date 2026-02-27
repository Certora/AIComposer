from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from graphcore.graph import build_workflow, FlowInput

from composer.assistant.types import OrchestratorState, OrchestratorContext
from composer.assistant.tools import build_tools
from composer.io.ide_bridge import IDEBridge
from composer.templates.loader import load_jinja_template


def build_orchestrator(
    llm: BaseChatModel,
    workspace: Path,
    ide: IDEBridge | None,
) -> CompiledStateGraph[OrchestratorState, OrchestratorContext, FlowInput, Any]:
    """Build and compile the orchestrator agent graph."""
    tools = build_tools(workspace)

    sys_prompt = load_jinja_template("orchestrator_system.j2")
    initial_prompt = load_jinja_template(
        "orchestrator_prompt.j2",
        workspace=str(workspace),
        ide_connected=ide is not None,
    )

    graph_builder = build_workflow(
        state_class=OrchestratorState,
        input_type=FlowInput,
        tools_list=tools,
        sys_prompt=sys_prompt,
        initial_prompt=initial_prompt,
        output_key="result",
        unbound_llm=llm,
        context_schema=OrchestratorContext,
    )

    return graph_builder[0].compile(checkpointer=MemorySaver())
