from typing import NotRequired, Protocol


from graphcore.graph import MessagesState, FlowInput
from langchain_core.tools import BaseTool


from composer.spec.context import (
    WorkflowContext, CacheKey,
    SystemDoc
)
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.system_model import Application
from composer.spec.tool_env import BasicAgentTools
from composer.spec.system_analysis import run_component_analysis as wrapped_analysis


SOURCE_ANALYSIS_KEY = CacheKey[None, Application]("source-analysis")

DESCRIPTION = "Component analysis"


class AnalysisEnv(BasicAgentTools, Protocol):
    @property
    def system_analysis_tools(self) -> tuple[BaseTool, ...]:
        ...

async def run_component_analysis(
    context: WorkflowContext[None],
    input: SystemDoc,
    tools: AnalysisEnv
) -> Application | None:
    """Analyze application components from a system doc and optionally source code."""
    return await wrapped_analysis(
        ty=Application,
        child_ctxt=context.child(SOURCE_ANALYSIS_KEY),
        env=tools,
        extra_input=[],
        input=input,
    )
