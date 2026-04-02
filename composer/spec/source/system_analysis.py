from typing import Protocol
from langchain_core.tools import BaseTool
from composer.spec.tool_env import SourceTools, BasicAgentTools

from composer.spec.system_model import SourceApplication
from composer.spec.context import WorkflowContext, SourceCode, CacheKey
from composer.spec.system_analysis import run_component_analysis as wrapped_analysis


class AnalysisEnv(BasicAgentTools, Protocol):
    @property
    def system_analysis_tools(self) -> tuple[BaseTool, ...]:
        ...

SOURCE_ANALYSIS_KEY = CacheKey[None, SourceApplication]("source-analysis")

async def run_component_analysis(
    context: WorkflowContext[None],
    input: SourceCode,
    env: AnalysisEnv
) -> SourceApplication | None:
    child_ctx = context.child(SOURCE_ANALYSIS_KEY)
    return await wrapped_analysis(
        ty=SourceApplication,
        child_ctxt=child_ctx,
        env=env,
        extra_input=[
            f"The main entry point of this application has been explicitly identified as {input.contract_name} at relative path {input.relative_path}"
        ],
        input=input
    )
