from typing import Any, Protocol

from langchain_core.tools import BaseTool

from composer.spec.context import (
    WorkflowContext, CacheKey,
    SystemDoc
)
from composer.spec.natspec.task_description import MentalModel
from composer.spec.system_model import NatspecApplication
from composer.spec.tool_env import BasicAgentTools
from composer.spec.system_analysis import run_component_analysis as wrapped_analysis


SOURCE_ANALYSIS_KEY = CacheKey[None, NatspecApplication]("source-analysis")

DESCRIPTION = "Component analysis"


class AnalysisEnv(BasicAgentTools, Protocol):
    @property
    def system_analysis_tools(self) -> tuple[BaseTool, ...]:
        ...

def source_analysis_key[A: NatspecApplication](
    s: MentalModel[A, Any, Any]
) -> CacheKey[None, A]:
    return CacheKey[None, A]("source-analysis")

async def run_component_analysis[A: NatspecApplication](
    context: WorkflowContext[None],
    input: SystemDoc,
    tools: AnalysisEnv,
    mental_model: MentalModel[A, Any, Any],
) -> A | None:
    """Analyze application components from a system doc and optionally source code.

    The concrete application subtype to produce (``Application`` for greenfield,
    ``FromSourceApplication`` for the from-source workflow) is taken from
    ``mental_model.model_ty``.
    """
    return await wrapped_analysis(
        ty=mental_model.model_ty,
        child_ctxt=context.child(source_analysis_key(mental_model)),
        env=tools,
        extra_input=[],
        input=input,
        tagged_contracts=mental_model.from_existing
    )
