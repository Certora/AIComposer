from composer.spec.system_model import SourceApplication
from composer.spec.context import WorkflowContext, SourceCode, CacheKey
from composer.spec.service_host import ServiceHost
from composer.spec.system_analysis import run_component_analysis as wrapped_analysis


SOURCE_ANALYSIS_KEY = CacheKey[None, SourceApplication]("source-analysis")


async def run_component_analysis(
    context: WorkflowContext[None],
    input: SourceCode,
    env: ServiceHost,
) -> SourceApplication | None:
    child_ctx = context.child(SOURCE_ANALYSIS_KEY)
    return await wrapped_analysis(
        ty=SourceApplication,
        child_ctxt=child_ctx,
        env=env,
        extra_input=[
            f"The main entry point of this application has been explicitly identified as {input.contract_name} at relative path {input.relative_path}. "
            "Your output MUST contain an explicit contract instance with this solidity identifier."
        ],
        input=input,
        expected_main_id=input.contract_name
    )
