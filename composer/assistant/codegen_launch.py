import traceback
from dataclasses import dataclass, field
import pathlib
from typing import Optional

from composer.assistant.launch_args import LaunchCodegenArgs, LaunchResumeArgs, CommonCodeGen
from composer.assistant.types import OrchestratorContext
from composer.input.files import upload_configuration
from composer.input.types import ResumeFSData
from composer.ui.codegen_rich import CodeGenRichApp
from composer.io.protocol import WorkflowPurpose
from composer.workflow.executor import execute_ai_composer_workflow
from composer.workflow.types import WorkflowResult, WorkflowSuccess, WorkflowFailure, WorkflowCrash
from composer.workflow.services import create_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class CodegenWorkflowArgs:
    """Satisfies WorkflowOptions protocol for programmatic invocation."""
    rag_db: str
    prover_capture_output: bool = True
    prover_keep_folders: bool = False
    local_prover: bool = False
    debug_prompt_override: Optional[str] = None
    recursion_limit: int = 100
    summarization_threshold: Optional[int] = 50
    requirements_oracle: list[str] = field(default_factory=list)
    set_reqs: Optional[str] = None
    skip_reqs: bool = False
    checkpoint_id: Optional[str] = None
    thread_id: Optional[str] = None
    model: str = "claude-opus-4-6"
    tokens: int = 10_000
    thinking_tokens: int = 2048
    memory_tool: bool = True
    interleaved_thinking: bool = False
    # ``prover_conf`` matches ``WorkflowOptions.prover_conf`` shape:
    # an inline dict resolved at the agent boundary (or ``None``).
    # Forwarded straight to the executor — no path-loading inline.
    prover_conf: Optional[dict] = None
    cache_namespace: Optional[str] = None


def _codegen_args(ctx: OrchestratorContext, cg: CommonCodeGen, thread_id: str | None = None) -> CodegenWorkflowArgs:
    return CodegenWorkflowArgs(
        rag_db=ctx.config.rag_db,
        model=ctx.config.model,
        tokens=ctx.config.tokens,
        thinking_tokens=ctx.config.thinking_tokens,
        memory_tool=ctx.config.memory_tool,
        recursion_limit=200,
        debug_prompt_override=cg.prompt_addition,
        prover_conf=cg.prover_conf,
        cache_namespace=cg.cache_namespace,
        thread_id=thread_id
    )


def _format_result(
    label: str,
    tid: str,
    result: WorkflowResult,
    memory_namespace: str | None,
    natreq_tid: str | None,
) -> str:
    ns_info = f" Memory namespace: {memory_namespace}." if memory_namespace else ""
    natreq_info = f" NatReq thread ID: {natreq_tid}." if natreq_tid else ""
    match result:
        case WorkflowCrash(resume_work_key=key, error=error):
            tb = "".join(traceback.format_exception(error))
            key_info = f" Resume work key: {key}." if key else ""
            return (
                f"{label} crashed with {type(error).__name__}: "
                f"{error}\nTraceback:\n{tb}\nThread ID: {tid}.{ns_info}{natreq_info}{key_info}"
            )
        case WorkflowSuccess():
            return (
                f"{label} completed successfully. Thread ID: {tid}.{ns_info}{natreq_info} "
                f"Save this to /memories/last_run.json for future resume."
            )
        case WorkflowFailure():
            return f"{label} finished without producing output. Thread ID: {tid}.{ns_info}{natreq_info}"


# ---------------------------------------------------------------------------
# Launch functions
# ---------------------------------------------------------------------------

async def launch_codegen_workflow(
    args: LaunchCodegenArgs,
    ctx: OrchestratorContext,
) -> str:
    input_data = await upload_configuration(args.launch_config, pathlib.Path(args.source_root))

    wf_args = _codegen_args(ctx, args)
    llm = create_llm(wf_args)

    app = CodeGenRichApp(ide=ctx.ide)

    async def work() -> None:
        app.result = await execute_ai_composer_workflow(
            handler=app, llm=llm, input=input_data,
            workflow_options=wf_args,
            memory_namespace=args.memory_namespace,
            resume_work_key=args.resume_work_key,
        )

    app.set_work(work)
    await app.run_async()

    tid = app.workflow_threads.get(WorkflowPurpose.CODEGEN, "unknown")
    return _format_result(
        label="Code generation",
        tid=tid,
        result=app.result or WorkflowFailure(),
        memory_namespace=args.memory_namespace,
        natreq_tid=app.workflow_threads.get(WorkflowPurpose.NATREQ),
    )


async def launch_resume_workflow(
    args: LaunchResumeArgs,
    ctx: OrchestratorContext,
) -> str:
    input_data = ResumeFSData(
        thread_id=args.thread_id,
        file_path=str(ctx.workspace / args.working_dir),
        comments=args.commentary or None,
        new_system=None,
    )

    wf_args = _codegen_args(ctx, args)
    llm = create_llm(wf_args)

    app = CodeGenRichApp(ide=ctx.ide)

    async def work() -> None:
        app.result = await execute_ai_composer_workflow(
            handler=app, llm=llm, input=input_data,
            workflow_options=wf_args,
            memory_namespace=args.memory_namespace,
            resume_work_key=args.resume_work_key,
        )

    app.set_work(work)
    await app.run_async()

    tid = app.workflow_threads.get(WorkflowPurpose.CODEGEN, args.thread_id)
    return _format_result(
        label="Resume",
        tid=tid,
        result=app.result or WorkflowFailure(),
        memory_namespace=args.memory_namespace,
        natreq_tid=app.workflow_threads.get(WorkflowPurpose.NATREQ),
    )
