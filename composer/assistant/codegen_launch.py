import traceback
from dataclasses import dataclass, field
from typing import Optional

from composer.assistant.launch_args import LaunchCodegenArgs, LaunchResumeArgs
from composer.assistant.types import OrchestratorContext
from composer.audit.db import DEFAULT_CONNECTION as AUDIT_DEFAULT
from composer.input.files import upload_input
from composer.input.types import ResumeFSData
from composer.io.codegen_rich import CodeGenRichApp
from composer.io.ide_bridge import IDEBridge
from composer.workflow.executor import execute_ai_composer_workflow
from composer.workflow.services import create_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _CodegenUploadPaths:
    spec_file: str
    interface_file: str
    system_doc: str


@dataclass
class CodegenWorkflowArgs:
    """Satisfies WorkflowOptions protocol for programmatic invocation."""
    prover_capture_output: bool = True
    prover_keep_folders: bool = False
    debug_prompt_override: Optional[str] = None
    recursion_limit: int = 50
    audit_db: str = ""
    summarization_threshold: Optional[int] = None
    requirements_oracle: list[str] = field(default_factory=list)
    set_reqs: Optional[str] = None
    skip_reqs: bool = False
    rag_db: str = ""
    checkpoint_id: Optional[str] = None
    thread_id: Optional[str] = None
    model: str = "claude-opus-4-6"
    tokens: int = 10_000
    thinking_tokens: int = 2048
    memory_tool: bool = True


class ThreadIdCapture(CodeGenRichApp):
    """CodeGen app that captures the assigned thread ID."""

    def __init__(self, ide: IDEBridge | None = None):
        super().__init__(ide=ide)
        self.captured_thread_id: str | None = None

    async def log_thread_id(self, tid: str, chosen: bool) -> None:
        self.captured_thread_id = tid
        await super().log_thread_id(tid, chosen)


def _codegen_args(ctx: OrchestratorContext) -> CodegenWorkflowArgs:
    return CodegenWorkflowArgs(
        audit_db=AUDIT_DEFAULT,
        rag_db=ctx.config.rag_db,
        model=ctx.config.model,
        tokens=ctx.config.tokens,
        thinking_tokens=ctx.config.thinking_tokens,
        memory_tool=ctx.config.memory_tool,
        recursion_limit=200,
    )


# ---------------------------------------------------------------------------
# Launch functions
# ---------------------------------------------------------------------------

async def launch_codegen_workflow(
    args: LaunchCodegenArgs,
    ctx: OrchestratorContext,
) -> str:
    paths = _CodegenUploadPaths(
        spec_file=str(ctx.workspace / args.spec_file),
        interface_file=str(ctx.workspace / args.interface_file),
        system_doc=str(ctx.workspace / args.system_doc),
    )
    input_data = upload_input(paths)

    wf_args = _codegen_args(ctx)
    llm = create_llm(wf_args)

    app = ThreadIdCapture(ide=ctx.ide)
    captured_error: Exception | None = None

    async def work() -> None:
        nonlocal captured_error
        try:
            result = await execute_ai_composer_workflow(
                handler=app, llm=llm, input=input_data,
                workflow_options=wf_args,
            )
            app.result_code = result
        except Exception as exc:
            app.result_code = 1
            captured_error = exc

    app.set_work(work)
    await app.run_async()

    tid = app.captured_thread_id or "unknown"
    code = app.result_code
    if captured_error is not None:
        tb = "".join(traceback.format_exception(captured_error))
        return (
            f"Code generation crashed with {type(captured_error).__name__}: "
            f"{captured_error}\nTraceback:\n{tb}\nThread ID: {tid}."
        )
    if code == 0:
        return (
            f"Code generation completed successfully. Thread ID: {tid}. "
            f"Save this to /memories/last_run.json for future resume."
        )
    return f"Code generation finished with exit code {code}. Thread ID: {tid}."


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

    wf_args = _codegen_args(ctx)
    llm = create_llm(wf_args)

    app = ThreadIdCapture(ide=ctx.ide)
    captured_error: Exception | None = None

    async def work() -> None:
        nonlocal captured_error
        try:
            result = await execute_ai_composer_workflow(
                handler=app, llm=llm, input=input_data,
                workflow_options=wf_args,
            )
            app.result_code = result
        except Exception as exc:
            app.result_code = 1
            captured_error = exc

    app.set_work(work)
    await app.run_async()

    tid = app.captured_thread_id or args.thread_id
    code = app.result_code
    if captured_error is not None:
        tb = "".join(traceback.format_exception(captured_error))
        return (
            f"Resume crashed with {type(captured_error).__name__}: "
            f"{captured_error}\nTraceback:\n{tb}\nThread ID: {tid}."
        )
    if code == 0:
        return (
            f"Resume completed successfully. Thread ID: {tid}. "
            f"Save this to /memories/last_run.json for future resume."
        )
    return f"Resume finished with exit code {code}. Thread ID: {tid}."
