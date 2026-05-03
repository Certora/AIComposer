import base64
import re
from pathlib import Path

from pydantic import Field

from langchain_core.tools import BaseTool
from langgraph.runtime import get_runtime
from langgraph.types import interrupt

from graphcore.tools.vfs import fs_tools
from graphcore.tools.results import result_tool_generator
from graphcore.tools.memory import FileSystemMemoryBackend, memory_tool
from graphcore.tools.schemas import WithAsyncImplementation

from composer.assistant.types import OrchestratorContext
from composer.assistant.launch_args import (
    LaunchCodegenArgs,
    LaunchNatSpecArgs,
    LaunchResumeArgs,
)
from composer.assistant.codegen_launch import launch_codegen_workflow, launch_resume_workflow
from composer.assistant.natspec_launch import launch_natspec_workflow
from composer.assistant.post_mortem import PostMortemTool
from composer.input.models import CmdlineCodegenConfiguration, CodegenConfiguration
from composer.workflow.recovery import recovery_from_thread
from composer.workflow.services import checkpointer_context, store_context


# ---------------------------------------------------------------------------
# Workspace path safety
# ---------------------------------------------------------------------------

# Same forbidden-read pattern fs_tools uses on the workspace, applied to the
# PDF tool too so the two surfaces stay consistent.
_FORBIDDEN_WORKSPACE_READ = re.compile(r"^\.composer/.+$")


def _resolve_workspace_path(workspace: Path, rel: str) -> Path | str:
    """Return the absolute path under ``workspace`` for ``rel``, or an error
    string if the path escapes the workspace, is absolute, or is forbidden."""
    if rel.startswith("/"):
        return f"Absolute paths are not allowed: {rel}"
    if ".." in Path(rel).parts:
        return f"Path traversal (..) is not allowed: {rel}"
    if _FORBIDDEN_WORKSPACE_READ.fullmatch(rel):
        return f"Path is forbidden: {rel}"
    return (workspace / rel).resolve()


# ---------------------------------------------------------------------------
# Confirmation gate
# ---------------------------------------------------------------------------

_APPROVED = "yes"
_REJECTED = "no"


def _check_confirmation(response: str) -> str | None:
    """Return None if approved, or a rejection message string."""
    if response == _APPROVED:
        return None
    if response == _REJECTED:
        return "User rejected the launch."
    return f"User rejected with feedback: {response}"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class LaunchCodegenTool(LaunchCodegenArgs, WithAsyncImplementation[str]):
    """Launch the code generation workflow. The user will be asked to confirm before proceeding."""

    async def run(self) -> str:
        ctx = get_runtime(OrchestratorContext).context
        response = interrupt(LaunchCodegenArgs.model_validate(self.model_dump()))
        if (r := _check_confirmation(response)) is not None:
            return r
        return await launch_codegen_workflow(self, ctx)


class LaunchResumeTool(LaunchResumeArgs, WithAsyncImplementation[str]):
    """Resume a previous code generation workflow. The user will be asked to confirm before proceeding."""

    async def run(self) -> str:
        ctx = get_runtime(OrchestratorContext).context
        response = interrupt(LaunchResumeArgs.model_validate(self.model_dump()))
        if (r := _check_confirmation(response)) is not None:
            return r
        return await launch_resume_workflow(self, ctx)


class GetPDFTool(WithAsyncImplementation[list[str | dict]]):
    """Read a PDF from the workspace and return its contents to the model.

    `get_file` is text-only and cannot read PDFs. Use this tool when the
    project has a `.pdf` system/design document, audit report, or other PDF
    you need to read. The result is delivered to you as a multimodal
    document content block — you can then reason about its contents the
    same way you would any other document.

    Path is workspace-relative; absolute paths, `..` traversal, and the
    forbidden-read pattern from your normal file tools are all rejected.
    """

    path: str = Field(description="Workspace-relative path to the PDF file (must end in `.pdf`).")

    async def run(self) -> list[str | dict]:
        ctx = get_runtime(OrchestratorContext).context
        resolved = _resolve_workspace_path(ctx.workspace, self.path)
        if isinstance(resolved, str):
            return [resolved]
        if not resolved.is_file():
            return [f"File not found: {self.path}"]
        if resolved.suffix.lower() != ".pdf":
            return [f"Not a PDF (suffix must be .pdf): {self.path}"]
        data = base64.standard_b64encode(resolved.read_bytes()).decode("utf-8")
        return [
            f"PDF contents of {self.path}:",
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                },
            },
        ]


class LaunchNatSpecTool(LaunchNatSpecArgs, WithAsyncImplementation[str]):
    """Launch the NatSpec multi-agent pipeline. The user will be asked to confirm before proceeding."""

    async def run(self) -> str:
        ctx = get_runtime(OrchestratorContext).context
        response = interrupt(LaunchNatSpecArgs(
            input_file=self.input_file,
            solc_version=self.solc_version,
            cache_namespace=self.cache_namespace,
            memory_namespace=self.memory_namespace,
            forbidden_read=self.forbidden_read,
            prover_conf=self.prover_conf,
            source_root=self.source_root,
            output_root=self.output_root,
            interactive=self.interactive,
        ))
        if (r := _check_confirmation(response)) is not None:
            return r
        return await launch_natspec_workflow(self, ctx)
    
class WriteLaunchConfigTool(WithAsyncImplementation[str]):
    """Serialize a code-generation launch configuration to a JSON file
    that the main pipeline accepts via ``--input-json``.

    Useful for debugging: skips the orchestrator confirmation flow and
    produces a file the user can re-run directly without going through
    the assistant. Field shape mirrors ``launch_codegen``'s
    ``launch_config`` + ``source_root`` — copy whatever you would have
    passed there. Run-time-only concerns (prover_conf, cache_namespace,
    memory_namespace, run_description, prompt_addition) are not part of
    the JSON; the user supplies them as CLI flags at invocation.
    """

    launch_config: CodegenConfiguration = Field(
        description="Same shape as the `launch_config` field on `launch_codegen`.",
    )
    source_root: str = Field(
        description="Absolute path to the codebase root the JSON should reference.",
    )
    output_path: str = Field(
        description="Workspace-relative path to write the JSON file to (e.g. `debug-run.json`).",
    )

    async def run(self) -> str:
        ctx = get_runtime(OrchestratorContext).context
        resolved = _resolve_workspace_path(ctx.workspace, self.output_path)
        if isinstance(resolved, str):
            return resolved
        cmdline_conf = CmdlineCodegenConfiguration(
            **self.launch_config.model_dump(),
            source_root=self.source_root,
        )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(cmdline_conf.model_dump_json(indent=2))
        return (
            f"Wrote launch config to {self.output_path}. Run with "
            f"`python tui_main.py --input-json {self.output_path}` (plus any "
            f"--prover-conf / --description / cache flags as needed)."
        )


class RecoverVFS(WithAsyncImplementation[str]):
    """
    Call this tool to create a resume key for a thread id if the workflow failed to create one.
    """
    thread_id : str = Field(description="The thread id of the run to create a recovery key from")

    async def run(self) -> str:
        async with (
            store_context() as store,
            checkpointer_context() as checkpointer
        ):
            recovery = await recovery_from_thread(
                thread_id=self.thread_id,
                checkpointer=checkpointer,
                store=store
            )
            if recovery is None:
                return "Recovery failed; is the thread id correct?"
            return f"Resume key: {recovery}"


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def build_tools(workspace: Path) -> list[BaseTool]:
    """Build all tools for the orchestrator agent."""
    project_tools = fs_tools(
        str(workspace), cache_listing=False, forbidden_read=r"^\.composer/.+$"
    )
    mem = memory_tool(FileSystemMemoryBackend(workspace / ".composer"))

    done = result_tool_generator(
        "result", (str, "Exit message"), "Call when the user wants to quit."
    )

    return [
        *project_tools,
        mem,
        GetPDFTool.as_tool("get_pdf"),
        LaunchCodegenTool.as_tool("launch_codegen"),
        LaunchResumeTool.as_tool("launch_resume"),
        LaunchNatSpecTool.as_tool("launch_natspec"),
        PostMortemTool.as_tool("post_mortem"),
        WriteLaunchConfigTool.as_tool("write_launch_config"),
        done,
        RecoverVFS.as_tool("recover_vfs")
    ]
