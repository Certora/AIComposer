from pathlib import Path

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
        response = interrupt(LaunchCodegenArgs(
            spec_file=self.spec_file,
            interface_file=self.interface_file,
            system_doc=self.system_doc,
        ))
        if (r := _check_confirmation(response)) is not None:
            return r
        return await launch_codegen_workflow(self, ctx)


class LaunchResumeTool(LaunchResumeArgs, WithAsyncImplementation[str]):
    """Resume a previous code generation workflow. The user will be asked to confirm before proceeding."""

    async def run(self) -> str:
        ctx = get_runtime(OrchestratorContext).context
        response = interrupt(LaunchResumeArgs(
            thread_id=self.thread_id,
            working_dir=self.working_dir,
            commentary=self.commentary,
        ))
        if (r := _check_confirmation(response)) is not None:
            return r
        return await launch_resume_workflow(self, ctx)


class LaunchNatSpecTool(LaunchNatSpecArgs, WithAsyncImplementation[str]):
    """Launch the NatSpec multi-agent pipeline. The user will be asked to confirm before proceeding."""

    async def run(self) -> str:
        ctx = get_runtime(OrchestratorContext).context
        response = interrupt(LaunchNatSpecArgs(
            input_file=self.input_file,
            contract_name=self.contract_name,
            solc_version=self.solc_version,
            cache_namespace=self.cache_namespace,
            memory_namespace=self.memory_namespace
        ))
        if (r := _check_confirmation(response)) is not None:
            return r
        return await launch_natspec_workflow(self, ctx)


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
        LaunchCodegenTool.as_tool("launch_codegen"),
        LaunchResumeTool.as_tool("launch_resume"),
        LaunchNatSpecTool.as_tool("launch_natspec"),
        done,
    ]
