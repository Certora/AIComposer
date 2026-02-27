from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

from graphcore.tools.vfs import fs_tools
from graphcore.tools.human import human_interaction_tool
from graphcore.tools.results import result_tool_generator
from graphcore.tools.memory import FileSystemMemoryBackend, memory_tool

from composer.assistant.types import OrchestratorState


class ProposeCodegen(BaseModel):
    """Propose running the code generation workflow to synthesize an implementation."""
    type: Literal["propose_codegen"]
    explanation: str = Field(description="Why codegen was selected and how args were inferred")
    spec_file: str = Field(description="Relative path to CVL spec file (.spec)")
    interface_file: str = Field(description="Relative path to Solidity interface file (.sol)")
    system_doc: str = Field(description="Relative path to system/design document (.md)")


class ProposeNatSpec(BaseModel):
    """Propose running the natural language to CVL specification generation workflow."""
    type: Literal["propose_natspec"]
    explanation: str = Field(description="Why natspec was selected and how args were inferred")
    input_file: str = Field(description="Relative path to design/system document")


class ProposeResume(BaseModel):
    """Propose resuming a previous code generation workflow with updated files."""
    type: Literal["propose_resume"]
    explanation: str = Field(description="Human-readable description of what will be resumed")
    thread_id: str = Field(description="Thread ID of the previous workflow (read from /memories/)")
    working_dir: str = Field(description="Path to the directory with current/updated files")
    commentary: str = Field(description="Description of changes since last run", default="")


class AskUser(BaseModel):
    """Ask the user a question when you need clarification."""
    type: Literal["ask_user"]
    question: str = Field(description="The question to ask")
    context: str = Field(description="Context for the question")


ProposalType = ProposeCodegen | ProposeNatSpec | ProposeResume
InteractionPayload = ProposalType | AskUser


def build_tools(workspace: Path) -> list[BaseTool]:
    """Build all tools for the orchestrator agent."""
    project_tools = fs_tools(str(workspace), cache_listing=False, forbidden_read=r"^\.composer/.+$")

    mem = memory_tool(FileSystemMemoryBackend(workspace / ".composer"))

    propose_codegen = human_interaction_tool(
        ProposeCodegen, OrchestratorState, "propose_codegen"
    )
    propose_natspec = human_interaction_tool(
        ProposeNatSpec, OrchestratorState, "propose_natspec"
    )
    propose_resume = human_interaction_tool(
        ProposeResume, OrchestratorState, "propose_resume"
    )
    ask_user = human_interaction_tool(
        AskUser, OrchestratorState, "ask_user"
    )

    done = result_tool_generator(
        "result", (str, "Exit message"), "Call when the user wants to quit."
    )

    return [*project_tools, mem, propose_codegen, propose_natspec, propose_resume, ask_user, done]
