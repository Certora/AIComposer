"""
Merge agent and publish tools for the natspec pipeline.

The merge agent is a lightweight sub-agent that merges a property agent's
working CVL copy into the master spec. It can run typecheck on the merged
result and adjust the merge if needed.

PublishSpec and GiveUp are tools injected into property agents as custom
result tools. PublishSpec spawns the merge agent and does a CAS update on
the master spec.
"""

import pathlib
import asyncio
import sys
from dataclasses import dataclass
from typing import NotRequired, Protocol, override

from pydantic import Field

from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.graph import MessagesState
from langgraph.types import Command

from graphcore.graph import FlowInput, tool_output, tool_return, tool_state_update
from graphcore.tools.schemas import WithInjectedState, WithInjectedId, WithAsyncImplementation

from composer.spec.natspec.cas import SharedArtifact
from composer.spec.natspec.pipeline_events import MasterSpecUpdate
from composer.spec.graph_builder import run_to_completion
from composer.spec.cvl_generation import CVLGenerationExtra, check_completion
from composer.spec.tool_env import BasicAgentTools
from composer.spec.util import uniq_thread_id
from composer.ui.tool_display import tool_display, suppress_ack
from composer.spec.natspec.task_description import Assembler, ConfigurationBuilder
from composer.spec.natspec.registry import FileRegistry, StubRegistry
from composer.spec.util import temp_certora_file
from composer.spec.natspec.async_result import AsyncResultTool


class MergeEnv(BasicAgentTools, Protocol):
    """Role-scoped env for the merge sub-agent: basic agent plumbing plus the
    ``merge_tools`` tool set (RAG only in the no-source flow; source + RAG
    when running against an existing codebase).
    """
    @property
    def merge_tools(self) -> tuple[BaseTool, ...]:
        ...


# ---------------------------------------------------------------------------
# Typecheck utility
# ---------------------------------------------------------------------------


async def typecheck_spec(
    files: list[str],
    *,
    spec: str,
    primary_contract: str,
    assembler: Assembler,
    config_builder: ConfigurationBuilder
) -> str | None:
    """Run certoraTypeCheck.py on spec + stubs. Returns None on success, error string on failure.

    When ``source_root`` is set, the existing codebase is copied into the temp dir so stubs
    that reference other contracts in the project can be resolved. Generated stubs are
    laid down at ``new_contracts_subdir`` and interfaces at ``interfaces_subdir``.

    ``conf_overrides`` is merged into the emitted certora.conf; dynamic keys (``files``,
    ``verify``, ``solc``, ``compilation_steps_only``) always win.
    """

    import logging
    logger = logging.getLogger(__name__)
    async with assembler.project_directory() as tmpdir:
        with (
            temp_certora_file(
                content=spec,
                root=str(tmpdir),
                ext="spec"
            ) as spec_file,
            (
                config_builder
                .with_files(files)
                .with_verify(main_contract=primary_contract, spec_file=spec_file)
                .build_to(tmpdir)
            ) as config_file
        ):
            p = (pathlib.Path(__file__).parent.parent / "certoraTypeCheck.py").absolute()
            proc = asyncio.subprocess.create_subprocess_exec(
                sys.executable, *[str(p), config_file],
                cwd=tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            result = await proc
            if result.returncode == 0:
                return None
            assert result.stderr is not None and result.stdout is not None
            stdout = (await result.stdout.read()).decode()
            stderr = (await result.stderr.read()).decode()
            return f"stdout:\n{stdout}\n\nstderr:\n{stderr}"


# ---------------------------------------------------------------------------
# Merge agent result
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    success: bool
    merged_spec: str = ""
    feedback: str = ""


# ---------------------------------------------------------------------------
# Merge agent
# ---------------------------------------------------------------------------

async def run_merge_agent(
    env: MergeEnv,
    files_registry: FileRegistry,
    stub_registry: StubRegistry,
    *,
    working_copy: str,
    master_content: str,
    primary_contract: str,
    assembler: Assembler,
    config_builder: ConfigurationBuilder
) -> MergeResult:
    """Spawn a merge agent to merge working_copy into master_content.

    The agent reasons about the merge: union imports, dedup methods, append rules.
    It can run typecheck on the merged result and adjust if needed.
    Returns MergeResult with the merged spec on success, or feedback on failure.
    """

    class ST(MessagesState):
        result: NotRequired[str]

    class ResultTool(AsyncResultTool[str]):
        """
        Call this tool to deliver the complete, merged specification
        """
        async def validate(self, res: str) -> str | None:
            r = await typecheck_spec(
                files=await files_registry.read_all(primary_contract),
                spec=res,
                assembler=assembler,
                config_builder=config_builder,
                primary_contract=primary_contract
            )
            if r is not None:
                return f"Merged spec failed typecheck:\n{r}\nPlease fix the merge and try again"
            else:
                return None

    workflow = (
        env.builder
        .with_default_summarizer(max_messages=50)
        .with_state(ST)
        .with_tools(
            [ResultTool.as_tool("result")]
        ).with_input(
            FlowInput
        ).with_tools(
            env.merge_tools
        ).with_sys_prompt(
            "You are a CVL specification merge assistant. Your job is to merge a new property's "
            "CVL rules into an existing master specification without breaking it."
        ).with_initial_prompt_template(
            "merge_prompt.j2",
        ).compile_async()
    )

    stubs = await stub_registry.read_all_stubs()
    stub_listing = "\n\n".join(
        f"--- {name} ({decl.path}) ---\n{decl.content}" for name, decl in stubs.items()
    )
    input_parts: list[str | dict] = [
        "The working copy (new property's CVL) is:",
        working_copy,
        "The current master spec is:",
        master_content if master_content else "(empty — this is the first property)",
        f"The current stubs (primary contract: {primary_contract}):",
        stub_listing,
    ]

    try:
        res = await run_to_completion(
            workflow,
            FlowInput(input=input_parts),
            thread_id=uniq_thread_id("spec_merge"),
            recursion_limit=30,
            description="Spec merge",
        )
        if "result" not in res:
            return MergeResult(
                success=False,
                feedback="Merge agent did not produce a result.",
            )
        return MergeResult(
            success=True,
            merged_spec=res["result"],
        )
    except Exception as e:
        return MergeResult(
            success=False,
            feedback=f"Merge agent failed with error: {e}",
        )


# ---------------------------------------------------------------------------
# Advisory typecheck tool for property agents
# ---------------------------------------------------------------------------

def make_advisory_typecheck_tool(
    files: FileRegistry,
    assembler: Assembler,
    config_builder: ConfigurationBuilder,
    primary_contract: str,
) -> BaseTool:
    """Create an advisory typecheck tool for property agents."""

    @tool_display("Type-checking spec", "Type-check result")
    class AdvisoryTypecheck(WithInjectedState[CVLGenerationExtra], WithAsyncImplementation[str]):
        """Run the CVL typechecker on your current working specification against the shared stub.
        This is advisory — use it to catch issues before attempting to publish.
        Reads the current spec from state (written via put_cvl / put_cvl_raw).
        """

        @override
        async def run(self) -> str:
            spec = self.state.get("curr_spec")
            if spec is None:
                return "No spec written yet. Use put_cvl or put_cvl_raw first."
            result = await typecheck_spec(
                files=await files.read_all(primary_contract),
                assembler=assembler,
                config_builder=config_builder,
                spec=spec,
                primary_contract=primary_contract,
            )
            if result is None:
                return "Typecheck passed."
            return f"Typecheck failed:\n{result}"

    return AdvisoryTypecheck.as_tool("advisory_typecheck")


# ---------------------------------------------------------------------------
# Publish + GiveUp tools
# ---------------------------------------------------------------------------

@dataclass
class ContractArtifacts:
    contract_name: str
    master_spec: SharedArtifact
    files_registry: FileRegistry
    stub_registry: StubRegistry

def make_publish_tools(
    contract: ContractArtifacts,
    env: MergeEnv,
    assembler: Assembler,
    config_builder: ConfigurationBuilder
) -> tuple[BaseTool, BaseTool]:
    """Construct PublishSpec + GiveUp tools for a property agent.

    PublishSpec acquires the master spec lock, spawns a merge agent,
    and writes the result. GiveUp lets the agent bail after repeated failures.

    ``validator`` is called before merge — if it returns a string, the publish
    is rejected with that message.
    """

    master_spec = contract.master_spec
    primary_contract = contract.contract_name

    @tool_display("Publishing to master spec", suppress_ack("Publish result"))
    class PublishSpec(WithInjectedState[CVLGenerationExtra], WithInjectedId, WithAsyncImplementation[Command]):
        """Publish your working CVL to the master spec. This spawns a merge agent that
        combines your working copy with the current master spec. If the merge succeeds
        and typechecks, your contribution is recorded and this task completes.
        If the merge fails, you'll receive feedback — address it and try again.
        """
        commentary: str = Field(
            description="A description of your generated CVL and what properties it formalizes"
        )

        @override
        async def run(self) -> Command:
            rejection = check_completion(self.state)
            if rejection is not None:
                return tool_return(self.tool_call_id, content=rejection)

            working_copy = self.state.get("curr_spec")
            if working_copy is None:
                return tool_return(self.tool_call_id, content="No spec written yet. Use put_cvl first.")

            async with master_spec.locked() as (master_content, write_master):
                merge_result = await run_merge_agent(
                    working_copy=working_copy,
                    master_content=master_content or "",
                    assembler=assembler,
                    config_builder=config_builder,
                    primary_contract=primary_contract,
                    env=env,
                    files_registry=contract.files_registry,
                    stub_registry=contract.stub_registry,
                )

                if not merge_result.success:
                    return tool_return(
                        self.tool_call_id,
                        content=f"Merge failed: {merge_result.feedback}",
                    )

                write_master(merge_result.merged_spec)
                evt: MasterSpecUpdate = {
                    "type": "master_spec_update",
                    "spec": merge_result.merged_spec,
                    "contract_id": primary_contract
                }
                get_stream_writer()(evt)
                return tool_output(
                    self.tool_call_id,
                    res={"result": self.commentary},
                )

    @tool_display("Giving up on property", suppress_ack("Give up result"))
    class GiveUp(WithInjectedId, WithAsyncImplementation[Command]):
        """Call this if you cannot formalize *any* of the properties after multiple merge attempts.
        This will end this task with a failure record.
        """
        reason: str = Field(
            description="Why you are giving up on this generation attempt"
        )

        @override
        async def run(self) -> Command:
            return tool_output(
                self.tool_call_id,
                res={"result": f"GAVE_UP: {self.reason}"},
            )

    return (
        PublishSpec.as_tool("publish_spec"),
        GiveUp.as_tool("give_up"),
    )
