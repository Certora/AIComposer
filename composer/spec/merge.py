"""
Merge agent and publish tools for the natspec pipeline.

The merge agent is a lightweight sub-agent that merges a property agent's
working CVL copy into the master spec. It can run typecheck on the merged
result and adjust the merge if needed.

PublishSpec and GiveUp are tools injected into property agents as custom
result tools. PublishSpec spawns the merge agent and does a CAS update on
the master spec.
"""

import contextlib
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import NotRequired, override

from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState
from langgraph.types import Command

from graphcore.graph import FlowInput, tool_output, tool_return
from graphcore.tools.schemas import WithInjectedState, WithInjectedId, WithAsyncImplementation

from composer.spec.cas import SharedArtifact
from composer.spec.context import WorkflowContext, PlainBuilder, CVLOnlyBuilder
from composer.spec.graph_builder import bind_standard, run_to_completion


# ---------------------------------------------------------------------------
# Typecheck utility
# ---------------------------------------------------------------------------

def typecheck_spec(
    spec: str,
    stub: str,
    interface: str,
    contract_name: str,
    solc_version: str,
) -> str | None:
    """Run certoraTypeCheck.py on spec + stub. Returns None on success, error string on failure."""
    solc_name = f"solc{solc_version}"
    with tempfile.TemporaryDirectory() as tmpdir:
        with contextlib.chdir(tmpdir):
            root = pathlib.Path(tmpdir)
            (root / "input.spec").write_text(spec)
            (root / "Intf.sol").write_text(interface)
            (root / "Impl.sol").write_text(stub)

            p = pathlib.Path(__file__).parent / "certoraTypeCheck.py"
            result = subprocess.run(
                [
                    sys.executable, str(p),
                    f"Impl.sol:{contract_name}",
                    "--verify", f"{contract_name}:./input.spec",
                    "--solc", solc_name,
                    "--compilation_steps_only",
                ],
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                return None
            return f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"


# ---------------------------------------------------------------------------
# Merge agent result
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    success: bool
    merged_spec: str = ""
    feedback: str = ""


class MergedSpec(BaseModel):
    """The merged CVL specification."""
    spec: str = Field(description="The complete merged CVL specification")


# ---------------------------------------------------------------------------
# Merge agent
# ---------------------------------------------------------------------------

async def run_merge_agent(
    working_copy: str,
    master_content: str,
    master_version: int,
    stub: str,
    interface: str,
    contract_name: str,
    solc_version: str,
    builder: PlainBuilder | CVLOnlyBuilder,
    ctx: WorkflowContext,
) -> MergeResult:
    """Spawn a merge agent to merge working_copy into master_content.

    The agent reasons about the merge: union imports, dedup methods, append rules.
    It can run typecheck on the merged result and adjust if needed.
    Returns MergeResult with the merged spec on success, or feedback on failure.
    """

    class ST(MessagesState):
        result: NotRequired[MergedSpec]

    def validate_merge(_s: ST, res: MergedSpec) -> str | None:
        tc_result = typecheck_spec(
            res.spec, stub, interface, contract_name, solc_version,
        )
        if tc_result is not None:
            return f"Merged spec failed typecheck:\n{tc_result}\nPlease fix the merge and try again."
        return None

    workflow = bind_standard(
        builder, ST, validator=validate_merge,
    ).with_input(
        FlowInput
    ).with_sys_prompt(
        "You are a CVL specification merge assistant. Your job is to merge a new property's "
        "CVL rules into an existing master specification without breaking it."
    ).with_initial_prompt_template(
        "merge_prompt.j2",
    ).compile_async(
        checkpointer=ctx.checkpointer
    )

    input_parts: list[str | dict] = [
        "The working copy (new property's CVL) is:",
        working_copy,
        "The current master spec is:",
        master_content if master_content else "(empty — this is the first property)",
        "The current stub is:",
        stub,
    ]

    try:
        res = await run_to_completion(
            workflow,
            FlowInput(input=input_parts),
            thread_id=ctx.uniq_thread_id(),
            recursion_limit=30,
            description="Spec merge",
        )
        if "result" not in res:
            return MergeResult(
                success=False,
                feedback="Merge agent did not produce a result.",
            )
        merged: MergedSpec = res["result"]
        return MergeResult(
            success=True,
            merged_spec=merged.spec,
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
    read_stub: Callable[[], str],
    interface: str,
    contract_name: str,
    solc_version: str,
) -> BaseTool:
    """Create an advisory typecheck tool for property agents."""

    class AdvisoryTypecheck(WithAsyncImplementation[str]):
        """Run the CVL typechecker on your current working specification against the shared stub.
        This is advisory — use it to catch issues before attempting to publish.
        """
        spec: str = Field(description="The CVL specification to typecheck")

        @override
        async def run(self) -> str:
            stub_content = read_stub()
            result = typecheck_spec(
                self.spec, stub_content, interface, contract_name, solc_version,
            )
            if result is None:
                return "Typecheck passed."
            return f"Typecheck failed:\n{result}"

    return AdvisoryTypecheck.as_tool("advisory_typecheck")


# ---------------------------------------------------------------------------
# Publish + GiveUp tools
# ---------------------------------------------------------------------------

# State type for property agents (must have curr_spec)
class _PropertyAgentState(MessagesState):
    curr_spec: NotRequired[str]
    result: NotRequired[str]


MAX_CAS_RETRIES = 3


def make_publish_tools(
    master_spec: SharedArtifact,
    stub_read: Callable[[], str],
    interface: str,
    contract_name: str,
    solc_version: str,
    builder: PlainBuilder | CVLOnlyBuilder,
    ctx: WorkflowContext,
) -> tuple[BaseTool, BaseTool]:
    """Construct PublishSpec + GiveUp tools for a property agent.

    PublishSpec spawns a merge agent, then CAS-updates the master spec.
    GiveUp lets the agent bail after repeated failures.
    """

    class PublishSpec(WithInjectedState[_PropertyAgentState], WithInjectedId, WithAsyncImplementation[Command]):
        """Publish your working CVL to the master spec. This spawns a merge agent that
        combines your working copy with the current master spec. If the merge succeeds
        and typechecks, your contribution is recorded and this task completes.
        If the merge fails, you'll receive feedback — address it and try again.
        """
        commentary: str = Field(
            description="A description of your generated CVL and what property it formalizes"
        )

        @override
        async def run(self) -> Command:
            working_copy = self.state.get("curr_spec")
            if working_copy is None:
                return tool_return(self.tool_call_id, content="No spec written yet. Use put_cvl first.")

            for attempt in range(MAX_CAS_RETRIES):
                master_content, master_version = await master_spec.read()
                stub_content = stub_read()

                merge_result = await run_merge_agent(
                    working_copy=working_copy,
                    master_content=master_content,
                    master_version=master_version,
                    stub=stub_content,
                    interface=interface,
                    contract_name=contract_name,
                    solc_version=solc_version,
                    builder=builder,
                    ctx=ctx,
                )

                if not merge_result.success:
                    return tool_return(
                        self.tool_call_id,
                        content=f"Merge failed: {merge_result.feedback}",
                    )

                cas_result = await master_spec.cas_update(
                    master_version, merge_result.merged_spec,
                )
                if cas_result.success:
                    return tool_output(
                        self.tool_call_id,
                        res={"result": self.commentary},
                    )
                # CAS conflict — retry with updated master

            return tool_return(
                self.tool_call_id,
                content="Failed to publish after multiple CAS retries due to concurrent edits. Try again.",
            )

    class GiveUp(WithInjectedId, WithAsyncImplementation[Command]):
        """Call this if you cannot formalize the property after multiple merge attempts.
        This will end this task with a failure record.
        """
        reason: str = Field(
            description="Why you are giving up on this property"
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
