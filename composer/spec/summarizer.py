"""
Custom summary generation for external contracts.

Given a ``Configuration`` with classified external contracts, produces a CVL
specification file containing summaries for all SUMMARIZABLE contracts.
"""

import json
import pathlib
import subprocess
import sys
from typing import NotRequired, override

from typing_extensions import TypedDict
from pydantic import Field

from langgraph.types import Command
from langchain_core.messages import HumanMessage, ToolMessage

from graphcore.graph import FlowInput, MessagesState, tool_state_update
from graphcore.tools.schemas import WithImplementation, WithInjectedState, WithInjectedId

from composer.spec.harness import Configuration, ERC20TokenGuidance
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.cvl_tools import get_cvl, put_cvl, put_cvl_raw
from composer.spec.cvl_generation import CVLResource
from composer.spec.context import WorkflowContext, CVLBuilder, SourceCode
from composer.spec.util import temp_certora_file
from composer.templates.loader import load_jinja_template


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ResolutionGuidance(WithImplementation[Command], WithInjectedId):
    """
    Retrieve guidance on resolution. You must NOT call this tool in parallel with other tools.
    """

    @override
    def run(self) -> Command:
        return Command(
            update={
                "messages": [
                    ToolMessage(tool_call_id=self.tool_call_id, content="Guidance is as follows..."),
                    HumanMessage(content=load_jinja_template("resolution_guidance.j2"))
                ]
            }
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_container(d: dict) -> str:
    c = d.get("containingContract", None)
    if c is None:
        return "at the top level"
    return f"in contract {c}"


def _format_type(s: dict) -> str | None:
    kind = s.get("typeCategory", None)
    if not kind:
        return None
    where_def = _format_container(s)
    ty_name = s.get("typeName", None)
    if not ty_name:
        return None
    qual_name = s.get("qualifiedName", None)
    match kind:
        case "UserDefinedStruct":
            return f"A struct {ty_name} {where_def}: use `{qual_name}`"
        case "UserDefinedEnum":
            return f"An enum {ty_name} {where_def}: use `{qual_name}`"
        case "UserDefinedValueType":
            base = s.get("baseType", None)
            if not base:
                return None
            return f"An alias for {base} called {ty_name} {where_def}: use `{qual_name}`"
        case _:
            return None


def _format_types(udts: list[dict]) -> str:
    to_format: list[str] = []
    for ty in udts:
        r = _format_type(ty)
        if r:
            to_format.append(r)
    return "\n".join(to_format)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

async def setup_summaries(
    ctx: WorkflowContext[None],
    source: SourceCode,
    config: Configuration,
    cvl_authorship: CVLBuilder,
) -> CVLResource:
    """Generate custom CVL summaries for SUMMARIZABLE external contracts.

    Runs an LLM agent that reads the summarization instructions from the harness
    classification and produces a type-checked CVL specification file containing
    the appropriate summaries.

    Args:
        ctx: Workflow context for threading, memory, and checkpointing.
        source: Source code metadata.
        config: Harness configuration with external contract classifications.
        cvl_authorship: Builder with CVL + source tools for the summary author.

    Returns:
        CVLResource pointing to the generated ``custom_summaries.spec`` file.
    """
    result_path = pathlib.Path(source.project_root) / "certora" / "custom_summaries.spec"

    to_ret = CVLResource(
        import_path="custom_summaries.spec",
        required=True,
        description="Protocol specific summaries",
        sort="import",
    )

    class SummarizerExtra(TypedDict):
        plan: str | None
        curr_spec: str | None
        typechecked: str

    class ST(MessagesState, SummarizerExtra):
        result: NotRequired[str]

    class Input(FlowInput, SummarizerExtra):
        pass

    class TypeChecker(
        WithImplementation[Command | str], WithInjectedState[ST], WithInjectedId
    ):
        """
        Typecheck your specification
        """
        @override
        def run(self) -> Command | str:
            if self.state["curr_spec"] is None:
                return "Spec not yet generated"
            with temp_certora_file(
                root=source.project_root,
                ext="spec",
                content=self.state["curr_spec"],
            ) as spec_file:
                to_check = config.config.copy()
                to_check["verify"] = f"{source.contract_name}:certora/{spec_file}"
                to_check["compilation_steps_only"] = True
                typechecker = pathlib.Path(__file__).parent / "certoraTypeCheck.py"
                with temp_certora_file(
                    root=source.project_root,
                    ext="conf",
                    content=json.dumps(to_check),
                ) as conf_file:
                    res = subprocess.run([
                        sys.executable, str(typechecker), f"certora/{conf_file}"
                    ], cwd=source.project_root, capture_output=True, text=True)
                    if res.returncode == 0:
                        return tool_state_update(
                            self.tool_call_id, "Typechecking passed", typechecked=self.state["curr_spec"]
                        )
                    else:
                        return f"Typechecking failed:\nstdout:\n{res.stdout}\n{res.stderr}"

    class PlanWrite(WithInjectedId, WithImplementation[Command]):
        """
        Write your summarization plan.
        """
        plan: str = Field(description="Your summarization plan")

        @override
        def run(self) -> Command:
            return tool_state_update(
                tool_call_id=self.tool_call_id,
                content="Accepted",
                plan=self.plan,
            )

    class PlanReader(WithInjectedState[ST], WithImplementation[str]):
        """
        Read your summarization plan
        """

        @override
        def run(self) -> str:
            if self.state["plan"] is None:
                return "No plan written"
            return self.state["plan"]

    def _validator(s: ST, _res: str) -> str | None:
        if s["curr_spec"] is None:
            return "Spec hasn't been written yet"
        if s["typechecked"] != s["curr_spec"]:
            return "Spec has not been typechecked"
        return None

    graph = bind_standard(
        cvl_authorship, ST, "The commentary on the generated specification", _validator
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_initial_prompt_template(
        "cvl_setup_prompt.j2"
    ).with_tools(
        [
            get_cvl(ST),
            put_cvl_raw,
            put_cvl,
            PlanReader.as_tool("read_plan"),
            PlanWrite.as_tool("plan_write"),
            TypeChecker.as_tool("typechecker"),
            ERC20TokenGuidance.as_tool("erc20_guidance"),
            ctx.get_memory_tool(),
            ResolutionGuidance.as_tool("resolution_guidance"),
        ]
    ).with_input(Input).compile_async(checkpointer=ctx.checkpointer)

    inputs: list[str] = []
    for ext in config.external_contracts:
        if ext.l == "SUMMARIZABLE":
            inputs.append(
                f"""
<component>
Name: {ext.name}
Description: {ext.description}
Source path: {ext.path}
Summarization instructions: {ext.suggested_summaries}
</component>
"""
            )

    udts = _format_types(config.user_types)

    st = await run_to_completion(
        graph,
        Input(
            typechecked="",
            plan=None,
            curr_spec=None,
            input=[
                "The summarization instructions are as follows:",
                "\n".join(inputs),
                "The prover input config is as follows",
                json.dumps(config.config, indent=4),
                "The following types are available for use in your spec",
                udts,
            ],
        ),
        thread_id=ctx.uniq_thread_id(),
        description="Custom summaries",
    )

    assert st["curr_spec"] is not None
    result_path.write_text(st["curr_spec"])
    return to_ret
