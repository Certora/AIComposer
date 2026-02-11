
import hashlib
from typing import NotRequired, override
import pathlib
import json
import subprocess
import sys

from pydantic import Field

from langgraph.types import Command
from langchain_core.messages import HumanMessage, ToolMessage

from graphcore.graph import Builder, FlowInput, MessagesState, tool_state_update
from graphcore.tools.schemas import WithImplementation, WithInjectedState, WithInjectedId

from composer.spec.harness import Configuration, ERC20TokenGuidance
from composer.spec.graph_builder import bind_standard
from composer.spec.cvl_tools import get_cvl, put_cvl, put_cvl_raw
from composer.spec.cvl_generation import CVLResource
from composer.spec.context import WorkspaceContext
from composer.spec.utils import temp_certora_file
from composer.workflow.services import get_checkpointer
from composer.templates.loader import load_jinja_template
from composer.spec.trunner import run_to_completion_sync

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


def format_container(d: dict) -> str:
    c = d.get("containingContract", None)
    if c is None:
        return "at the top level"
    else:
        return f"in contract {c}"

def format_type(s: dict) -> str | None:
    kind = s.get("typeCategory", None)
    if not kind:
        return None
    where_def = format_container(s)
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

def format_types(udts: list[dict]) -> str:
    to_format: list[str] = []
    for ty in udts:
        r = format_type(ty)
        if not r:
            continue
        to_format.append(r)
    return "\n".join(to_format)


def setup_summaries(
    ctx: WorkspaceContext,
    d: Configuration,
    cvl_builder: Builder[None, None, FlowInput]
) -> CVLResource:
    cacher = hashlib.sha256(d.model_dump_json().encode()).hexdigest()[:16]
    
    summary_context = ctx.child(f"summary-{cacher}", d.model_dump())

    result_path = (pathlib.Path(ctx.project_root) / "certora" / "custom_summaries.spec")

    to_ret = CVLResource(
        import_path="custom_summaries.spec",
        required=True,
        description="Protocol specific summaries",
        sort="import"
    )

    if (cached := summary_context.cache_get()) is not None:
        c = cached["content"]
        result_path.write_text(c)
        return to_ret

    class ST(MessagesState):
        plan: NotRequired[str]
        curr_spec: NotRequired[str]
        result: NotRequired[str]
        typechecked: str

    class Input(FlowInput):
        typechecked: str

    class TypeChecker(
        WithImplementation[Command | str], WithInjectedState[ST], WithInjectedId
    ):
        """
        Typecheck your specification
        """
        @override
        def run(self) -> Command | str:
            if "curr_spec" not in self.state:
                return "Spec not yet generated"
            with temp_certora_file(
                root=ctx.project_root,
                ext="spec",
                content=self.state["curr_spec"]
            ) as spec_file:
                to_check = d.config.copy()
                to_check["verify"] = f"{ctx.contract_name}:certora/{spec_file}"
                to_check["compilation_steps_only"] = True
                typechecker = pathlib.Path(__file__).parent / "certoraTypeCheck.py"
                with temp_certora_file(
                    root=ctx.project_root,
                    ext="conf",
                    content=json.dumps(to_check)
                ) as conf_file:
                    res = subprocess.run([
                        sys.executable, str(typechecker), f"certora/{conf_file}"
                ], cwd=ctx.project_root, capture_output=True, text=True)
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
                plan=self.plan
            )
        
    class PlanReader(WithInjectedState[ST], WithImplementation[str]):
        """
        Read your summarization plan
        """

        @override
        def run(self) -> str:
            if "plan" not in self.state:
                return "No plan written"
            return self.state["plan"]

    def validator(
        s: ST, res: str
    ) -> str | None:
        if "curr_spec" not in s:
            return "Spec hasn't been written yet"
        if s["typechecked"] != s["curr_spec"]:
            return "Spec has not been typechecked"
        return None

    g = bind_standard(
        cvl_builder, ST, "The commentary on the generated specification", validator
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
            summary_context.get_memory_tool(),
            ResolutionGuidance.as_tool("resolution_guidance")
        ]
    ).with_input(Input).compile(checkpointer=get_checkpointer())
    
    inputs = []

    for ext in d.external_contracts:
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

    udts = format_types(d.user_types)

    res = run_to_completion_sync(g, Input(
        typechecked="",
        input=[
            "The summarization instructions are as follows:",
            "\n".join(inputs),
            "The prover input config is as follows",
            json.dumps(d.config, indent=4),
            "The following types are avialable for use in your spec",
            udts
        ]
    ), thread_id=summary_context.thread_id)
    assert "curr_spec" in res
    assert "result" in res

    summary_context.cache_put({"content": res["curr_spec"]})
    result_path.write_text(res["curr_spec"])
    return to_ret
