"""
Custom summary generation for external contracts.

Given a ``Configuration`` with classified external contracts, produces a CVL
specification file containing summaries for all SUMMARIZABLE contracts.
"""

from dataclasses import dataclass
import json
import pathlib
import subprocess
import sys
from typing import NotRequired, override

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.types import Command
from langgraph.runtime import get_runtime

from graphcore.graph import FlowInput, MessagesState, tool_state_update
from graphcore.tools.schemas import WithImplementation, WithInjectedState, WithInjectedId

from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.cvl.tools import get_cvl, put_cvl, put_cvl_raw
from composer.spec.gen_types import CVLResource
from composer.spec.context import WorkflowContext, SourceCode, CacheKey
from composer.spec.util import temp_certora_file, string_hash
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.source.harness import ContractSetup, ExternalInterface, HarnessDef
from composer.spec.system_model import HarnessedApplication, ExternalActor
from composer.spec.gen_types import TypedTemplate
from composer.ui.tool_display import tool_display, suppress_ack


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


class SummarizerExtra(TypedDict):
    plan: str | None
    curr_spec: str | None
    typechecked: str

class ST(MessagesState, SummarizerExtra):
    result: NotRequired[str]

class Input(FlowInput, SummarizerExtra):
    pass

@dataclass
class SummaryContext:
    config: dict
    source: SourceCode

@tool_display("Type checking", "Typecheck result")
class _TypeChecker(
    WithImplementation[Command | str], WithInjectedState[ST], WithInjectedId
):
    """
    Typecheck your specification
    """
    @override
    def run(self) -> Command | str:
        ctxt = get_runtime(SummaryContext).context
        source = ctxt.source
        config = ctxt.config
        if self.state["curr_spec"] is None:
            return "Spec not yet generated"
        with temp_certora_file(
            root=source.project_root,
            ext="spec",
            content=self.state["curr_spec"],
        ) as spec_file:
            to_check = config.copy()
            to_check["verify"] = f"{source.contract_name}:certora/{spec_file}"
            to_check["compilation_steps_only"] = True
            typechecker = pathlib.Path(__file__).parent.parent / "certoraTypeCheck.py"
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

@tool_display("Writing Plan", None)
class _PlanWrite(WithInjectedId, WithImplementation[Command]):
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

@tool_display("Reading plan", "Summarization Plan")
class _PlanReader(WithInjectedState[ST], WithImplementation[str]):
    """
    Read your summarization plan
    """

    @override
    def run(self) -> str:
        if self.state["plan"] is None:
            return "No plan written"
        return self.state["plan"]

# Summary API

class LocatedHarness(BaseModel):
    path: str
    name: str

class LocatedExternalInterface(ExternalInterface):
    path: str

class SummarizationParams(TypedDict):
    context: HarnessedApplication
    erc20_contracts: list[str]
    interfaces: list[LocatedExternalInterface]
    contract_name: str
    contract_path: str
    included_contracts: list[str]
    config: dict

_SummarizationTemplate = TypedTemplate[SummarizationParams]("cvl_setup_summarization_prompt.j2")

async def _setup_summaries_impl(
    ctx: WorkflowContext["_SummaryCache"],
    env: SourceEnvironment,
    setup: ContractSetup,
    application: HarnessedApplication,
    source: SourceCode
) -> str:
    def _validator(s: ST, _res: str) -> str | None:
        if s["curr_spec"] is None:
            return "Spec hasn't been written yet"
        if s["typechecked"] != s["curr_spec"]:
            return "Spec has not been typechecked"
        return None

    tools = [
        get_cvl(ST),
        put_cvl_raw,
        put_cvl,
        _PlanReader.as_tool("read_plan"),
        _PlanWrite.as_tool("plan_write"),
        _TypeChecker.as_tool("typechecker")
    ]

    intf_summaries = []
    intf_paths = {
        i.name: i.path for i in application.components if
        isinstance(i, ExternalActor) and i.path is not None
    }
    for i in setup.system_description.external_interfaces:
        if i.name not in intf_paths:
            continue # I'm tired boss
            # raise ValueError(f"Told to summarize {i.name}, but no path exists?")
        intf_summaries.append(
            LocatedExternalInterface(
                path=intf_paths[i.name],
                name=i.name,
                behavioral_spec=i.behavioral_spec
            )
        )

    bound = _SummarizationTemplate.bind({
        "config": setup.config.prover_config,
        "context": application,
        "contract_name": source.contract_name,
        "contract_path": source.relative_path,
        "erc20_contracts": setup.system_description.erc20_contracts,
        "included_contracts": [
            c.name for c in setup.system_description.transitive_closure
        ],
        "interfaces": intf_summaries
    })

    graph = bind_standard(
        env.builder, ST, "The commentary on the generated specification", _validator
    ).with_sys_prompt_template(
        "cvl_summarization_system_prompt.j2"
    ).inject(
        lambda g: bound.render_to(g.with_initial_prompt_template)
    ).with_tools(
        [ctx.get_memory_tool(), *env.cvl_authorship_tools]
    ).with_tools(
        tools
    ).with_input(Input).with_context(SummaryContext).compile_async()

    udts = _format_types(setup.config.user_types)

    st = await run_to_completion(
        graph,
        Input(
            typechecked="",
            plan=None,
            curr_spec=None,
            input=[
                "The following types are available for use in your spec",
                udts,
            ],
        ),
        thread_id=ctx.thread_id,
        description="Custom summaries",
        context=SummaryContext(
            config=setup.config.prover_config,
            source=source
        )
    )
    assert st["curr_spec"] is not None
    return st["curr_spec"]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class _SummaryCache(BaseModel):
    content: str


def _summary_key(d: ContractSetup) -> CacheKey[None, _SummaryCache]:
    cacher = string_hash(d.model_dump_json())[:16]
    return CacheKey("summary-" + cacher)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

async def setup_summaries(
    ctx: WorkflowContext[None],
    source: SourceCode,
    env: SourceEnvironment,
    config: ContractSetup,
    app: HarnessedApplication
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
        cvl_research: Builder with CVL manual tools for the research sub-agent.

    Returns:
        CVLResource pointing to the generated ``custom_summaries.spec`` file.
    """

    summary_context = ctx.child(_summary_key(config))
    result_path = pathlib.Path(source.project_root) / "certora" / "custom_summaries.spec"

    to_ret = CVLResource(
        import_path="custom_summaries.spec",
        required=True,
        description="Protocol specific summaries",
        sort="import",
    )

    if (cached := await summary_context.cache_get(_SummaryCache)) is not None:
        result_path.write_text(cached.content)
        return to_ret

    result = await _setup_summaries_impl(
        summary_context, env, config, app, source
    )

    await summary_context.cache_put(_SummaryCache(content=result))
    result_path.write_text(result)
    return to_ret
