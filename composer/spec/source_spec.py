"""
Source-based spec generation.

Generate CVL specs for existing smart contracts using PreAudit for
compilation analysis and verification.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import tempfile
import shutil
import sys
import subprocess

import composer.certora as _

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Awaitable, Callable, Iterable, NotRequired, TypeVar, Literal, Generic

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool, BaseTool
from langgraph.prebuilt import InjectedState
from langgraph.runtime import get_runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from composer.input.types import ModelOptions, RAGDBOptions, LangraphOptions
from composer.input.parsing import add_protocol_args
from composer.prover.analysis import analyze_cex
from composer.prover.ptypes import RuleResult
from composer.prover.results import read_and_format_run_result
from composer.rag.db import PostgreSQLRAGDatabase
from graphcore.graph import MessagesState
from graphcore.tools.vfs import VFSState, VFSToolConfig

import uuid
from typing import cast

from langchain_core.runnables import RunnableConfig

from composer.spec.preaudit_setup import run_preaudit_setup, SetupFailure, SetupSuccess
from composer.spec.cvl_tools import put_cvl_raw, put_cvl
from composer.tools.search import cvl_manual_search
from composer.human.handlers import handle_human_interrupt
from graphcore.tools.results import result_tool_generator
from langgraph.store.postgres import PostgresStore
from composer.workflow.services import create_llm, get_checkpointer, get_memory, get_store
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.templates.loader import load_jinja_template
from graphcore.graph import build_workflow, FlowInput
from graphcore.tools.vfs import vfs_tools, VFSInput, VFSAccessor
from graphcore.tools.memory import memory_tool
from graphcore.summary import SummaryConfig
from graphcore.types import VMWithResult, resolve_generics

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class SourceSpecContext:
    """Context for source-based spec generation."""
    project_root: Path
    main_contract: str
    main_contract_path: str
    compilation_config: dict
    summaries_import: str | None
    rag_db: PostgreSQLRAGDatabase
    unbound_llm: BaseChatModel

class SourceSpecInput(FlowInput):
    vfs: dict
    curr_spec: None

class SourceSpecState(MessagesState, VFSState):
    """State for source-based spec generation workflow."""
    curr_spec: str | None
    result: NotRequired[dict]


class SourceSpecArgs(ModelOptions, RAGDBOptions, LangraphOptions):
    """Arguments for source-based spec generation."""
    project_root: str
    main_contract: str
    system_doc: str


def apply_async_parallel(
    func: Callable[[T], Awaitable[R]],
    items: Iterable[T]
) -> list[R]:
    """
    Apply an async function to items in parallel and return results.

    Works whether or not there's an active event loop.
    """
    async def _gather_results():
        tasks = [func(item) for item in items]
        return await asyncio.gather(*tasks)

    in_loop = False
    try:
        # Check if there's a running event loop
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        pass
    if not in_loop:
        return asyncio.run(_gather_results())
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, _gather_results())
            return future.result()


class VerifySpecSchema(BaseModel):
    """
    Run the Certora prover to verify the current spec against the source code.

    Returns verification results:
    - VERIFIED: Rule holds for all inputs
    - VIOLATED: Counterexample found (with CEX analysis)
    - TIMEOUT: Verification did not complete in time

    Use these results to refine your spec.
    """
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[SourceSpecState, InjectedState]

    rules: list[str] | None = Field(
        default=None,
        description="Specific rules to verify. If None, verifies all rules."
    )
    timeout: int = Field(
        default=300,
        description="Per-rule timeout in seconds"
    )


async def _analyze(
    llm: BaseChatModel, state, res: RuleResult, tool_call_id: str
) -> tuple[RuleResult, str | None]:
    cex_analysis = None
    if res.status == "VIOLATED":
        cex_analysis = await analyze_cex(llm, state, res, tool_call_id=tool_call_id)
    return (res, cex_analysis)


@tool(args_schema=VerifySpecSchema)
def verify_spec(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated["SourceSpecState", InjectedState],
    rules: list[str] | None = None,
    timeout: int = 300
) -> str | Command:
    """Run Certora prover to verify the spec against source code."""
    context = get_runtime(SourceSpecContext).context

    curr_spec = state.get("curr_spec")
    if curr_spec is None:
        return "No spec has been written yet. Use put_cvl_raw or put_cvl to create a spec first."

    # Create temp directory and copy source

    certora_dir = Path(context.project_root) / "certora"

    (certora_dir / "generated.spec").write_text(curr_spec)

    config = {
        **context.compilation_config,
        "verify": f"{context.main_contract}:certora/generated.spec",
        "parametric_contracts": context.main_contract,
        "optimistic_loop": True,
        "rule_sanity": "basic",
    }

    if rules:
        config["rule"] = rules

    # Write config file
    config_path = certora_dir / "verify.conf"
    config_path.write_text(json.dumps(config, indent=2))

    # Run certoraRunWrapper
    wrapper_script = Path(__file__).parent.parent / "prover" / "certoraRunWrapper.py"
    with tempfile.NamedTemporaryFile("rb", suffix=".pkl") as output_file:

        proc_result = subprocess.run(
            [sys.executable, str(wrapper_script), str(output_file.name), str(config_path)],
            cwd=context.project_root,
            capture_output=True,
            text=True,
            timeout=timeout * 2,
        )

        # Read the pickled output
        import pickle
        run_result = pickle.load(output_file)


    # Check for errors
    if proc_result.returncode != 0:
        return f"Verification failed:\nstdout:\n{proc_result.stdout}\nstderr:\n{proc_result.stderr}"


    # Check if it's an exception
    if isinstance(run_result, Exception):
        return f"Certora prover raised exception: {str(run_result)}\nstdout:\n{proc_result.stdout}"

    if run_result is None or not run_result.is_local_link or run_result.link is None:
        return f"Prover did not produce local results.\nstdout:\n{proc_result.stdout}"

    emv_path = Path(run_result.link)

    # Parse results using existing infrastructure
    results = read_and_format_run_result(emv_path)

    if isinstance(results, str):
        # Error occurred during parsing
        return f"Failed to parse prover results: {results}"

    # Run CEX analysis for violated rules using apply_async_parallel
    results_with_analysis = apply_async_parallel(
        lambda res: _analyze(context.unbound_llm, state, res, tool_call_id),
        list(results.values())
    )

    # Format results for LLM
    lines = ["## Verification Results\n"]
    verified, violated, timeout_count = 0, 0, 0

    for rule_result, cex_analysis in results_with_analysis:
        status = rule_result.status
        name = rule_result.name

        if status == "VERIFIED":
            verified += 1
            lines.append(f"✓ **{name}**: VERIFIED")
        elif status == "VIOLATED":
            violated += 1
            lines.append(f"✗ **{name}**: VIOLATED")
            if cex_analysis:
                lines.append(f"  Analysis: {cex_analysis}")
        elif status == "TIMEOUT":
            timeout_count += 1
            lines.append(f"⏱ **{name}**: TIMEOUT")
        else:
            lines.append(f"? **{name}**: {status}")

    lines.append(f"\n**Summary**: {verified} verified, {violated} violated, {timeout_count} timeout")

    return "\n".join(lines)

class GetCVLSchema(BaseModel):
    """
    View the (pretty-printed) version of the CVL file.
    """
    state: Annotated[SourceSpecState, InjectedState]

@tool(args_schema=GetCVLSchema)
def get_cvl(
    state: Annotated[SourceSpecState, InjectedState]
) -> str:
    if state["curr_spec"] is None:
        return "No spec file on VFS"
    return state["curr_spec"]

@dataclass
class ContractSpec:
    relative_path: str
    contract_name: str

class ComponentInteraction(BaseModel):
    """
    Describes an interaction between some component and another
    """
    other_component: str = Field(description="The name of the other component with which this component interacts")
    interaction_description: str = Field(description="Why the interaction occurs, and a brief description of what the interaction looks like")

class ExternalDependency(BaseModel):
    """
    A single external dependency for a component
    """
    name: str = Field(description="A succint name for the external dependency (e.g., 'Price Oracle', 'Off-chain oracle', 'ERC20 asset token', etc.)")
    requirements: list[str] = Field(description="A list of assumptions/requirements that this external dependency must satisfy (e.g., 'Honest validator', 'implements a standard erc20 interface', etc.)")

class ApplicationComponent(BaseModel):
    """
    A single component within the application
    """
    name: str = Field(description="The brief, concise name of the component (e.g., Price Tracking/Token Management/etc.)")
    description: str = Field(description="A longer description of *what* the component does, not *how* it does it.")
    requirements: list[str] = Field(description="A list of short, succint natural language requirements describing the component's *intended* behavior")
    external_entry_points: list[str] = Field(description="The signatures/names of any external methods that comprise this component")
    state_variables: list[str] = Field(description="State variables involved in the component")
    interactions: list[ComponentInteraction] = Field(description="A list of interactions with other components")

    dependencies: list[ExternalDependency] = Field(description="A list of external dependencies for this component")

class ApplicationSummary(BaseModel):
    """
    A summary of your analysis of the application
    """
    application_type : str = Field(description="A short, concise description of the type of application (AMM/Liquidity Provider/etc.)")
    components: list[ApplicationComponent] = Field(description="The list of components in the application")

def get_system_doc(sys_path: Path) -> dict | str | None:
    """Load a system document from a file path, returning base64-encoded PDF or text."""
    if not sys_path.is_file():
        print("System file not found")
        return None
    if sys_path.suffix == ".pdf":
        file_data = base64.standard_b64encode(sys_path.read_bytes()).decode("utf-8")
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": file_data
            }
        }
    else:
        return sys_path.read_text()


def _hash_file(path: Path) -> str:
    """Return SHA256 hash of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_key_source_analysis(args: "SourceSpecArgs", spec: "ContractSpec") -> str:
    """Generate a cache key for source analysis based on inputs."""
    components = [
        args.project_root,
        _hash_file(Path(args.system_doc)),
        spec.relative_path,
        spec.contract_name,
    ]
    combined = "|".join(str(c) for c in components)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _cache_key_bug_analysis(
    args: "SourceSpecArgs",
    spec: "ContractSpec",
    component: "ApplicationComponent",
    summ: str
) -> str:
    """Generate a cache key for bug analysis based on inputs."""
    components = [
        args.project_root,
        spec.relative_path,
        spec.contract_name,
        component.model_dump_json(),
        summ,
    ]
    combined = "|".join(str(c) for c in components)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def format_summary_xml(summary: ApplicationSummary) -> str:
    """Pretty print an ApplicationSummary to XML-like format for LLM consumption."""
    lines = []
    lines.append("<application-summary>")
    lines.append(f"  <application-type>{summary.application_type}</application-type>")

    for component in summary.components:
        _format_component_xml(lines, component)

    lines.append("</application-summary>")
    return "\n".join(lines)

def _format_component_xml(lines: list[str], component: ApplicationComponent):
    lines.append("  <component>")
    lines.append(f"    <name>{component.name}</name>")
    lines.append(f"    <description>{component.description}</description>")

    if component.requirements:
        lines.append("    <requirements>")
        for req in component.requirements:
            lines.append(f"      <requirement>{req}</requirement>")
        lines.append("    </requirements>")

    if component.external_entry_points:
        lines.append("    <entry-points>")
        for entry in component.external_entry_points:
            lines.append(f"      <entry-point>{entry}</entry-point>")
        lines.append("    </entry-points>")

    if component.state_variables:
        lines.append("    <state-variables>")
        for var in component.state_variables:
            lines.append(f"      <variable>{var}</variable>")
        lines.append("    </state-variables>")

    if component.interactions:
        lines.append("    <interactions>")
        for interaction in component.interactions:
            lines.append("      <interaction>")
            lines.append(f"        <other-component>{interaction.other_component}</other-component>")
            lines.append(f"        <description>{interaction.interaction_description}</description>")
            lines.append("      </interaction>")
        lines.append("    </interactions>")

    if component.dependencies:
        lines.append("    <external-dependencies>")
        for dep in component.dependencies:
            lines.append("      <dependency>")
            lines.append(f"        <name>{dep.name}</name>")
            for req in dep.requirements:
                lines.append(f"        <requirement>{req}</requirement>")
            lines.append("      </dependency>")
        lines.append("    </external-dependencies>")

    lines.append("  </component>")

def format_component_xml(component: ApplicationComponent) -> str:
    l = []
    _format_component_xml(l, component)
    return "\n".join(l)

def run_source_analysis(
    args: SourceSpecArgs,
    thread_id: str,
    spec: ContractSpec,
    store: PostgresStore,
    vfs: "VFSFactory"
) -> ApplicationSummary | None:
    # Check cache first
    cache_key = _cache_key_source_analysis(args, spec)
    cached = store.get(("source_analysis",), cache_key)
    if cached is not None:
        print(f"Using cached source analysis (key={cache_key})")
        return ApplicationSummary.model_validate(cached.value)

    llm = create_llm(args)

    memory = memory_tool(get_memory(f"source-summary-{thread_id}"))

    generation_complete = result_tool_generator(
        outkey="result",
        result_schema=ApplicationSummary,
        doc="Used to indicate successful result of your analysis."
    )

    @resolve_generics
    class AnalysisState(VMWithResult[ApplicationSummary]):
        pass

    v_tools = vfs(AnalysisState)

    task_thread_id = "summary-extraction-" + thread_id

    graph = build_workflow(
        initial_prompt=load_jinja_template(
            "source_summarization.j2",
            main_contract_name=spec.contract_name,
            relative_path=spec.relative_path

        ),
        sys_prompt="You are an expert software analyst, who is very methodical in their work.",
        input_type=VFSInput,
        output_key="result",
        state_class=AnalysisState,
    
        unbound_llm=llm,
        summary_config=SummaryConfig(max_messages=50),
        tools_list=[
            *v_tools,
            memory,
            generation_complete
        ]
    )[0].compile(checkpointer=get_checkpointer())

    system_doc = get_system_doc(Path(args.system_doc))
    if system_doc is None:
        return None


    input : VFSInput = VFSInput(
        vfs={},
        input=[
            "The system document is as follows",
            system_doc
        ]
    )

    res = graph.invoke(
        input=input,
        config={"configurable": {
            "thread_id": task_thread_id
        }, "recursion_limit": 50}
    )
    result: ApplicationSummary = res["result"]

    # Cache the result
    store.put(("source_analysis",), cache_key, result.model_dump())
    print(f"Cached source analysis (key={cache_key})")

    return result

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

class PropertyFormulation(BaseModel):
    """
    A property or invariant that must hold for the component
    """
    methods: list[str] | Literal["invariant"] = Field(description="A list of external methods involved in the property, or 'invariant' if the property is an invariant on the contract state")
    description: str = Field(description="A description of the property/invariant. " \
    "In the case of a 'negative' property (something that should nto be possible), the statement could look like: 'a user should not be able to ...'. " \
    "For an invariant, the statement could look like: 'the sum of all balances in X should always be ...'.")

def run_bug_analysis(
    args: SourceSpecArgs,
    spec: ContractSpec,
    component: "ComponentInst",
    vfs_f: "VFSFactory",
    store: PostgresStore,
) -> list[PropertyFormulation] | None:
    # Check cache first
    cache_key = _cache_key_bug_analysis(args, spec, component.component, component.summ.application_type)
    cached = store.get(("bug_analysis",), cache_key)
    if cached is not None:
        print(f"Using cached bug analysis (key={cache_key})")
        return [PropertyFormulation.model_validate(p) for p in cached.value["items"]]

    m = create_llm(args)

    @resolve_generics
    class ST(VMWithResult[list[PropertyFormulation]]):
        pass

    result_tool = result_tool_generator("result", (list[PropertyFormulation], "The list of properties you have formulated"), "Used to indicate the success of your analysis")

    vfs = vfs_f(ST)

    d = build_workflow(
        context_schema=None,
        initial_prompt="""
Analyze the implementation and documentation of the following component and formulate potential issues that the implementation might suffer from.

DO NOT try to find the actual issues, but focus your analysis on determining the types of errors, security vulnerabilities, or bugs the component (and its implementation) might suffer from.

For example, if you determine that the implementation relies on some sort of rounding for converting prices, you might conclude that an attack involving incorrect rounding *could* be an issue.

When formulating your scenarios, try to be as specific as possible. An attack like "a user could steal all funds" is NOT helpful, but "A user might manipulate an oracle to unbalance the pool, forcing one of the
token prices to 0, letting them mint tokens for free." Again, it is important to stress you do not need to find evidence that this is an actual error in the implementation, simply that these bugs/attacks/etc. are *plausible*
given the feature and its implementation.
""",
        sys_prompt="You are an expert security and software analyst, with extensive knowledge of the types of issues and vulnerabilities found in DeFi protocols",
        input_type=VFSInput,
        state_class=ST,
        output_key="result",
        tools_list=[*vfs, result_tool],
        unbound_llm=m
    )[0].compile()

    r = d.invoke(input=VFSInput(vfs={}, input=[
        f"The application in question is described as: {component.summ.application_type}",
        f"The target of your analysis is contract {spec.contract_name} at {spec.relative_path}",
        "The component description is as follows",
        format_component_xml(component.component)
    ]))

    result: list[PropertyFormulation] = r["result"]

    # Cache the result
    store.put(("bug_analysis",), cache_key, {"items": [p.model_dump() for p in result]})
    print(f"Cached bug analysis (key={cache_key})")

    return result

B = TypeVar("B", bound=VFSState)

type VFSFactory = Callable[[type[VFSState]], list[BaseTool]]

@dataclass
class ComponentInst:
    summ: ApplicationSummary
    ind: int

    @property
    def component(self) -> ApplicationComponent:
        return self.summ.components[self.ind]

def vfs_factory(
    args: SourceSpecArgs,
) -> Callable[[type[VFSState]], list[BaseTool]]:
    def to_ret(
        b: type[B]
    ) -> list[BaseTool]:
        vfs, _ = vfs_tools(VFSToolConfig(immutable=True, fs_layer=args.project_root, forbidden_read="(^lib/.*)|(^\\.certora_internal.*)|(^\\.git.*)|(^test/.*)|(^emv-.*)"), b)
        return vfs
    return to_ret

class PropertyFeedback(BaseModel):
    """
    The feedback on the properties
    """
    good: bool = Field(description="Whether the properties are good as is, or if there is room for improvement")
    feedback: str = Field(description="The feedback on the rule if work is needed. Can be empty if there is no feedback")
    
type FeedbackTool = Callable[[PostgreSQLRAGDatabase, ComponentInst, PropertyFormulation, str], PropertyFeedback]

def property_feedback_judge(
    args: SourceSpecArgs,
    vfs: VFSFactory
) -> FeedbackTool:
    result_tool = result_tool_generator("result", PropertyFeedback, "Used to output your feedback on the rule(s)/invariant(s)")

    @resolve_generics
    class ST(VMWithResult[PropertyFeedback]):
        memory: NotRequired[str]

    @dataclass
    class Ctxt:
        rag_db: PostgreSQLRAGDatabase

    class GetMemory(BaseModel):
        """
        Retrieve the rough draft of the feedback
        """
        st: Annotated[ST, InjectedState]

    @tool(args_schema=GetMemory)
    def get_rough_draft(st: Annotated[ST, InjectedState]) -> str:
        return st.get("memory", "Rough draft not yet written")
    
    class SetMemory(BaseModel):
        """
        Write your rough draft for review
        """
        rough_draft : str = Field(description="The new rough draft of your feedback")
        tool_call_id: Annotated[str, InjectedToolCallId]
    
    @tool(args_schema=SetMemory)
    def write_rough_draft(rough_draft: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        return Command(update={
            "memory": rough_draft,
            "messages": [ToolMessage(tool_call_id=tool_call_id, content="Success")]
        })

    workflow = build_workflow(
        state_class=ST,
        input_type=VFSInput,
        context_schema=Ctxt,
        initial_prompt=load_jinja_template("property_judge_prompt.j2"),
        sys_prompt=load_jinja_template("cvl_system_prompt.j2"),
        output_key="result",
        summary_config=SummaryConfig(max_messages=50),
        tools_list=[result_tool, *vfs(ST), cvl_manual_search(Ctxt), write_rough_draft, get_rough_draft],
        unbound_llm=create_llm(args)
    )[0].compile()

    def the_tool(
        db: PostgreSQLRAGDatabase,
        inst: ComponentInst, prop: PropertyFormulation, cvl: str
    ) -> PropertyFeedback:
        import yaml
        print("HIYA")
        print(cvl)
        r = workflow.invoke(VFSInput(vfs={}, input=[
            f"The component in question is located within an application described as {inst.summ.application_type}"
            "The component's description is", format_component_xml(inst.component),
            "The property is", yaml.dump(prop.model_dump()),
            "The proposed CVL file is",
            cvl
        ]), context=Ctxt(rag_db=db))["result"]
        import yaml
        print(f"Returning feedback \n{r.feedback}\nOn:\n{cvl}\nFor:{prop}")
        return r

    return the_tool

def generate_property_cvl(
    args: SourceSpecArgs,
    spec: ContractSpec,
    feat: ComponentInst,
    prop: PropertyFormulation,
    vfs_fact: VFSFactory,
    db: PostgreSQLRAGDatabase,
    feedback: FeedbackTool
) -> tuple[str, str]:
    m = create_llm(args)

    @resolve_generics
    class ST(VMWithResult[str]):
        curr_spec: NotRequired[str]

    result_tool = result_tool_generator("result", (str, "An explanation of the CVL rule/invariant you have written"), "Used to indicate the succesful generation of the rule.")

    vfs = vfs_fact(ST)

    thing : str
    init_message : str

    if prop.methods == "invariant":
        thing = "invariant"
        init_message = f"The proposed invariant is: {prop.description}"
    else:
        thing = "rule"
        init_message = f"The potential issue/attack/etc. is: {prop.description}"

    @dataclass
    class RagCtxt:
        rag_db: PostgreSQLRAGDatabase

    class FeedbackSchema(BaseModel):
        """
        Receive feedback on your CVL
        """
        st: Annotated[ST, InjectedState]
        tool_call_id: Annotated[str, InjectedToolCallId]

    @tool(args_schema=FeedbackSchema)    
    def feedback_tool(
        st: Annotated[ST, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> str:
        spec = st.get("curr_spec", None)
        if spec is None:
            return "No spec put yet"
        import yaml
        return yaml.dump(feedback(db, feat, prop, spec).model_dump())
    
    d = build_workflow(
        context_schema=None,
        initial_prompt=load_jinja_template("property_generation_prompt.j2", thing=thing),
        sys_prompt="You are an expert security and software analyst, with extensive knowledge of the types of issues and vulnerabilities found in DeFi protocols",
        input_type=VFSInput,
        state_class=ST,
        output_key="result",
        tools_list=[*vfs, result_tool, put_cvl, put_cvl_raw, feedback_tool, cvl_manual_search(RagCtxt)],
        unbound_llm=m,
        summary_config=SummaryConfig(max_messages=50)
    )[0].compile()

    r = d.invoke(input=VFSInput(vfs={}, input=[
        f"The application in question is described as: {feat.summ.application_type}", 
        f"The target of your analysis is contract {spec.contract_name} at {spec.relative_path}",
        "The component description is as follows",
        format_component_xml(feat.component),
        init_message
    ]), config={"recursion_limit": 100}, context=RagCtxt(rag_db=db))

    return (r["result"], r["curr_spec"])


def execute(args: SourceSpecArgs) -> int:
    """Execute source-based spec generation workflow."""

    thread_id = args.thread_id if args.thread_id else f"source_spec_{uuid.uuid4().hex}"
    print(f"Thread ID: {thread_id}")

    store = get_store()

    project_root = Path(args.project_root)

    main_contract_path, main_contract_name = args.main_contract.split(":", 1)

    full_contract_path = Path(main_contract_path).resolve()

    if not full_contract_path.is_relative_to(project_root.resolve()):
        print(f"Invalid path: {full_contract_path} doesn't appear in project root {project_root}")
        return 1
    
    relativized_main = full_contract_path.relative_to(project_root.resolve())

    spec = ContractSpec(str(relativized_main), main_contract_name)

    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db,
        model=get_model(),
        skip_test=True
    )

    def vfs_factory(
        ty: type[VFSState]
    ) -> list[BaseTool]:
        return vfs_tools(
            conf=VFSToolConfig(immutable=True, fs_layer=args.project_root, forbidden_read="(^lib/.*)|(^\\.certora_internal.*)|(^\\.git.*)|(^test/.*)|(^emv-.*)"),
            ty=ty
        )[0]

    analysis = run_source_analysis(args, thread_id, spec, store, vfs_factory)

    if analysis is None:
        print("Oh well")
        return 1

    feedback_t = property_feedback_judge(
        args, vfs_factory
    )

    for feature_idx in range(0, len(analysis.components)):
        feat = ComponentInst(analysis, feature_idx)
        res = run_bug_analysis(args, spec, feat, vfs_factory, store)
        if res is None:
            print("Didn't work")
            continue
        for prop in res:
            import yaml
            print(yaml.dump(prop.model_dump()))
            r, cvl = generate_property_cvl(
                args, spec, feat, prop, vfs_factory, rag_db, feedback_t
            )
            print(cvl)
            print(r)

    sys.exit(1)

    # Step 1: Run PreAudit setup
    print("Running PreAudit compilation analysis...")
    setup_result = run_preaudit_setup(
        project_root=Path(args.project_root),
        main_contract=main_contract_name,
        relative_path=str(relativized_main)
    )
    match setup_result:
        case SetupFailure(error=e):
            print(f"Auto setup failed: {e}")
            return 1
        case _:
            pass

    # Step 2: Create LLM and thread
    llm = create_llm(args)

    # Step 3: Build context
    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db,
        model=get_model(),
        skip_test=True
    )

    summaries_import = f'import "{setup_result.summaries_path}";'

    context = SourceSpecContext(
        project_root=Path(args.project_root),
        main_contract=main_contract_name,
        main_contract_path=main_contract_path,
        compilation_config=setup_result.config,
        summaries_import=summaries_import,
        rag_db=rag_db,
        unbound_llm=llm
    )

    # Step 4: Build workflow
    manual = cvl_manual_search(SourceSpecContext)
    checkpointer = get_checkpointer()

    memory = memory_tool(get_memory(f"memory-{thread_id}"))
    generation_complete = result_tool_generator(
        outkey="result",
        result_schema=(dict, "Final result"),
        doc="Used to indicate successful result of your analysis."
    )

    v_tools, _ = vfs_tools(
        conf=VFSToolConfig(immutable=True, fs_layer=args.project_root, forbidden_read="(^lib/.*)|(^\\.certora_internal.*)|(^\\.git.*)|(^test/.*)|(^emv-.*)"),
        ty=SourceSpecState
    )

    graph = build_workflow(
        context_schema=SourceSpecContext,
        initial_prompt=load_jinja_template(
            "source_spec_prompt.j2",
            main_contract=args.main_contract,
            summaries_import=summaries_import
        ),
        sys_prompt=load_jinja_template("cvl_system_prompt.j2"),
        input_type=SourceSpecInput,
        output_key="result",
        state_class=SourceSpecState,
    
        unbound_llm=llm,
        summary_config=SummaryConfig(max_messages=50),
        tools_list=[
            # VFS tools for source code access
            *v_tools,

            # CVL spec tools
            put_cvl,
            put_cvl_raw,
            get_cvl,

            # Verification
            verify_spec,

            # Support tools
            manual,
            memory,
            generation_complete
        ]
    )[0].compile(checkpointer=checkpointer, store=get_store())

    # Step 5: Run workflow
    def fresh_config() -> RunnableConfig:
        return {
            "recursion_limit": args.recursion_limit,
            "configurable": {
                "thread_id": thread_id
            }
        }

    runnable_conf = fresh_config()

    sys_doc = get_system_doc(Path(args.system_doc))
    assert sys_doc is not None

    if args.checkpoint_id is not None:
        runnable_conf["configurable"]["checkpoint_id"] = args.checkpoint_id #type: ignore

    formatted_types = format_types(setup_result.user_types)

    graph_input = SourceSpecInput(input=[
        "The major components of the software is as follows",
        format_summary_xml(analysis),
        "The system document is as follows",
        sys_doc,
        "User defined types can be referenced in your specification, according to the following guidance",
        formatted_types
        ], curr_spec=None, vfs={}) if args.checkpoint_id is None else None

    while True:
        t = graph_input
        graph_input = None
        for (tag, payload) in graph.stream(
            input=t,
            config=runnable_conf,
            context=context,
            stream_mode=["updates", "checkpoints"]
        ):
            assert isinstance(payload, dict)
            if tag == "checkpoints":
                print("current checkpoint: " + payload["config"]["configurable"]["checkpoint_id"])
                continue
            if "__interrupt__" in payload:
                if "configurable" in runnable_conf and "checkpoint_id" in runnable_conf["configurable"]:
                    del runnable_conf["configurable"]["checkpoint_id"]
                interrupt_data = cast(dict, payload["__interrupt__"][0].value)
                def debug_thunk() -> None:
                    pass  # TODO: implement debug console if needed
                human_response = handle_human_interrupt(interrupt_data, debug_thunk)
                graph_input = Command(resume=human_response)
                break
            else:
                print(payload)
        if graph_input is None:
            break

    final_state = cast(SourceSpecState, graph.get_state(fresh_config()).values)
    if "result" not in final_state:
        return 1

    print("Spec file generation complete")
    print(final_state["result"])

    return 0

def auto_prover() -> int:
    parser = argparse.ArgumentParser()
    add_protocol_args(parser, ModelOptions)
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, LangraphOptions)
    parser.add_argument("project_root")
    parser.add_argument("main_contract")
    parser.add_argument("system_doc")

    res = cast(SourceSpecArgs, parser.parse_args())

    return execute(res)
