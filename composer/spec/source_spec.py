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
import sqlite3
import sys
import subprocess

import composer.certora as _

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Awaitable, Callable, Iterable, NotRequired, TypeVar, Literal, get_args, get_origin, Any, override, Coroutine, Awaitable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
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
from graphcore.graph import MessagesState, Builder
from graphcore.tools.vfs import VFSState, VFSToolConfig, fs_tools
from graphcore.tools.schemas import WithInjectedState, WithImplementation, WithInjectedId

import uuid
from typing import cast

from langchain_core.runnables import RunnableConfig

from composer.spec.preaudit_setup import run_preaudit_setup, SetupFailure
from composer.spec.cvl_tools import put_cvl_raw, put_cvl
from composer.tools.search import cvl_manual_search
from composer.human.handlers import handle_human_interrupt
from graphcore.tools.results import result_tool_generator, ValidationResult
from langgraph.store.postgres import PostgresStore
from composer.workflow.services import create_llm, get_checkpointer, get_memory, get_store
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.templates.loader import load_jinja_template
from graphcore.graph import build_workflow, FlowInput
from graphcore.tools.vfs import vfs_tools
from graphcore.tools.memory import memory_tool, SqliteMemoryBackend
from langgraph._internal._typing import StateLike
from graphcore.summary import SummaryConfig

T = TypeVar('T')
R = TypeVar('R')
_S = TypeVar('_S', bound=MessagesState)
_C = TypeVar('_C', bound=StateLike | None)
_I = TypeVar('_I', bound=FlowInput | None)

def bind_standard(
    builder: Builder[Any, _C, _I],
    state_type: type[_S],
    doc: str | None = None,
    validator: Callable[[_S], str | None] | None = None
) -> Builder[_S, _C, _I]:
    """
    Bind a state type to the builder and generate a result tool based on the state's `result` annotation.

    Extracts the result type from the state's `result: NotRequired[T]` annotation and generates
    a result tool using `result_tool_generator`. The tool is then attached to the builder.

    Args:
        builder: The builder to modify
        state_type: The state type to bind, must have a `result: NotRequired[T]` annotation
        doc: Description for the result field. Required if the result type is not a BaseModel.

    Returns:
        Builder with state bound and result tool attached, preserving context and input types

    Raises:
        ValueError: If state_type has no 'result' annotation, or if doc is missing for non-BaseModel result types
    """
    annotations = getattr(state_type, '__annotations__', {})
    if 'result' not in annotations:
        raise ValueError(f"State type {state_type.__name__} must have a 'result' annotation")

    result_annotation = annotations['result']

    # Extract inner type from NotRequired[T]
    origin = get_origin(result_annotation)
    if origin is NotRequired:
        result_type = get_args(result_annotation)[0]
    else:
        result_type = result_annotation

    # Check if result_type is a BaseModel
    is_basemodel = isinstance(result_type, type) and issubclass(result_type, BaseModel)

    if not is_basemodel and doc is None:
        raise ValueError(f"doc parameter is required when result type {result_type} is not a BaseModel")

    tool_doc = "Used to indicate successful completion with result."

    valid : tuple[type[_S], Callable[[_S, Any, str], ValidationResult]] | None = None
    if validator:
        valid = (state_type, lambda s, r, id: validator(s))

    # Generate the result tool
    if is_basemodel:
        result_tool = result_tool_generator("result", result_type, tool_doc, valid)
    else:
        assert doc is not None
        result_tool = result_tool_generator("result", (result_type, doc), tool_doc, valid)

    # Bind state and add tool
    return builder.with_state(state_type).with_tools([result_tool]).with_output_key("result").with_default_summarizer(
        max_messages=50
    )


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

# Common forbidden read pattern for source analysis
FS_FORBIDDEN_READ = "(^lib/.*)|(^\\.certora_internal.*)|(^\\.git.*)|(^test/.*)|(^emv-.*)"

def run_source_analysis(
    args: SourceSpecArgs,
    thread_id: str,
    spec: ContractSpec,
    store: PostgresStore,
    builder: Builder[None, None, FlowInput]
) -> ApplicationSummary | None:
    # Check cache first
    cache_key = _cache_key_source_analysis(args, spec)
    cached = store.get(("source_analysis",), cache_key)
    if cached is not None:
        print(f"Using cached source analysis (key={cache_key})")
        return ApplicationSummary.model_validate(cached.value)
    
    system_doc = get_system_doc(Path(args.system_doc))
    if system_doc is None:
        return None

    memory = memory_tool(get_memory(f"source-summary-{thread_id}"))

    class AnalysisState(MessagesState):
        result: NotRequired[ApplicationSummary]

    graph = bind_standard(
        builder=builder,
        state_type=AnalysisState
    ).with_sys_prompt(
        "You are an expert software analyst, who is very methodical in their work."
    ).with_tools(
        [memory]
    ).with_initial_prompt_template(
        "source_summarization.j2",
        main_contract_name=spec.contract_name,
        relative_path=spec.relative_path
    ).build()[0].compile(
        checkpointer=get_checkpointer()
    )
    task_thread_id = "summary-extraction-" + thread_id

    input: FlowInput = FlowInput(
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
    sort: Literal["attack_vector", "safety_property", "invariant"] = Field(description="The type of property you are describing.")
    description: str = Field(description="The description of the property")

    def to_template_args(self) -> dict:
        thing : str
        what_formal : str
        prop = self
        match prop.sort:
            case "attack_vector":
                thing = "potential attack vector/exploit"
                what_formal = f"that a {thing} is not possible"
            case "invariant":
                thing = "invariant"
                what_formal = "that an invariant holds"
            case "safety_property":
                thing = "safety property"
                what_formal = "that a safety property holds"
        return {
            "thing": thing,
            "what_formal": what_formal,
            "thing_tag": self.sort,
            "thing_descr": self.description
        }

def run_bug_analysis(
    args: SourceSpecArgs,
    component: "ComponentInst",
    builder: Builder[None, None, FlowInput],
    store: PostgresStore,
) -> list[PropertyFormulation] | None:
    # Check cache first
    cache_key = _cache_key_bug_analysis(args, component, component.component, component.summ.application_type)
    cached = store.get(("bug_analysis_2",), cache_key)
    if cached is not None:
        print(f"Using cached bug analysis (key={cache_key})")
        return [PropertyFormulation.model_validate(p) for p in cached.value["items"]]

    class ST(MessagesState):
        result: NotRequired[list[PropertyFormulation]]

    d = bind_standard(
        builder, ST, "The security properties you have extracted about the component"
    ).with_initial_prompt_template(
        "property_analysis_prompt.j2",
        context=component
    ).with_sys_prompt(
        "You are an expert security and software analyst, with extensive knowledge of the types of issues and vulnerabilities found in DeFi protocols"
    ).build()[0].compile()

    r = d.invoke(input=FlowInput(input=[]))

    result: list[PropertyFormulation] = r["result"]

    # Cache the result
    store.put(("bug_analysis_2",), cache_key, {"items": [p.model_dump() for p in result]})
    print(f"Cached bug analysis (key={cache_key})")

    return result

@dataclass
class ComponentInst(ContractSpec):
    summ: ApplicationSummary
    ind: int

    @property
    def component(self) -> ApplicationComponent:
        return self.summ.components[self.ind]
    
    @property
    def application_type(self) -> str:
        return self.summ.application_type

class PropertyFeedback(BaseModel):
    """
    The feedback on the properties
    """
    good: bool = Field(description="Whether the properties are good as is, or if there is room for improvement")
    feedback: str = Field(description="The feedback on the rule if work is needed. Can be empty if there is no feedback")

type FeedbackTool = Callable[[str], PropertyFeedback]

def property_feedback_judge(
    builder: Builder[None, None, FlowInput],
    inst: ComponentInst, prop: PropertyFormulation
) -> FeedbackTool:
    
    class ST(MessagesState):
        memory: NotRequired[str]
        result: NotRequired[PropertyFeedback]
        did_read: NotRequired[bool]

    class GetMemory(WithInjectedState[ST], WithImplementation[Command | str], WithInjectedId):
        """
        Retrieve the rough draft of the feedback
        """
        @override
        def run(self) -> str | Command:
            mem = self.state.get("memory", None)
            if mem is None:
                return "Rough draft not yet written"
            return Command(update={
                "messages": [ToolMessage(tool_call_id=self.tool_call_id, content=mem)],
                "did_read": True
            })

    class SetMemory(WithInjectedId, WithImplementation[Command]):
        """
        Write your rough draft for review
        """
        rough_draft : str = Field(description="The new rough draft of your feedback")

        @override
        def run(self) -> Command:
            return Command(update={
                "memory": self.rough_draft,
                "messages": [ToolMessage(tool_call_id=self.tool_call_id, content="Success")]
            })
        
    def did_rough_draft_read(s: ST) -> str | None:
        h = s.get("did_read", None) is None
        if h is None:
            return "Completion REJECTED: never read rough draft for review"
        return None

    db = sqlite3.connect(":memory:", check_same_thread=False)
    memory = memory_tool(SqliteMemoryBackend("dummy", db))
    workflow = bind_standard(
        builder, ST, validator=did_rough_draft_read
    ).with_initial_prompt_template(
        "property_judge_prompt.j2",
        context=inst,
        **prop.to_template_args()
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_tools([SetMemory.as_tool("write_rough_draft"), GetMemory.as_tool("read_rough_draft"), memory]).build()[0].compile()

    def the_tool(
        cvl: str
    ) -> PropertyFeedback:
        print("HIYA")
        print(cvl)
        res = workflow.invoke(FlowInput(input=[
            "The proposed CVL file is",
            cvl
        ]))
        r = res["result"]
        print(f"Returning feedback \n{r.feedback}\nFor:{prop}")
        return r

    return the_tool

def generate_property_cvl(
    feat: ComponentInst,
    prop: PropertyFormulation,
    builder: Builder[None, None, FlowInput],
    store: PostgresStore
) -> tuple[str, str]:
    
    class ST(MessagesState):
        curr_spec: NotRequired[str]
        result: NotRequired[str]

    feedback = property_feedback_judge(
        builder, feat, prop
    )

    class FeedbackSchema(WithInjectedState[ST], WithImplementation[str]):
        """
        Receive feedback on your CVL
        """
        @override
        def run(self) -> str:
            st = self.state
            spec = st.get("curr_spec", None)
            if spec is None:
                return "No spec put yet"
            t = feedback(spec)
            return f"""
Good? {str(t.good)}
Feedback {t.feedback}
"""

    d = bind_standard(
        builder, ST, "A description of your generated CVL"
    ).with_tools(
        [put_cvl, put_cvl_raw, FeedbackSchema.as_tool("feedback_tool")]
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_initial_prompt_template(
        "property_generation_prompt.j2",
        context=feat,
        **prop.to_template_args()
    ).build()[0].compile()

    r = d.invoke(input=FlowInput(input=[]), config={"recursion_limit": 100})

    return (r["result"], r["curr_spec"])

type PropertyFormalization = tuple[PropertyFormulation, str, str]

def _analyze_component(
    args: SourceSpecArgs,
    feat: ComponentInst,
    b: Builder[None, None, FlowInput],
    cvl_builder: Builder[None, None, FlowInput],
    store: PostgresStore
) -> None | list[tuple[PropertyFormulation, str, str]]:
    res = run_bug_analysis(args, feat, b, store)
    if res is None:
        print("Didn't work")
        return None
    work : list[tuple[PropertyFormulation, str, str]] = []
    for prop in res:
        print(prop)
        r, cvl = generate_property_cvl(
            feat, prop, cvl_builder, store
        )
        print(cvl)
        print(r)
        work.append((prop, r, cvl))
    return work


def execute(args: SourceSpecArgs) -> int:
    """Execute source-based spec generation workflow."""

    thread_id = args.thread_id if args.thread_id else f"source_spec_{uuid.uuid4().hex}"
    print(f"Thread ID: {thread_id}")

    project_root = Path(args.project_root)

    main_contract_path, main_contract_name = args.main_contract.split(":", 1)

    full_contract_path = Path(main_contract_path).resolve()

    if not full_contract_path.is_relative_to(project_root.resolve()):
        print(f"Invalid path: {full_contract_path} doesn't appear in project root {project_root}")
        return 1
    
    relativized_main = full_contract_path.relative_to(project_root.resolve())

    spec = ContractSpec(str(relativized_main), main_contract_name)


    store = get_store()

    llm = create_llm(args)

    b : Builder[None, None, FlowInput] = Builder().with_llm(
        llm
    ).with_input(
        FlowInput
    ).with_loader(
        load_jinja_template
    ).with_tools(
        fs_tools(args.project_root, forbidden_read=FS_FORBIDDEN_READ)
    )
    
    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db,
        model=get_model(),
        skip_test=True
    )

    analysis = run_source_analysis(args, thread_id, spec, store, b)

    cvl_builder = b.with_tools(
        [cvl_manual_search(rag_db)]
    )

    if analysis is None:
        print("Oh well")
        return 1


    work : list[tuple[ComponentInst, None | list[PropertyFormalization]]] = []
    for feature_idx in range(0, len(analysis.components)):
        def thunk() -> tuple[ComponentInst, None | list[PropertyFormalization]]:
            feat = ComponentInst(
                contract_name=spec.contract_name,
                relative_path=spec.relative_path,
                summ=analysis,
                ind=feature_idx
            )
            l = _analyze_component(
                args, feat, b, cvl_builder, store
            )
            return (feat, l)
        work.append(thunk())



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
