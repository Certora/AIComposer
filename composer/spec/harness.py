"""
Harness analysis and prover setup.

Identifies external contracts, classifies them, generates harness files
for contracts needing multiple instances, and runs PreAudit compilation
to produce a ``Configuration`` for downstream phases.

Two entry points:

``analyze_external_interactions``
    Library entry point: runs the classification agent and returns a
    ``HarnessSetup`` with classifications + generated VFS files.  Uses
    an in-memory checkpointer and no memory tool.

``setup_and_harness_agent``
    Pipeline entry point: runs the classification agent within a
    ``WorkflowContext``, writes harness files to disk, runs PreAudit
    compilation, and returns a ``Configuration``.
"""

from dataclasses import dataclass
from typing import Any, Literal, Annotated, cast, override, NotRequired, Protocol
from pathlib import Path

from pydantic import Field, BaseModel, Discriminator

from langgraph.types import Command, Checkpointer
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel

from graphcore.graph import Builder
from graphcore.tools.schemas import WithImplementation, WithInjectedId
from graphcore.tools.vfs import VFSState, VFSInput, VFSToolConfig, vfs_tools, VFSAccessor
from graphcore.tools.results import result_tool_generator, ValidationResult

from composer.templates.loader import load_jinja_template
from composer.spec.graph_builder import run_to_completion
from composer.spec.preaudit_setup import run_preaudit_setup, SetupFailure
from composer.spec.context import WorkflowContext, SourceCode


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ExternalActor(BaseModel):
    """Base class for any external contract that the main contract directly interacts with."""
    name: str = Field(description="A short, descriptive name of the external contract being interacted with")
    description: str = Field(description="A short, precise description of what the external actor does and its interaction with the main contract")


class Summarizable(ExternalActor):
    """An external contract whose behavior can be adequately modelled using CVL summaries."""
    l: Literal["SUMMARIZABLE"]
    suggested_summaries: str = Field(description="A natural language description of the suggested summaries to use for this contract")
    path: str | None = Field(description="The relative path to the source of the contract to be summarized; null if the implementation isn't available")


class SourceAvailable(ExternalActor):
    """Intermediate base for classifications where the external contract's source code is available."""
    path: str = Field(description="The relative path to the source of the contract being described")


class NotFoundHavoc(ExternalActor):
    """An external contract whose source code is not available.

    The prover will apply its default pessimistic (havoc) assumptions to any calls
    into this contract.
    """
    l: Literal["NOTFOUND_HAVOC"]


class Singleton(SourceAvailable):
    """An external contract for which exactly one instance is sufficient to model the
    protocol in a non-trivial state."""
    l: Literal["SINGLETON"]


class HarnessDef(BaseModel):
    """A generated harness file that creates a uniquely-named contract extending an external contract."""
    path: str = Field(description="Path to the harness definition")
    harness_name: str = Field(description="The name of the contract defined in the harness file")
    suggested_role: str = Field(description="The suggested role of this harness; e.g., 'the first token' of the pool, etc.")


class WithHarnesses(BaseModel):
    """Mixin for classifications that require multiple prover instances and therefore harness files."""
    harnesses: list[HarnessDef] = Field(description="The harnesses created to model this contract.")


class Dynamic(SourceAvailable, WithHarnesses):
    """An external contract that is dynamically instantiated by the main contract at runtime."""
    l: Literal["DYNAMIC"]


class Multiple(SourceAvailable, WithHarnesses):
    """An external contract that is NOT instantiated by the main contract, but of which
    multiple pre-existing instances are needed to model a non-trivial state."""
    l: Literal["MULTIPLE"]


type ContractClassification = Annotated[
    Summarizable |
    NotFoundHavoc |
    Dynamic |
    Singleton |
    Multiple,
    Discriminator("l")
]


class ContractSetup(BaseModel):
    """The result of harness analysis."""
    external_contracts: list[ContractClassification] = Field(description="The external actors classified by your analysis")
    primary_entity: str = Field(description="A description of the primary entity managed by this contract")
    non_trivial_state: str = Field(description="A semi-formal description of a `non-trivial state`. Reference the external "
    "contracts you identified during the harnessing step as necessary.")


class HarnessSetup(BaseModel):
    """Complete output of the harness analysis workflow."""
    setup: ContractSetup
    vfs: dict[str, str]


class Configuration(ContractSetup):
    """Full prover configuration produced by harness + PreAudit setup."""
    config: dict
    summaries_path: str
    user_types: list[dict]


# ---------------------------------------------------------------------------
# Protocols and standalone params
# ---------------------------------------------------------------------------

class HarnessProtocol(Protocol):
    """Minimum context needed for harness analysis.

    ``SourceCode`` satisfies this structurally.
    """
    @property
    def project_root(self) -> str: ...
    @property
    def contract_name(self) -> str: ...
    @property
    def relative_path(self) -> str: ...


@dataclass
class LLMParams:
    """Parameters to construct an LLM via ``create_llm`` for standalone use.

    Satisfies the ``ModelOptions`` protocol.
    """
    model: str
    tokens: int
    thinking_tokens: int
    memory_tool: bool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ERC20TokenGuidance(WithImplementation[Command], WithInjectedId):
    """
    Invoke this tool to receive guidance on how ERC20 is usually modelled using the prover.

    You MUST NOT invoke this tool in parallel with other tools.
    """
    @override
    def run(self) -> Command:
        return Command(update={
            "messages": [ToolMessage(
                tool_call_id=self.tool_call_id,
                content="Advice is as follows..."
            ), HumanMessage(
                content=[load_jinja_template(
                    "erc20_advice.j2"
                ), "Carefully consider if explicit ERC20 contract instances are necessary for this protocol, or if the 'standard summarization' is sufficient."]
            )]
        })


# ---------------------------------------------------------------------------
# Shared agent construction
# ---------------------------------------------------------------------------

class _HarnessST(VFSState, MessagesState):
    result: NotRequired[ContractSetup]


type _HarnessGraph = CompiledStateGraph[_HarnessST, None, VFSInput, Any]


def _build_harness_graph(
    llm: BaseChatModel,
    source: HarnessProtocol,
    vfs_conf: VFSToolConfig,
    memory: BaseTool | None,
    checkpointer: Checkpointer,
) -> tuple[_HarnessGraph, VFSAccessor[_HarnessST]]:
    """Build the harness analysis graph and VFS accessor."""
    fs_tool_list, mat = vfs_tools(conf=vfs_conf, ty=_HarnessST)

    def _validate_mocks(
        s: _HarnessST,
        r: ContractSetup,
        _tid: str,
    ) -> ValidationResult:
        errors = []
        for m in r.external_contracts:
            if m.l == "DYNAMIC" or m.l == "MULTIPLE":
                for h in m.harnesses:
                    if mat.get(s, h.path) is None:
                        errors.append(f"Harness {h.path} for {m.path} not found on the VFS")
        if errors:
            return "Update rejected:\n" + "\n".join(errors)
        return None

    result_tool = result_tool_generator(
        "result",
        ContractSetup,
        "Tool to communicate the result of your analysis",
        validator=(_HarnessST, _validate_mocks),
    )

    tools: list[BaseTool] = [
        *fs_tool_list,
        result_tool,
        ERC20TokenGuidance.as_tool("erc20_guidance"),
    ]
    if memory is not None:
        tools.append(memory)

    base = Builder().with_llm(llm).with_loader(load_jinja_template)

    graph = base.with_input(
        VFSInput
    ).with_output_key(
        "result"
    ).with_state(
        _HarnessST
    ).with_default_summarizer(
        max_messages=50
    ).with_sys_prompt(
        "You are an expert Solidity developer who is very good at following instructions who works at Certora, Inc."
    ).with_tools(
        tools
    ).with_initial_prompt_template(
        "harness_prompt.j2",
        contract_spec=source,
    ).compile_async(checkpointer=checkpointer)

    return graph, mat


def _extract_harness_setup(
    st: _HarnessST,
    mat: VFSAccessor[_HarnessST],
) -> HarnessSetup:
    """Extract HarnessSetup from the final agent state."""
    res: ContractSetup = st["result"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    vfs: dict[str, str] = {}
    for c in res.external_contracts:
        if c.l == "MULTIPLE" or c.l == "DYNAMIC":
            for h in c.harnesses:
                cont = mat.get(st, h.path)
                assert cont is not None
                vfs[h.path] = cont.decode()
    return HarnessSetup(vfs=vfs, setup=res)


# ---------------------------------------------------------------------------
# Standalone library entry point
# ---------------------------------------------------------------------------

async def analyze_external_interactions(
    source: HarnessProtocol,
    forbidden_reads: str,
    llm: BaseChatModel | LLMParams,
) -> HarnessSetup:
    """Run harness analysis as a standalone library call.

    Uses an in-memory checkpointer and no memory tool. Returns the raw
    ``HarnessSetup`` with classifications and any generated harness source
    files, without writing to disk or running PreAudit.

    Args:
        source: Provides project_root, contract_name, and relative_path.
        forbidden_reads: Regex for file paths the agent may not read.
        llm: A chat model instance or LLMParams to create one.

    Returns:
        HarnessSetup with classifications and any generated harness source files.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from composer.workflow.services import create_llm

    the_llm = create_llm(llm) if isinstance(llm, LLMParams) else llm

    vfs_conf = VFSToolConfig(
        immutable=False,
        forbidden_read=forbidden_reads,
        forbidden_write=r"^(?!certora/)",
        put_doc_extra="You may only write files in the certora/ subdirectory",
    )

    graph, mat = _build_harness_graph(
        the_llm, source, vfs_conf, None, InMemorySaver(),
    )

    st = cast(_HarnessST, await graph.ainvoke(
        VFSInput(vfs={}, input=[]),
        config={
            "configurable": {"thread_id": "harness-generation"},
            "recursion_limit": 100,
        },
    ))

    return _extract_harness_setup(st, mat)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

async def setup_and_harness_agent(
    ctx: WorkflowContext[None],
    source: SourceCode,
    llm: BaseChatModel,
) -> Configuration:
    """Run harness analysis, write harness files, and run PreAudit setup.

    Constructs a VFS-backed agent that reads source files from ``source.project_root``
    and writes generated harness files to ``certora/harnesses/``. After the agent
    classifies all external contracts, harness files are materialized to disk and
    PreAudit is run to produce the compilation config.

    Requires ``with_handler()`` to be active (pipeline mode).

    Args:
        ctx: Workflow context for thread IDs, memory, and checkpointing.
        source: Source code metadata (project root, contract name, relative path).
        llm: Chat model for the harness analysis agent.

    Returns:
        Configuration with external contract classifications, prover config,
        summaries path, and user-defined types.

    Raises:
        RuntimeError: If PreAudit compilation fails.
    """
    vfs_conf = VFSToolConfig(
        immutable=False,
        fs_layer=source.project_root,
        forbidden_write=r"^(?!certora/)",
        put_doc_extra="You may only write files in the certora/ subdirectory",
    )

    graph, mat = _build_harness_graph(
        llm, source, vfs_conf, ctx.get_memory_tool(), ctx.checkpointer,
    )

    st = await run_to_completion(
        graph,
        input=VFSInput(vfs={}, input=[]),
        thread_id=ctx.uniq_thread_id(),
        description="Harness setup",
    )

    h_setup = _extract_harness_setup(st, mat)

    # Write harness files to disk and collect extra input files
    root = Path(source.project_root)
    extra_files: list[str] = []

    for c in h_setup.setup.external_contracts:
        match c.l:
            case "MULTIPLE" | "DYNAMIC":
                for h in c.harnesses:
                    output_path = root / h.path
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    output_path.write_text(h_setup.vfs[h.path])
                    extra_files.append(h.path)
            case "SINGLETON":
                extra_files.append(c.path)

    # Run PreAudit compilation analysis
    setup_result = run_preaudit_setup(
        root,
        source.relative_path,
        source.contract_name,
        *extra_files,
    )
    if isinstance(setup_result, SetupFailure):
        raise RuntimeError(f"PreAudit setup failed: {setup_result.error}")

    return Configuration(
        external_contracts=h_setup.setup.external_contracts,
        primary_entity=h_setup.setup.primary_entity,
        non_trivial_state=h_setup.setup.non_trivial_state,
        config=setup_result.config,
        summaries_path=str(setup_result.summaries_path),
        user_types=setup_result.user_types,
    )
