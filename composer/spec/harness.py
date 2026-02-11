from typing import Literal, Annotated, override, NotRequired, Protocol, Any, cast
from pathlib import Path
from dataclasses import dataclass
import sys

from pydantic import Field, BaseModel, Discriminator

from langgraph.types import Command, Checkpointer
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import BaseTool

from graphcore.tools.schemas import WithImplementation, WithInjectedId
from graphcore.tools.vfs import VFSState, VFSInput, VFSAccessor
from graphcore.tools.results import result_tool_generator, ValidationResult
from graphcore.graph import Builder, BaseChatModel

from composer.templates.loader import load_jinja_template
from composer.spec.trunner import run_to_completion_sync
from composer.spec.preaudit_setup import run_preaudit_setup, SetupFailure
from composer.spec.context import WorkspaceContext
from composer.workflow.services import get_checkpointer, create_llm

class ExternalActor(BaseModel):
    """Base class for any external contract that the main contract directly interacts with.

    Each external actor is identified during the harness analysis step and carries a human-readable
    name and description of its role relative to the contract under verification.
    """
    name: str = Field(description="A short, descriptive name of the external contract being interacted with")
    description: str = Field(description="A short, precise description of what the external actor does and its interaction with the main contract")

class Summarizable(ExternalActor):
    """An external contract whose behavior can be adequately modelled using CVL summaries
    without including its full implementation in the prover.

    Typical examples: price oracles (return an arbitrary number), simple registries.
    The ``suggested_summaries`` field contains a natural-language description of how the
    agent recommends summarizing the contract's methods (e.g., "return an arbitrary uint256
    for getPrice"). ``path`` is provided when the source is available for reference, but the
    implementation itself is NOT included in the prover input.
    """
    l: Literal["SUMMARIZABLE"]
    suggested_summaries: str = Field(description="A natural language description of the suggested summaries to use for this contract")
    path: str | None = Field(description="The relative path to the source of the contract to be summarized; null if the implementation isn't available")

class SourceAvailable(ExternalActor):
    """Intermediate base for classifications where the external contract's source code is available."""
    path: str = Field(description="The relative path to the source of the contract being described")

class NotFoundHavoc(ExternalActor):
    """An external contract whose source code is not available.

    The prover will apply its default pessimistic (havoc) assumptions to any calls
    into this contract — i.e., the return value and any side effects are unconstrained.
    """
    l: Literal["NOTFOUND_HAVOC"]


class Singleton(SourceAvailable):
    """An external contract for which exactly one instance is sufficient to model the
    protocol in a non-trivial state.

    The source at ``path`` is included directly in the prover input as a single
    contract instance; no harnesses are needed.
    """
    l: Literal["SINGLETON"]

class HarnessDef(BaseModel):
    """A generated harness file that creates a uniquely-named contract extending an external contract.

    Because the Certora Prover maps each contract declaration to exactly one on-chain instance,
    multiple instances of the same contract require distinct harness contracts
    (e.g., ``contract TokenA is ERC20 {}``). Each harness is a minimal Solidity file placed in
    ``certora/harnesses/``.
    """
    path: str = Field(description="Path to the harness definition")
    harness_name: str = Field(description="The name of the contract defined in the harness file")
    suggested_role: str = Field(description="The suggested role of this harness; e.g., 'the first token' of the pool, etc.")


class WithHarnesses(BaseModel):
    """Mixin for classifications that require multiple prover instances and therefore harness files."""
    harnesses: list[HarnessDef] = Field(description="The harnesses created to model this contract.")

class Dynamic(SourceAvailable, WithHarnesses):
    """An external contract that is dynamically instantiated by the main contract at runtime
    (e.g., factory-created wrapper tokens).

    The number of harness instances reflects how many such dynamic instances are needed to
    represent the protocol in a non-trivial state.
    """
    l: Literal["DYNAMIC"]

class Multiple(SourceAvailable, WithHarnesses):
    """An external contract that is NOT instantiated by the main contract, but of which
    multiple pre-existing instances are needed to model a non-trivial state
    (e.g., two distinct ERC20 tokens in a swap pool).

    Each required instance gets its own harness file.
    """
    l: Literal["MULTIPLE"]

type ContractClassification = Annotated[
    Summarizable |
    NotFoundHavoc |
    Dynamic |
    Singleton |
    Multiple,
    Discriminator("l")
]
"""Discriminated union of all external contract classifications.

The ``l`` field serves as the discriminator tag:
- ``SUMMARIZABLE`` — model via CVL summaries, don't include implementation
- ``NOTFOUND_HAVOC`` — source unavailable, prover uses pessimistic assumptions
- ``SINGLETON`` — include source directly as a single prover instance
- ``DYNAMIC`` — contract is instantiated by the main contract; generate N harnesses
- ``MULTIPLE`` — multiple pre-existing instances needed; generate N harnesses
"""

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

class ContractSetup(BaseModel):
    """
    The result of your analysis
    """
    external_contracts: list[ContractClassification] = Field(description="The external actors classified by your analysis")
    primary_entity: str = Field(description="A description of the primary entity managed by this contract")
    non_trivial_state: str = Field(description="A semi-formal description of a `non-trivial state`. Reference the external " \
    "contracts you identified during the harnessing step as necessary.")

class Configuration(ContractSetup):
    config: dict  # Contents of compilation_config.conf
    summaries_path: str  # Path to summaries-{Contract}.spec, if generated
    user_types: list[dict]

class HarnessSetup(BaseModel):
    """The complete output of the harness analysis workflow.

    Attributes:
        setup: The LLM-produced analysis containing classified external contracts,
            the primary entity description, and a non-trivial state characterization.
        vfs: Virtual filesystem snapshot mapping harness file paths (e.g.,
            ``certora/harnesses/TokenA.sol``) to their generated Solidity source.
            Only populated for DYNAMIC and MULTIPLE classifications that required
            harness generation. SINGLETON and SUMMARIZABLE entries have no VFS artifacts.
        is_v2: Literal sentinel tag used to distinguish the current schema version
            from earlier cached formats (see ``_harness_setup`` cache migration).
    """
    setup: ContractSetup
    vfs: dict[str, str]
    is_v2: Literal["is_v2"]

class HarnessProtocol(Protocol):
    @property
    def project_root(self) -> str:
        """
        Path to the root of the project
        """
        ...

    @property
    def contract_name(self) -> str:
        """
        The name of the contract to analyze
        """
        ...

    @property
    def relative_path(self) -> str:
        """
        The relative path to `contract_name` within `project_root`
        """
        ...

from langgraph._internal._typing import StateLike
from langgraph.graph.state import CompiledStateGraph

class _ServiceProtocol(Protocol):
    def vfs_tools[T: VFSState](
        self,
        ty: type[T],
        forbidden_write: str | None = None,
        put_doc_extra: str | None = None
    ) -> tuple[list[BaseTool], VFSAccessor[T]]:
        ...

    def get_checkpointer(self) -> Checkpointer:
        ...

    def graph_runner[S: StateLike, I: StateLike](
        self,
        graph: CompiledStateGraph[S, None, I, Any],
        input: I,
        thread_id: str,
        recursion_limit: int
    ) -> S:
        ...

    def memory_tool(self) -> BaseTool | None:
        ...

    def tid(self) -> str:
        ...

@dataclass
class LLMParams:
    """Parameters to construct an LLM via ``create_llm`` for standalone use.

    Mirrors the ``ModelOptions`` protocol fields. Passed to ``ChatAnthropic`` as:
    ``model`` → ``model_name``, ``tokens`` → ``max_tokens_to_sample``,
    ``thinking_tokens`` → ``thinking.budget_tokens``.
    """
    model: str
    tokens: int
    thinking_tokens: int
    memory_tool: bool
    

def analyze_external_interactions(
    basic_ctx: HarnessProtocol,
    forbidden_reads: str,
    llm: BaseChatModel | LLMParams
) -> HarnessSetup:
    """Run the harness analysis agent to classify external contract interactions.

    An LLM agent inspects the contract's source, identifies every external contract
    it directly interacts with, and classifies each one (SUMMARIZABLE, NOTFOUND_HAVOC,
    SINGLETON, DYNAMIC, or MULTIPLE). For DYNAMIC/MULTIPLE contracts, minimal harness
    Solidity files are generated so the Certora Prover can model multiple instances.

    Args:
        basic_ctx: Provides project_root, contract_name, and relative_path.
        forbidden_reads: Regex for file paths the agent may not read.
        llm: A chat model instance or LLMParams to create one.

    Returns:
        HarnessSetup with classifications and any generated harness source files.
    """
    the_llm : BaseChatModel = create_llm(llm) if isinstance(llm, LLMParams) else llm

    b = Builder[None, None, None]().with_llm(the_llm).with_loader(
        load_jinja_template
    )

    class SVC():
        def memory_tool(self) -> None:
            return None
        
        def tid(self) -> str:
            return "harness-generation"
        
        def get_checkpointer(self) -> Checkpointer:
            from langgraph.checkpoint.memory import InMemorySaver
            return InMemorySaver()
            
        def vfs_tools[T: VFSState](
            self,
            ty: type[T],
            forbidden_write: str | None = None,
            put_doc_extra: str | None = None
        ) -> tuple[list[BaseTool], VFSAccessor[T]]:
            from graphcore.tools.vfs import VFSToolConfig, vfs_tools
            conf = VFSToolConfig(
                immutable=False, forbidden_read=forbidden_reads
            )
            if forbidden_write:
                conf["forbidden_write"] = forbidden_write
            if put_doc_extra:
                conf["put_doc_extra"] = put_doc_extra
            
            return vfs_tools(conf=conf, ty=ty)

        def graph_runner[S: StateLike, I: StateLike](
            self,
            graph: CompiledStateGraph[S, None, I, Any],
            input: I,
            thread_id: str,
            recursion_limit: int
        ) -> S:
            res = graph.invoke(input, config={"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit})
            return cast(S, res)

    return _harness_setup_inner(
        SVC(), basic_ctx, b
    )

def _harness_setup_inner(
    svc: _ServiceProtocol,
    ctx: HarnessProtocol,
    b: Builder[None, None, None]
) -> HarnessSetup:
    class ST(VFSState, MessagesState):
        result: NotRequired[ContractSetup]

    fs_tools, mat = svc.vfs_tools(
        forbidden_write=r"^(?!certora/)",
        put_doc_extra="You may only write files in the certora/ subdirectory",
        ty=ST
    )

    def validate_mocks(
        s: ST,
        r: ContractSetup,
        tid: str
    ) -> ValidationResult:
        errors = []
        for m in r.external_contracts:
            if m.l == "DYNAMIC" or m.l == "MULTIPLE":
                for h in m.harnesses:
                    path = h.path
                    harness_cont = mat.get(s, path)
                    if harness_cont is None:
                        errors.append(f"Harness {path} for {m.path} not found on the VFS")
        if errors:
            return "Update rejected:\n" + "\n".join(errors)
        return None

    result = result_tool_generator(
        "result",
        ContractSetup,
        "Tool to communicate the result of your analysis",
        validator=(ST, validate_mocks)
    )

    mem = svc.memory_tool()

    tools : list[BaseTool] = [*fs_tools, result, ERC20TokenGuidance.as_tool("erc20_guidance")]

    if mem is not None:
        tools.append(mem)

    graph = b.with_input(
        VFSInput
    ).with_output_key(
        "result"
    ).with_state(
        ST
    ).with_default_summarizer(
        max_messages=50
    ).with_sys_prompt(
        "You are an expert Solidity developer who is very good at following instructions who works at Certora, Inc."
    ).with_tools(
        tools
    ).with_initial_prompt_template(
        "harness_prompt.j2",
        contract_spec=ctx
    ).compile_async(checkpointer=svc.get_checkpointer())

    st = svc.graph_runner(
        graph,
        input=VFSInput(vfs={}, input=[]),
        thread_id=svc.tid(),
        recursion_limit=100,
    )

    res : ContractSetup = st["result"] # pyright: ignore[reportTypedDictNotRequiredAccess]

    vfs = {}
    for r in res.external_contracts:
        if r.l == "MULTIPLE" or r.l == "DYNAMIC":
            for h in r.harnesses:
                cont = mat.get(st, h.path)
                assert cont is not None
                vfs[h.path] = cont.decode()

    to_ret = HarnessSetup(vfs=vfs, setup=res, is_v2="is_v2")
    return to_ret

def _harness_setup(
    ctx: WorkspaceContext,
    b: Builder[None, None, None]
) -> HarnessSetup:
    harness_ctx = ctx.child("harnessing")
    if (cached := harness_ctx.cache_get()) is not None:
        adapted = cached
        if "is_v2" not in cached:
            adapted = adapted.copy()
            adapted["is_v2"] = "is_v2"
        return HarnessSetup.model_validate(adapted)
    
    class SVC():
        def tid(self) -> str:
            return harness_ctx.thread_id
        
        def get_checkpointer(self) -> Checkpointer:
            return get_checkpointer()
        
        def memory_tool(self) -> BaseTool:
            return harness_ctx.get_memory_tool()
        
        def vfs_tools[T: VFSState](self,
            ty: type[T],
            forbidden_write: str | None = None,
            put_doc_extra: str | None = None
        ) -> tuple[list[BaseTool], VFSAccessor[T]]:
            return ctx.vfs_tools(ty, forbidden_write, put_doc_extra)
    
        def graph_runner[S: StateLike, I: StateLike](self,
            graph: CompiledStateGraph[S, None, I, Any],
            input: I,
            thread_id: str,
            recursion_limit: int
        ) -> S:
            return run_to_completion_sync(graph, input, thread_id, recursion_limit=recursion_limit)
    
    svc : _ServiceProtocol = SVC()

    to_ret = _harness_setup_inner(
        svc, ctx, b
    )

    harness_ctx.cache_put(
        to_ret.model_dump()
    )
    return to_ret

def setup_and_harness_agent(
    ctx: WorkspaceContext,
    b: Builder[None, None, None],
    *,
    ignore_existing_config: bool
) -> Configuration:
    child_ctxt = ctx.child(
        "setup"
    )
    if (cached := child_ctxt.cache_get()) is not None:
        return Configuration.model_validate(cached)

    certora_dir = Path(ctx.project_root) / "certora"
    if certora_dir.exists() and not ignore_existing_config:
        raise RuntimeError(
            f"Directory {certora_dir} already exists. "
            "Use --ignore-existing-config to proceed anyway."
        )

    h_setup = _harness_setup(child_ctxt, b)
    vfs = h_setup.vfs
    res = h_setup.setup

    root = Path(ctx.project_root)
    extra_files = []
    for r in res.external_contracts:
        print("=" * 80)
        print("Name: " + r.name)
        print("> " + r.description)
        print("Sort: " + r.l)
        print("-" * 80)
        if r.l != "NOTFOUND_HAVOC":
            print(f"path = {r.path}")
        match r.l:
            case "MULTIPLE" | "DYNAMIC":
                for h in r.harnesses:
                    print(f"Harness of {r.path}")
                    output_path = root / h.path
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    cont = vfs[h.path]
                    output_path.write_text(cont)
                    extra_files.append(h.path)
                    print(h.path)
                    print(vfs[h.path])
            case "SINGLETON":
                extra_files.append(r.path)
            case "SUMMARIZABLE":
                print(r.suggested_summaries)

    print("Running pre-audit tool")

    r = run_preaudit_setup(
        Path(ctx.project_root),
        ctx.relative_path,
        ctx.contract_name,
        *extra_files
    )
    if isinstance(r, SetupFailure):
        print(f"Setup failed :( {r.error}")
        sys.exit(1)

    to_ret = Configuration(
        external_contracts=res.external_contracts,
        primary_entity=res.primary_entity,
        non_trivial_state=res.non_trivial_state,
        config=r.config,
        summaries_path=str(r.summaries_path),
        user_types=r.user_types
    )
    child_ctxt.cache_put(to_ret.model_dump())
    return to_ret
