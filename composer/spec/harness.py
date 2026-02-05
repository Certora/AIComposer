from typing import Literal, Annotated, override, NotRequired
from pathlib import Path
import sys

from pydantic import Field, BaseModel, Discriminator

from langgraph.types import Command
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage, HumanMessage

from graphcore.tools.schemas import WithImplementation, WithInjectedId
from graphcore.tools.vfs import VFSState, VFSInput, VFSToolConfig, vfs_tools
from graphcore.tools.results import result_tool_generator, ValidationResult
from graphcore.graph import Builder

from composer.templates.loader import load_jinja_template
from composer.spec.trunner import run_to_completion_sync
from composer.spec.preaudit_setup import run_preaudit_setup, SetupFailure
from composer.spec.context import WorkspaceContext
from composer.workflow.services import get_checkpointer

class ExternalActor(BaseModel):
    name: str = Field(description="A short, descriptive name of the external contract being interacted with")
    description: str = Field(description="A short, precise description of what the external actor does and its interaction with the main contract")

class Summarizable(ExternalActor):
    l: Literal["SUMMARIZABLE"]
    suggested_summaries: str = Field(description="A natural language description of the suggested summaries to use for this contract")
    path: str | None = Field(description="The relative path to the source of the contract to be summarized; null if the implementation isn't available")

class SourceAvailable(ExternalActor):
    path: str = Field(description="The relative path to the source of the contract being described")

class NotFoundHavoc(ExternalActor):
    l: Literal["NOTFOUND_HAVOC"]


class Singleton(SourceAvailable):
    l: Literal["SINGLETON"]

class HarnessDef(BaseModel):
    path: str = Field(description="Path to the harness definition")
    harness_name: str = Field(description="The name of the contract defined in the harness file")
    suggested_role: str = Field(description="The suggested role of this harness; e.g., 'the first token' of the pool, etc.")


class WithHarnesses(BaseModel):
    harnesses: list[HarnessDef] = Field(description="The harnesses created to model this contract.")

class Dynamic(SourceAvailable, WithHarnesses):
    l: Literal["DYNAMIC"]

class Multiple(SourceAvailable, WithHarnesses):
    l: Literal["MULTIPLE"]

type ContractClassification = Annotated[
    Summarizable |
    NotFoundHavoc |
    Dynamic |
    Singleton |
    Multiple,
    Discriminator("l")
]

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
    setup: ContractSetup
    vfs: dict[str, str]
    is_v2: Literal["is_v2"]

def _harness_setup(
    ctx: WorkspaceContext,
    b: Builder[None, None, None]
) -> HarnessSetup:
    if (cached := ctx.cache_get()) is not None:
        adapted = cached
        if "is_v2" not in cached:
            adapted = adapted.copy()
            adapted["is_v2"] = "is_v2"
        return HarnessSetup.model_validate(adapted)
    
    class ST(VFSState, MessagesState):
        result: NotRequired[ContractSetup]

    fs_tools, _ = vfs_tools(conf=VFSToolConfig(
        immutable=False,
        forbidden_read=ctx.fs_filter,
        fs_layer=ctx.project_root,
        forbidden_write=r"^(?!certora/)",
        put_doc_extra="You may only write files in the certora/ subdirectory"
    ), ty=ST)

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
                    if path not in s["vfs"]:
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

    mem = ctx.get_memory_tool()

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
        [*fs_tools, result, mem, ERC20TokenGuidance.as_tool("erc20_guidance")]
    ).with_initial_prompt_template(
        "harness_prompt.j2",
        contract_spec=ctx
    ).build_async()[0].compile(checkpointer=get_checkpointer())

    harness_ctx = ctx.child("harness-setup")
    st = run_to_completion_sync(
        graph,
        input=VFSInput(vfs={}, input=[]),
        thread_id=harness_ctx.thread_id,
        recursion_limit=100,
    )

    res : ContractSetup = st["result"] # pyright: ignore[reportTypedDictNotRequiredAccess]
    to_ret = HarnessSetup(vfs=st["vfs"], setup=res, is_v2="is_v2")
    ctx.cache_put(
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
    
    h_setup = _harness_setup(ctx, b)
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
