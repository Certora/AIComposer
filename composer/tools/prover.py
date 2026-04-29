from typing import Annotated, Optional

from pydantic import Field

from graphcore.graph import tool_return, tool_state_update
from graphcore.tools.schemas import WithInjectedId, WithAsyncImplementation

from langchain_core.messages import AIMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.runtime import get_runtime

from composer.core.state import AIComposerState
from composer.core.context import AIComposerContext, compute_state_digest
from composer.core.validation import prover as prover_key
from composer.prover.runner import certora_prover as prover_impl, RawReport, SummarizedReport
from composer.ui.tool_display import tool_display


@tool_display(
    lambda p: (
        "Running prover: " + p.get("target_contract", "")
        + (f" — rule {p['rule']}" if p.get("rule") else "")
    ),
    None,
)
class CertoraProverTool(WithInjectedId, WithAsyncImplementation[Command | str]):
    """
    Invoke the Certora Prover, a powerful symbolic reasoning tool for verifying the correctness of smart contracts.

    The Certora Prover operates on one or more smart contracts files, and a specification for their behavior, written in a domain
    specific language called CVL, for which you have the documentation. A specification for the code you are generating
    has been provided for you, and is composed of multiple `rules`. Each rule defines the acceptable behavior
    of the smart contract in terms of assertions; a violated assertion means the smart contract's behavior is
    unacceptable.

    The Certora Prover will automatically check whether a smart contract instance (the "contract under verifiction")
    satisfies the provided specification on a per rule basis.

    For each rule, the prover will give one of the following result:
    1. VERIFIED: the smart contract satisfies the rule for all possible inputs
    2. VIOLATED: The smart contract violates the specification. As part of this result, the prover will provide
    a concrete counter example for the input/states which lead to the violation
    3. TIMEOUT: The automated reasoning used by the prover timed out before giving a response either way
    4. SANITY_FAIL: The rule succeeded, but was vacuously true, perhaps due to too specific requirements
    5. ERROR/other: There was some internal error within the prover

    When there are large numbers of failures, these result may be truncated. If this occurs, a summary of results from
    the prover will be provided.
    """

    source_files: list[str] = Field(description="""
      The (relative) filenames to verify. These files MUST have been put into the virtual filesystem with
      prior invocations of the PutFile tool.

      IMPORTANT: Only the files containing the contracts to be verified need to be passed as this
      argument. Thus, if verifying `MyContract`, only `src/MyContract.sol` needs to be provided here.
      HOWEVER: any files necessary to the compilation of `MyContract` must also have been placed on the
      virtual filesystem before calling the prover.
    """)

    target_contract: str = Field(description="""
       The name of the contract to check the specification against. This contract should be
       the main "entry point" for the functionality being synthesized. NB: the name of this
       contract should match one of the names provided in `source_files` according to that input's rules.
    """)

    compiler_version: str = \
        Field(description="The compiler to use when compiling the Solidity source code."
              "This parameter is specified as 'solcX.Y' where X.Y indicate solidity version 0.X.Y, or simply `solc`, which uses the system "
              "default. For example, solc8.29 uses the Solidity compiler for version 0.8.29, solc7.0 uses 0.7.0, etc." \
              "When possible, use specific compiler versions, falling back on the system default `solc` only as a last resort." \
              "Attempt to use the most recent compiler version if you can, which is solc8.29.")

    loop_iter: int = \
        Field(description="The Certora Prover uses bounded verification for looping code; "
              "any statically unbounded loops are unrolled a fixed number time, and then the loop condition is *assumed*. "
              "You should set this number as low as possible so that non-trivial looping behavior is observed. "
              "While values above 3 are technically supported, performance becomes exponentially worse, and thus"
              "should be avoided whenever possible.")

    rule: Optional[str] = \
        Field(description="The specific rule to check from the target spec file. If unspecified,"
              "all rules are run. Before delivering the finished code to the user, ensure that all rules pass on the most"
              "up to date version of the code. However, when iteratively developing code, it may be useful to focus on a"
              "single, 'problematic' rule.")

    target_spec: Optional[str] = Field(description=(
        "The VFS path of the committed spec file to verify against. Required "
        "when ``use_working_spec`` is false; the prover reads this spec from "
        "the VFS and records a stamp at ``validation[\"prover:<target_spec>\"]`` "
        "on success. Ignored when ``use_working_spec`` is true (the transient "
        "working draft has no target until committed; runs against it do not "
        "produce a stamp)."
    ))

    use_working_spec : bool = Field(description=(
        "If true, verify against the current working spec draft instead of a "
        "committed spec file. The draft is transient and has no VFS path, so "
        "``target_spec`` is ignored in this mode and no prover stamp is "
        "produced. Use this mode to iterate on a draft before committing it."
    ))

    state: Annotated[AIComposerState, InjectedState]

    async def run(self) -> Command | str:
        if not self.use_working_spec and not self.target_spec:
            return (
                "target_spec is required when use_working_spec is false. "
                "Pass the VFS path of one of the registered spec files."
            )
        last = self.state["messages"][-1]
        if isinstance(last, AIMessage):
            tcs = last.tool_calls
            if any(tc["id"] == self.tool_call_id for tc in tcs) and len(tcs) > 1:
                return "Error: certora_prover must be the only tool call in its turn. Re-issue this call alone."
        result = await prover_impl(
            self.source_files,
            self.target_contract,
            self.compiler_version,
            self.loop_iter,
            self.rule,
            self.state,
            self.tool_call_id,
            self.use_working_spec,
            self.target_spec,
        )
        match result:
            case str():
                return result
            case RawReport():
                # Stamp only on: a full (non-ruled) run against a committed
                # spec that verified. Working-spec runs never stamp.
                if (
                    result.all_verified
                    and not self.use_working_spec
                    and not self.rule
                    and self.target_spec is not None
                ):
                    ctxt = get_runtime(AIComposerContext).context
                    state_digest = compute_state_digest(c=ctxt, state=self.state)
                    return tool_state_update(
                        self.tool_call_id, result.report, validation={
                            # One stamp per spec — the overall task is
                            # complete when every registered spec has its
                            # own stamped entry.
                            f"{prover_key}:{self.target_spec}": state_digest
                        }
                    )
                return tool_return(tool_call_id=self.tool_call_id, content=result.report)
            case SummarizedReport():
                return result.todo_list


certora_prover = CertoraProverTool.as_tool("certora_prover")
