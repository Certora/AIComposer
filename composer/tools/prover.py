from typing import Annotated, Optional
import hashlib
from pydantic import Field

from graphcore.graph import WithToolCallId, tool_return

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.runtime import get_runtime

from composer.core.state import AIComposerState
from composer.core.context import AIComposerContext, compute_state_digest
from composer.core.validation import prover as prover_key
from composer.prover.runner import certora_prover as certora_prover_impl, RawReport, SummarizedReport

class CertoraProverArgs(WithToolCallId):
    """
    Invoke the Certora Prover, a powerful symbolic reasoning tool for verifying the correctness of smart contracts.

    The Certora Prover operates on one or more smart contracts files, and a specification for their behavior, written in a domain
    specific language called CVL, for which you have the documentation. A specification for the code you are generating
    has been provided for you, and is composed of multiple `rules`. Each rule defines the acceptable behavior
    of the smart contract in terms of assertions; a violated assertion means the smart contract's behavior is
    unacceptable.

    The Certora Prover will automatically check whether a smart contract instance (the "contract under verification")
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
        Field(description="The specific rule to check from the `spec_file`. If unspecified,"
              "all rules are run. Before delivering the finished code to the user, ensure that all rules pass on the most"
              "up to date version of the code. However, when iteratively developing code, it may be useful to focus on a"
              "single, 'problematic' rule.")

    state: Annotated[AIComposerState, InjectedState]


class SolanaProverArgs(WithToolCallId):
    """
    Invoke the Certora Solana Prover, a powerful symbolic reasoning tool for verifying the correctness of Solana programs.

    The Certora Solana Prover operates on one or more Rust files, and a specification for their behavior, written in a domain
    specific language called CVLR (Certora Verification Language for Rust), for which you have the documentation.
    A specification for the code you are generating has been provided for you, and is composed of multiple `rules`.
    Each rule, marked with the `#[rule]` attribute, defines the acceptable behavior of the Solana program in terms of assertions.

    The Certora Solana Prover will automatically invoke cargo to compile the code before actually running the verification steps.
    Afterwards, the Certora Solana Prover will check whether a Solana program satisfies the provided specification
    on a per rule basis.
     
    For each rule, the prover will give one of the following result:
    1. VERIFIED: the program satisfies the rule for all possible inputs
    2. VIOLATED: The program violates the specification. As part of this result, the prover will provide
    a concrete counter example for the input/states which lead to the violation
    3. TIMEOUT: The automated reasoning used by the prover timed out before giving a response either way
    4. SANITY_FAIL: The rule succeeded, but was vacuously true, perhaps due to too specific requirements
    5. ERROR/other: There was some internal error within the prover

    IMPORTANT: The Solana prover requires rules to be specified via the --rule flag. You must specify
    which rule to verify.

    When there are large numbers of failures, these result may be truncated. If this occurs, a summary of results from
    the prover will be provided.
    """

    rule: str = Field(description="""
      The specific rule to check. This is REQUIRED for the Solana prover.
      The rule name should match a function marked with #[rule] in the specification.
      For example: "rule_vault_solvency_deposit" or "rule_correct_add_performs_addition".
    """)

    state: Annotated[AIComposerState, InjectedState]


def _handle_prover_result(
    result: str | RawReport | SummarizedReport,
    tool_call_id: str,
    state: AIComposerState,
) -> Command:
    match result:
        case str():
            return tool_return(tool_call_id=tool_call_id, content=result)
        case RawReport():
            if result.all_verified:
                ctxt = get_runtime(AIComposerContext).context
                state_digest = compute_state_digest(c=ctxt, state=state)
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                tool_call_id=tool_call_id,
                                content=result.report
                            )
                        ],
                        "validation": {
                            prover_key: state_digest
                        }
                    }
                )
            return tool_return(tool_call_id=tool_call_id, content=result.report)
        case SummarizedReport():
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            tool_call_id=tool_call_id,
                            content="... Output truncated ..."
                        ),
                        HumanMessage(
                            content=[
                                "The prover output was too large for the context window. A TODO list extracted from its output is as follows",
                                result.todo_list
                            ]
                        )

                    ]
                }
            )

@tool(args_schema=CertoraProverArgs)
def certora_prover(
    source_files: list[str],
    target_contract: str,
    compiler_version: str,
    loop_iter: int,
    rule: Optional[str],
    state: Annotated[AIComposerState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    result = certora_prover_impl(
        state=state,
        tool_call_id=tool_call_id,
        rule=rule,
        source_files=source_files,
        target_contract=target_contract,
        compiler_version=compiler_version,
        loop_iter=loop_iter,
    )
    return _handle_prover_result(result, tool_call_id, state)


@tool(args_schema=SolanaProverArgs)
def solana_prover(
    rule: str,
    state: Annotated[AIComposerState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    result = certora_prover_impl(
        state=state,
        tool_call_id=tool_call_id,
        rule=rule,
    )
    return _handle_prover_result(result, tool_call_id, state)
