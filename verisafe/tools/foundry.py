from graphcore.graph import WithToolCallId
from pydantic import Field
from typing import Annotated, Optional
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from verisafe.core.state import CryptoStateGen
from verisafe.core.context import CryptoContext
from langgraph.config import get_stream_writer
from langgraph.runtime import get_runtime
from verisafe.diagnostics.stream import FoundryResult
import subprocess
import json
import contextlib



class FoundryTestArgs(WithToolCallId):
    """
    Run Foundry tests on smart contracts to verify their behavior.

    Foundry is a fast, portable and modular toolkit for Ethereum application development.
    This tool allows you to run specific test files or individual test functions to verify
    that smart contracts behave correctly according to their test specifications.

    The tool will:
    1. materialize the virtual filesystem into a temporary directory
    2. build the project
    3. Run the foundry tests
    4. Return the test results including pass/fail status and any error messages

    This is particularly useful for:
    - Verifying contract functionality after code generation
    - Running regression tests
    - Validating contract behavior against expected outcomes
    - Debugging contract issues through test failures
    """
    test_function: Optional[str] = Field(default=None, description="""
        The specific test function to run within the test file. It uses --match-test flag of foundry and support regex.
        If not specified, all test functions in the file will be run.
        
        Example: "testMint" or "testTransfer"
    """)

    verbosity: int = Field(default=1, description="""
        Verbosity level for test output (0-5). Higher numbers provide more detailed output.
        Default is 1, which provides a good balance of information and readability.
    """)

    state: Annotated[CryptoStateGen, InjectedState]


@tool(args_schema=FoundryTestArgs)
def foundry_test(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[CryptoStateGen, InjectedState],
    test_function: Optional[str] = None,
    verbosity: int = 1,
) -> str:
    """
    Run Foundry tests on smart contracts.
    
    This function materializes the virtual filesystem into a temporary directory,
    builds the project using forge build, and runs the specified tests using forge test.
    Test results including pass/fail status, stdout, stderr and return codes are returned
    as a JSON string.
    """
    runtime = get_runtime(CryptoContext)
    ctxt = runtime.context
    writer = get_stream_writer()
    with ctxt.vfs_materializer.materialize(state, debug=ctxt.prover_opts.keep_folder) as temp_dir:
        with contextlib.chdir(temp_dir):
            # Build the project first
            build_result = subprocess.run(
                ["forge", "build"],
                capture_output=True,
                text=True
            )
            
            if build_result.returncode != 0:
                return f"Failed to build project: {build_result.stderr}"
            
            # Run the tests
            test_cmd = ["forge", "test"]

            # Map verbosity int (0–5) to -v, -vv, ... flags
            level = max(0, min(5, int(verbosity or 0)))
            if level > 0:
                test_cmd.append("-" + "v" * level)

            if test_function:
                test_cmd.extend(["--match-test", test_function])
            
            test_result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True
            )
            print(test_result.stdout)
            print(test_result.stderr)
            # Format the output
            output = {
                "test_function": test_function,
                "return_code": test_result.returncode,
                "stdout": test_result.stdout,
                "stderr": test_result.stderr,
                "success": test_result.returncode == 0
            }
            
            # Send progress update
            writer(FoundryResult(
                type="foundry_test",
                tool_id=tool_call_id,
                test_function=test_function,
                success=(test_result.returncode == 0),
                return_code=test_result.returncode,
                stdout=test_result.stdout,
                stderr=test_result.stderr,
                error_message=None
            ))
            return json.dumps(output)
   
