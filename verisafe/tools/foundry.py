from graphcore.graph import WithToolCallId
from pydantic import Field
from typing import List, Annotated, Optional, Literal, Generic, TypeVar, TypedDict, Union
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from verisafe.core.state import CryptoStateGen
from verisafe.core.context import CryptoContext
from langgraph.config import get_stream_writer
from langgraph.runtime import get_runtime
from verisafe.diagnostics.stream import ManualSearchResult, FoundryResult
from verisafe.rag.types import ManualRef
import subprocess
import os
import tempfile
import json
from pathlib import Path
from pydantic import Discriminator
from verisafe.prover.types import StatusCodes


class FoundryTestArgs(WithToolCallId):
    """
    Run Foundry tests on smart contracts to verify their behavior.

    Foundry is a fast, portable and modular toolkit for Ethereum application development.
    This tool allows you to run specific test files or individual test functions to verify
    that smart contracts behave correctly according to their test specifications.

    The tool will:
    1. Set up a temporary Foundry project structure
    2. Copy the necessary contract files to the test environment
    3. Run the specified test file or test function
    4. Return the test results including pass/fail status and any error messages

    This is particularly useful for:
    - Verifying contract functionality after code generation
    - Running regression tests
    - Validating contract behavior against expected outcomes
    - Debugging contract issues through test failures
    """

    test_file: str = Field(description="""
        The path to the Foundry test file to run. This should be a .t.sol file containing
        test functions. The file MUST have been put into the virtual filesystem with
        prior invocations of the PutFile tool.
        
        Example: "test/ERC20.t.sol" or "test/MyContract.t.sol"
    """)

    test_function: Optional[str] = Field(default=None, description="""
        The specific test function to run within the test file. If not specified,
        all test functions in the file will be run.
        
        Example: "testMint" or "testTransfer"
    """)

    contract_files: List[str] = Field(description="""
        List of contract files that the test depends on. These files MUST have been
        put into the virtual filesystem with prior invocations of the PutFile tool.
        
        Include all contracts that the test file imports or uses.
        Example: ["src/ERC20.sol", "src/IERC20.sol"]
    """)

    forge_version: str = Field(default="latest", description="""
        The Foundry version to use for running tests. Can be "latest" or a specific
        version like "0.2.0". Defaults to "latest".
    """)

    verbosity: int = Field(default=2, description="""
        Verbosity level for test output (0-5). Higher numbers provide more detailed output.
        Default is 2, which provides a good balance of information and readability.
    """)

    state: Annotated[CryptoStateGen, InjectedState]


@tool(args_schema=FoundryTestArgs)
def foundry_test(
    test_file: str,
    contract_files: List[str],
    test_function: Optional[str] = None,
    forge_version: str = "latest",
    verbosity: int = 2,
    state: Annotated[CryptoStateGen, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None
) -> str:
    """
    Run Foundry tests on smart contracts.
    
    This function creates a temporary Foundry project, copies the necessary files,
    and runs the specified tests, returning detailed results.
    """
    runtime = get_runtime(CryptoContext)
    writer = get_stream_writer()
    
    try:
        # Create a temporary directory for the Foundry project
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize Foundry project
            result = subprocess.run(
                ["forge", "init", "--no-git", "--no-commit", str(temp_path)],
                capture_output=True,
                text=True,
                cwd=temp_path
            )
            
            if result.returncode != 0:
                return f"Failed to initialize Foundry project: {result.stderr}"
            
            # Copy contract files to src/ directory
            src_dir = temp_path 
            for contract_file in contract_files:
                if contract_file in state["virtual_fs"]:
                    file_content = state["virtual_fs"][contract_file]
                    # Create subdirectories if needed
                    target_path = src_dir / contract_file
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(file_content)
                else:
                    return f"Contract file {contract_file} not found in virtual filesystem"
            
            # Copy test file to test/ directory
            test_dir = temp_path 
            test_file = "test.t.sol"
            file_content = state["virtual_fs"][test_file]
            # Create subdirectories if needed
            target_path = test_dir / "test" / test_file
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(file_content)
            
            # Build the project first
            build_result = subprocess.run(
                ["forge", "build"],
                capture_output=True,
                text=True,
                cwd=temp_path
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
                text=True,
                cwd=temp_path
            )
            print(test_result.stdout)
            print(test_result.stderr)
            # Format the output
            output = {
                "test_file": test_file,
                "test_function": test_function,
                "contract_files": contract_files,
                "return_code": test_result.returncode,
                "stdout": test_result.stdout,
                "stderr": test_result.stderr,
                "success": test_result.returncode == 0
            }
            
            # Send progress update
            writer(FoundryResult(
                type="foundry_test",
                tool_id=tool_call_id,
                test_file=test_file,
                test_function=test_function,
                success=(test_result.returncode == 0),
                return_code=test_result.returncode,
                stdout=test_result.stdout,
                stderr=test_result.stderr,
                error_message=None
            ))
            print(json.dumps(output, indent=2))
            return json.dumps(output, indent=2)
            
    except Exception as e:
        error_msg = f"Error running Foundry tests: {str(e)}"
        writer(FoundryResult(
            type="foundry_test", 
            tool_id=tool_call_id,
            test_file=test_file,
            test_function=test_function,
            success=False,
            return_code=-1,
            stdout="",
            stderr="",
            error_message=error_msg
        ))
        return error_msg

