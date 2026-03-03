"""
Stub generation agent: produces a minimal Solidity stub from an interface.

The stub imports the interface, declares the contract, and compiles.
No storage variables — those come from the semantic registry during CVL generation.
"""

import subprocess
import tempfile
from typing import NotRequired
from pydantic import BaseModel

from graphcore.graph import FlowInput

from langgraph.graph import MessagesState

from composer.spec.context import WorkflowContext, PlainBuilder, CVLOnlyBuilder
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.context import CacheKey
from composer.spec.util import string_hash

DESCRIPTION = "Stub generation"

class _CachedStub(BaseModel):
    stub: str

STUB_KEY = CacheKey[None, _CachedStub]("STUB")

async def generate_stub(
    ctx: WorkflowContext[None],
    interface: str,
    contract_name: str,
    builder: PlainBuilder | CVLOnlyBuilder,
    solc_version: str,
) -> str:
    """Generate a minimal Solidity stub that imports the interface and compiles.

    Returns validated Solidity stub source code.
    """

    key = CacheKey[None, _CachedStub](f"stub-for-{string_hash(interface)}")

    child = ctx.child(key, {"intf": interface})

    if (c := child.cache_get(_CachedStub)) is not None:
        return c.stub

    solc_name = f"solc{solc_version}"

    class ST(MessagesState):
        result: NotRequired[str]

    def validate_stub(_s: ST, stub: str) -> str | None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                import pathlib
                root = pathlib.Path(tmpdir)
                (root / "Intf.sol").write_text(interface)
                (root / "Impl.sol").write_text(stub)
                proc = subprocess.run(
                    [solc_name, str(root / "Impl.sol")],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if proc.returncode != 0:
                    return (
                        f"Stub compilation failed:\n"
                        f"stdout:\n{proc.stdout}\n"
                        f"stderr:\n{proc.stderr}"
                    )
                if "Intf.sol" not in stub:
                    return "Stub must import the interface file (Intf.sol)."
                if contract_name not in stub:
                    return f"Stub must declare a contract named {contract_name}."
        except FileNotFoundError:
            return f"Solidity compiler {solc_name} not found on this system"
        return None

    workflow = bind_standard(
        builder, ST,
        "The complete Solidity stub source code",
        validator=validate_stub,
    ).with_input(
        FlowInput
    ).with_sys_prompt(
        "You are an expert Solidity developer. Generate minimal stub implementations "
        "for formal verification."
    ).with_initial_prompt_template(
        "stub_generation_prompt.j2",
        contract_name=contract_name,
        solc_version=solc_version,
    ).compile_async(
        checkpointer=ctx.checkpointer
    )

    input_parts: list[str | dict] = [
        "The interface to implement is:",
        interface,
    ]

    res = await run_to_completion(
        workflow,
        FlowInput(input=input_parts),
        thread_id=ctx.uniq_thread_id(),
        recursion_limit=20,
        description=DESCRIPTION,
    )
    assert "result" in res
    child.cache_put(_CachedStub(stub=res["result"]))
    return res["result"]
