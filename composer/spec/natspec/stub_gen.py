"""
Stub generation agent: produces a minimal Solidity stub from an interface.

The stub imports the interface, declares the contract, and compiles.
No storage variables — those come from the semantic registry during CVL generation.
"""

import subprocess
import tempfile
from typing import NotRequired
from pydantic import BaseModel, Field

from graphcore.graph import FlowInput

from langgraph.graph import MessagesState

from composer.spec.context import WorkflowContext, PlainBuilder
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.context import CacheKey
from composer.spec.util import string_hash
from composer.spec.natspec.interface_gen import InterfaceResult
from composer.spec.util import uniq_thread_id

DESCRIPTION = "Stub generation"

class StubDeclaration(BaseModel):
    """
    The generated stub
    """
    solidity_identifier: str = Field(description="The contract name (solidity identifier) chosen for the stub")
    content: str = Field(description="The complete Solidity file which declares the stub implementation")

    @property
    def path(self) -> str:
        return f"{self.solidity_identifier}.sol"

STUB_KEY = CacheKey[None, StubDeclaration]("STUB")

async def generate_stub(
    ctx: WorkflowContext[None],
    interface: InterfaceResult,
    contract_name: str,
    builder: PlainBuilder,
    solc_version: str,
) -> StubDeclaration:
    """Generate a minimal Solidity stub that imports the interface and compiles.

    Returns validated Solidity stub source code.
    """

    key = CacheKey[None, StubDeclaration](f"stub-for-{string_hash(interface.model_dump_json())}-{contract_name}")

    child = await ctx.child(key, {"intf": interface.model_dump(), "contract": contract_name})

    if (c := await child.cache_get(StubDeclaration)) is not None:
        return c

    solc_name = f"solc{solc_version}"

    interface_to_implement = interface.name_to_interface[contract_name]

    interface_name = interface_to_implement.solidity_identifier,


    class ST(MessagesState):
        result: NotRequired[StubDeclaration]

    def validate_stub(_s: ST, stub: StubDeclaration) -> str | None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                import pathlib
                root = pathlib.Path(tmpdir)
                for (_, intf) in interface.name_to_interface.items():
                    rel_path = pathlib.Path("interfaces") / intf.path
                    final_path = root / rel_path
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    final_path.write_text(intf.content)
                
                contract_rel_path = pathlib.Path("contracts") / stub.path
                contract_path = root / contract_rel_path
                contract_path.parent.mkdir(parents=True, exist_ok=True)
                contract_path.write_text(stub.content)
                proc = subprocess.run(
                    [solc_name, str(contract_rel_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=tmpdir
                )
                if proc.returncode != 0:
                    return (
                        f"Stub compilation failed:\n"
                        f"stdout:\n{proc.stdout}\n"
                        f"stderr:\n{proc.stderr}"
                    )
                if interface_to_implement.path not in stub.content:
                    return f"Stub must import the interface file ({interface_to_implement.path})."
                if stub.solidity_identifier not in stub.content:
                    return f"Stub must declare a contract named {stub.solidity_identifier}."
        except FileNotFoundError:
            return f"Solidity compiler {solc_name} not found on this system"
        return None

    workflow = bind_standard(
        builder, ST, validator=validate_stub,
    ).with_input(
        FlowInput
    ).with_sys_prompt(
        "You are an expert Solidity developer. You are tasked with generating stub implementations "
        "for formal verification."
    ).with_initial_prompt_template(
        "stub_generation_prompt.j2",
        contract_name=contract_name,
        interface_name=interface_name,
        the_interface=interface.name_to_interface[contract_name].content,
        solc_version=solc_version,
    ).compile_async()

    input_parts : list[str | dict] = []

    res = await run_to_completion(
        workflow,
        FlowInput(input=input_parts),
        thread_id=uniq_thread_id("stub-gen"),
        recursion_limit=20,
        description=f"{DESCRIPTION}: {contract_name}",
    )
    assert "result" in res
    await child.cache_put(res["result"])
    return res["result"]
