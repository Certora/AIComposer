"""
Interface generation agent: produces a Solidity interface from component analysis.

Takes the ApplicationSummary and system document, generates an interface that
covers all external entry points, and validates it with the Solidity compiler.
"""

import subprocess
import tempfile
import pathlib
from pydantic import BaseModel, Field
from typing import NotRequired

from graphcore.graph import FlowInput

from langgraph.graph import MessagesState

from composer.spec.context import WorkflowContext, PlainBuilder, CacheKey
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.system_model import Application
from composer.spec.util import string_hash, uniq_thread_id

from logging import getLogger

_logger = getLogger(__name__)

DESCRIPTION = "Interface generation"

class InterfaceDecl(BaseModel):
    content: str = Field(description="The contents of `path`, which should hold a complete Solidity " \
    "interface describing the external entry points of the described contract(s)")
    solidity_identifier: str = Field(description="The solidity identifier of the interface")

    @property
    def path(self) -> str:
        return f"{self.solidity_identifier}.sol"

class InterfaceResult(BaseModel):
    """
    The result of your interface generation.
    """
    name_to_interface: dict[str, InterfaceDecl] = Field(description=
        "A mapping from the explicit contract name to the interface describing the behavior of that component"
    )

    def dump_to_path(self, p: pathlib.Path) -> list[pathlib.Path]:
        to_ret = []
        for (_, i) in self.name_to_interface.items():
            rel_path = pathlib.Path("interfaces") / i.path
            to_ret.append(to_ret)
            full_path = p / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(i.content)
        return to_ret


async def generate_interface(
    ctx: WorkflowContext[None],
    summary: Application,
    builder: PlainBuilder,
    solc_version: str,
) -> InterfaceResult:
    """Generate a Solidity interface from component analysis and system document.

    Returns validated Solidity interface source code.
    """
    cache_key = CacheKey[None, InterfaceResult](
        f"interface-{string_hash(summary.model_dump_json())}"
    )

    child = ctx.child(cache_key, summary.model_dump())

    if (cached := child.cache_get(InterfaceResult)) is not None:
        return cached

    solc_name = f"solc{solc_version}"

    class ST(MessagesState):
        result: NotRequired[InterfaceResult]

    external_contracts = { c.name for c in summary.contract_components  }

    def validate_interface(_s: ST, interface: InterfaceResult) -> str | None:
        seen = set()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                _logger.info(f"Writing to {temp_dir}")
                _logger.info(f"Interfaces: {interface.name_to_interface.keys()}")
                root = pathlib.Path(temp_dir)
                compile_inputs = []
                for (nm, i) in interface.name_to_interface.items():
                    _logger.info(f"{nm} -> {i.path}")
                    if nm not in external_contracts:
                        return f"Invalid entry found; no external contract with name {nm} appears in input"
                    rel_path = pathlib.Path("interfaces") / i.solidity_identifier 
                    target = root / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(i.content)
                    seen.add(nm)
                    compile_inputs.append(str(rel_path))
                if seen != external_contracts:
                    return f"Missing results for contract(s): {external_contracts - seen}"
                
                proc = subprocess.run(
                    [solc_name] + compile_inputs,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=temp_dir
                )
                if proc.returncode != 0:
                    return (
                        f"Interface compilation failed:\n"
                        f"stdout:\n{proc.stdout}\n"
                        f"stderr:\n{proc.stderr}"
                    )
        except FileNotFoundError:
            return f"Solidity compiler {solc_name} not found on this system"
        return None

    workflow = bind_standard(
        builder, ST,
        validator=validate_interface,
    ).with_input(
        FlowInput
    ).with_sys_prompt(
        "You are an expert Solidity developer specializing in interface design for "
        "formal verification of smart contracts."
    ).with_initial_prompt_template(
        "interface_generation_prompt.j2",
        summary=summary,
        solc_version=solc_version,
    ).compile_async()

    res = await run_to_completion(
        workflow,
        FlowInput(input=[]),
        thread_id=uniq_thread_id("interface-gen"),
        recursion_limit=30,
        description=DESCRIPTION,
    )
    assert "result" in res
    child.cache_put(res["result"])
    return res["result"]
