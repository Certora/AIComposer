"""
Interface generation agent: produces a Solidity interface from component analysis.

Takes the ApplicationSummary and system document, generates an interface that
covers all external entry points, and validates it with the Solidity compiler.
"""

import subprocess
import tempfile
from typing import NotRequired

from graphcore.graph import FlowInput

from langgraph.graph import MessagesState

from composer.spec.context import WorkflowContext, PlainBuilder, CVLOnlyBuilder, SystemDoc
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.component import ApplicationSummary

DESCRIPTION = "Interface generation"


async def generate_interface(
    ctx: WorkflowContext,
    summary: ApplicationSummary,
    input: SystemDoc,
    builder: PlainBuilder | CVLOnlyBuilder,
    solc_version: str,
) -> str:
    """Generate a Solidity interface from component analysis and system document.

    Returns validated Solidity interface source code.
    """

    solc_name = f"solc{solc_version}"

    class ST(MessagesState):
        result: NotRequired[str]

    def validate_interface(_s: ST, interface: str) -> str | None:
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sol") as f:
                f.write(interface)
                f.flush()
                proc = subprocess.run(
                    [solc_name, f.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
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
        "The complete Solidity interface source code",
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
    ).compile_async(
        checkpointer=ctx.checkpointer
    )

    input_parts: list[str | dict] = [
        "The system/design document is:",
        input.content,
    ]

    res = await run_to_completion(
        workflow,
        FlowInput(input=input_parts),
        thread_id=ctx.uniq_thread_id(),
        recursion_limit=30,
        description=DESCRIPTION,
    )
    assert "result" in res
    return res["result"]
