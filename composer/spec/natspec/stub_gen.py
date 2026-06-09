"""
Stub generation agent: produces a minimal Solidity stub from an interface.

The stub imports the interface, declares the contract, and compiles.
No storage variables — those come from the semantic registry during CVL generation.
"""

import asyncio
import pathlib
from collections.abc import Callable
from typing import NotRequired, cast, override

from graphcore.graph import FlowInput

from langgraph.graph import MessagesState

from composer.spec.context import WorkflowContext, PlainBuilder
from composer.spec.graph_builder import run_to_completion
from composer.spec.context import CacheKey
from composer.spec.util import string_hash
from composer.spec.natspec.async_result import AsyncResultTool
from composer.spec.natspec.models import (
    InterfaceResult,
    StubDeclarationModel,
)
from composer.spec.system_model import ContractName, SolidityIdentifier
from composer.spec.natspec.task_description import (
    AgentDescription,
    Assembler,
    StubGenCallParams,
    resolve_extra_input,
)
from composer.spec.util import uniq_thread_id
from composer.spec.service_host import ServiceHost

DESCRIPTION = "Stub generation"


async def generate_stub[S: StubDeclarationModel](
    ctx: WorkflowContext[None],
    interface: InterfaceResult,
    env: ServiceHost,
    contract_name: ContractName,
    solidity_identifier: SolidityIdentifier,
    solc_version: str,
    materializer: Assembler,
    description: AgentDescription[S, StubGenCallParams],
) -> S:
    """Generate a minimal Solidity stub that imports the interface and compiles.

    The candidate stub is validated by laying it out through the caller-supplied
    ``assembler_for_candidate`` factory (which produces a fresh ``Assembler``
    seeded with the candidate), then invoking solc inside the assembled project.
    ``description`` fixes the concrete stub subtype and the prompt (with any
    workflow-constant params pre-bound).

    ``solidity_identifier`` is both the lookup key into
    ``interface.name_to_interface`` and the Solidity identifier the stub MUST
    declare — caller-supplied (from the contract record), validator-enforced.
    ``contract_name`` is the conceptual label used only for the task
    description string.
    """
    stub_ty : type[S] = description.output_ty

    key = CacheKey[None, StubDeclarationModel](
        f"stub-for-{string_hash(interface.model_dump_json())}-{solidity_identifier}-{stub_ty.__name__}"
    )

    child = await ctx.child(key, {"intf": interface.model_dump(), "contract": solidity_identifier})

    if (c := await child.cache_get(stub_ty)) is not None:
        return cast(S, c)

    solc_name = f"solc{solc_version}"

    interface_to_implement = interface.name_to_interface[solidity_identifier]
    interface_name = interface_to_implement.solidity_identifier

    ST = type("ST", (MessagesState,), {
        "__annotations__": {"result": NotRequired[stub_ty]}
    })

    class ResultTool(AsyncResultTool[stub_ty]):
        """Submit your completed stub declaration. Triggers a solc compile
        against the assembled project tree; a compile failure is reported
        back to you for a retry.
        """

        @override
        async def validate(self, res: S) -> str | None:
            interface_basename = pathlib.Path(interface_to_implement.path).name
            if interface_basename not in res.content:
                return f"Stub must import the interface file ({interface_basename})."
            if res.solidity_identifier != solidity_identifier:
                return (
                    f"Stub must declare the contract with the exact identifier "
                    f"`{solidity_identifier}` (got `{res.solidity_identifier}`)."
                )
            if f"contract {solidity_identifier}" not in res.content:
                return f"Stub content must declare `contract {solidity_identifier}`."
            if pathlib.PurePosixPath(res.path).stem != solidity_identifier:
                return f"Stub filename must match the contract identifier: `{solidity_identifier}.sol`."

            try:
                async with materializer.project_directory() as tmpdir:
                    if (tmpdir / res.path).exists():
                        return f"Path {res.path} already exists, pick a different name"
                    (tmpdir / res.path).parent.mkdir(exist_ok=True, parents=True)
                    (tmpdir / res.path).write_text(res.content)
                    proc = await asyncio.create_subprocess_exec(
                        solc_name, res.path,
                        cwd=str(tmpdir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout_b, stderr_b = await proc.communicate()
            except FileNotFoundError:
                import logging
                logging.getLogger(__name__).exception("Stub compilation failed")
                return f"Solidity compiler {solc_name} not found on this system"
            if proc.returncode != 0:
                return (
                    f"Stub compilation failed:\n"
                    f"stdout:\n{stdout_b.decode()}\n"
                    f"stderr:\n{stderr_b.decode()}"
                )
            return None

    final_prompt = description.prompt.inject(
        StubGenCallParams(
            solidity_identifier=solidity_identifier,
            interface_name=interface_name,
            interface_path=interface_to_implement.path,
            the_interface=interface_to_implement.content,
            solc_version=solc_version,
        )
    )

    workflow = (
        env.builder
        .with_state(ST)
        .with_tools([ResultTool.as_tool("result"), *env.source_tools])
        .with_output_key("result")
        .with_default_summarizer()
        .with_input(FlowInput)
        .with_sys_prompt(
            "You are an expert Solidity developer. You are tasked with generating stub implementations "
            "for formal verification."
        )
        .inject(lambda b: final_prompt.render_to(b.with_initial_prompt_template))
        .compile_async()
    )

    res = await run_to_completion(
        workflow,
        FlowInput(input=await resolve_extra_input(description.extra_input)),
        thread_id=uniq_thread_id("stub-gen"),
        recursion_limit=ctx.recursion_limit,
        description=f"{DESCRIPTION}: {contract_name}",
    )
    assert "result" in res
    output = res["result"]
    if isinstance(output, dict):
        res_value = stub_ty.model_validate(output)
    else:
        res_value = cast(S, output)
    await child.cache_put(res_value)
    return res_value
