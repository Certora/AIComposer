"""
Interface generation agent: produces a Solidity interface from component analysis.

Takes the ApplicationSummary and system document, generates an interface that
covers all external entry points, and validates it with the Solidity compiler.
"""

import asyncio
from logging import getLogger
from typing import NotRequired, cast, override

from graphcore.graph import FlowInput

from langgraph.graph import MessagesState

from composer.spec.context import WorkflowContext, CacheKey
from composer.spec.graph_builder import run_to_completion
from composer.spec.natspec.async_result import AsyncResultTool
from composer.spec.natspec.models import (
    InterfaceDeclModel,
    InterfaceResult,
)
from composer.spec.natspec.task_description import (
    AgentDescription,
    Assembler,
    InterfaceGenCallParams,
    resolve_extra_input,
)
from composer.spec.system_model import NatspecApplication, SolidityIdentifier
from composer.spec.util import string_hash, uniq_thread_id
from composer.spec.service_host import ServiceHost

_logger = getLogger(__name__)

DESCRIPTION = "Interface generation"


async def generate_interface[I: InterfaceDeclModel](
    ctx: WorkflowContext[None],
    summary: NatspecApplication,
    env: ServiceHost,
    solc_version: str,
    materializer: Assembler,
    description: AgentDescription[InterfaceResult[I], InterfaceGenCallParams],
    *,
    target_identifiers: set[SolidityIdentifier] | None = None,
) -> InterfaceResult[I]:
    """Generate a Solidity interface from component analysis and system document.

    The candidate interface set is validated by laying it out through the
    caller-supplied ``assembler_for_candidate`` factory, then invoking solc
    inside the assembled project. ``description`` fixes the concrete decl
    subtype and the prompt (with any workflow-constant params pre-bound).

    ``target_identifiers`` restricts which contracts the agent is expected to
    produce interfaces for. Defaults to every contract in ``summary`` (the
    greenfield case). In from-source mode the caller passes the subset of
    Solidity identifiers tagged ``new`` — contracts tagged
    ``unchanged``/``edited`` already have their interfaces in the source tree.
    """
    result_ty = description.output_ty

    cache_key = CacheKey[None, InterfaceResult](
        f"interface-{string_hash(summary.model_dump_json())}-{result_ty.__name__}"
    )

    child = await ctx.child(cache_key, summary.model_dump())

    if (cached := await child.cache_get(result_ty)) is not None:
        return cached

    solc_name = f"solc{solc_version}"

    external_contracts = (
        target_identifiers
        if target_identifiers is not None
        else {c.solidity_identifier for c in summary.contract_components}
    )

    ST = type("ST", (MessagesState,), {
        "__annotations__": {"result": NotRequired[InterfaceResult[I]]}
    })

    class ResultTool(AsyncResultTool[result_ty]):
        """Submit your completed interface set. Triggers a solc compile
        against the assembled project tree; a compile failure is reported
        back to you for a retry.
        """

        @override
        async def validate(self, res: InterfaceResult[I]) -> str | None:
            seen: set[SolidityIdentifier] = set()
            for nm, i in res.name_to_interface.items():
                if nm not in external_contracts:
                    return f"Invalid entry found; no external contract with solidity identifier {nm} appears in input"
                if not i.path.endswith(".sol"):
                    return f"Interface path '{i.path}' for {nm} must end in '.sol'."
                seen.add(nm)
            if seen != external_contracts:
                return f"Missing results for contract(s): {external_contracts - seen}"

            compile_inputs = [i.path for i in res.name_to_interface.values()]
            try:
                async with materializer.project_directory() as tmpdir:
                    for (_, v) in res.name_to_interface.items():
                        if (tmpdir / v.path).exists():
                            return f"Path {v.path} already exists; pick another one"
                        (tmpdir / v.path).parent.mkdir(parents=True, exist_ok=True)
                        (tmpdir / v.path).write_text(v.content)
                    _logger.info(f"Compiling interfaces in {tmpdir}: {compile_inputs}")
                    proc = await asyncio.create_subprocess_exec(
                        solc_name, *compile_inputs,
                        cwd=str(tmpdir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout_b, stderr_b = await proc.communicate()
            except FileNotFoundError:
                _logger.exception("Execution failed!")
                return f"Solidity compiler {solc_name} not found on this system"
            if proc.returncode != 0:
                return (
                    f"Interface compilation failed:\n"
                    f"stdout:\n{stdout_b.decode()}\n"
                    f"stderr:\n{stderr_b.decode()}"
                )
            return None

    target_contracts = [
        c for c in summary.contract_components if c.solidity_identifier in external_contracts
    ]
    existing_contracts = [
        c for c in summary.contract_components if c.solidity_identifier not in external_contracts
    ]

    final_prompt = description.prompt.inject(
        InterfaceGenCallParams(
            summary=summary,
            target_contracts=target_contracts,
            existing_contracts=existing_contracts,
            solc_version=solc_version,
        )
    )

    workflow = (
        env.builder
        .with_state(ST)
        .with_tools([ResultTool.as_tool("result"), *env.analysis_tools])
        .with_output_key("result")
        .with_default_summarizer()
        .with_input(FlowInput)
        .with_sys_prompt(
            "You are an expert Solidity developer specializing in interface design for "
            "formal verification of smart contracts."
        )
        .inject(lambda b: final_prompt.render_to(b.with_initial_prompt_template))
        .compile_async()
    )

    res = await run_to_completion(
        workflow,
        FlowInput(input=await resolve_extra_input(description.extra_input)),
        thread_id=uniq_thread_id("interface-gen"),
        recursion_limit=ctx.recursion_limit,
        description=DESCRIPTION,
    )
    assert "result" in res
    res_raw = res["result"]
    if isinstance(res_raw, dict):
        res_value = result_ty.model_validate(res_raw)
    else:
        res_value = cast(InterfaceResult[I], res_raw)
    await child.cache_put(res_value)
    return res_value
