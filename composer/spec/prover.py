import os
import asyncio
import json
import tempfile
import sys
import uuid
from contextlib import contextmanager

import composer.certora as _

from pathlib import Path
from typing import Annotated, Awaitable, Callable, NotRequired, Awaitable, TypedDict, Iterator

from langchain_core.messages import AnyMessage
from langchain_core.tools import InjectedToolCallId, tool, BaseTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from composer.prover.analysis import analyze_cex_raw
from composer.prover.ptypes import RuleResult
from composer.prover.results import read_and_format_run_result
from composer.spec.trunner import log_message

from graphcore.graph import LLM

from pydantic import create_model

class WithCVL(TypedDict):
    curr_spec: NotRequired[str]
    messages: list[AnyMessage]

class VerifySpecSchema[T: WithCVL](BaseModel):
    """
    Run the Certora prover to verify the current spec against the source code.

    Returns verification results:
    - VERIFIED: Rule holds for all inputs
    - VIOLATED: Counterexample found (with CEX analysis)
    - TIMEOUT: Verification did not complete in time

    Use these results to refine your spec.
    """
    tool_call_id: Annotated[str, InjectedToolCallId]

    rules: list[str] | None = Field(
        default=None,
        description="Specific rules to verify. If None, verifies all rules."
    )

@contextmanager
def tmp_spec(
    *,
    root: str,
    content: str,
    prefix: str = "generated"
) -> Iterator[str]:
    t_name = f"{prefix}_{uuid.uuid4().hex[:16]}.spec"

    rel_path = "certora/" + t_name

    full_path = (Path(root) / "certora" / t_name)

    full_path.write_text(content)

    try:
        yield rel_path
    finally:
        os.unlink(full_path)

async def _prover_tool_internal(
    sem: asyncio.Semaphore,
    curr_spec: str,
    project_root: str,
    main_contract: str,
    compilation_config: dict,
    rules: list[str] | None,
    cex_analyzer: Callable[[RuleResult], Awaitable[tuple[RuleResult, str | None]]]
) -> str:
    
    certora_dir = Path(project_root) / "certora"

    with tmp_spec(
        root=project_root,
        content=curr_spec
    ) as generated:
        config = {
            **compilation_config,
            "verify": f"{main_contract}:{generated}",
            "parametric_contracts": main_contract,
            "optimistic_loop": True,
            "rule_sanity": "basic",
        }

        if rules:
            config["rule"] = rules

        # Write config file
        config_path = certora_dir / "verify.conf"
        config_path.write_text(json.dumps(config, indent=2))

        # Run certoraRunWrapper
        wrapper_script = Path(__file__).parent.parent / "prover" / "certoraRunWrapper.py"
        log_message(f"starting command: {str(config)}", "info")

        with tempfile.NamedTemporaryFile("rb", suffix=".pkl") as output_file:
            async with sem: 
                proc_result = await asyncio.subprocess.create_subprocess_exec(
                    sys.executable,
                    str(wrapper_script), str(output_file.name), str(config_path),
                    cwd=project_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout_raw, stderr_raw = await proc_result.communicate()
            stdout = stdout_raw.decode()
            stderr = stderr_raw.decode()

            # Read the pickled output
            import pickle
            run_result = pickle.load(output_file)

    # Check for errors
    if proc_result.returncode != 0:
        return f"Verification failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"


    # Check if it's an exception
    if isinstance(run_result, Exception):
        return f"Certora prover raised exception: {str(run_result)}\nstdout:\n{stdout}"

    if run_result is None or not run_result.is_local_link or run_result.link is None:
        return f"Prover did not produce local results.\nstdout:\n{stdout}"

    emv_path = Path(run_result.link)

    # Parse results using existing infrastructure
    results = read_and_format_run_result(emv_path)

    if isinstance(results, str):
        # Error occurred during parsing
        return f"Failed to parse prover results: {results}"
    
    jobs = [
        cex_analyzer(res) for res in results.values()
    ]

    results_with_analysis = await asyncio.gather(*jobs)

    # Format results for LLM
    lines = ["## Verification Results\n"]
    verified, violated, timeout_count = 0, 0, 0

    for rule_result, cex_analysis in results_with_analysis:
        status = rule_result.status
        name = rule_result.name

        if status == "VERIFIED":
            verified += 1
            lines.append(f"✓ **{name}**: VERIFIED")
        elif status == "VIOLATED":
            violated += 1
            lines.append(f"✗ **{name}**: VIOLATED")
            if cex_analysis:
                lines.append(f"  Analysis: {cex_analysis}")
        elif status == "TIMEOUT":
            timeout_count += 1
            lines.append(f"⏱ **{name}**: TIMEOUT")
        else:
            lines.append(f"? **{name}**: {status}")

    lines.append(f"\n**Summary**: {verified} verified, {violated} violated, {timeout_count} timeout")

    return "\n".join(lines)


def get_prover_tool[T: WithCVL](
    llm: LLM,
    ty: type[T],
    conf: dict,
    main_contract: str,
    project_root: str
) -> BaseTool:
    sem = asyncio.Semaphore(1)
    args_spec = create_model(
        "VerifySpecSchemaInst",
        __doc__=VerifySpecSchema.__doc__,
        __base__=VerifySpecSchema,
        state=Annotated[ty, InjectedState]
    )

    @tool(args_schema=args_spec)
    async def verify_spec(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[T, InjectedState],
        rules: list[str] | None = None
    ) -> str:
        m = state["messages"]
        if "curr_spec" not in state:
            return "Specification not yet put on VFS"
        
        log_message("YES HELLO", "error")

        async def analyzer(
            rule_result: RuleResult
        ) -> tuple[RuleResult, str | None]:
            res = await analyze_cex_raw(
                llm, m, rule_result, tool_call_id
            )
            return (rule_result, res)
        
        return await _prover_tool_internal(
            sem, state["curr_spec"], project_root, main_contract, conf, rules, analyzer
        )
    
    return verify_spec
