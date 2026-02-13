import os
import asyncio
import json
import logging
import tempfile
import sys
import uuid
from contextlib import asynccontextmanager, contextmanager

import composer.certora as _

from pathlib import Path
from typing import AsyncIterator, Annotated, Awaitable, Callable, Literal, NotRequired, Awaitable, TypedDict, Iterator

from langchain_core.messages import AnyMessage
from langchain_core.tools import InjectedToolCallId, tool, BaseTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from composer.prover.analysis import analyze_cex_raw
from composer.prover.cloud import cloud_results
from composer.prover.ptypes import RuleResult
from composer.prover.results import read_and_format_run_result

from langgraph.config import get_stream_writer
from graphcore.graph import LLM

from pydantic import create_model

logger = logging.getLogger("composer.spec")


class ProverOutputEvent(TypedDict):
    type: Literal["prover_output"]
    tool_call_id: str
    line: str


class CloudPollingEvent(TypedDict):
    type: Literal["cloud_polling"]
    tool_call_id: str
    status: str
    message: str


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
    cex_analyzer: Callable[[RuleResult], Awaitable[tuple[RuleResult, str | None]]],
    line_callback: Callable[[str], Awaitable[None]] | None = None,
    cloud: bool = False,
    poll_callback: Callable[[str, str], Awaitable[None]] | None = None,
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

        if cloud:
            config["server"] = "prover"
            config["prover_version"] = "master"

        if rules:
            config["rule"] = rules

        # Write config file
        config_path = certora_dir / "verify.conf"
        config_path.write_text(json.dumps(config, indent=2))

        # Run certoraRunWrapper
        wrapper_script = Path(__file__).parent.parent / "prover" / "certoraRunWrapper.py"

        with tempfile.NamedTemporaryFile("rb", suffix=".pkl") as output_file:
            async with sem:
                proc = await asyncio.subprocess.create_subprocess_exec(
                    sys.executable,
                    str(wrapper_script), str(output_file.name), str(config_path),
                    cwd=project_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout_lines: list[str] = []
                assert proc.stdout is not None
                while True:
                    raw = await proc.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode()
                    stdout_lines.append(line)
                    if line_callback:
                        await line_callback(line.rstrip("\n"))
                stderr_raw = await proc.stderr.read() if proc.stderr else b""
                await proc.wait()
            stdout = "".join(stdout_lines)
            stderr = stderr_raw.decode()

            # Read the pickled output
            import pickle
            run_result = pickle.load(output_file)

    # Check for errors
    if proc.returncode != 0:
        return f"Verification failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"

    # Check if it's an exception
    if isinstance(run_result, Exception):
        return f"Certora prover raised exception: {str(run_result)}\nstdout:\n{stdout}"

    if run_result is None or run_result.link is None:
        return f"Prover did not produce results.\nstdout:\n{stdout}"

    # Diverge: local vs cloud result retrieval
    if cloud:
        results_cm = cloud_results(run_result.link, poll_callback=poll_callback)
    else:
        if not run_result.is_local_link:
            return f"Prover did not produce local results.\nstdout:\n{stdout}"
        results_cm = _local_results(Path(run_result.link))

    async with results_cm as emv_path:
        return await _parse_and_format(emv_path, cex_analyzer)


async def _parse_and_format(
    emv_path: Path,
    cex_analyzer: Callable[[RuleResult], Awaitable[tuple[RuleResult, str | None]]],
) -> str:
    """Parse prover results from a local directory and format for the LLM."""
    results = read_and_format_run_result(emv_path)

    if isinstance(results, str):
        return f"Failed to parse prover results: {results}"

    jobs = [cex_analyzer(res) for res in results.values()]
    results_with_analysis = await asyncio.gather(*jobs)

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


@asynccontextmanager
async def _local_results(path: Path) -> AsyncIterator[Path]:
    """Trivial context manager that yields a local results path unchanged."""
    yield path


def get_prover_tool[T: WithCVL](
    llm: LLM,
    ty: type[T],
    conf: dict,
    main_contract: str,
    project_root: str,
    cloud: bool = False,
    semaphore: asyncio.Semaphore | None = None,
) -> BaseTool:
    sem = semaphore or asyncio.Semaphore(1)
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

        async def analyzer(
            rule_result: RuleResult
        ) -> tuple[RuleResult, str | None]:
            res = await analyze_cex_raw(
                llm, m, rule_result, tool_call_id
            )
            return (rule_result, res)

        writer = get_stream_writer()

        async def on_prover_line(line: str) -> None:
            evt: ProverOutputEvent = {
                "type": "prover_output",
                "tool_call_id": tool_call_id,
                "line": line,
            }
            writer(evt)

        async def on_poll_status(status: str, message: str) -> None:
            evt: CloudPollingEvent = {
                "type": "cloud_polling",
                "tool_call_id": tool_call_id,
                "status": status,
                "message": message,
            }
            writer(evt)

        return await _prover_tool_internal(
            sem, state["curr_spec"], project_root, main_contract, conf, rules, analyzer,
            line_callback=on_prover_line,
            cloud=cloud,
            poll_callback=on_poll_status if cloud else None,
        )

    return verify_spec
