"""Unified prover execution core.

Extracts the shared logic between composer/spec/prover.py (async, cloud-enabled)
and composer/prover/runner.py (sync, local-only) into a single async function.
Both callers become thin wrappers that define callbacks and call run_prover().
"""

import asyncio
import pickle
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Protocol, overload

from langchain_core.messages import AnyMessage, ToolMessage, HumanMessage
from langgraph.graph import MessagesState

from graphcore.graph import LLM
from graphcore.utils import acached_invoke

from composer.prover.analysis import analyze_cex_raw
from composer.prover.cloud import cloud_results
from composer.prover.ptypes import RuleResult
from composer.prover.results import read_and_format_run_result
from composer.templates.loader import load_jinja_template


@dataclass
class CloudConfig:
    server: str = "prover"
    prover_version: str = "master"


@dataclass
class ProverOptions:
    cloud: CloudConfig | None = None


@dataclass
class RawReport:
    report: str
    all_verified: bool


@dataclass
class SummarizedReport:
    report: str
    todo_list: str


class ProverCallbacks:
    """Base class with no-op defaults. Subclass and override only what you need."""
    async def on_stdout_line(self, line: str) -> None: pass
    async def on_cloud_poll(self, status: str, message: str) -> None: pass
    async def on_prover_run(self, args: list[str]) -> None: pass
    async def on_prover_result(self, results: dict[str, RuleResult]) -> None: pass
    async def on_analysis_start(self, rule: RuleResult) -> None: pass
    async def on_rule_result(self, rule: RuleResult, analysis: str | None) -> None: pass


class AnalysisCache(Protocol):
    def get(self, rule: RuleResult) -> str | None: ...
    def put(self, rule: RuleResult, analysis: str) -> None: ...


@asynccontextmanager
async def _local_results(path: Path) -> AsyncIterator[Path]:
    """Trivial context manager that yields a local results path unchanged."""
    yield path


async def _report_to_todo_list(
    llm: LLM,
    messages: list[AnyMessage],
    report: str,
    tool_call_id: str,
) -> str:
    new_messages = messages.copy()
    new_messages.append(ToolMessage(
        content=report,
        tool_call_id=tool_call_id,
    ))
    new_messages.append(HumanMessage(
        content="""\
Analyze the counterexamples returned by the prover.
Investigate the rule-by-rule feedback and create a TODO list of the code changes you
need to perform to address the identified issues.
In particular, find common issues exhibited between the different counterexamples and identify
the common root causes and what changes are necessary.""",
    ))
    res = await acached_invoke(llm, new_messages)
    return res.text()

@overload
async def run_prover(
    state: MessagesState,
    folder: Path,
    args: list[str],
    llm: LLM,
    tool_call_id: str,
    options: ProverOptions,
    callbacks: ProverCallbacks,
    *,
    analysis_cache: AnalysisCache | None = None
) -> str | RawReport:
    ...

@overload
async def run_prover(
    state: MessagesState,
    folder: Path,
    args: list[str],
    llm: LLM,
    tool_call_id: str,
    options: ProverOptions,
    callbacks: ProverCallbacks,
    *,
    analysis_cache: AnalysisCache | None = None,
    summarization_threshold: int
) -> str | RawReport | SummarizedReport:
    ...


async def run_prover(
    state: MessagesState,
    folder: Path,
    args: list[str],
    llm: LLM,
    tool_call_id: str,
    options: ProverOptions,
    callbacks: ProverCallbacks,
    *,
    analysis_cache: AnalysisCache | None = None,
    summarization_threshold: int | None = None,
) -> RawReport | SummarizedReport | str:
    """Execute the Certora prover and return structured results.

    Returns:
        RawReport — normal result with report text and all_verified flag
        SummarizedReport — when failures exceed summarization_threshold
        str — error message
    """

    # 1. Build effective args
    effective_args = args.copy()
    if options.cloud is not None:
        effective_args.extend(["--server", options.cloud.server])
        effective_args.extend(["--prover_version", options.cloud.prover_version])

    # 2. Notify callback
    await callbacks.on_prover_run(effective_args)

    # 3-5. Spawn async subprocess, stream stdout, collect stderr
    wrapper_script = Path(__file__).parent / "certoraRunWrapper.py"

    with tempfile.NamedTemporaryFile("rb", suffix=".pkl") as output_file:
        proc = await asyncio.subprocess.create_subprocess_exec(
            sys.executable,
            str(wrapper_script), str(output_file.name), *effective_args,
            cwd=str(folder),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_lines: list[str] = []
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode()
            stdout_lines.append(line)
            await callbacks.on_stdout_line(line.rstrip("\n"))

        stderr_raw = await proc.stderr.read() if proc.stderr else b""
        await proc.wait()

        stdout = "".join(stdout_lines)
        stderr = stderr_raw.decode()

        # 5. Unpickle result
        run_result = pickle.load(output_file)

    # 6. Error handling
    if proc.returncode != 0:
        return f"Verification failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"

    if isinstance(run_result, Exception):
        return f"Certora prover raised exception: {run_result!s}\nstdout:\n{stdout}"

    if run_result is None or run_result.link is None:
        return f"Prover did not produce results.\nstdout:\n{stdout}"

    # 7. Result retrieval: cloud vs local
    if options.cloud is not None:
        results_cm = cloud_results(run_result.link, poll_callback=callbacks.on_cloud_poll)
    else:
        if not run_result.is_local_link:
            return f"Prover did not produce local results.\nstdout:\n{stdout}"
        results_cm = _local_results(Path(run_result.link))

    # 8. Parse results
    async with results_cm as emv_path:
        parsed = read_and_format_run_result(emv_path)

    if isinstance(parsed, str):
        return f"Failed to parse prover results: {parsed}"

    # 9. Notify prover_result callback
    await callbacks.on_prover_result(parsed)

    # 10. Parallel CEX analysis
    messages = state["messages"]

    async def _analyze(rule: RuleResult) -> tuple[RuleResult, str | None]:
        if analysis_cache is not None:
            cached = analysis_cache.get(rule)
            if cached is not None:
                return (rule, cached)
        await callbacks.on_analysis_start(rule)
        analysis = await analyze_cex_raw(llm, messages, rule, tool_call_id)
        if analysis is not None and analysis_cache is not None:
            analysis_cache.put(rule, analysis)
        return (rule, analysis)

    jobs = [_analyze(res) for res in parsed.values()]
    results_with_analysis: list[tuple[RuleResult, str | None]] = await asyncio.gather(*jobs)

    # 11. Notify per-rule callbacks
    for rule_result, analysis in results_with_analysis:
        await callbacks.on_rule_result(rule_result, analysis)

    # 12. Format results as markdown
    report = load_jinja_template("rule_feedback.j2", results=results_with_analysis)

    # 13. Count failures, possibly summarize
    failed_count = sum(1 for r, _ in results_with_analysis if r.status == "VIOLATED")

    if summarization_threshold is not None and failed_count > summarization_threshold:
        todo_list = await _report_to_todo_list(llm, messages, report, tool_call_id)
        return SummarizedReport(report=report, todo_list=todo_list)

    # 14. Normal return
    return RawReport(report=report, all_verified=(failed_count == 0))
