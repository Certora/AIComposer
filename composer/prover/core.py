"""Unified prover execution core.

Extracts the shared logic between composer/spec/prover.py (async, cloud-enabled)
and composer/prover/runner.py (sync, local-only) into a single async function.
Both callers become thin wrappers that define callbacks and call run_prover().
"""

import asyncio
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, cast, override
from abc import ABC, abstractmethod
import json
import logging


from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import MessagesState

from graphcore.graph import LLM
from graphcore.utils import acached_invoke

from composer.prover.analysis import analyze_cex_raw
from composer.prover.cloud import cloud_results
from composer.prover.ptypes import RuleResult
from composer.prover.results import read_and_format_run_result
from composer.templates.loader import load_jinja_template
from composer.prover.prover_protocol import ProverResult

_logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    server: str = "prover"
    prover_version: str = "master"


@dataclass
class ProverOptions:
    cloud: CloudConfig | None = None

@dataclass
class ProverReport:
    rule_status: dict[str, bool]

@dataclass
class RawReport(ProverReport):
    report: str
    @property
    def all_verified(self) -> bool:
        return all(self.rule_status.values())

@dataclass
class SummarizedReport(ProverReport):
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
    async def on_analysis_complete(self, rule: RuleResult, analysis: str) -> None: pass

class CexHandler(ABC):
    @abstractmethod
    async def analyze_cex(
        self, rule: RuleResult
    ) -> str | None:
        ...

class SummarizingCexHandler(CexHandler):
    def __init__(self, summarization_threshold: int):
        super().__init__()
        self.summarization_threshold = summarization_threshold

    @abstractmethod
    async def summarize(
        self, report: str
    ) -> str:
        ...

class DefaultCexHandler(SummarizingCexHandler):
    def __init__(self, tid: str, llm: LLM, state: MessagesState, summarization_threshold: int = 10):
        super().__init__(summarization_threshold)
        self.state = state
        self.llm = llm
        self.tid = tid

    @override
    async def analyze_cex(self, rule: RuleResult) -> str | None:
        messages = self.state["messages"]
        analysis = await analyze_cex_raw(self.llm, messages, rule, self.tid)
        return analysis

    async def summarize(self, report: str) -> str:
        return await _report_to_todo_list(self.llm, report)

@asynccontextmanager
async def _local_results(path: Path) -> AsyncIterator[Path]:
    """Trivial context manager that yields a local results path unchanged."""
    yield path


async def _report_to_todo_list(
    llm: LLM,
    report: str,
) -> str:
    fresh_messages: list[AnyMessage] = [
        HumanMessage(content=f"""\
Below is a rule-by-rule prover report with counterexample analyses. Your job is to produce
a detailed, actionable TODO list of code changes needed to fix the violations.

For each TODO item:
- Identify the root cause (which rules share it)
- Describe the specific code change needed
- Note which file/function to modify

Group related violations that share a common root cause into a single TODO item.

PROVER REPORT:
{report}"""),
    ]
    # Disable thinking for summarization — adaptive thinking can burn the entire
    # max_tokens budget on reasoning, leaving nothing for actual text output.
    if isinstance(llm, BaseChatModel):
        llm = llm.model_copy(update={"thinking": None})
    res = await acached_invoke(llm, fresh_messages)
    return res.text

async def run_prover(
    folder: Path,
    args: list[str],
    options: ProverOptions,
    callbacks: ProverCallbacks,
    cex: CexHandler
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

    with tempfile.NamedTemporaryFile("rb", suffix=".json") as output_file:
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
        if proc.returncode != 0:
            _logger.error("Process failed %d\nstdout:%s\nstderr:%s", proc.returncode, stdout, stderr)
            return f"Verification failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        
        run_result = cast(ProverResult, json.load(output_file))

    if run_result is None or (run_result["sort"] == "success" and run_result["link"] is None):
        _logger.warning("Prover failed: %s", run_result)
        return f"Prover did not produce results.\nstdout:\n{stdout}"
    
    if run_result["sort"] == "failure":
        _logger.info("Prover run failed: %s", run_result['exc_str'])
        return f"Certora prover raised exception: {run_result['exc_str']}\nstdout:\n{stdout}"

    assert run_result is not None and run_result["sort"] == "success" and run_result["link"] is not None

    # 7. Result retrieval: cloud vs local
    if options.cloud is not None:
        results_cm = cloud_results(run_result["link"], poll_callback=callbacks.on_cloud_poll)
    else:
        if not run_result["is_local_link"]:
            return f"Prover did not produce local results.\nstdout:\n{stdout}"
        results_cm = _local_results(Path(run_result["link"]))

    # 8. Parse results
    async with results_cm as emv_path:
        parsed = read_and_format_run_result(emv_path)

    if isinstance(parsed, str):
        return f"Failed to parse prover results: {parsed}"

    # 9. Notify prover_result callback
    await callbacks.on_prover_result(parsed)

    async def _analyze(rule: RuleResult) -> tuple[RuleResult, str | None]:
        if rule.status != "VIOLATED":
            return (rule, None)
        await callbacks.on_analysis_start(rule)
        res = await cex.analyze_cex(rule)
        if res is not None:
            await callbacks.on_analysis_complete(rule, res)
        return (rule, analysis)

    jobs = [_analyze(res) for res in parsed.values()]
    results_with_analysis: list[tuple[RuleResult, str | None]] = await asyncio.gather(*jobs)

    # 11. Notify per-rule callbacks
    for rule_result, analysis in results_with_analysis:
        await callbacks.on_rule_result(rule_result, analysis)

    # 12. Format results as markdown
    report = load_jinja_template("rule_feedback.j2", results=results_with_analysis)

    prover_report = {}
    failed_count = 0
    for i in parsed.values():
        rule_name = i.path.rule
        if i.status != "VERIFIED":
            failed_count += 1
        if rule_name in prover_report and not prover_report[rule_name]:
            continue
        prover_report[rule_name] = i.status == "VERIFIED"

    if isinstance(cex, SummarizingCexHandler) and cex.summarization_threshold < failed_count:
        todo_list = await cex.summarize(report)
        return SummarizedReport(report=report, todo_list=todo_list, rule_status=prover_report)

    # 14. Normal return
    return RawReport(report=report, rule_status=prover_report)
