"""Unified prover execution core.

Extracts the shared logic between composer/spec/prover.py (async, cloud-enabled)
and composer/prover/runner.py (sync, local-only) into a single async function.
Both callers become thin wrappers that define callbacks and call run_prover().

CEX analysis flow:

* The prover produces a ``dict[str, RuleResult]`` of per-rule outcomes.
* Failing instances are grouped into ``FailingRule`` (one per rule name,
  with one or more parametric instances). The grouping is the input
  shape every handler consumes.
* A ``CexHandler`` exposes a single ``analyze`` entry point that
  consumes the grouped failures and returns the fully-rendered report
  string. The handler decides its rendering, its summarization (if
  any), and the shape of its per-rule UI events. The mainline flow
  doesn't round-trip intermediate analysis types — handlers that
  produce structured data (e.g. report_keys for downstream lookup)
  persist it themselves via injected dependencies.
"""

import asyncio
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Protocol, cast, override, Callable
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
    """Single return shape from ``run_prover`` (alongside the ``str``
    error path).

    ``rule_status`` maps rule name to verified-or-not. ``result_str`` is
    the fully-rendered text the calling tool hands back to the LLM —
    constructed by the ``CexHandler``, including any summarization,
    diagnosis indexing, or shape choices the handler made internally.
    No structured analysis fields on this type: handlers that produce
    keyed records (e.g. report_keys for ``cex_remediation`` to look up)
    persist them via their own injected stores rather than bubbling
    them through the return value.
    """
    rule_status: dict[str, bool]
    result_str: str

    @property
    def all_verified(self) -> bool:
        return all(self.rule_status.values())

def zip_results[T](
    l: list[RuleResult], m: Callable[[RuleResult], T | None]
) -> list[tuple[RuleResult, T | None]]:
    return [
        (r, m(r) if r.status == "VIOLATED" else None) for r in l
    ]

# ---------------------------------------------------------------------------
# CEX-analysis protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailingRule:
    """A logical rule that failed verification, possibly with multiple
    parametric instances. The ``rule_name`` is the base rule (``path.rule``);
    ``instances`` lists the per-binding ``RuleResult`` records (one per
    parametric arg, each with its own CEX trace). Non-parametric rules
    have ``len(instances) == 1``.

    Per-rule grouping is the natural scope for an analyzer's scratchpad —
    parametric instances of the same rule typically share root causes."""

    rule_name: str
    instances: list[RuleResult]

class CexProgressCallbacks(Protocol):
    """The narrow callback surface ``CexHandler`` fires into as it works
    through CEX analysis.

    For each failing rule the handler analyzes, it fires
    ``on_analysis_start`` once and then ``on_analysis_complete`` once
    with the *displayed explanation* for that rule — what the UI
    surfaces to the user. The explanation isn't constrained to be a
    fresh per-rule analysis; a handler that already saw the same root
    cause for another rule can emit a back-reference (e.g. "same root
    cause as rule X: …") here. The contract is "every failing rule
    gets a start/complete pair, and the complete payload is renderable
    to the user."

    ``ProverCallbacks`` satisfies this structurally, so ``run_prover``
    hands its callbacks through without conversion.
    """

    async def on_analysis_start(self, rule: RuleResult) -> None: ...
    async def on_analysis_complete(self, rule: RuleResult, explanation: str) -> None: ...


class ProverCallbacks:
    """Base class with no-op defaults. Subclass and override only what you need.

    Per-rule lifecycle events fire from inside the ``CexHandler`` as it
    processes individual ``RuleResult`` instances — every failing rule
    gets a ``on_analysis_start`` / ``on_analysis_complete`` pair, where
    the explanation payload is whatever the handler wants the UI to
    show for that rule.

    The class structurally satisfies ``CexProgressCallbacks`` — the subset
    the handler actually needs.
    """
    async def on_stdout_line(self, line: str) -> None: pass
    async def on_cloud_poll(self, status: str, message: str) -> None: pass
    async def on_prover_run(self, args: list[str]) -> None: pass
    async def on_prover_result(self, results: dict[str, RuleResult]) -> None: pass
    async def on_analysis_start(self, rule: RuleResult) -> None: pass
    async def on_analysis_complete(self, rule: RuleResult, explanation: str) -> None: pass

class CexHandler(ABC):
    """Strategy for turning a batch of failing rules into a fully-rendered
    report string. Implementations decide their own analysis approach
    (per-CEX fanout, cross-rule clustering, anything in between), their
    own per-rule UI events, their own rendering, and any post-processing
    (summarization, key minting, store writes) they need.
    """

    @abstractmethod
    async def analyze(
        self,
        failing_rules: list[FailingRule],
        all_results: list[RuleResult],
        tool_call_id: str,
        callbacks: CexProgressCallbacks,
        report_dir: Path,
    ) -> str:
        """Analyze the failing rules and return the final report string.

        ``report_dir`` is the prover's report directory (containing
        ``inputs/.certora_sources``). Implementations that read source
        for grounding can scope their reads inside that path; the
        directory is guaranteed live for the duration of this call.

        ``callbacks`` is the ``CexProgressCallbacks`` slice — fire
        ``on_analysis_start`` / ``on_analysis_complete`` for every
        failing rule the user should see in the UI. The wider
        ``ProverCallbacks`` surface (stdout streaming, cloud polling,
        etc.) is intentionally not exposed here.
        """
        ...


class TrivialFanoutCexHandler(CexHandler):
    """Per-CEX single-shot analyzer. Each parametric instance gets its own
    analysis; no cross-CEX clustering. The default for callers that
    don't want a smarter handler.

    Owns its own report-volume summarization: when failed-rule count
    exceeds ``summarization_threshold`` the rendered report is
    post-processed into a TODO-list summary. Local to this strategy —
    aggregating handlers don't run into runaway-output problems.
    """

    def __init__(
        self,
        llm: LLM,
        state: MessagesState,
        summarization_threshold: int = 10,
    ) -> None:
        super().__init__()
        self.state = state
        self.llm = llm
        self.summarization_threshold = summarization_threshold

    @override
    async def analyze(
        self,
        failing_rules: list[FailingRule],
        all_results: list[RuleResult],
        tool_call_id: str,
        callbacks: CexProgressCallbacks,
        report_dir: Path,
    ) -> str:
        async def _one(instance: RuleResult) -> tuple[RuleResult, str | None]:
            await callbacks.on_analysis_start(instance)
            analysis = await analyze_cex_raw(
                self.llm, self.state["messages"], instance, tool_call_id
            )
            if analysis is not None:
                await callbacks.on_analysis_complete(instance, analysis)
            return (instance, analysis)

        jobs = [
            _one(instance)
            for rule_group in failing_rules
            for instance in rule_group.instances
            if instance.status == "VIOLATED"
        ]
        results = await asyncio.gather(*jobs)

        to_cex_explanation = {
            r.name: stat for (r, stat) in results if stat is not None
        }

        # Render every rule (verified + violated) so the agent sees the
        # full picture, not just failures. Trivial fanout uses
        # ``flat_rule_feedback.j2`` — explanations inline under each
        # violated rule, no diagnosis-key indirection. The keyed-
        # diagnosis ``rule_feedback.j2`` belongs to the agentic path,
        # which mints opaque ``report_key``s for ``cex_remediation``
        # to look up.
        results_for_template = zip_results(
            all_results, lambda r: to_cex_explanation.get(r.name)
        )
        report = load_jinja_template(
            "flat_rule_feedback.j2",
            results=results_for_template,
        )

        failed_count = sum(
            1 for instance, _ in results
            if instance.status != "VERIFIED"
        )
        if failed_count > self.summarization_threshold:
            return await _report_to_todo_list(self.llm, report)
        return report


# Compatibility alias for the legacy name. New code should reach for
# TrivialFanoutCexHandler directly; the agentic codegen handler lives
# elsewhere.
DefaultCexHandler = TrivialFanoutCexHandler

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
    tool_call_id: str,
    options: ProverOptions,
    callbacks: ProverCallbacks,
    cex: CexHandler
) -> ProverReport | str:
    """Execute the Certora prover and return structured results.

    Returns:
        ProverReport — rule outcomes + rendered result text + diagnoses
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

    # 8. Parse results + run analysis. Both happen inside ``results_cm`` so
    # the report directory (which contains ``inputs/.certora_sources`` —
    # the source files actually compiled into this verification problem)
    # stays alive for the analyzer to read. Cloud runs unzip into a tmpdir
    # whose lifetime is bound to ``cloud_results``; local runs hand back
    # a stable path. Either way the analyzer must complete before the
    # context manager exits.
    async with results_cm as emv_path:
        parsed = read_and_format_run_result(emv_path)

        if isinstance(parsed, str):
            return f"Failed to parse prover results: {parsed}"

        # 9. Notify prover_result callback
        await callbacks.on_prover_result(parsed)

        # 10. Group failing instances by rule name (parametric instances
        # of the same rule cluster here as the natural unit a handler
        # consumes).
        failing_by_rule: dict[str, list[RuleResult]] = {}
        all_rules = []
        for rule_result in parsed.values():
            all_rules.append(rule_result)
            if rule_result.status != "VIOLATED":
                continue
            failing_by_rule.setdefault(rule_result.path.rule, []).append(rule_result)

        failing_rules = [
            FailingRule(rule_name=name, instances=instances)
            for name, instances in failing_by_rule.items()
        ]

        # 11. Hand off to the handler. It owns the analysis approach,
        # per-rule UI events, rendering, summarization (if any), and
        # storage of any keyed records (e.g. report_keys for
        # ``cex_remediation`` lookup). We get back the final report
        # string. ``emv_path`` is the prover's report directory;
        # implementations that read source narrow further to
        # ``emv_path / "inputs" / ".certora_sources"``.
        if failing_rules:
            result_str = await cex.analyze(
                failing_rules, list(parsed.values()), tool_call_id, callbacks, emv_path
            )
        else:
            result_str = load_jinja_template(
                "rule_feedback.j2",
                rule_entries=[(r, None) for r in parsed.values()],
            )

        prover_report = {}
        for i in parsed.values():
            rule_name = i.path.rule
            if rule_name in prover_report and not prover_report[rule_name]:
                continue
            prover_report[rule_name] = i.status == "VERIFIED"

        return ProverReport(
            rule_status=prover_report,
            result_str=result_str,
        )
