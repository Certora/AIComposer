"""
Spec-side prover tool: wraps composer/prover/core.py into a LangGraph tool.

Provides get_prover_tool() which creates a verify_spec tool that:
- Reads curr_spec from injected state
- Writes a temporary .spec file
- Runs the Certora prover via run_prover()
- Streams output/polling events via custom stream writer
"""

import asyncio
import json
import logging
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Annotated, Callable, Iterator, override, AsyncContextManager
from typing_extensions import TypedDict, NotRequired

from langchain_core.tools import InjectedToolCallId, tool, BaseTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from langgraph.config import get_stream_writer
from langgraph.types import Command
from composer.prover.ptypes import RuleResult
from graphcore.graph import LLM

from composer.prover.core import (
    ProverOptions, ProverCallbacks, run_prover, SummarizedReport, DefaultCexHandler
)
from composer.ui.tool_display import tool_display
from composer.diagnostics.stream import (
    ProverOutputEvent, CloudPollingEvent, RuleAnalysisResult,
    CEXAnalysisStart, ProverRun, ProverLink, ProverResult
)
from composer.spec.cvl_generation import CVLGenerationState, make_validation_stamper
from composer.diagnostics.timing import RunSummary, get_run_summary
from graphcore.graph import tool_state_update
from composer.spec.util import temp_certora_file, ensure_dir
from composer.spec.gen_types import CERTORA_DIR, SPECS_DIR, under_project


_logger = logging.getLogger("composer.prover")


def dump_final_conf(
    *,
    project_root: str,
    main_contract: str,
    task_id: str,
    spec_path: Path,
    conf: dict | None = None,
) -> None:
    """Write *task_id*'s last prover conf to ``certora/confs/{stem}.conf``,
    rewriting the ``verify`` line to point at *spec_path*.

    *spec_path* is the path to the persisted spec **relative to the project root**
    (e.g. ``certora/specs/invariants.spec``). It is used verbatim in the conf's
    ``verify`` entry, which the prover resolves relative to the project root.

    ``conf`` may be supplied explicitly (e.g. a conf persisted in the generation cache so
    a cache hit can still produce the conf); when omitted it falls back to the conf
    recorded by a live prover run this session. No-op if neither is available.
    """
    if conf is None:
        conf = get_run_summary().get_latest_conf(task_id=task_id)
    if conf is None:
        _logger.warning(f"Attempting to dump the conf for task_id {task_id} but it doesn't exist")
        return
    conf["verify"] = f"{main_contract}:{spec_path}"
    confs_dir = ensure_dir(under_project(project_root, CERTORA_DIR / "confs"))
    out_path = confs_dir / f"{Path(spec_path).stem}.conf"
    out_path.write_text(json.dumps(conf, indent=2))
    _logger.info(f"wrote final conf for task={task_id} to {out_path}")


DELETE_SKIP = "__delete_skip"

VALIDATION_KEY = "prover"

def _merge_rule_skips(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    to_ret = left.copy()
    for (k,v) in right.items():
        if v == DELETE_SKIP:
            if k in to_ret:
                del to_ret[k]
            continue
        to_ret[k] = v
    return to_ret


class ProverStateExtra(TypedDict):
    rule_skips: Annotated[dict[str, str], _merge_rule_skips]
    config: dict
    # Basename the spec is materialized/persisted under (e.g. "autospec_<slug>").
    # NotRequired so other ProverStateExtra injectors (e.g. config_edit) needn't set it.
    spec_stem: NotRequired[str]

type ProverEvents = CEXAnalysisStart | CloudPollingEvent | ProverOutputEvent | RuleAnalysisResult | ProverRun | ProverLink | ProverResult

class StateWithSkips(CVLGenerationState, ProverStateExtra):
    pass

class _SpecCallbacks(ProverCallbacks):
    def __init__(
        self,
        writer: Callable[[ProverEvents], None],
        tool_call_id: str,
        summary: RunSummary,
        config: dict,
    ) -> None:
        self._writer = writer
        self._tool_call_id = tool_call_id
        self._summary = summary
        self._config = config
        self._started_mono: float | None = None

    @override
    async def on_stdout_line(self, line: str) -> None:
        self._writer({
            "type": "prover_output",
            "tool_call_id": self._tool_call_id,
            "line": line,
        })

    @override
    async def on_cloud_poll(self, status: str, message: str) -> None:
        elapsed = (time.perf_counter() - self._started_mono) if self._started_mono else 0.0
        _logger.info(
            f"cloud poll tool_call={self._tool_call_id} status={status} "
            f"elapsed={elapsed:.1f}s msg={message}"
        )
        self._writer({
            "type": "cloud_polling",
            "tool_call_id": self._tool_call_id,
            "status": status,
            "message": message,
        })

    @override
    async def on_analysis_start(self, rule: RuleResult) -> None:
        self._writer({
            "type": "cex_analysis",
            "rule_name": rule.path.pprint(),
            "tool_call_id": self._tool_call_id
        })

    @override
    async def on_analysis_complete(self, rule: RuleResult, analysis: str) -> None:
        self._writer({
            "type": "rule_analysis",
            "analysis": analysis,
            "tool_call_id": self._tool_call_id,
            "rule": rule.path.pprint()
        })

    @override
    async def on_prover_run(self, args: list[str]) -> None:
        self._started_mono = time.perf_counter()
        _logger.info(f"prover start tool_call={self._tool_call_id} args={args}")
        self._writer({
            "type": "prover_run",
            "tool_call_id": self._tool_call_id,
            "args": args,
            "config": self._config,
        })

    @override
    async def on_prover_link(self, link: str) -> None:
        _logger.info(f"prover link tool_call={self._tool_call_id} link={link}")
        self._summary.record_prover_link(link)
        self._writer({
            "type": "prover_link",
            "tool_call_id": self._tool_call_id,
            "link": link,
        })

    @override
    async def on_prover_result(self, results: dict[str, RuleResult]) -> None:
        elapsed = (time.perf_counter() - self._started_mono) if self._started_mono else 0.0
        status_summary = { k: v.status for (k,v) in results.items() }
        _logger.info(
            f"prover done tool_call={self._tool_call_id} "
            f"elapsed={elapsed:.1f}s status={status_summary}"
        )
        self._summary.add_prover_call(elapsed)
        result_evt: ProverResult = {
            "type": "prover_result",
            "tool_call_id": self._tool_call_id,
            "status": { k: v.status for (k,v) in results.items() },
        }
        self._writer(result_evt)


class VerifySpecSchema(BaseModel):
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
    state: Annotated[StateWithSkips, InjectedState]


@contextmanager
def tmp_spec(
    *,
    root: str,
    content: str,
    name: str | None = None,
) -> Iterator[str]:
    # Materialize under the canonical specs dir -- the same directory the spec is
    # ultimately persisted to -- so the prover resolves the spec's CVL imports
    # (e.g. ``summaries/X.spec``) identically at verify-time and after dumping.
    with temp_certora_file(
        root=root,
        ext="spec",
        content=content,
        name=name,
        dest_dir=SPECS_DIR,
    ) as tmp:
        yield tmp

def _prover_sem(cloud: bool) -> AsyncContextManager[None]:
    if not cloud:
        return asyncio.Semaphore(1)

    class ToRet():
        async def __aenter__(self):
            return

        async def __aexit__(self, exc_type, exc, tb):
            return

    return ToRet()

def get_prover_tool(
    llm: LLM,
    main_contract: str,
    project_root: str,
    prover_opts: ProverOptions,
) -> BaseTool:
    sem = _prover_sem(prover_opts.cloud)
    stamper = make_validation_stamper(VALIDATION_KEY)
    # Serialize verify calls targeting the same spec name: the spec/conf are written
    # under a deterministic name and unlinked on exit, so two overlapping same-stem
    # calls (e.g. parallel verify_spec for one component) would race. Distinct stems
    # stay concurrent (notably on cloud, where ``sem`` is a no-op).
    # Not pruned: bounded by this run's stems (per-component + invariants) and dies with
    # the per-run tool; popping a held lock would let a later same-stem call mint a fresh,
    # non-excluding one.
    spec_locks: dict[str, asyncio.Lock] = {}

    @tool_display("Running prover", None)
    @tool(args_schema=VerifySpecSchema)
    async def verify_spec(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[StateWithSkips, InjectedState],
        rules: list[str] | None = None
    ) -> str | Command:
        if state["curr_spec"] is None:
            return "Specification not yet put on VFS"
        conf = state["config"]
        # With a seeded stem, name the spec/conf after it (so on-disk names match the
        # dump) under a lock; else fall back to unique uid names (no lock needed).
        spec_stem = state.get("spec_stem")
        conf_dir = (CERTORA_DIR / "confs") if spec_stem is not None else CERTORA_DIR
        lock = spec_locks.setdefault(spec_stem, asyncio.Lock()) if spec_stem is not None else nullcontext()
        async with lock:
            with tmp_spec(root=project_root, content=state["curr_spec"], name=spec_stem) as generated:
                config = {
                    **conf,
                    "verify": f"{main_contract}:{generated}",
                    "parametric_contracts": main_contract,
                    "optimistic_loop": True,
                    "rule_sanity": "basic",
                }

                if rules:
                    config["rule"] = rules

                summary = get_run_summary()
                summary.record_prover_conf(config)

                with temp_certora_file(
                    root=project_root,
                    content=json.dumps(config, indent=2),
                    ext="conf",
                    name=spec_stem,
                    prefix="verify",
                    dest_dir=conf_dir,
                ) as config_path:
                    async with sem:
                        result = await run_prover(
                            Path(project_root),
                            [config_path],
                            tool_call_id,
                            prover_opts,
                            _SpecCallbacks(get_stream_writer(), tool_call_id, summary, config),
                            DefaultCexHandler(llm, state, summarization_threshold=10)
                        )

                if isinstance(result, str):
                    return result
                if isinstance(result, SummarizedReport):
                    return result.todo_list
                all_verified = True
                for (r, stat) in result.rule_status.items():
                    if r in state["rule_skips"]:
                        continue
                    if not stat:
                        all_verified = False
                        break
                if rules is None and all_verified:
                    return tool_state_update(tool_call_id=tool_call_id, content=result.report, validations=stamper(state))
                return result.report

    return verify_spec
