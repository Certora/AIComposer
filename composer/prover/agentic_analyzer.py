"""Agentic CEX analyzer for the codegen workflow.

Replaces single-shot per-rule analysis (``TrivialFanoutCexHandler``) with
two layered sub-agents:

1. **Per-rule analyzer** — runs once per failing CEX, with a Python-side
   scratchpad shared across all CEXes for the same rule. Each invocation
   commits to a discriminated union: either match an existing scratchpad
   entry (with evidence) or open a new entry (with text + evidence).
   This amortizes cost across parametric instances of the same rule
   (CEX 2 of rule foo can match CEX 1's root cause without re-deriving)
   while structurally preventing lazy over-clustering — the model must
   point at a specific prior entry and articulate evidence to merge.

2. **Cross-rule aggregator** — runs once after all per-rule passes, sees
   the union of per-CEX analyses across rules, partitions them into
   ``K`` ``AnalyzedDiagnosis`` records. Bias mitigation: aggregator runs
   fresh on per-CEX texts; may un-group within a rule if the analysis
   texts diverge enough.

Both agents read source via fs tools scoped to the prover's report
directory (the materialized VFS containing only the source files
compiled into this verification run). No write access. No access to the
codegen author's broader VFS — the analyzer should not be answering
"what does the design intend"; it should be reasoning from the trace
and the actual source compiled into the verification problem.
"""


import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, NotRequired, Union, override
import uuid

from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState

from graphcore.graph import Builder, FlowInput
from graphcore.tools.vfs import fs_tools

from composer.prover.core import (
    CexHandler,
    CexProgressCallbacks,
    FailingRule,
    zip_results
)
from composer.prover.ptypes import RuleResult, RulePath, AnalyzedDiagnosis
from composer.prover.report_store import ReportStore
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.util import uniq_thread_id
from composer.templates.loader import load_jinja_template
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools


# ---------------------------------------------------------------------------
# Per-CEX commit shape (discriminated union)
# ---------------------------------------------------------------------------


class _MatchedExisting(BaseModel):
    """Commit: this CEX shares its root cause with a prior scratchpad entry."""

    decision: Literal["match"] = "match"
    entry_index: int = Field(
        description=(
            "0-based index into the scratchpad's existing entries that this CEX "
            "shares a root cause with. Must be a valid index of an existing entry."
        ),
    )
    evidence: str = Field(
        description=(
            "Why this CEX matches the named scratchpad entry. Cite specific aspects "
            "of the trace (storage state, call sequence, operands) that mirror the "
            "prior entry's evidence. A vague 'looks similar' is not acceptable."
        ),
    )


class _NewEntry(BaseModel):
    """Commit: this CEX has a root cause not yet recorded in the scratchpad."""

    decision: Literal["new"] = "new"
    text: str = Field(
        description=(
            "The root cause for this CEX. Be specific: name the construct (HAVOC "
            "of variable X, missing invariant on storage Y, ghost mismodeling, "
            "spec assertion contradicts the implementation, etc.) and the part of "
            "the trace that points to it."
        ),
    )
    evidence: str = Field(
        description=(
            "Concrete evidence from this CEX trace that supports the root cause. "
            "Cite operands, storage values, call sequence."
        ),
    )


_PerCexCommit = Annotated[
    Union[_MatchedExisting, _NewEntry],
    Field(discriminator="decision"),
]


class _PerCexCommitWrapper(BaseModel):
    """Your structured commitment to your result."""

    commit: _PerCexCommit


# ---------------------------------------------------------------------------
# Aggregator shape
# ---------------------------------------------------------------------------


class _AggregatedPartition(BaseModel):
    """One root cause covering one or more per-rule root causes."""

    diagnosis: str = Field(
        description=(
            "The consolidated root-cause description. Should subsume the "
            "per-rule root causes you attribute to it, at a level of "
            "abstraction that's specific enough to be actionable yet general "
            "enough to cover all the per-rule causes in this partition."
        ),
    )
    cause_indices: list[int] = Field(
        description=(
            "0-based indices into the input list of per-rule root causes "
            "that this partition covers. Every input index MUST appear in "
            "exactly one partition; partitions cover the input exactly."
        ),
    )


class _AggregatorResult(BaseModel):
    """Output of the cross-rule aggregator: a partition of per-rule root
    causes by cross-rule equivalence."""

    partitions: list[_AggregatedPartition] = Field(
        description=(
            "Disjoint partition of the input per-rule root causes by "
            "cross-rule equivalence. Every input index appears in exactly "
            "one partition. Length is K, the number of distinct root causes "
            "across all rules."
        ),
    )


# ---------------------------------------------------------------------------
# Scratchpad (orchestrator-owned)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScratchpadEntry:
    text: str
    evidence: str


@dataclass(frozen=True)
class _PerRuleRootCause:
    """One root cause identified by the per-rule analyzer, covering one or
    more CEXes of a single rule.

    This is the unit the cross-rule aggregator partitions over — the
    per-rule agent's match/new commits already settled within-rule
    clustering, and the aggregator only handles cross-rule equivalence.
    ``attributed_rule_paths`` is the list of ``RuleResult.name`` values
    for every CEX of this rule that the per-rule agent attributed to
    this root cause.
    """

    text: str
    evidence: str
    attributed_rule_paths: list[RulePath]


# ---------------------------------------------------------------------------
# State / input types for the sub-agents
# ---------------------------------------------------------------------------


class _PerCexState(MessagesState, RoughDraftState):
    result: NotRequired[_PerCexCommitWrapper]


class _PerCexInput(FlowInput, RoughDraftState):
    pass


class _AggregatorState(MessagesState, RoughDraftState):
    result: NotRequired[_AggregatorResult]


class _AggregatorInput(FlowInput, RoughDraftState):
    pass


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _per_cex_initial_prompt(
    *,
    rule_name: str,
    instance: RuleResult,
    scratchpad: list[_ScratchpadEntry],
) -> str:
    parts: list[str] = [
        f"# Counterexample on rule `{rule_name}`",
        "",
        f"Rule path: `{instance.name}`",
        "",
        "## Counterexample trace",
        "",
        "```",
        instance.cex_dump or "(no cex dump)",
        "```",
        "",
    ]
    if scratchpad:
        parts += [
            "## Root causes already recorded for this rule",
            "",
        ]
        for i, entry in enumerate(scratchpad):
            parts += [
                f"### Entry {i}",
                "",
                entry.text,
                "",
                f"_Evidence:_ {entry.evidence}",
                "",
            ]
    else:
        parts += [
            "## Root causes already recorded for this rule",
            "",
            "None yet. Commit `new` with the root cause text and evidence.",
            "",
        ]
    return "\n".join(parts)


def _aggregator_initial_prompt(causes: list[_PerRuleRootCause]) -> str:
    parts: list[str] = []
    for i, cause in enumerate(causes):
        rules = ", ".join(f"`{r}`" for r in cause.attributed_rule_paths)
        parts += [
            f"## Root cause {i}",
            "",
            f"_Covers rule(s):_ {rules}",
            "",
            cause.text,
            "",
            f"_Evidence:_ {cause.evidence}",
            "",
        ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# AgenticCexHandler
# ---------------------------------------------------------------------------


def _per_rule_validator(scratchpad: list[_ScratchpadEntry]):
    """Build the per-rule commit validator. Closes over the current
    scratchpad so the agent's ``entry_index`` (when matching) gets
    bounds-checked at commit time. Validation failures land back in the
    sub-agent's conversation as a rejection — the agent retries with a
    valid index rather than the orchestrator silently fudging."""
    n = len(scratchpad)

    def validate(state: _PerCexState, result: _PerCexCommitWrapper) -> str | None:
        if not state.get("did_read", False):
            return "Completion REJECTED: read your rough draft before delivering."
        commit = result.commit
        if isinstance(commit, _MatchedExisting):
            if n == 0:
                return (
                    "Completion REJECTED: no prior results, you cannot "
                    "match an existing entry. Commit `new` with the root "
                    "cause for this CEX."
                )
            if not (0 <= commit.entry_index < n):
                return (
                    f"Completion REJECTED: entry_index {commit.entry_index} "
                    f"is out of range. Valid indices are 0..{n - 1} "
                    f"({n} prior results)."
                )
        return None

    return validate


def _aggregator_validator(input_count: int):
    """Build the aggregator's partition validator. Closes over the input
    length so the partition's ``cause_indices`` are checked for:
    in-range, no duplicates, every input covered. Rejection messages
    name the specific failure (out-of-range index, missing index,
    duplicated index) so the agent's retry has actionable feedback."""

    def validate(state: _AggregatorState, result: _AggregatorResult) -> str | None:
        if not state.get("did_read", False):
            return "Completion REJECTED: read your rough draft before delivering."
        seen: dict[int, int] = {}  # idx → first partition that claimed it
        out_of_range: list[int] = []
        for p_idx, partition in enumerate(result.partitions):
            for cause_idx in partition.cause_indices:
                if not (0 <= cause_idx < input_count):
                    out_of_range.append(cause_idx)
                    continue
                if cause_idx in seen:
                    return (
                        f"Completion REJECTED: cause_index {cause_idx} appears "
                        f"in partition {seen[cause_idx]} AND partition "
                        f"{p_idx}. Every input index must appear in EXACTLY "
                        f"ONE partition."
                    )
                seen[cause_idx] = p_idx
        if out_of_range:
            return (
                f"Completion REJECTED: cause_indices {sorted(set(out_of_range))} "
                f"are out of range. Valid indices are 0..{input_count - 1} "
                f"({input_count} per-rule root causes)."
            )
        missing = [i for i in range(input_count) if i not in seen]
        if missing:
            return (
                f"Completion REJECTED: cause_indices {missing} are not covered "
                f"by any partition. Every input index must appear in exactly "
                f"one partition."
            )
        return None

    return validate


class AgenticCexHandler(CexHandler):
    """Codegen-side CEX analyzer. Iterates a per-rule sub-agent per CEX
    with a shared scratchpad, then runs a cross-rule aggregator.

    ``builder`` should have the LLM, the CVL research / manual / KB
    tools, and the loader/checkpointer bound — those are reused across
    every per-rule and aggregator sub-agent. Source-side reads are
    added per call via fs tools scoped to the prover's report
    directory; not on the builder.

    ``report_store`` is the codegen-side persistence layer for keyed
    diagnoses; the handler writes each produced ``AnalyzedDiagnosis``
    here so ``cex_remediation`` can look them up by ``report_key``
    later. No summarization layer — the handler produces ``K << N``
    diagnoses by construction.
    """

    def __init__(
        self,
        builder: Builder,
        report_store: ReportStore,
        *,
        max_concurrent_analyses: int = 8,
    ) -> None:
        super().__init__()
        self._builder = builder
        self._report_store = report_store
        self._max_concurrent = max_concurrent_analyses

    @override
    async def analyze(
        self,
        failing_rules: list[FailingRule],
        all_rules: list[RuleResult],
        tool_call_id: str,
        callbacks: CexProgressCallbacks,
        report_dir: Path,
    ) -> str:
        # fs tools scoped to the report's ``inputs/.certora_sources``
        # subdirectory — the canonical location of source files actually
        # compiled into this verification problem. The standalone CEX
        # analyzer at ``analyzer/analysis.py`` uses the same path; we
        # mirror it here so the agentic analyzer sees exactly what the
        # standalone tool sees, neither more (e.g. .certora_internal
        # build artifacts) nor less. Forbidden-read pattern matches the
        # standalone tool too.
        sources_dir = report_dir / "inputs" / ".certora_sources"
        report_fs = fs_tools(
            fs_layer=str(sources_dir),
            forbidden_read=r"^\..*$",
        )

        # Different rules don't share scratchpad state, so the outer loop
        # over ``failing_rules`` is parallelizable. Per-CEX iteration
        # within a single rule stays sequential — the scratchpad is
        # built up across iterations and each commit decision depends on
        # the entries placed by prior commits. The per-rule output is the
        # scratchpad lifted to (text, evidence, attributed-rule-paths)
        # records — the within-rule clustering the per-rule agent
        # already committed to. The aggregator only handles cross-rule
        # equivalence on top of those.
        #
        # Concurrency bound: a per-call Semaphore caps how many per-rule
        # sub-agents run concurrently. The parallel-prover guard already
        # ensures only one prover tool runs at a time within a graph, so
        # this also bounds the total live sub-agents per graph.
        sem = asyncio.Semaphore(self._max_concurrent)

        # Ordering note: ``prover_run`` was emitted long ago and has had
        # the prover's await-loop to flush through to the queue. The
        # Analysis Agents collapsible / within_tool override are
        # installed in the renderer's ``prover_run`` handler. This
        # ``analyze`` runs from inside the tool coroutine and emits
        # ``Start(subagent_tid)`` synchronously via ``run_to_completion``;
        # by then the override is already in place.
        async def _process_rule(
            rule_group: FailingRule,
        ) -> list[_PerRuleRootCause]:
            og_rule : dict[int, RulePath] = {}
            scratchpad: list[_ScratchpadEntry] = []
            attribution: dict[int, list[RulePath]] = {}
            for instance in rule_group.instances:
                if instance.status != "VIOLATED":
                    continue
                async with sem:
                    await callbacks.on_analysis_start(instance)
                    commit = await self._run_per_rule(
                        rule_name=rule_group.rule_name,
                        instance=instance,
                        scratchpad=scratchpad,
                        report_fs=report_fs,
                        tool_call_id=tool_call_id,
                    )
                # Resolve commit to a scratchpad index. The validator
                # (see ``_per_rule_validator``) already enforced
                # index-in-range for matches; trust it. The displayed
                # explanation is back-referenced for matches (so the UI
                # makes the decision visible) and the raw root cause for
                # new entries.
                match commit.commit:
                    case _MatchedExisting() as m:
                        idx = m.entry_index
                        explanation = (
                            f"Same root cause as a prior counterexample "
                            f"on the analysis of rule `{og_rule[idx].pprint()}:\n\n"
                            f"{scratchpad[idx].text}\n\n"
                            f"_Match evidence:_ {m.evidence}"
                        )
                    case _NewEntry() as n:
                        idx = len(scratchpad)
                        og_rule[idx] = instance.path
                        scratchpad.append(_ScratchpadEntry(
                            text=n.text, evidence=n.evidence,
                        ))
                        explanation = n.text
                attribution.setdefault(idx, []).append(instance.path)
                await callbacks.on_analysis_complete(instance, explanation)
            return [
                _PerRuleRootCause(
                    text=entry.text,
                    evidence=entry.evidence,
                    attributed_rule_paths=attribution.get(i, []),
                )
                for i, entry in enumerate(scratchpad)
            ]

        per_rule = await asyncio.gather(
            *(_process_rule(rg) for rg in failing_rules)
        )
        # Flatten preserving rule order (gather preserves input order).
        all_causes: list[_PerRuleRootCause] = [
            c for rule_results in per_rule for c in rule_results
        ]

        if not all_causes:
            # No CEXes — render a no-failures report from the rule
            # statuses we have. Caller still gets a non-empty string.
            return ""

        # Cross-rule aggregator. Sees the per-rule root causes (already
        # within-rule-clustered) and partitions them by cross-rule
        # equivalence.
        agg = await self._run_aggregator(
            all_causes, report_fs=report_fs, tool_call_id=tool_call_id,
        )

        # Validator (see ``_aggregator_validator``) already enforced that
        # every cause_index is in range, no duplicates, every input
        # covered. We attribute by union over the referenced per-rule
        # root causes' rule paths.
        diagnoses: list[AnalyzedDiagnosis] = []
        rule_to_diagnosis_key: dict[RulePath, str] = {}

        for partition in agg.partitions:
            attributed: list[RulePath] = []
            report_key = uuid.uuid4().hex
            for i in partition.cause_indices:
                for attr in all_causes[i].attributed_rule_paths:
                    attributed.append(attr)
                    rule_to_diagnosis_key[attr] = report_key
            diagnoses.append(AnalyzedDiagnosis(
                report_key=report_key,
                diagnosis=partition.diagnosis,
                attributed_rules=attributed,
            ))

        # Persist diagnoses keyed by report_key so cex_remediation can
        # look them up later. Codegen author only ever sees the keys
        # (rendered into the report below); paraphrase-corruption is
        # blocked by construction.
        await self._report_store.record(diagnoses)

        # Render the final report. Every failing rule that the per-rule
        # iteration touched ends up attributed to one of the produced
        # diagnoses (via attribution above); we map back here for the
        # template's ``rule_entries`` shape.
        rule_entries = zip_results(
            all_rules, lambda r: rule_to_diagnosis_key.get(r.path)
        )
        return load_jinja_template(
            "rule_feedback.j2",
            rule_entries=rule_entries,
            diagnoses=diagnoses,
        )

    # ── sub-agent runners ───────────────────────────────────────

    async def _run_per_rule(
        self,
        *,
        rule_name: str,
        instance: RuleResult,
        scratchpad: list[_ScratchpadEntry],
        report_fs: "list[BaseTool]",
        tool_call_id: str,
    ) -> _PerCexCommitWrapper:
        rough = get_rough_draft_tools(
            _PerCexState, review_reminder=_PER_RULE_REVIEW_REMINDER,
        )
        graph = (
            bind_standard(
                self._builder,
                _PerCexState,
                validator=_per_rule_validator(scratchpad),
            )
            .with_input(_PerCexInput)
            .with_tools([*rough, *report_fs])
            .with_initial_prompt(
                _per_cex_initial_prompt(
                    rule_name=rule_name,
                    instance=instance,
                    scratchpad=scratchpad,
                )
            )
            .with_sys_prompt_template("cex_analyzer_per_rule_system.j2")
            .compile_async()
        )
        st = await run_to_completion(
            graph,
            _PerCexInput(input=[], did_read=False, memory=None),
            thread_id=uniq_thread_id("cex-analyzer"),
            description=f"CEX analysis: {instance.name}",
            within_tool=tool_call_id,
        )
        assert "result" in st
        return st["result"]

    async def _run_aggregator(
        self,
        causes: list[_PerRuleRootCause],
        *,
        report_fs: "list[BaseTool]",
        tool_call_id: str,
    ) -> _AggregatorResult:
        rough = get_rough_draft_tools(
            _AggregatorState,
        )
        graph = (
            bind_standard(
                self._builder,
                _AggregatorState,
                validator=_aggregator_validator(len(causes)),
            )
            .with_input(_AggregatorInput)
            .with_tools([*rough, *report_fs])
            .with_initial_prompt(_aggregator_initial_prompt(causes))
            .with_sys_prompt_template("cex_analyzer_aggregator_system.j2")
            .compile_async()
        )
        st = await run_to_completion(
            graph,
            _AggregatorInput(input=[], did_read=False, memory=None),
            thread_id=uniq_thread_id("cex-aggregator"),
            description="CEX aggregation",
            within_tool=tool_call_id,
        )
        assert "result" in st
        return st["result"]


# ---------------------------------------------------------------------------
# Review reminders — emitted alongside read_rough_draft to re-state at
# review time what the agent should be checking. Defeats long-context
# drift where the agent reviews its draft without remembering the
# original criteria.
# ---------------------------------------------------------------------------


_PER_RULE_REVIEW_REMINDER = """\
Re-read the draft above and double check, before delivering. You should be particularly careful in finding:
1. Unjustified assertions about CVL behavior
2. Applications of overly broad matching
3. Unjustified or speculative assertions about prover behavior
4. Assertions about the scenario modeled in the counterexample that is not backed up by verifiable, unambiguous evidence in the XML dump.
"""


_AGGREGATOR_REVIEW_REMINDER = """\
Re-read the draft above and check, before delivering:

- Does every input root-cause index appear in EXACTLY ONE partition?
  Missing indices and double-counted indices both make the result
  invalid.
- Did you err toward more partitions when uncertain? The failure mode is
  over-grouping — two per-rule causes sharing a vague label like
  "missing invariant" are SEPARATE partitions when the invariants
  differ.
- Is each partition's `diagnosis` actionable + specific enough that
  a concrete fix can be reached without further re-derivation?

If any check fails, revise the draft (`write_rough_draft`) and re-read
before delivering."""
