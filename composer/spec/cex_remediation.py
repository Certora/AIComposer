"""
CEX-driven CVL remediation sub-agent for the codegen workflow.

Two sub-agents live here:

* ``cex_remediation_tool`` — the codegen author calls this when a prover run
  returns a counterexample whose root cause it has decided needs a CVL-side
  fix (summary, ghost model, invariant) rather than a code change. The
  remediator drafts a proposed full-spec replacement + rationale and returns
  it as a string. The codegen author stays in charge of the working-spec
  flow (``write_working_spec`` → verify → ``commit_working_spec``); the
  remediator never writes to the VFS or runs the prover itself.

* ``summary_critic_tool`` — a sub-agent the remediator can call to
  pre-flight a candidate change for soundness and design-faithfulness.
  Specifically tuned to catch the footguns the codegen author has been
  reaching for: ``_.transfer => NONDET`` (NONDET treats the body as a
  no-op, making every token movement a silent zero-effect call), naked
  ``DISPATCHER`` on ERC20-shaped polymorphic calls, ``persistent`` ghosts
  used to escape HAVOC, etc.

The system document is plumbed into both agents at construction time so
the critic can judge faithfulness of a proposed summary against the
protocol's stated design — not just local CVL soundness.
"""

from typing import NotRequired, override
from typing_extensions import TypedDict
from dataclasses import dataclass
import difflib
import uuid

from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState

from graphcore.graph import Builder, FlowInput
from graphcore.tools.schemas import WithAsyncDependencies, WithInjectedState, WithAsyncImplementation, WithInjectedId
from graphcore.tools.vfs import VFSState, VFSAccessor
from graphcore.tools.memory import AsyncMemoryBackend, async_memory_tool

from composer.core.state import AIComposerState
from composer.input.types import InputFileLike
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.util import uniq_thread_id
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.ui.tool_display import tool_display
from composer.prover.report_store import ReportStore
from .proposal_store import ProposalStore

class _CommonRemediationExtra(TypedDict):
    vfs: dict[str, str]

# ---------------------------------------------------------------------------
# Summary critic
# ---------------------------------------------------------------------------


class SummaryCritique(BaseModel):
    """The critic's verdict on a proposed CVL summary."""
    sound: bool = Field(
        description="True only if the proposed summary is both locally sound and faithful to the system document. False if any concerns."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific, actionable issues. Cite the offending CVL fragment, system-doc passage, or call site. Empty when sound."
    )
    suggested_direction: str | None = Field(
        default=None,
        description="One-paragraph guidance on how to address the issues. None when sound."
    )


def _render_critique(c: SummaryCritique) -> str:
    """Render the critic's verdict for the calling LLM. Tools must return
    strings; structured returns don't survive the langchain tool boundary."""
    if c.sound:
        return (
            "Soundness verdict: SOUND.\n"
            "The proposed change passed soundness and faithfulness checks."
        )
    parts = ["Soundness verdict: UNSOUND"]
    if c.issues:
        parts.append("")
        parts.append("Issues:")
        for i, issue in enumerate(c.issues, 1):
            parts.append(f"  {i}. {issue}")
    if c.suggested_direction:
        parts.append("")
        parts.append(f"Suggested direction: {c.suggested_direction}")
    return "\n".join(parts)

class _CritiqueExtra(_CommonRemediationExtra, RoughDraftState):
    pass

class _CritiqueState(MessagesState, _CritiqueExtra):
    result: NotRequired[SummaryCritique]

class _CritiqueInput(FlowInput, _CritiqueExtra):
    pass


def _critic_validator(s: _CritiqueState, _: SummaryCritique) -> str | None:
    if not s.get("did_read", False):
        return "Completion REJECTED: read your rough draft before delivering. Call read_rough_draft."
    return None


def summary_critic_tool(
    builder: Builder,
    system_doc: InputFileLike,
) -> BaseTool:
    """Build the summary-critic sub-agent as a tool.

    ``builder`` should already have the LLM, CVL manual + KB tools, and
    immutable VFS tools bound. ``system_doc`` is the protocol's design
    document; it's spliced into every critic invocation so the critic can
    judge faithfulness, not just local CVL well-formedness.
    """
    rough_draft_tools = get_rough_draft_tools(_CritiqueState)

    graph = (
        bind_standard(builder, _CritiqueState, validator=_critic_validator)
        .with_input(_CritiqueInput)
        .with_tools(rough_draft_tools)
        .with_sys_prompt_template("summary_critic_system.j2")
        .with_initial_prompt_template("summary_critic_initial.j2")
        .compile_async()
    )

    @tool_display("Critiquing proposed summary", "Summary critique")
    class SummaryCritic(WithAsyncImplementation[str], WithInjectedId, WithInjectedState[_RemediationState]):
        """Review a proposed CVL summary or spec change for soundness and
        faithfulness to the system document. Returns a verdict + issue
        list. Always invoke before delivering a remediation."""

        proposed_cvl: str = Field(
            description="The full proposed CVL spec contents after your changes to review."
        )
        proposed_addendum: str | None = Field(
            default=None,
            description=(
                "Free-form non-spec artifacts that accompany the proposed CVL change — "
                "e.g. a Solidity stub for a not-yet-implemented callee plus a CVL rule that "
                "validates the eventual implementation matches the stub's behavior. Pass "
                "exactly what you intend to put in the result's `addendum` field; the critic "
                "will scrutinize it alongside the CVL change (stubs without a paired "
                "validation rule are an anti-pattern). Leave null when the proposal has no "
                "addendum."
            )
        )
        target_call: str = Field(
            description="The external call(s) being summarized, e.g. `IERC20.transfer(address,uint256)` or `_.unwrapExcessWstEth()`."
        )
        rule_under_repair: str = Field(
            description="Name of the rule whose verification failure motivated this change."
        )
        cex_diagnosis: str = Field(
            description="The CEX analyzer's root-cause diagnosis for the failure."
        )

        @override
        async def run(self) -> str:
            diff_lines = list(difflib.unified_diff(
                self.state["draft_against_version"].splitlines(keepends=True),
                self.proposed_cvl.splitlines(keepends=True),
                fromfile="prior.spec",
                tofile="proposed.spec",
                n=3,
            ))
            diff_text = (
                "".join(diff_lines)
                if diff_lines
                else "(no diff — proposed is byte-identical to prior; this is almost always a remediator bug)"
            )

            input_parts: list[str | dict] = [
                "System document (read carefully — your job is to judge faithfulness to the design it describes):",
                system_doc.to_document_dict(),
                f"Rule under repair: {self.rule_under_repair}",
                f"CEX diagnosis: {self.cex_diagnosis}",
                f"Call(s) being summarized: {self.target_call}",
                (
                    "Diff against the prior version (start your review here — issues most "
                    "often live in the changed lines):"
                ),
                f"```diff\n{diff_text}```",
                "Full proposed CVL spec text (for completeness checks):",
                self.proposed_cvl,
            ]
            if self.proposed_addendum is not None:
                input_parts.append(
                    "Proposed addendum (non-spec artifacts — stubs, validation rules):"
                )
                input_parts.append(self.proposed_addendum)
            inp = _CritiqueInput(input=input_parts, did_read=False, memory=None, vfs=self.state["vfs"])
            st = await run_to_completion(
                graph, inp,
                thread_id=uniq_thread_id("summary-critic"),
                description="Summary critic",
                within_tool=self.tool_call_id,
            )
            assert "result" in st
            return _render_critique(st["result"])

    return SummaryCritic.as_tool("summary_critic")


# ---------------------------------------------------------------------------
# CEX remediation
# ---------------------------------------------------------------------------


class CEXRemediationResult(BaseModel):
    """The remediator's proposed change."""
    proposed_cvl: str = Field(
        description="Full proposed contents of the spec file under repair."
    )
    rationale: str = Field(
        description="One-paragraph explanation of the change and why it addresses the root cause, citing the root-cause category (A/B/C/D) and the specific strategy applied."
    )
    addendum: str | None = Field(
        default=None,
        description=(
            "Free-form non-spec artifacts the remediation requires — e.g. a proposed "
            "validation rule for a stub callee, or a Solidity stub the codegen author "
            "should consider adding. Use this when the fix requires changes outside the "
            "spec file (Strategy B.b in particular). Leave null when no extra artifacts "
            "are needed."
        ),
    )


def _render_remediation(
    r: CEXRemediationResult,
    *,
    diff_against: str,
    proposal_key: str,
    target_spec_path: str,
) -> str:
    """Render the remediator's proposal for the calling LLM.

    The proposed CVL is delivered as a unified diff against the version
    the remediator drafted from (working_version if present, else the
    committed ground truth). The full text is retrievable via
    ``proposal_key`` — ``write_working_spec(proposal_key=<key>)`` looks
    it up.

    Why a diff instead of the full text: when the full proposed spec is
    inlined, by the time the codegen author finishes reading it,
    attention has shifted away from the rationale and (importantly) the
    addendum's Solidity-side instructions. A diff is much shorter, so
    the rationale and addendum stay in the agent's active attention
    window. The working-spec flow doesn't suffer because the agent
    doesn't re-emit the full text — it passes the key.
    """
    diff_lines = list(difflib.unified_diff(
        diff_against.splitlines(keepends=True),
        r.proposed_cvl.splitlines(keepends=True),
        fromfile="prior.spec",
        tofile="proposed.spec",
        n=3,
    ))
    diff_text = (
        "".join(diff_lines)
        if diff_lines
        else "(no diff — proposal is byte-identical to the prior spec; "
             "this is almost always a remediator bug — surface it in your turn)"
    )
    parts = [
        "## Rationale",
        "",
        r.rationale,
        "",
        "## Proposed change (diff against prior spec)",
        "",
        f"```diff\n{diff_text}```",
        "",
        "## Apply via",
        "",
        f"`apply_remediation_proposal(proposal_key=\"{proposal_key}\")`",
        "",
        f"Then verify with `certora_prover(use_working_spec=True, target_spec=\"{target_spec_path}\", ...)`. "
        "The full proposed spec text is stored under the key; "
        "`apply_remediation_proposal` fetches it and stages it as your "
        "working draft. If the proposed CVL fails to typecheck or you need "
        "a small tweak, fall through to `write_working_spec` with the "
        "modified text.",
    ]
    if r.addendum is not None:
        parts.extend([
            "",
            "## Addendum (non-spec artifacts)",
            "",
            r.addendum,
        ])
    return "\n".join(parts)

class _RemediationExtra(_CommonRemediationExtra):
    draft_against_version: str

class _RemediationState(MessagesState, _RemediationExtra):
    result: NotRequired[CEXRemediationResult]


class _RemediationInput(FlowInput, _RemediationExtra):
    pass

@dataclass
class _CEXRemediationDeps:
    mat: VFSAccessor[VFSState]
    # Builder is fully bound except for the initial prompt — state, input,
    # tools (including the per-spec memory tool), system prompt, summarizer,
    # and result tool are all in place. The initial prompt is rendered per
    # call so the per-CEX context (rule, diagnosis, ground truth, working
    # version, system doc text) lives inline in the prompt body, where it
    # survives a summarization round; ``state["input"]`` does not survive,
    # since ``_get_summarizer_pure`` rebuilds the post-summary message list
    # from the rendered initial prompt template alone.
    builder: Builder[_RemediationState, None, _RemediationInput]
    reports: ReportStore
    proposals: ProposalStore


@tool_display("Drafting CEX remediation plan", "CEX remediation plan")
class CEXRemediator(
    WithAsyncDependencies[str, _CEXRemediationDeps],
    WithInjectedState[AIComposerState],
    WithInjectedId,
):
    """Delegate spec-side remediation of a (single) counterexample to a sub-agent.

    Use when a `certora_prover` run returned a CEX you've decided needs
    a CVL-side fix (summary, ghost model, invariant, etc.) rather than a
    code change. Pass the `report_key` printed alongside the failing
    rule in the prover's reportit. The agent returns a proposed full-spec replacement +
    rationale. You stay in charge of the working-spec flow: stage the
    proposal via `write_working_spec`/`apply_remediation_proposal`, verify with
    `certora_prover(use_working_spec=True)`, then commit.

    Do NOT use for: code-side bug fixes, spec corrections that weaken
    the property (use `propose_spec_change` for genuine spec bugs,
    with user review)."""

    report_key: str = Field(
        description=(
            "The opaque report_key printed in the prover's report under "
            "'Diagnoses'. Identifies the analyzed root cause; the remediator "
            "looks up the diagnosis text itself."
        ),
    )
    target_spec_path: str = Field(
        description="VFS path to the spec file under repair.",
    )

    @override
    async def run(self) -> str:
        with self.tool_deps() as deps:
            # Look up the analyzer's diagnosis by report_key. Stale keys
            # (from a prior prover run that's been superseded, or from a
            # different thread) return None — surface as a recoverable
            # error so the agent can re-run the prover for fresh keys.

            diagnosis_record = await deps.reports.lookup(self.report_key)
            if diagnosis_record is None:
                return (
                    f"No report found for report_key {self.report_key!r}."
                )

            # Ground truth — the committed spec on the VFS. Authoritative
            # baseline; the codegen author cannot mutate this except via
            # `commit_working_spec` (an observable event the remediator's
            # memory should track across rounds).
            ground_truth_bytes = deps.mat.get(self.state, self.target_spec_path)
            if ground_truth_bytes is None:
                return f"No file exists at path {self.target_spec_path}"
            try:
                ground_truth = ground_truth_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return f"Could not read file at path {self.target_spec_path}"

            # Working version — the most recent draft the codegen author
            # staged via `write_working_spec`, which is what the prover
            # most recently ran against. Author-mutable; treat as a
            # description of what the prover saw, NOT as authoritative.
            # When this diverges from your memory of your last proposal,
            # the codegen author has either edited or committed something
            # else — flag the divergence in your rationale.
            working_version = self.state.get("working_spec")

            draft_against = working_version if working_version is not None else ground_truth

            # Render and compile per call so the per-CEX inputs sit inside
            # the rendered initial prompt template — the only part of the
            # message history the summarizer replays after a summarization
            # round (see graphcore's _get_summarizer_pure). The system
            # document is intentionally NOT inlined here; the agent fetches
            # it on demand via the `read_system_document` tool.
            graph = deps.builder.with_initial_prompt_template(
                "cex_remediation_initial.j2",
                attributed_rules=diagnosis_record.attributed_rules,
                target_spec_path=self.target_spec_path,
                cex_diagnosis=diagnosis_record.diagnosis,
                ground_truth=ground_truth,
                working_version=working_version,
            ).compile_async()

            inp = _RemediationInput(
                input=[],
                vfs=self.state["vfs"],
                draft_against_version=draft_against,
            )
            st = await run_to_completion(
                graph, inp,
                thread_id=uniq_thread_id("cex-remediation"),
                description="CEX Remediation Agent",
                within_tool=self.tool_call_id,
                recursion_limit=300,
            )
            assert "result" in st
            result: CEXRemediationResult = st["result"]

            # Persist the full proposed_cvl under an opaque key so
            # apply_remediation_proposal can fetch it without the
            # codegen author having to re-emit (or paraphrase) the spec
            # text. The codegen author only ever sees the diff + the key.
            proposal_key = uuid.uuid4().hex
            await deps.proposals.record(proposal_key, result.proposed_cvl)
            return _render_remediation(
                result,
                diff_against=draft_against,
                proposal_key=proposal_key,
                target_spec_path=self.target_spec_path,
            )


def cex_remediation_tool(
    builder: Builder,
    mat: VFSAccessor[VFSState],
    system_doc: InputFileLike,
    mem: AsyncMemoryBackend,
    proposals: ProposalStore,
    reports: ReportStore
) -> BaseTool:
    """Build the CEX-remediation sub-agent as a tool.

    ``builder`` should have: LLM, CVL manual/KB/researcher tools,
    immutable VFS tools, and the ``summary_critic`` tool already bound.
    ``system_doc`` is spliced into the agent's input so it can weigh the
    protocol's design when proposing changes (the system doc is the
    authoritative description of intent — a summary that contradicts it
    is wrong even if locally well-formed CVL).

    ``mem`` is the remediator's dedicated memory backend — namespaced
    per codegen session, distinct from the codegen author's own memory
    so the two don't pollute each other's context windows. The agent
    organizes its notes by spec path under ``/memories/work/{path}``;
    the convention is enforced at the prompt level, not the framework.

    All builder configuration except the initial prompt happens here;
    the initial prompt is rendered per call inside ``CEXRemediator.run``
    so the per-CEX context survives summarization. See the comment on
    ``_CEXRemediationDeps.builder``.
    """
    memory_tool = async_memory_tool(mem)

    @tool_display("Reading system document", None)
    class ReadSystemDocument(WithAsyncImplementation[list[str | dict]]):
        """Read the protocol's system document — the authoritative
        description of the system you're verifying. Returns the document
        as a multimodal content block (text or PDF, whichever the upstream
        run supplied), so binary system docs round-trip correctly.

        The contents do not change across calls. Cache key passages in
        your memory after the first read; do not re-call this every round.
        """

        @override
        async def run(self) -> list[str | dict]:
            return ["System document:", system_doc.to_document_dict()]

    read_system_document = ReadSystemDocument.as_tool("read_system_document")

    prebuilt = (
        bind_standard(builder, _RemediationState)
        .with_input(_RemediationInput)
        .with_tools([memory_tool, read_system_document])
        .with_sys_prompt_template("cex_remediation_system.j2")
    )

    return CEXRemediator.bind(_CEXRemediationDeps(
        mat=mat, builder=prebuilt, proposals=proposals, reports=reports
    )).as_tool("cex_remediation")
