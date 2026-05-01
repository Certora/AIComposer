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

from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState

from graphcore.graph import Builder, FlowInput
from graphcore.tools.schemas import WithAsyncImplementation

from composer.input.types import InputFileLike
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.util import uniq_thread_id
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.ui.tool_display import tool_display


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


class _CritiqueState(MessagesState, RoughDraftState):
    result: NotRequired[SummaryCritique]


class _CritiqueInput(FlowInput, RoughDraftState):
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
    class SummaryCritic(WithAsyncImplementation[str]):
        """Review a proposed CVL summary or spec change for soundness and
        faithfulness to the system document. Returns a verdict + issue
        list. Always invoke before delivering a remediation."""

        proposed_cvl: str = Field(
            description="The full proposed CVL spec contents to review."
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
            input_parts: list[str | dict] = [
                "System document (read carefully — your job is to judge faithfulness to the design it describes):",
                system_doc.to_document_dict(),
                f"Rule under repair: {self.rule_under_repair}",
                f"CEX diagnosis: {self.cex_diagnosis}",
                f"Call(s) being summarized: {self.target_call}",
                "Proposed CVL spec text:",
                self.proposed_cvl,
            ]
            inp = _CritiqueInput(input=input_parts, did_read=False, memory=None)
            st = await run_to_completion(
                graph, inp,
                thread_id=uniq_thread_id("summary-critic"),
                description="Summary critic",
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
        description="One-paragraph explanation of the change and why it addresses the root cause."
    )


def _render_remediation(r: CEXRemediationResult) -> str:
    """Render the remediator's proposal for the calling LLM. Tools must
    return strings; the proposed CVL is delivered as plain text after the
    rationale so the calling agent can lift it into the working-spec
    flow."""
    return (
        "## Rationale\n\n"
        f"{r.rationale}\n\n"
        "## Proposed spec contents\n\n"
        f"{r.proposed_cvl}"
    )


class _RemediationState(MessagesState, RoughDraftState):
    result: NotRequired[CEXRemediationResult]


class _RemediationInput(FlowInput, RoughDraftState):
    pass


def _remediation_validator(s: _RemediationState, _: CEXRemediationResult) -> str | None:
    if not s.get("did_read", False):
        return "Completion REJECTED: read your rough draft before delivering. Call read_rough_draft."
    return None


def cex_remediation_tool(
    builder: Builder,
    system_doc: InputFileLike,
) -> BaseTool:
    """Build the CEX-remediation sub-agent as a tool.

    ``builder`` should have: LLM, CVL manual/KB/researcher tools,
    immutable VFS tools, and the ``summary_critic`` tool already bound.
    ``system_doc`` is spliced into the agent's input so it can weigh the
    protocol's design when proposing changes (the system doc is the
    authoritative description of intent — a summary that contradicts it
    is wrong even if locally well-formed CVL).
    """
    rough_draft_tools = get_rough_draft_tools(_RemediationState)

    graph = (
        bind_standard(builder, _RemediationState, validator=_remediation_validator)
        .with_input(_RemediationInput)
        .with_tools(rough_draft_tools)
        .with_sys_prompt_template("cex_remediation_system.j2")
        .with_initial_prompt_template("cex_remediation_initial.j2")
        .compile_async()
    )

    @tool_display("Drafting CEX remediation", "CEX remediation")
    class CEXRemediator(WithAsyncImplementation[str]):
        """Delegate spec-side remediation of a counterexample to a sub-agent.

        Use when a `certora_prover` run returned a CEX you've decided needs
        a CVL-side fix (summary, ghost model, invariant) rather than a
        code change. The sub-agent reads the diagnosis, explores via the
        VFS and CVL research tools, drafts a candidate, runs the summary
        critic for soundness review, and returns a proposed full-spec
        replacement + rationale. You stay in charge of the working-spec
        flow: stage the proposal via `write_working_spec`, verify with
        `certora_prover(use_working_spec=True)`, then commit.

        Do NOT use for: code-side bug fixes, spec corrections that weaken
        the property (use `propose_spec_change` for genuine spec bugs,
        with user review), or counterexamples whose root cause is an
        impossible starting state (handle inline with invariants /
        justification rules)."""

        rule_name: str = Field(description="The rule that failed.")
        target_spec_path: str = Field(description="VFS path to the spec file under repair.")
        current_spec: str = Field(description="Current contents of the spec file.")
        cex_diagnosis: str = Field(description="The CEX analyzer's diagnosis.")

        @override
        async def run(self) -> str:
            input_parts: list[str | dict] = [
                "System document (the authoritative description of the protocol's design):",
                system_doc.to_document_dict(),
                f"Rule under repair: {self.rule_name}",
                f"Target spec VFS path: {self.target_spec_path}",
                f"CEX diagnosis: {self.cex_diagnosis}",
                "Current spec file contents:",
                self.current_spec,
            ]
            inp = _RemediationInput(input=input_parts, did_read=False, memory=None)
            st = await run_to_completion(
                graph, inp,
                thread_id=uniq_thread_id("cex-remediation"),
                description="CEX remediation",
            )
            assert "result" in st
            return _render_remediation(st["result"])

    return CEXRemediator.as_tool("cex_remediation")
