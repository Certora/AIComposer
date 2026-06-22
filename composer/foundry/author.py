"""Foundry test author — generates ``.t.sol`` tests for property formulations.

The single home for the foundry authoring workflow: the author's tools, the
feedback judge, the publish gate, and the batch entry point
(``batch_foundry_test_generation``). The ``forge test`` runner lives in
``composer.foundry.runner``; state types and the publish-gate checks live in
``composer.foundry.state``.

Workflow shape:

* Single ``curr_test: str`` buffer per batch (one ``.t.sol`` file), written
  via ``put_test_raw``. No put-time compile check; ``forge_test`` is the gate.
* A feedback judge (``feedback_tool``) reviews the draft against the batch's
  properties. Publish requires both a green unseeded ``forge_test`` run AND a
  judge acceptance stamped on the *current* buffer.
* The publish-time property→test mapping is validated against the test names
  forge actually ran (from its JSON output), in both directions: every
  non-skipped property is demonstrated by a test that ran, and every test
  that ran is tied back to a property.
* Per-test expected-failure marking via ``expect_test_failure``.
* No prover-config editor — foundry projects are assumed pre-configured.
"""

from dataclasses import dataclass
import asyncio
from typing import (
    Awaitable, Callable, Literal, NotRequired, Protocol, override
)
from typing_extensions import TypedDict

from langgraph.graph import MessagesState
from langgraph.types import Command
from pydantic import BaseModel, Field

from graphcore.graph import FlowInput, tool_state_update
from graphcore.summary import SummaryConfig
from graphcore.tools.schemas import (
    WithAsyncDependencies, WithAsyncImplementation, WithImplementation,
    WithInjectedId, WithInjectedState,
)

from composer.spec.context import CVLGeneration, CVLJudge, WorkflowContext
from composer.spec.cvl_generation import (
    PropertyFeedbackProtocol, RebuttalBase, SkippedProperty,
)
from composer.spec.feedback import PropertyFeedback
from composer.spec.gen_types import TypedTemplate
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.prop import PropertyFormulation
from composer.spec.system_model import ContractComponentInstance
from composer.spec.tool_env import BasicAgentTools
from composer.spec.service_host import ServiceHost
from composer.spec.util import uniq_thread_id
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.ui.tool_display import (
    suppress_ack, tool_display,
)

from composer.foundry.runner import get_forge_test_tool
from composer.foundry.state import (
    FEEDBACK,
    FORGE_TEST_VALIDATION_KEY,
    FOUNDRY_JUDGE_KEY,
    FoundryTestExtra,
    FoundryGenerationInput,
    FoundryGenerationState,
    PropertyTestMapping,
    check_foundry_completion,
    make_foundry_validation_stamper,
    validate_property_tests,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class GeneratedFoundryTest(BaseModel):
    """Successful output of the foundry author for a batch."""
    commentary: str
    test_source: str
    skipped: list[SkippedProperty] = Field(default_factory=list)
    property_tests: list[PropertyTestMapping] = Field(default_factory=list)
    # Forge ground truth at publish time: the tests that actually ran in the
    # gating unseeded run, and the author's expected-failure markings (test
    # name -> reason). Together they give every test a pass / expected-failure
    # status without trusting the model's own transcription.
    expected_failures: dict[str, str] = Field(default_factory=dict)
    ran_tests: list[str] = Field(default_factory=list)


class GaveUp(BaseModel):
    reason: str


type BatchFoundryResult = GeneratedFoundryTest | GaveUp


# ---------------------------------------------------------------------------
# Author tools
# ---------------------------------------------------------------------------


@tool_display(
    label=lambda p: f"Putting test draft ({len(p.get('test_source', ''))} chars)",
    result=suppress_ack("Put test result", ("Accepted",)),
)
class PutTestRaw(WithImplementation[Command], WithInjectedId):
    """
    Put a foundry test file into the working buffer.

    The provided source replaces the entire ``curr_test`` buffer. There is no
    put-time compile check — call ``forge_test`` to verify the draft actually
    builds and passes. ``forge_test``'s green stamp is invalidated by any
    subsequent ``put_test_raw``, so call ``forge_test`` *after* you're done
    iterating.
    """
    test_source: str = Field(
        description=(
            "The full source of the foundry test file (a single ``.t.sol`` "
            "file's contents). Must declare a contract that extends "
            "``forge-std/Test.sol``'s ``Test`` and contain ``test_*`` "
            "functions for the properties being verified."
        )
    )

    @override
    def run(self) -> Command:
        return tool_state_update(
            tool_call_id=self.tool_call_id,
            content="Accepted",
            curr_test=self.test_source,
        )


@tool_display("Reading current test draft", None)
class GetTestTool(WithInjectedState[FoundryTestExtra], WithImplementation):
    """
    Retrieve the textual representation of the current foundry test.
    """
    def run(self) -> str:
        if self.state["curr_test"] is None:
            return "No test draft written"
        return self.state["curr_test"]

@tool_display(
    lambda p: f"Skipping property `{p.get('property_title', '?')}`",
    suppress_ack("Skip result", ("Recorded skip",)),
)
class _RecordSkipSchema(
    WithInjectedId,
    # deps: the batch's property titles
    WithAsyncDependencies[Command, list[str]],
):
    """
    Declare that you are skipping a property from the batch.

    You must provide the property's title and a justification. Skipping
    excludes the property from the publish-time property→test mapping
    check; only use after a genuine attempt to formalize.
    """
    property_title: str = Field(
        description="The snake_case title of the property from the batch listing"
    )
    reason: str = Field(
        description="Justification for why this property cannot be formalized as a foundry test"
    )

    @override
    async def run(self) -> Command:
        with self.tool_deps() as titles:
            if self.property_title not in titles:
                return tool_state_update(
                    self.tool_call_id,
                    f"Unknown property title {self.property_title!r}. Must be one "
                    f"of: {', '.join(titles)}.",
                )
        if not self.reason.strip():
            return tool_state_update(
                self.tool_call_id,
                "A non-empty justification is required when skipping a property.",
            )
        skip = SkippedProperty(
            property_title=self.property_title,
            reason=self.reason,
        )
        return tool_state_update(
            self.tool_call_id,
            f"Recorded skip for property {self.property_title}.",
            skipped=[skip],
        )


@tool_display(
    lambda p: f"Un-skipping property `{p.get('property_title', '?')}`",
    suppress_ack("Unskip result", ("Removed skip",)),
)
class _UnskipSchema(
    WithInjectedId,
    # deps: the batch's property titles
    WithAsyncDependencies[Command, list[str]],
):
    """
    Remove a previously declared skip for a property. Use this if you later
    find a way to formalize a property you previously skipped.
    """
    property_title: str = Field(
        description="The snake_case title of the property to un-skip"
    )

    @override
    async def run(self) -> Command:
        with self.tool_deps() as titles:
            if self.property_title not in titles:
                return tool_state_update(
                    self.tool_call_id,
                    f"Unknown property title {self.property_title!r}. Must be one "
                    f"of: {', '.join(titles)}.",
                )
        # Sentinel reason "" — _merge_skips drops empty-reason entries.
        skip = SkippedProperty(property_title=self.property_title, reason="")
        return tool_state_update(
            self.tool_call_id,
            f"Removed skip for property {self.property_title}.",
            skipped=[skip],
        )


@tool_display(lambda p: f"Expecting test `{p['test_name']}` to fail", None)
class ExpectTestFailure(WithAsyncImplementation[Command], WithInjectedId):
    """
    Mark a foundry test as expected to fail.

    The ``forge_test`` runner excludes expected-fail tests from the
    all-green check, so a failing test marked here will not block the
    publish gate. Use only when the failure is the *demonstration* of
    a property (e.g., a regression test that proves a negation).
    """
    test_name: str = Field(
        description="The name of the test function (e.g., `test_RevertWhen_Unauthorized`)"
    )
    reason: str = Field(description="Why this test is expected to fail")

    @override
    async def run(self) -> Command:
        # The merge treats an empty reason as "remove the marking" (see
        # _merge_expected_failures), so an empty reason must not get through.
        if not self.reason.strip():
            return tool_state_update(
                self.tool_call_id,
                "A non-empty reason is required when marking a test as expected to fail.",
            )
        return tool_state_update(
            tool_call_id=self.tool_call_id,
            content="Success",
            expected_failures={self.test_name: self.reason},
        )


@tool_display(lambda p: f"Expecting test `{p['test_name']}` to pass", None)
class ExpectTestPassage(WithAsyncImplementation[Command], WithInjectedId):
    """
    Unmark a test previously marked expected-to-fail.

    By default every test is expected to pass; only call this to revert a
    prior ``expect_test_failure``.
    """
    test_name: str = Field(
        description="The name of the test function previously marked expected-to-fail"
    )

    @override
    async def run(self) -> Command:
        # Empty reason = remove the marking (see _merge_expected_failures).
        return tool_state_update(
            tool_call_id=self.tool_call_id,
            content="Success",
            expected_failures={self.test_name: ""},
        )


# ---------------------------------------------------------------------------
# Feedback judge
# ---------------------------------------------------------------------------


class Rebuttal(RebuttalBase):
    """A rebuttal to a specific piece of feedback from a prior round, backed
    by evidence.

    File a rebuttal when a prior-round suggestion was tried and provably does
    not work — the suggested construction does not compile, the suggested
    test demonstrably does not exercise the property, etc. Do NOT file
    rebuttals for feedback you merely disagree with; address those by
    revising the tests.
    """
    evidence_type: Literal[
        "compilation_failure",
        "test_run_output",
        "execution_trace",
        "manual_citation",
        "reasoned",
    ] = Field(
        description=(
            "What backs this rebuttal. Use 'compilation_failure' for forge/solc "
            "build errors hit when trying the suggestion, 'test_run_output' for "
            "forge test output demonstrating the suggestion's actual behavior, "
            "'execution_trace' for a trace showing what a run actually did, "
            "'manual_citation' for a foundry / forge-std documentation citation, "
            "and 'reasoned' for an argument not backed by tool output or "
            "documentation."
        )
    )


type _FeedbackImplThunk = Callable[
    [str, list[SkippedProperty], list[Rebuttal], str],
    Awaitable[PropertyFeedbackProtocol],
]
"""``(test_source, skipped, rebuttals, within_tool) -> PropertyFeedback``.
``within_tool`` is the calling ``FeedbackTool``'s ``tool_call_id``, plumbed
through to the judge's ``run_to_completion`` so its UI panel anchors under
the parent tool widget."""


@dataclass
class FeedbackDependencies:
    thunk: _FeedbackImplThunk
    stamper: Callable[[FoundryGenerationState], dict[str, str]]


@tool_display("Getting feedback", "Feedback")
class FeedbackTool(
    WithInjectedId, WithInjectedState[FoundryGenerationState],
    WithAsyncDependencies[Command | str, FeedbackDependencies],
):
    """
    Receive feedback on your foundry tests and any skip declarations.
    The judge will evaluate whether the tests meaningfully demonstrate the
    batch's properties, coverage (all properties accounted for), and the
    validity of any skip justifications.

    If a prior-round suggestion from the judge was tried and provably does not
    work, file it in `rebuttals` with concrete evidence (compile error text,
    test run output, an execution trace, a documentation citation). Do NOT
    file rebuttals for feedback you merely disagree with — address those by
    revising the tests. An empty rebuttal list is the expected default; only
    populate it when you have ground-truth evidence against a prior point.
    """
    rebuttals: list[Rebuttal] = Field(
        default_factory=list,
        description=(
            "Optional rebuttals to specific pieces of prior-round feedback. Each "
            "entry identifies the prior point being rebutted, classifies the "
            "evidence (`compilation_failure` / `test_run_output` / "
            "`execution_trace` / `manual_citation` / `reasoned`), and supplies "
            "the concrete evidence text. Empirical types outweigh reasoned ones "
            "with the judge. Leave empty if you have nothing to rebut."
        ),
    )

    @override
    async def run(self) -> Command | str:
        if self.state["curr_test"] is None:
            return "No test written"
        with self.tool_deps() as deps:
            res = await deps.thunk(
                self.state["curr_test"],
                self.state["skipped"],
                self.rebuttals,
                self.tool_call_id,
            )
            result = f"Good? {res.good}\nFeedback:\n{res.feedback}"
            if res.good:
                return tool_state_update(
                    content=result,
                    tool_call_id=self.tool_call_id,
                    validations=deps.stamper(self.state),
                )
            return result


def _build_feedback_thunk(
    judge_ctx: WorkflowContext[CVLJudge],
    env: ServiceHost,
    props: list[PropertyFormulation],
    component: ContractComponentInstance | None,
) -> _FeedbackImplThunk:
    """Compile the feedback-judge graph and wrap it in the thunk
    ``FeedbackTool`` invokes. The judge follows the CVL property judge's
    review protocol (rough draft + persistent memory + read-back of the
    artifact under review) with the foundry tool surface: project source
    tools + the cheatcode RAG."""

    class JudgeExtra(RoughDraftState):
        curr_test: str

    class ST(MessagesState, JudgeExtra):
        result: NotRequired[PropertyFeedback]

    class TestJudgeInput(FlowInput, JudgeExtra):
        pass

    def did_rough_draft_read(s: ST, _) -> str | None:
        if not s["did_read"]:
            return "Completion REJECTED: never read rough draft for review"
        return None

    workflow = bind_standard(
        env.builder_heavy().with_tools(env.source_tools).with_tools(env.rag_tools),
        ST,
        validator=did_rough_draft_read,
    ).with_input(
        TestJudgeInput
    ).with_initial_prompt_template(
        "foundry_feedback_prompt.j2", properties=props, context=component,
    ).with_sys_prompt_template(
        "foundry_property_judge_system_prompt.j2"
    ).with_tools(
        [*get_rough_draft_tools(ST), judge_ctx.get_memory_tool(), GetTestTool.as_tool("get_test")]
    ).compile_async()

    async def thunk(
        test_source: str,
        skipped: list[SkippedProperty],
        rebuttals: list[Rebuttal],
        within_tool: str,
    ) -> PropertyFeedbackProtocol:
        input_parts: list[str | dict] = [
            "The proposed foundry test file is",
            test_source,
        ]
        if skipped:
            input_parts.append("The following properties were explicitly skipped by the author:")
            for s in skipped:
                input_parts.append(f"  Property {s.property_title}: {s.reason}")
        if rebuttals:
            input_parts.append(
                "The author has filed the following rebuttals against feedback "
                "from prior rounds. Evaluate each per the rebuttal rules in your "
                "instructions. Empirical evidence types (`compilation_failure`, "
                "`test_run_output`, `execution_trace`, `manual_citation`) carry "
                "near-binding weight; `reasoned` rebuttals are a conversation, "
                "not a veto."
            )
            for i, r in enumerate(rebuttals, 1):
                input_parts.append(
                    f"  Rebuttal {i} [{r.evidence_type}]\n"
                    f"    Addressing: {r.prior_feedback_reference}\n"
                    f"    Evidence: {r.evidence}"
                )
        res = await run_to_completion(
            workflow,
            TestJudgeInput(
                input=input_parts, curr_test=test_source,
                memory=None, did_read=False,
            ),
            thread_id=uniq_thread_id("foundry-feedback"),
            recursion_limit=judge_ctx.recursion_limit,
            description="Foundry test feedback judge",
            within_tool=within_tool,
        )
        assert "result" in res
        return res["result"]

    return thunk


# ---------------------------------------------------------------------------
# Publish / give-up tools
# ---------------------------------------------------------------------------


@tool_display(label="Publishing foundry test", result=None)
class PublishResultTool(
    WithInjectedState[FoundryGenerationState],
    WithInjectedId,
    # deps: the batch's property titles
    WithAsyncDependencies[Command | str, list[str]],
):
    """
    Call to signal completion. The publish is gated on the required
    validations: ``forge_test`` must have reported a clean unseeded run AFTER
    your latest ``put_test_raw``, and the feedback judge must have accepted
    the current draft.

    ``property_tests`` is checked against the tests forge actually ran:
    every non-skipped property from the batch must be demonstrated by at
    least one test that ran, and every test that ran must be tied back to
    one of the batch's properties.
    """
    commentary: str = Field(
        description="Human-readable commentary on the generated test file"
    )
    property_tests: list[PropertyTestMapping] = Field(
        description=(
            "The property→tests mapping. For every property you did NOT skip "
            "(referenced by its unique snake_case title from the batch listing), "
            "list the name(s) of the foundry test function(s) in your draft "
            "that demonstrate it (e.g., ``test_RevertWhen_Unauthorized``)."
        )
    )

    @override
    async def run(self) -> Command | str:
        if (err := check_foundry_completion(self.state)) is not None:
            return err
        ran = self.state["last_test_names"]
        if ran is None:
            # Unreachable in practice — the forge_test stamp required above
            # implies a run recorded its test names — but defend anyway.
            return "Completion REJECTED: no forge_test run has been recorded."
        with self.tool_deps() as titles:
            err = validate_property_tests(
                self.property_tests, self.state["skipped"], titles, ran,
            )
        if err is not None:
            return err
        return tool_state_update(
            self.tool_call_id,
            "Accepted",
            result=self.commentary,
            property_tests=self.property_tests,
            failed=False,
        )


@tool_display(
    label=lambda p: f"Giving up on foundry-test generation: {p['reason']}",
    result=None,
)
class GiveUpTool(WithImplementation[Command], WithInjectedId):
    """
    Last-resort exit when you've exhausted other mechanisms to complete
    the task. The batch will be reported as failed with your ``reason``.
    """
    reason: str = Field(description="Why you are giving up on this batch")

    @override
    def run(self) -> Command:
        return tool_state_update(
            self.tool_call_id,
            "Accepted",
            failed=True,
            result=self.reason,
        )


# ---------------------------------------------------------------------------
# Summary config (context compaction)
# ---------------------------------------------------------------------------


class FoundryGenerationSummaryConfig(SummaryConfig[FoundryGenerationState]):
    """Summarization prompts for the foundry author when the context window
    fills up. Same role as ``PropertyGenerationConfig`` in the CVL author,
    reworded for the foundry workflow (``curr_test`` not ``curr_spec``)."""

    @override
    def get_summarization_prompt(self, state: FoundryGenerationState) -> str:
        return """
You are approaching the context limit for your task. After this point your
context will be cleared and the task restarted from the initial prompt.

To enable you to continue effectively, summarize the current state of your
task. In particular, summarize:
1. The current state of your test draft (high-level structure, which
   properties you have formalized, which you have skipped and why).
2. Which test functions verify which properties (the property→test mapping
   you intend to declare at publish).
3. Any tests you have marked as expected-to-fail and why.
4. Any unresolved feedback — from the last ``forge_test`` run (compile
   errors, failing tests, etc.) or from the feedback judge — that you still
   need to address.
5. Foundry cheatcode patterns / idioms you discovered during this batch
   so the next iteration does not re-research them.

If your current task itself began with a summary, include the salient parts
of that summary in your new summary.
"""

    @override
    def get_resume_prompt(self, state: FoundryGenerationState, summary: str) -> str:
        return f"""
You are resuming this task already in progress. The current version of your
test draft (if any) is available via the ``get_test`` tool.

A summary of your work up to this point:

BEGIN SUMMARY:
{summary}
END SUMMARY

**IMPORTANT**: Nothing has changed since the summary was produced. You do
NOT need to re-research foundry cheatcode patterns already captured in the
summary. If you have outstanding ``forge_test`` failures or judge feedback to
address, proceed directly with addressing them.
"""


# ---------------------------------------------------------------------------
# Top-level batch entry
# ---------------------------------------------------------------------------


class FoundryPropertyGenParams(TypedDict):
    """Per-batch render variables for ``foundry_property_generation_prompt.j2``.
    Mirror of ``PropertyGenParams`` in the CVL author, minus ``resources``
    (no CVL-importable resource concept here)."""
    context: ContractComponentInstance | None
    properties: list[PropertyFormulation]
    contract_name: str


_FoundryPropertyGenTemplate = TypedTemplate[FoundryPropertyGenParams](
    "foundry_property_generation_prompt.j2"
)


async def batch_foundry_test_generation(
    ctx: WorkflowContext[CVLGeneration],
    *,
    project_root: str,
    contract_name: str,
    props: list[PropertyFormulation],
    component: ContractComponentInstance | None,
    env: ServiceHost,
    description: str,
    forge_binary: str = "forge",
    forge_timeout_s: int = 600,
    forge_sem : asyncio.Semaphore
) -> BatchFoundryResult:
    """Author one batch of foundry tests covering ``props``.

    The graph terminates when the agent calls ``result`` (publish) or
    ``give_up``. Both ``forge_test`` and the feedback judge must have stamped
    the *current* ``curr_test`` for ``result`` to be accepted.

    Caller responsibilities:

    * ``project_root`` is a fully-configured foundry project (has
      ``foundry.toml`` and any required deps under ``lib/``). The author
      stages its draft into ``<project_root>/test/`` and deletes the
      staged file after each ``forge test`` invocation.
    * ``env`` carries the foundry RAG (``rag_tools``) + project source
      tools (``source_tools``). Typically built via
      ``composer.foundry.env.build_foundry_env``.
    * ``contract_name`` / ``component`` / ``props`` are bound into the
      initial prompt (``foundry_property_generation_prompt.j2``).

    Reuses ``CVLGeneration`` as the cache marker for the workflow context
    purely because that's what the surrounding ``WorkflowContext`` API
    expects; the cache namespacing should differ between CVL and foundry
    runs at the caller level.
    """
    forge_test_tool = get_forge_test_tool(
        project_root, forge_binary=forge_binary, timeout_s=forge_timeout_s, forge_sem=forge_sem
    )

    bound_template = _FoundryPropertyGenTemplate.bind({
        "context": component,
        "properties": props,
        "contract_name": contract_name,
    })

    titles = [p.title for p in props]
    judge_ctx = ctx.child(FOUNDRY_JUDGE_KEY)
    feedback_deps = FeedbackDependencies(
        thunk=_build_feedback_thunk(judge_ctx, env, props, component),
        stamper=make_foundry_validation_stamper(FEEDBACK),
    )

    builder = (
        env.builder_heavy()
        .with_state(FoundryGenerationState)
        .with_input(FoundryGenerationInput)
        .with_output_key("result")
        .with_tools(env.source_tools)
        .with_tools(env.rag_tools)
        .with_tools([
            PutTestRaw.as_tool("put_test_raw"),
            GetTestTool.as_tool("get_test"),
            _RecordSkipSchema.bind(titles).as_tool("record_skip"),
            _UnskipSchema.bind(titles).as_tool("unskip_property"),
            ExpectTestFailure.as_tool("expect_test_failure"),
            ExpectTestPassage.as_tool("expect_test_passage"),
            forge_test_tool,
            FeedbackTool.bind(feedback_deps).as_tool("feedback_tool"),
            PublishResultTool.bind(titles).as_tool("result"),
            GiveUpTool.as_tool("give_up"),
            ctx.get_memory_tool(),
        ])
        .with_sys_prompt_template("foundry_property_generation_system_prompt.j2")
        .inject(lambda b: bound_template.render_to(b.with_initial_prompt_template))
        .with_summary_config(FoundryGenerationSummaryConfig())
    )
    graph = builder.compile_async()

    init_state = FoundryGenerationInput(
        curr_test=None,
        input=[],
        required_validations=[FORGE_TEST_VALIDATION_KEY, FEEDBACK],
        skipped=[],
        property_tests=[],
        validations={},
        expected_failures={},
        last_test_names=None,
        failed=None,
    )

    tid, mnem = await ctx.thread_and_mnemonic()
    res_state = await run_to_completion(
        graph,
        init_state,
        thread_id=tid,
        description=f"{description} ({mnem})",
        recursion_limit=ctx.recursion_limit,
    )

    assert "result" in res_state
    assert res_state["failed"] is not None
    if res_state["failed"]:
        return GaveUp(reason=res_state["result"])
    draft = res_state["curr_test"]
    assert draft is not None
    return GeneratedFoundryTest(
        commentary=res_state["result"],
        test_source=draft,
        skipped=res_state["skipped"],
        property_tests=res_state["property_tests"],
        expected_failures=res_state["expected_failures"],
        ran_tests=res_state["last_test_names"] or [],
    )
