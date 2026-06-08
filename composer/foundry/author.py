"""Foundry test author — generates ``.t.sol`` tests for property formulations.

Direct analog of ``composer/spec/source/author.py`` for the foundry backend:

* Single ``curr_test: str`` buffer per batch (one ``.t.sol`` file).
* No feedback judge — the only publish gate is a green ``forge test`` run.
* No CVL research sub-agent — the foundry cheatcode RAG tools are
  installed directly so the agent can look them up itself.
* No prover-config editor — foundry projects are assumed pre-configured.
* Per-test expected-failure marking via ``expect_test_failure`` (mirror
  of ``ExpectRuleFailure``).
* Publish enforces a property→test-function mapping the same way CVL's
  ``PublishResultTool`` enforces property→rule mapping.

The high-level entry point is ``batch_foundry_test_generation``. It mirrors
``batch_cvl_generation``'s shape: takes a workflow context, a batch of
properties, a project root, and a builder/rag-tools env, and returns either
a ``GeneratedFoundryTest`` (commentary + source + skips + mapping) or a
``GaveUp``.
"""

from typing import Callable, Protocol, override

from langgraph.runtime import get_runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from graphcore.graph import Builder, tool_state_update
from graphcore.summary import SummaryConfig
from graphcore.tools.schemas import (
    WithImplementation, WithInjectedId, WithInjectedState,
)

from composer.spec.cvl_generation import (
    PropertyRuleMapping, SkippedProperty, validate_property_rules,
)
from composer.spec.graph_builder import run_to_completion
from composer.spec.prop import PropertyFormulation
from composer.spec.context import WorkflowContext, CVLGeneration
from composer.spec.tool_env import BasicAgentTools, RAGTools, SourceTools
from composer.ui.tool_display import tool_display

from composer.foundry.runner import get_forge_test_tool
from composer.foundry.state import (
    FORGE_TEST_VALIDATION_KEY,
    FoundryGenerationInput,
    FoundryGenerationState,
    check_foundry_completion,
)
from composer.foundry.tools import (
    FoundryGenerationContext, foundry_static_tools,
)


class _FoundryEnv(RAGTools, SourceTools, BasicAgentTools, Protocol):
    """Minimum the foundry author needs from the caller's env: a builder
    (BasicAgentTools), foundry RAG (RAGTools), and source-tree exploration
    tools (SourceTools — fs + optionally a code-explorer sub-agent). The
    caller is expected to wire foundry cheatcode tools into ``rag_tools``
    and project-source tools into ``source_tools`` when constructing the
    env (see ``composer.foundry.env.build_foundry_env`` for the standard
    wiring)."""
    ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class GeneratedFoundryTest(BaseModel):
    """Successful output of the foundry author for a batch."""
    commentary: str
    test_source: str
    skipped: list[SkippedProperty] = Field(default_factory=list)
    property_tests: list[PropertyRuleMapping] = Field(default_factory=list)


class GaveUp(BaseModel):
    reason: str


type BatchFoundryResult = GeneratedFoundryTest | GaveUp


# ---------------------------------------------------------------------------
# Publish / give-up tools
# ---------------------------------------------------------------------------


@tool_display(label="Publishing foundry test", result=None)
class PublishResultTool(
    WithImplementation[Command | str],
    WithInjectedState[FoundryGenerationState],
    WithInjectedId,
):
    """
    Call to signal completion. The publish is gated on the ``forge_test``
    validation: ``forge_test`` must have been run AFTER your latest
    ``put_test_raw`` and reported a clean run (excluding any tests marked
    via ``expect_test_failure``).

    Every non-skipped property from the batch must be claimed by at least
    one test function in the spec via ``property_tests``.
    """
    commentary: str = Field(
        description="Human-readable commentary on the generated test file"
    )
    property_tests: list[PropertyRuleMapping] = Field(
        description=(
            "The property→tests mapping. For every property you did NOT skip "
            "(referenced by its unique snake_case title from the batch listing), "
            "list the name(s) of the foundry test function(s) in your draft "
            "that verify it (e.g., ``test_RevertWhen_Unauthorized``). Every "
            "non-skipped property must appear with at least one test name."
        )
    )

    @override
    def run(self) -> Command | str:
        if (err := check_foundry_completion(self.state)) is not None:
            return err
        titles = get_runtime(FoundryGenerationContext).context.titles
        # `validate_property_rules` is workflow-agnostic: it checks coverage
        # against (titles, skipped) and the mapping format. The "rules"
        # field on PropertyRuleMapping holds test_ function names here.
        err = validate_property_rules(
            self.property_tests, self.state["skipped"], titles,
        )
        if err is not None:
            return err
        return tool_state_update(
            self.tool_call_id,
            "Accepted",
            result=self.commentary,
            property_rules=self.property_tests,
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
    fills up. Same role as ``PropertyGenerationConfig`` in the CVL author
    but reworded for the foundry workflow (no judge, ``curr_test`` not
    ``curr_spec``)."""

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
4. Any unresolved feedback from the last ``forge_test`` run (compile
   errors, failing tests, etc.) that you still need to address.
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
summary. If you have outstanding ``forge_test`` failures to address, proceed
directly with addressing them.
"""


# ---------------------------------------------------------------------------
# Top-level batch entry
# ---------------------------------------------------------------------------


async def batch_foundry_test_generation(
    ctx: WorkflowContext[CVLGeneration],
    *,
    project_root: str,
    props: list[PropertyFormulation],
    env: _FoundryEnv,
    description: str,
    inject_initial_prompt: Callable[
        [Builder[FoundryGenerationState, FoundryGenerationContext, FoundryGenerationInput]],
        Builder[FoundryGenerationState, FoundryGenerationContext, FoundryGenerationInput],
    ],
    system_prompt_template: str = "foundry_property_generation_system_prompt.j2",
    forge_binary: str = "forge",
    forge_timeout_s: int = 600,
) -> BatchFoundryResult:
    """Author one batch of foundry tests covering ``props``.

    The graph terminates when the agent calls ``result`` (publish) or
    ``give_up``. ``forge_test`` must have produced a green stamp on the
    *current* ``curr_test`` for ``result`` to be accepted.

    Caller responsibilities:

    * ``project_root`` is a fully-configured foundry project (has
      ``foundry.toml`` and any required deps under ``lib/``). The author
      stages its draft into ``<project_root>/test/`` and deletes the
      staged file after each ``forge test`` invocation.
    * ``env`` is a ``ToolEnvironment``-ish object whose ``rag_tools`` slot
      has been populated with the foundry cheatcode tools (typically via
      ``composer.tools.foundry_rag.get_tools``). The caller chooses what
      RAG surface to bind — the author makes no assumption beyond
      ``rag_tools + builder``.
    * ``inject_initial_prompt`` is a builder-mutation that installs the
      per-batch initial prompt (template + binding). Mirrors the pattern
      used by ``batch_cvl_generation``'s ``_PropertyGenTemplate.bind(...)
      .render_to(d.with_initial_prompt_template)`` but pushed out to the
      caller since the foundry prompt template is workflow-specific.
    * ``system_prompt_template`` is the system-prompt jinja template
      installed via ``with_sys_prompt_template``. Defaults to
      ``foundry_property_generation_system_prompt.j2`` — the caller
      must ensure that template exists in the loader's search path.

    Reuses ``CVLGeneration`` as the cache marker for the workflow context
    purely because that's what the surrounding ``WorkflowContext`` API
    expects; the cache namespacing should differ between CVL and foundry
    runs at the caller level.
    """
    forge_test_tool = get_forge_test_tool(
        project_root, forge_binary=forge_binary, timeout_s=forge_timeout_s,
    )

    builder = (
        env.builder
        .with_state(FoundryGenerationState)
        .with_input(FoundryGenerationInput)
        .with_context(FoundryGenerationContext)
        .with_output_key("result")
        .with_tools(env.source_tools)
        .with_tools(env.rag_tools)
        .with_tools(foundry_static_tools())
        .with_tools([
            forge_test_tool,
            PublishResultTool.as_tool("result"),
            GiveUpTool.as_tool("give_up"),
            ctx.get_memory_tool(),
        ])
        .with_sys_prompt_template(system_prompt_template)
        .inject(inject_initial_prompt)
        .with_summary_config(FoundryGenerationSummaryConfig())
    )
    graph = builder.compile_async()

    init_state = FoundryGenerationInput(
        curr_test=None,
        input=[],
        required_validations=[FORGE_TEST_VALIDATION_KEY],
        skipped=[],
        property_rules=[],
        validations={},
        expected_failures={},
        failed=None,
    )

    titles = [p.title for p in props]
    tid, mnem = await ctx.thread_and_mnemonic()
    res_state = await run_to_completion(
        graph,
        init_state,
        thread_id=tid,
        context=FoundryGenerationContext(titles=titles),
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
        property_tests=res_state["property_rules"],
    )
