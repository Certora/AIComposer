"""
Fake-LLM end-to-end UI harness for ``tui_autoprove.py`` (auto-prove
multi-agent pipeline).

Substitutes the real ``ChatAnthropic`` built by
``composer.workflow.services.create_llm`` / ``create_llm_base`` with a
``FakeMessagesListChatModel`` preloaded with a hand-authored tape of
responses. Every other part of the pipeline runs normally — ``AutoProveApp``
TUI, real tool execution (solc, Typechecker.jar, certoraTypeCheck.py,
the real Certora prover, PreAudit subprocess), workflow graphs,
checkpointing, Postgres-backed store/checkpointer, RAG.

Scenario inputs and wiring instructions live under
``composer/testing/scenarios/autoprove_counter/``.

The scenario is deliberately constrained to one contract with one component
so that the per-component ``asyncio.gather`` fan-outs in phases 5 and 6 of
``run_generation_pipeline`` collapse to linear execution. Multiple
invariants and multiple properties are still authored per-phase — a single
authoring agent services them sequentially, so the tape remains linear.

There is no HITL in auto-prove (``AutoProveTaskHandler.format_hitl_prompt``
raises ``NotImplementedError``). The tape is pure tool_calls + plain-text
AIMessages.

Global call order across phases:

    run_component_analysis
      → run_setup / classifier_agent
      → get_invariant_formulation
          └─ invariant_feedback sub-agent ×3
      → batch_cvl_generation (invariant CVL)
          ├─ cvl_research sub-agent
          ├─ explore_code sub-agent
          ├─ feedback-judge sub-agent ×2
          └─ CEX analyzer (1 LLM call for the violated rule)
      → run_bug_analysis
      → batch_cvl_generation (component CVL, streamlined)
          └─ feedback-judge sub-agent ×1
"""

from typing import Any, Callable, Sequence, override
import uuid

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Fake LLM plumbing
# ---------------------------------------------------------------------------


class _AutoProveFakeLLM(FakeMessagesListChatModel):
    """``FakeMessagesListChatModel`` tolerant of attribute access the
    auto-prove pipeline performs on the bound LLM.

    Mirrors ``_CodegenFakeLLM`` / ``_NatspecFakeLLM``:

    * ``thinking`` — declared as a field so ``llm.model_copy(update={"thinking": ...})``
      (used by the CEX summarizer path, if ever triggered) is a well-formed
      no-op.
    * ``betas`` — empty so no ``context-management-2025-06-27`` beta branches
      attach extra tools.
    * ``bind_tools`` — no-op so Builders can attach tool schemas without the
      fake raising ``NotImplementedError``.
    """

    thinking: Any = None
    betas: list[str] = []

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return self


def _tc(name: str, **args: Any) -> ToolCall:
    """Tool-call dict with a unique ``id`` (LangGraph binds tool responses back
    to calls by id, so every entry needs its own)."""
    return {
        "id": f"toolu_{uuid.uuid4().hex[:20]}",
        "name": name,
        "args": args,
        "type": "tool_call",
    }


def _ai(text: str = "", *tool_calls: ToolCall) -> AIMessage:
    """Build a tape entry: optional text + zero or more tool_calls. LangGraph's
    agent loop transitions to the tools node when ``tool_calls`` is non-empty,
    and to END (returning to output_key extraction) otherwise."""
    content: list[str | dict] = []
    if text:
        content.append(text)
    content.extend(
        {"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["args"]}
        for t in tool_calls
    )
    return AIMessage(content=content, tool_calls=list(tool_calls))


# ---------------------------------------------------------------------------
# Scenario artifacts (Solidity + CVL)
# ---------------------------------------------------------------------------
#
# The Solidity source is staged on disk in
# ``composer/testing/scenarios/autoprove_counter/src/Counter.sol``. These CVL
# strings are emitted as ``put_cvl_raw`` arguments during the invariant-CVL
# and component-CVL phases. Real tools validate them:
#
#   - Typechecker.jar  — gatekeeps ``put_cvl_raw`` (rejects parse errors).
#   - Certora prover   — gatekeeps ``verify_spec`` (proves or CEXes).


# Intentionally malformed surface-syntax CVL. Triggers the Typechecker.jar
# rejection path on the first ``put_cvl_raw`` of the invariant-CVL phase;
# the tape's next turn resubmits valid CVL.
BROKEN_PARSE_CVL = """\
invariant not_valid_cvl()
    this is definitely not valid CVL syntax;
"""

# Typechecks but the invariant is obviously false: after ``increment()`` runs,
# ``count`` is 1, so ``count == 0`` no longer holds. Used as the first
# (easy-to-catch) semantic-error candidate — the feedback judge rejects this
# on first pass without involving the prover at all.
BAD_INV_CVL = """\
invariant count_zero()
    currentContract.count == 0;
"""

# Typechecks and declares the two ostensibly-correct invariant names, but the
# body of ``count_nonneg`` is subtly wrong (``count > 0`` instead of ``>= 0``).
# The feedback judge approves by name-coverage; the prover catches it on the
# base case (initial state has ``count == 0``, violating ``count > 0``).
# This is the artifact that drives the verify_spec → analyze_cex_raw round-trip
# in the tape — exactly one failing rule (``count_nonneg``), so exactly one
# CEX LLM call is consumed.
SUBTLE_INV_CVL = """\
invariant count_nonneg()
    currentContract.count > 0;

invariant increments_nonneg(address a)
    currentContract.increments[a] >= 0;
"""

# Two trivially-true invariants over the Counter state. Both should verify
# against Counter.sol on first try, so verify_spec stamps the prover digest
# and the author can call `result` to terminate the invariant-CVL author graph.
GOOD_INV_CVL = """\
invariant count_nonneg()
    currentContract.count >= 0;

invariant increments_nonneg(address a)
    currentContract.increments[a] >= 0;
"""

# Component-CVL spec: two rules, one per property from bug analysis. Both
# hold against Counter.increment() on first try so the prover stamps the
# digest without a CEX detour.
COMPONENT_CVL = """\
methods {
    function count() external returns (uint256) envfree;
    function increments(address) external returns (uint256) envfree;
    function increment() external;
}

rule increment_increases_count {
    env e;
    mathint before = count();
    increment(e);
    assert to_mathint(count()) == before + 1,
        "increment() must increase count by exactly 1";
}

rule increment_increases_sender_tally {
    env e;
    address s = e.msg.sender;
    mathint before = increments(s);
    increment(e);
    assert to_mathint(increments(s)) == before + 1,
        "increment() must increase increments[msg.sender] by exactly 1";
}
"""


# ---------------------------------------------------------------------------
# SourceApplication payload — emitted by the component-analysis result tool
# ---------------------------------------------------------------------------
#
# Shape must satisfy pydantic validation of
# ``composer.spec.system_model.SourceApplication`` AND the
# ``_validate_connectivity`` validator: unique names + all referenced
# components / external actors exist. One SourceExplicitContract ("Counter")
# with one ContractComponent ("Increment"), no interactions, no external
# actors — minimal valid shape.

_APP_RESULT = {
    "application_type": "Counter",
    "description": (
        "A minimal singleton Counter application that maintains a global "
        "count and a per-caller tally of invocations via a single external "
        "entry point."
    ),
    "components": [
        {
            "sort": "singleton",
            "name": "Counter",
            "path": "src/Counter.sol",
            "description": (
                "The only contract in the system; owns the count and per-"
                "caller tally state and the increment entry point."
            ),
            "components": [
                {
                    "name": "Increment",
                    "description": (
                        "Handles all count updates through the single "
                        "``increment()`` external entry point."
                    ),
                    "external_entry_points": ["increment()"],
                    "state_variables": [
                        "uint256 count",
                        "mapping(address => uint256) increments",
                    ],
                    "interactions": [],
                    "requirements": [
                        "Each call to increment() increases count by exactly 1.",
                        "Each call to increment() increases increments[msg.sender] by exactly 1.",
                        "increment() must not revert under normal operation.",
                    ],
                }
            ],
        }
    ],
}


# ---------------------------------------------------------------------------
# AgentSystemDescription payload — emitted by the classifier-agent result tool
# ---------------------------------------------------------------------------
#
# Shape must satisfy pydantic validation of
# ``composer.spec.source.harness.AgentSystemDescription`` AND the
# ``classifier_agent`` validator: every ``transitive_closure[*].name`` must
# map to a known SourceExplicitContract, every ``external_interfaces[*].name``
# must map to a known SourceExternalActor (with a path).
#
# We use ``num_instances=None`` so ``needs_harnessing()`` returns False and
# the harness-generation sub-agent is skipped. ``erc20_contracts=[]`` and
# ``external_interfaces=[]`` so the summaries sub-agent is skipped.

_CLASSIFIER_RESULT = {
    "non_trivial_state": (
        "A non-trivial state has been reached once at least one call to "
        "increment() has executed: count > 0 and increments[msg.sender] > 0 "
        "for that sender."
    ),
    "transitive_closure": [
        {
            "name": "Counter",
            "link_fields": [],
            "num_instances": None,
        }
    ],
    "erc20_contracts": [],
    "external_interfaces": [],
}


# ---------------------------------------------------------------------------
# PropertyFormulation payloads — emitted by the bug-analysis result tool
# ---------------------------------------------------------------------------
#
# The bug-analysis agent's result schema is ``list[PropertyFormulation]``
# wrapped via the ``(type, doc)`` overload of ``result_tool_generator``, so
# the tool args are ``{"value": [...]}``.

_BUG_ANALYSIS_PROPS = [
    {
        "methods": ["increment()"],
        "sort": "safety_property",
        "description": (
            "After calling increment(), the global count must be exactly "
            "one greater than before the call."
        ),
    },
    {
        "methods": ["increment()"],
        "sort": "safety_property",
        "description": (
            "After calling increment(), increments[msg.sender] must be "
            "exactly one greater than before the call."
        ),
    },
]


# ---------------------------------------------------------------------------
# The tape
# ---------------------------------------------------------------------------
#
# Global call order (section headers mark boundaries, NOT separate tapes).
# Every AIMessage below is popped by ``FakeMessagesListChatModel`` on a
# single LLM call, in order. If the real pipeline issues a call not shown
# here, the fake runs off the end and raises. If real dispatch order drifts
# from this layout, edit the tape — that's the cheap loop.

_AUTOPROVE_TAPE: list[BaseMessage] = [

    # ───────────────────────────────────────────────────────────────────
    # P1. Component analysis (run_component_analysis → SourceApplication)
    # ───────────────────────────────────────────────────────────────────
    # Tools available: memory, write_rough_draft, read_rough_draft,
    #   source_tools = list_files, get_file, grep_files, code_explorer,
    #                  code_document_ref.
    # Validator: _validate_connectivity (graph wellformedness only; no
    #   did_read requirement — we can hit `result` at any time once the
    #   application shape is correct).

    # P1.1 — exercise memory + list_files + get_file. Memory paths must sit
    # under /memories; `view /memories` is the harmless exercise.
    _ai(
        "Cataloguing memory and surveying the project layout.",
        _tc("memory", command="view", path="/memories"),
        _tc("list_files"),
        _tc("get_file", path="src/Counter.sol"),
    ),

    # P1.2 — exercise grep_files. Returns matches for `increment` in the
    # source; the agent uses the result to narrow understanding.
    _ai(
        "Grepping for the entry point symbol.",
        _tc(
            "grep_files",
            search_string="increment",
            matching_lines=False,
        ),
    ),

    # P1.3 — exercise code_explorer. This spawns the code-explorer sub-agent
    # (CE.1..CE.2 below). The indexed variant caches by normalized question
    # hash; subsequent code_explorer calls with the same question return
    # without an LLM call, so we only pay for it here. Tool is registered
    # as ``code_explorer`` by ``indexed_code_explorer_tool`` — note the
    # ``source_displays()`` mapping uses the stale key ``explore_code``,
    # but the tool itself is dispatched under ``code_explorer``.
    _ai(
        "Delegating a state-shape question to the code-explorer sub-agent.",
        _tc(
            "code_explorer",
            question=(
                "What storage state does the Counter contract maintain, and "
                "which function modifies it?"
            ),
        ),
    ),

    # CE.1 — code-explorer sub-agent turn 1. Tools: base_source_tools
    # (list_files, get_file, grep_files) + result. The sub-agent has no
    # memory/rough_draft tools (see composer/spec/code_explorer.py).
    _ai(
        "Explorer: inspecting Counter.sol.",
        _tc("get_file", path="src/Counter.sol"),
    ),

    # CE.2 — code-explorer result. Schema is (str, "Your findings about
    # the source code"), so args are {"value": "..."}.
    _ai(
        "Explorer: findings ready.",
        _tc(
            "result",
            value=(
                "Counter stores `uint256 public count` and "
                "`mapping(address => uint256) public increments`. Both are "
                "mutated by the single external entry point `increment()`, "
                "which adds 1 to `count` and 1 to `increments[msg.sender]`."
            ),
        ),
    ),

    # P1.4 — exercise rough_draft tools before result. No did_read validator
    # in this phase, so the order is just for coverage.
    _ai(
        "Drafting a one-paragraph summary for self-reference.",
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Counter is a singleton with one component (Increment). "
                "State: count (uint256) + increments (address→uint256). "
                "One entry: increment(). No external interactions."
            ),
        ),
    ),
    _ai(
        "Reading back the draft before emitting the application model.",
        _tc("read_rough_draft"),
    ),

    # P1.5 — emit the SourceApplication. Satisfies _validate_connectivity
    # (unique names, no dangling interaction references since there are no
    # interactions).
    _ai(
        "Application model ready.",
        _tc("result", **_APP_RESULT),
    ),

    # ───────────────────────────────────────────────────────────────────
    # P2. Classifier agent (run_setup → classifier_agent →
    #     AgentSystemDescription)
    # ───────────────────────────────────────────────────────────────────
    # Tools available: memory, source_tools, result.
    # Validator: every transitive_closure[*].name must be a known
    #   SourceExplicitContract and every external_interfaces[*].name must
    #   be a known SourceExternalActor with a non-None path. We return zero
    #   external interfaces and only "Counter" in the closure.
    #
    # After this result, `needs_harnessing()` returns False
    # (num_instances=None) so generate_harnesses is skipped. Empty erc20 +
    # empty external_interfaces means setup_summaries is skipped by
    # `run_autoprove_pipeline`.
    #
    # The preaudit subprocess runs between this phase and the invariant
    # phase — it's a real `python -m orchestrator` call and does not
    # consume LLM calls.

    # P2.1 — exercise list_files in this agent's thread (different from
    # the P1 thread, so the listing call re-runs against the real fs).
    _ai(
        "Classifier: surveying project contents before classifying.",
        _tc("list_files"),
    ),

    # P2.2 — emit the AgentSystemDescription. Empty external_interfaces +
    # empty erc20_contracts + num_instances=None short-circuits the next
    # two pipeline phases (harnessing + summaries).
    _ai(
        "Counter is standalone — no harnessing, no external summaries.",
        _tc("result", **_CLASSIFIER_RESULT),
    ),

    # ───────────────────────────────────────────────────────────────────
    # P3. Structural invariant formulation (get_invariant_formulation)
    # ───────────────────────────────────────────────────────────────────
    # Main-agent tools: memory, source_tools, invariant_feedback, result.
    # Feedback sub-agent tools: memory, rough_draft, source_tools, result
    #   (schema: InvariantFeedback{sort, explanation}).
    # Validator `_validate_invariants`: every inv in the final result must
    #   appear in state["invariant_data"] with (description, "GOOD") matching
    #   exactly. The state dict merges on name, so resubmitting the same
    #   name with a different description overwrites the prior entry.
    #
    # The tape uses 3 invariant_feedback rounds (1 bad + 2 good) to exercise
    # the NOT_INDUCTIVE → resubmit recovery path, and delivers 2 invariants
    # in the final result.

    # P3.1 — exercise source_tools in the main invariant agent.
    _ai(
        "Reading Counter.sol to understand the state shape.",
        _tc("get_file", path="src/Counter.sol"),
    ),

    # P3.2 — first invariant_feedback call: candidate "count_zero" (count is
    # always 0) — intentionally bad. This spawns F1.{1-3}.
    _ai(
        "Proposing count_zero as a structural candidate.",
        _tc(
            "invariant_feedback",
            inv={
                "name": "count_zero",
                "description": "The global count is always zero.",
            },
        ),
    ),

    # F1.1 — invariant feedback judge, first invocation, turn 1. Judge tools:
    # memory, rough_draft, source_tools, result. Validator on this sub-agent
    # is the standard `bind_standard` without custom checks — the only
    # implicit requirement is providing `result` to set output_key.
    _ai(
        "Judge: inspecting the source + drafting a verdict.",
        _tc("get_file", path="src/Counter.sol"),
        _tc(
            "write_rough_draft",
            rough_draft=(
                "count_zero claims count is always 0, but increment() "
                "mutates count upward. The post-state of any increment() "
                "call already violates this claim. Verdict: NOT_INDUCTIVE."
            ),
        ),
    ),

    # F1.2 — judge: read the draft before emitting result.
    _ai(
        "Judge: re-reading the draft.",
        _tc("read_rough_draft"),
    ),

    # F1.3 — judge: NOT_INDUCTIVE verdict. This stores
    # state["invariant_data"]["count_zero"] = ("The global count is always
    # zero.", "NOT_INDUCTIVE"). The main agent sees the ToolMessage and can
    # try a different candidate.
    _ai(
        "Judge: delivering NOT_INDUCTIVE verdict.",
        _tc(
            "result",
            sort="NOT_INDUCTIVE",
            explanation=(
                "The claim fails immediately after any call to increment(): "
                "count transitions from k to k+1 and the invariant does not "
                "hold in the post-state. Consider a non-negativity "
                "invariant (count >= 0) or a correlation between count and "
                "the increments mapping instead."
            ),
        ),
    ),

    # P3.3 — main agent resubmits with a stronger invariant name:
    # "count_nonneg" (trivially true on uint256). Spawns F2.{1-3}.
    _ai(
        "Addressing the feedback — proposing count_nonneg instead.",
        _tc(
            "invariant_feedback",
            inv={
                "name": "count_nonneg",
                "description": (
                    "The global count is always non-negative."
                ),
            },
        ),
    ),

    # F2.1 — judge, second invocation, turn 1.
    _ai(
        "Judge: evaluating count_nonneg.",
        _tc(
            "write_rough_draft",
            rough_draft=(
                "count is a uint256 — the type guarantees non-negativity. "
                "Trivial but formal and inductive. Verdict: GOOD."
            ),
        ),
    ),
    _ai(
        "Judge: reading the draft.",
        _tc("read_rough_draft"),
    ),
    # F2.3 — GOOD verdict. Stamps state["invariant_data"]["count_nonneg"].
    _ai(
        "Judge: GOOD verdict on count_nonneg.",
        _tc(
            "result",
            sort="GOOD",
            explanation=(
                "uint256 arithmetic guarantees the invariant holds "
                "trivially and inductively at every reachable state."
            ),
        ),
    ),

    # P3.4 — main agent proposes second invariant. Spawns F3.{1-3}.
    _ai(
        "Proposing the second invariant.",
        _tc(
            "invariant_feedback",
            inv={
                "name": "increments_nonneg",
                "description": (
                    "For every address a, increments[a] is always "
                    "non-negative."
                ),
            },
        ),
    ),

    # F3.1 — judge, third invocation.
    _ai(
        "Judge: evaluating increments_nonneg.",
        _tc(
            "write_rough_draft",
            rough_draft=(
                "increments is mapping(address => uint256). uint256 is "
                "non-negative by type. Same shape as count_nonneg — "
                "trivial but formal and inductive. Verdict: GOOD."
            ),
        ),
    ),
    _ai(
        "Judge: reading the draft.",
        _tc("read_rough_draft"),
    ),
    _ai(
        "Judge: GOOD verdict on increments_nonneg.",
        _tc(
            "result",
            sort="GOOD",
            explanation=(
                "uint256 mapping values are non-negative by type. The "
                "invariant is trivially and inductively true."
            ),
        ),
    ),

    # P3.5 — main agent delivers both invariants. Descriptions must match
    # the ones in state["invariant_data"] verbatim (merged on name).
    _ai(
        "Delivering the validated invariants.",
        _tc(
            "result",
            inv=[
                {
                    "name": "count_nonneg",
                    "description": (
                        "The global count is always non-negative."
                    ),
                },
                {
                    "name": "increments_nonneg",
                    "description": (
                        "For every address a, increments[a] is always "
                        "non-negative."
                    ),
                },
            ],
        ),
    ),

    # ───────────────────────────────────────────────────────────────────
    # P4. Invariant CVL generation (batch_cvl_generation, component=None)
    # ───────────────────────────────────────────────────────────────────
    # Author-agent tools:
    #   - cvl_authorship_tools (source_tools + rag_tools): list_files,
    #     get_file, grep_files, code_explorer, code_document_ref,
    #     cvl_manual_search, cvl_keyword_search, get_cvl_manual_section,
    #     scan_knowledge_base, get_knowledge_base_article, cvl_research,
    #     cvl_document_ref.
    #   - static_tools: put_cvl, put_cvl_raw, feedback_tool, record_skip,
    #     unskip_property, get_cvl, erc20_guidance, unresolved_call_guidance.
    #   - prover_tool: verify_spec.
    #   - ExpectRuleFailure.as_tool("expect_rule_failure"),
    #     ExpectRulePassage.as_tool("expect_rule_passage").
    #   - result (str commentary), memory.
    #
    # Result digest: validations[feedback] AND validations[prover] must
    # both equal digest(curr_spec, skipped) before `result` is accepted.
    # feedback_tool (good=True) stamps feedback; verify_spec (rules=None,
    # all_verified) stamps prover. Any put_cvl_raw / record_skip /
    # unskip_property invalidates both stamps.
    #
    # num_props=2 (2 invariants) — record_skip / unskip_property accept
    # property_index in {1, 2}.

    # Q1 — exercise the similarity + keyword search paths.
    _ai(
        "Surveying the CVL manual for invariant patterns.",
        _tc(
            "cvl_manual_search",
            question=(
                "What is the syntax for declaring a parametric invariant "
                "in CVL?"
            ),
            similarity_cutoff=0.5,
            max_results=5,
            manual_section=[],
        ),
        _tc("cvl_keyword_search", query="invariant", min_depth=0, limit=5),
    ),

    # Q2 — exercise section retrieval + knowledge-base scan.
    _ai(
        "Fetching the Invariants section and scanning the knowledge base.",
        _tc("get_cvl_manual_section", headers=["Invariants"]),
        _tc(
            "scan_knowledge_base",
            symptom="structural invariant authoring",
            limit=5,
            offset=0,
        ),
    ),

    # Q3 — exercise the direct KB fetch + both guidance tools + memory view.
    # The KB article title is expected to miss — the harness only cares
    # about exercising the tool dispatch, not the result value.
    _ai(
        "Checking KB for prior notes and pulling guidance.",
        _tc("get_knowledge_base_article", title="Structural invariant patterns"),
        _tc("erc20_guidance"),
        _tc("unresolved_call_guidance"),
        _tc("memory", command="view", path="/memories"),
    ),

    # Q4 — delegate a CVL-syntax question to the research sub-agent.
    # Spawns CR.{1-3}.
    _ai(
        "Delegating an invariant-syntax question to the researcher.",
        _tc(
            "cvl_research",
            question=(
                "What is the correct syntax to write an invariant over a "
                "single top-level uint256 storage field using "
                "currentContract?"
            ),
        ),
    ),

    # CR.1 — research sub-agent, turn 1. Tools: write_rough_draft,
    # read_rough_draft, base_rag_tools (cvl_manual_*, kb_*), result.
    # Validator `_did_rough_draft_read` rejects result until did_read=True.
    _ai(
        "Researcher: sketching an answer + pulling the manual section.",
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Plan: quote the parametric-invariant syntax from the "
                "Invariants section of the manual. Give a worked example "
                "against a uint256 storage field called `count`."
            ),
        ),
        _tc(
            "cvl_manual_search",
            question="invariant syntax currentContract storage field",
            similarity_cutoff=0.5,
            max_results=5,
            manual_section=[],
        ),
    ),

    # CR.2 — research: read the draft so did_read flips true.
    _ai(
        "Researcher: reading the draft before answering.",
        _tc("read_rough_draft"),
    ),

    # CR.3 — research result. Schema is (str, "Your research findings"), so
    # args are {"value": "..."}.
    _ai(
        "Researcher: answer ready.",
        _tc(
            "result",
            value=(
                "An invariant over a single storage field uses the form:\n"
                "  invariant <name>()\n"
                "      currentContract.<field> <relational-op> <expr>;\n"
                "For a uint256 `count`, non-negativity is expressed as:\n"
                "  invariant count_nonneg()\n"
                "      currentContract.count >= 0;\n"
                "Parametric invariants quantify over free variables in the "
                "parameter list (e.g., `invariant f(address a) ...`)."
            ),
        ),
    ),

    # Q5 — intentionally malformed CVL on the first put_cvl_raw.
    # Typechecker.jar rejects the parse and the tool returns the error text
    # without mutating curr_spec.
    _ai(
        "Attempting an initial draft.",
        _tc("put_cvl_raw", cvl_file=BROKEN_PARSE_CVL),
    ),

    # Q6 — put the BAD_INV_CVL. Typechecks fine — the bug is semantic
    # (the invariant is false), not syntactic. Mutates state["curr_spec"]
    # and resets did_read.
    _ai(
        "Putting an initial count_zero-style invariant.",
        _tc("put_cvl_raw", cvl_file=BAD_INV_CVL),
    ),

    # Q7 — exercise get_cvl + record_skip. num_props=2 so property_index=1
    # is valid.
    _ai(
        "Reading back the draft + recording a tentative skip.",
        _tc("get_cvl"),
        _tc(
            "record_skip",
            property_index=1,
            reason=(
                "Tentative — will be undone on the next turn to exercise "
                "unskip_property."
            ),
        ),
    ),

    # Q8 — exercise unskip_property. Empty-reason sentinel in _merge_skips
    # filters the entry out, so state["skipped"] returns to [].
    _ai(
        "Undoing the tentative skip.",
        _tc("unskip_property", property_index=1),
    ),

    # Q9 — exercise expect_rule_failure + expect_rule_passage. The rule
    # name here needn't match any actual rule in curr_spec — both tools just
    # record a rule_skips entry. `expect_rule_passage` then removes it with
    # the DELETE_SKIP sentinel, so state["rule_skips"] returns to {}.
    _ai(
        "Marking a rule expected-to-fail then unmarking it.",
        _tc(
            "expect_rule_failure",
            rule_name="count_zero",
            reason=(
                "Tentative mark — about to unmark to exercise the paired "
                "expect_rule_passage tool."
            ),
        ),
        _tc("expect_rule_passage", rule_name="count_zero"),
    ),

    # Q10 — first feedback_tool invocation against BAD_INV_CVL. Spawns the
    # feedback judge sub-agent (J1.{1-3}). The judge returns good=False so
    # validations["feedback"] is NOT stamped.
    _ai(
        "Seeking judge feedback on the current (bad) draft.",
        _tc("feedback_tool"),
    ),

    # J1.1 — feedback judge, first invocation, turn 1. Tools: memory,
    # rough_draft, get_cvl, feedback_tools (= cvl_authorship_tools), result
    # (PropertyFeedback). Validator `did_rough_draft_read` rejects result
    # until did_read=True.
    _ai(
        "Judge: gathering the spec + drafting a verdict.",
        _tc("memory", command="view", path="/memories"),
        _tc("get_cvl"),
        _tc(
            "write_rough_draft",
            rough_draft=(
                "First-pass: the current spec encodes `count == 0` as an "
                "invariant, which directly contradicts the property that "
                "increment() increases count by 1. Verdict: BAD — spec does "
                "not faithfully express the two target invariants "
                "(count_nonneg, increments_nonneg)."
            ),
        ),
    ),

    # J1.2 — judge: read the draft.
    _ai(
        "Judge: reading the draft before verdict.",
        _tc("read_rough_draft"),
    ),

    # J1.3 — judge: good=False verdict. Does NOT stamp the feedback digest.
    _ai(
        "Judge: delivering the first (rejecting) verdict.",
        _tc(
            "result",
            good=False,
            feedback=(
                "The submitted spec states `count == 0` as an invariant "
                "but the properties to formalize are `count_nonneg` and "
                "`increments_nonneg`. Please replace the spec with "
                "invariants that match the approved property list."
            ),
        ),
    ),

    # Q11 — author addresses the feedback by replacing the spec with
    # SUBTLE_INV_CVL (has the two expected invariant names but `count_nonneg`
    # is subtly wrong — body says ``count > 0`` instead of ``>= 0``).
    # Mutates curr_spec, resets did_read. The feedback digest stamped for
    # BAD_INV_CVL (if any — here J1 returned good=False so there was no
    # stamp) is now stale regardless.
    _ai(
        "Addressing the judge feedback with the two named invariants.",
        _tc("put_cvl_raw", cvl_file=SUBTLE_INV_CVL),
    ),

    # Q12 — second feedback_tool invocation against SUBTLE_INV_CVL. Spawns
    # J2.{1-3}. The judge approves by name-coverage (both expected names
    # present, both trivially typecheck) — missing the subtle `count > 0`
    # semantic bug in the first invariant. good=True stamps
    # validations["feedback"] = digest(SUBTLE_INV_CVL, skipped=[]).
    _ai(
        "Re-running the judge on the updated draft.",
        _tc("feedback_tool"),
    ),

    # J2.1 — feedback judge, second invocation, turn 1.
    _ai(
        "Judge: re-evaluating the updated spec.",
        _tc("get_cvl"),
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Second pass: the spec declares both count_nonneg and "
                "increments_nonneg as separate invariants matching the "
                "approved property list. Coverage looks complete. "
                "Verdict: GOOD."
            ),
        ),
    ),
    _ai(
        "Judge: reading the draft.",
        _tc("read_rough_draft"),
    ),
    # J2.3 — good=True verdict. Stamps validations["feedback"] =
    # digest(SUBTLE_INV_CVL, []). Judge did not catch the `count > 0`
    # typo; the prover will.
    _ai(
        "Judge: approving the spec.",
        _tc(
            "result",
            good=True,
            feedback="",
        ),
    ),

    # Q13 — run verify_spec against SUBTLE_INV_CVL. The base-case check
    # for `count_nonneg` fires on the initial state (count == 0), where
    # the body `count > 0` is false. One rule violated → one
    # ``analyze_cex_raw`` LLM call fires INSIDE verify_spec (between this
    # tape entry and the next author turn). ``all_verified=False`` so
    # the tool returns the raw report string; validations[prover] is NOT
    # stamped.
    _ai(
        "Running the prover on the updated draft.",
        _tc("verify_spec", rules=None),
    ),

    # CEX.1 — inline counter-example analysis. ``analyze_cex_raw`` in
    # ``composer/prover/analysis.py`` calls ``llm.ainvoke(messages)`` (via
    # ``acached_invoke``) with a human-framed instruction template. It
    # expects a plain-text AIMessage back — NO tool_calls, because the
    # call bypasses the LangGraph agent loop entirely.
    #
    # Placement is critical: ``FakeMessagesListChatModel`` has a single
    # global cursor, so this entry must sit between the verify_spec turn
    # (Q13) and the next author turn (Q14). If the author reorders or
    # verify_spec is invoked twice without an intervening CEX, the tape
    # will drift.
    _ai(
        "Counter-example analysis for rule ``count_nonneg``:\n\n"
        "The prover found a violation at the base case. The invariant body "
        "``currentContract.count > 0`` does not hold in the initial state "
        "where ``count == 0``. Intent from the property list was "
        "non-negativity (``>= 0``), not strict positivity (``> 0``) — a "
        "one-character typo in the operator. Suggested fix: change the "
        "body of ``count_nonneg`` to ``currentContract.count >= 0``."
    ),

    # Q14 — author responds to the CEX by replacing SUBTLE_INV_CVL with
    # GOOD_INV_CVL (uses ``>=`` instead of ``>``). Mutates curr_spec,
    # invalidates validations["feedback"] (digest changes).
    _ai(
        "Fixing the count_nonneg operator as the CEX suggests.",
        _tc("put_cvl_raw", cvl_file=GOOD_INV_CVL),
    ),

    # Q15 — third feedback_tool invocation. Spawns J3.{1-3}. Digest stale
    # since curr_spec changed; re-stamping is required before result.
    _ai(
        "Re-running the judge to re-stamp the feedback digest.",
        _tc("feedback_tool"),
    ),

    # J3.1 — feedback judge, third invocation, turn 1.
    _ai(
        "Judge: re-evaluating with the operator fix applied.",
        _tc("get_cvl"),
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Third pass: both invariants now use the ``>= 0`` operator, "
                "which matches uint256 type semantics and makes them "
                "trivially inductive. Verdict: GOOD."
            ),
        ),
    ),
    _ai(
        "Judge: reading the draft.",
        _tc("read_rough_draft"),
    ),
    # J3.3 — good=True. Stamps validations["feedback"] =
    # digest(GOOD_INV_CVL, []).
    _ai(
        "Judge: approving the fixed spec.",
        _tc("result", good=True, feedback=""),
    ),

    # Q16 — run verify_spec on GOOD_INV_CVL. Both invariants reduce to
    # uint256 non-negativity and hold trivially. all_verified=True with
    # rules=None → validations["prover"] stamped with
    # digest(GOOD_INV_CVL, []) — same digest as feedback.
    _ai(
        "Running the prover on the fixed invariants.",
        _tc("verify_spec", rules=None),
    ),

    # Q17 — final result. Both validations current, curr_spec unchanged
    # since Q14 / J3. result schema is (str, "Commentary on your
    # generated spec") so args are {"value": "..."}.
    _ai(
        "Finalizing the invariant CVL.",
        _tc(
            "result",
            value=(
                "Formalized the two structural invariants (count_nonneg, "
                "increments_nonneg) as uint256 non-negativity assertions. "
                "The first authoring attempt used a strict inequality for "
                "count_nonneg which the prover CEXed at the base case; "
                "fixed the operator to ``>=`` and both invariants verified."
            ),
        ),
    ),

    # ───────────────────────────────────────────────────────────────────
    # P5. Bug analysis (run_bug_analysis, 1 component)
    # ───────────────────────────────────────────────────────────────────
    # Tools available: rough_draft (via get_rough_draft_tools),
    #   bug_analysis_tools (= source_tools), result.
    # Validator: standard bind_standard (output_key). Result schema is
    #   (list[PropertyFormulation], "The security properties ..."), so args
    #   are {"value": [...]}.
    #
    # `refinement` is None from the pipeline, so there is NO refinement-loop
    # conversation after this — once `result` fires, the phase ends.

    # P5.1 — exercise source_tools + rough_draft. No did_read requirement,
    # kept for coverage.
    _ai(
        "Bug analysis: inspecting the entry point source.",
        _tc("get_file", path="src/Counter.sol"),
        _tc(
            "write_rough_draft",
            rough_draft=(
                "increment() unconditionally adds 1 to count and 1 to "
                "increments[msg.sender]. The two relevant safety "
                "properties are: (a) count increases by exactly 1, "
                "(b) increments[msg.sender] increases by exactly 1."
            ),
        ),
    ),

    # P5.2 — read draft before emitting result.
    _ai(
        "Bug analysis: re-reading the draft.",
        _tc("read_rough_draft"),
    ),

    # P5.3 — emit both properties in one result call.
    _ai(
        "Delivering the two extracted properties.",
        _tc("result", value=_BUG_ANALYSIS_PROPS),
    ),

    # ───────────────────────────────────────────────────────────────────
    # P6. Component CVL generation (batch_cvl_generation, component=<one>)
    # ───────────────────────────────────────────────────────────────────
    # Same author-agent shape as P4 but streamlined — we do not re-exercise
    # every tool. Tool coverage is satisfied by P4; P6 just exercises the
    # happy path.
    #
    # num_props=2 (2 properties) — same record_skip bounds.

    # R1 — put the two-rule component spec. Typechecks; covers both props
    # from P5 (`increment_increases_count`, `increment_increases_sender_tally`).
    _ai(
        "Writing the component spec covering both properties.",
        _tc("put_cvl_raw", cvl_file=COMPONENT_CVL),
    ),

    # R2 — request feedback. Spawns J3.{1-3}. Judge returns good=True on
    # first pass (no bad-feedback detour in this phase).
    _ai(
        "Requesting judge feedback on the component spec.",
        _tc("feedback_tool"),
    ),

    # J3.1 — feedback judge, single pass, turn 1.
    _ai(
        "Judge: inspecting the component spec.",
        _tc("get_cvl"),
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Two rules, each asserting the exact increment relationship "
                "for its respective storage field. Both assertions are "
                "annotated. Coverage is complete. Verdict: GOOD."
            ),
        ),
    ),
    _ai(
        "Judge: reading the draft.",
        _tc("read_rough_draft"),
    ),
    # J3.3 — good=True verdict. Stamps validations["feedback"] with
    # digest(COMPONENT_CVL, skipped=[]).
    _ai(
        "Judge: approving the component spec.",
        _tc("result", good=True, feedback=""),
    ),

    # R3 — run prover. rules=None + all_verified (both rules hold over
    # increment()) → validations["prover"] stamped.
    _ai(
        "Running the prover on the component spec.",
        _tc("verify_spec", rules=None),
    ),

    # R4 — final result. Both validations current, curr_spec unchanged.
    _ai(
        "Finalizing the component CVL.",
        _tc(
            "result",
            value=(
                "Formalized the two extracted safety properties as direct "
                "pre/post equalities on count and increments[msg.sender]. "
                "Both verified on the first prover run."
            ),
        ),
    ),
]


# ---------------------------------------------------------------------------
# Install / configuration API
# ---------------------------------------------------------------------------
#
# The CEX analyzer's response is inlined at its exact global position in the
# main tape (see the ``CEX.1`` entry right after Q13's verify_spec). There is
# no side-channel tape — ``FakeMessagesListChatModel`` has a single global
# cursor, so any out-of-band entry would be consumed by the wrong LLM call.


def get_autoprove_llm() -> _AutoProveFakeLLM:
    """Return a fresh fake LLM loaded with the autoprove counter tape.

    Each call returns an independent instance (the tape list is shared
    but ``FakeMessagesListChatModel``'s internal cursor is per-instance),
    so tests can run multiple scenarios without cross-contamination.
    """
    return _AutoProveFakeLLM(responses=list(_AUTOPROVE_TAPE))


def install_autoprove_tape() -> _AutoProveFakeLLM:
    """Monkey-patch ``composer.workflow.services.create_llm`` and
    ``create_llm_base`` so the real autoprove pipeline receives the fake.

    Call this BEFORE importing ``tui_autoprove`` — the entry path
    (``composer.cli.tui_autoprove`` → ``composer.spec.source.autoprove_common``)
    imports ``create_llm`` at module load time, so the local binding is
    captured the first time the module is imported. Calling
    ``install_autoprove_tape()`` after that import would be a no-op at
    the real call site.

    Returns the fake instance so the caller can inspect ``.i`` /
    ``.responses`` for debugging.
    """
    fake = get_autoprove_llm()
    import composer.workflow.services as services

    services.create_llm = lambda args: fake  # type: ignore[assignment]
    services.create_llm_base = lambda args: fake  # type: ignore[assignment]
    return fake


__all__ = [
    "BAD_INV_CVL",
    "BROKEN_PARSE_CVL",
    "COMPONENT_CVL",
    "GOOD_INV_CVL",
    "SUBTLE_INV_CVL",
    "get_autoprove_llm",
    "install_autoprove_tape",
]
