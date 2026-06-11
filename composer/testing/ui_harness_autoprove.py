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
so that the per-component ``asyncio.gather`` fan-outs in the extraction and
CVL phases collapse to a single lane each. Multiple invariants and multiple
properties are still authored per-phase — a single authoring agent services
them sequentially, so each lane stays linear.

``AutoProveTaskHandler.format_hitl_prompt`` raises ``NotImplementedError``
— there is no Textual-side HITL prompt in this pipeline. The interactive
post-bug-analysis *refinement conversation* is a different mechanism: it
runs through a ``RichConsoleConversationClient`` outside the Textual
screen and consumes plain-text human input from stdin. **Run the pipeline
with ``--interactive``** to exercise it — the refinement conversation's four
tape entries live in the ``bug-0-Increment`` lane, after the property-extraction
entries. Routing is per-lane now, so skipping ``--interactive`` just leaves
those entries unconsumed rather than corrupting another phase. Every
expected human reply is embedded as a ``[TAPE EXPECTATION: respond ...]``
marker inside the preceding AI message so the operator running the harness
knows what to type.

Lanes and call order
--------------------
The pipeline runs several phases concurrently (``asyncio.gather``), so there
is no single global call order any more. ``HarnessFakeLLM`` routes each call
to a per-phase *lane* keyed by the ``run_task`` task_id (read from the
``get_current_task_id`` ContextVar that ``run_task`` sets). Within a lane the
calls happen in the order authored below; sub-agents (invariant_feedback, CEX
analyzer, cvl_research, code_explorer) inherit their parent phase's task_id,
so their responses live in the parent's lane.

    system-analysis : run_component_analysis (+ code_explorer sub-agent)
    harness         : run_harness_creation / classifier_agent
    autosetup       : run_autosetup_phase — a subprocess, makes NO LLM calls,
                      so it has no lane
    ── after harness creation, these lanes run concurrently ──
    invariants       : get_invariant_formulation (+ invariant_feedback ×3)
    bug-0-Increment  : run_property_inference (+ refinement when --interactive)
    ── staged CVL join, after the concurrent branch completes ──
    invariant-cvl    : batch_cvl_generation, component=None
                        (+ cvl_research, code_explorer, feedback ×2, CEX ×1)
    cvl-0-Increment  : batch_cvl_generation, component=<one>
                        (+ feedback ×1, CEX ×1 — surfaces the real
                        ``incrementOther`` implementation bug)

    Per-component lanes are ``{bug,cvl}-{component index}-{slugified_name}``;
    here the Counter's sole component is "Increment".
"""

from typing import Any
import uuid

from composer.testing.harness_tape import HarnessFakeLLM
from composer.spec.source.task_ids import (
    SYSTEM_ANALYSIS_TASK_ID, HARNESS_TASK_ID, INVARIANTS_TASK_ID,
    INVARIANT_CVL_TASK_ID, bug_analysis_task_id, cvl_gen_task_id,
)

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import ToolCall


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
invariant increments_sum_is_count() currentContract.count == 0;
"""

# Typechecks and declares the two ostensibly-correct invariant names, but the
# ``increments_sum_is_count`` is subtly wrong; without an init state axiom, the
# prover can choose an initial value of incrementsSum that violates the base case.
# The feedback judge approves by name-coverage; the prover catches it on the
# base case (initial state has ``count == 0``, violating ``count > 0``).
# This is the artifact that drives the verify_spec → analyze_cex_raw round-trip
# in the tape — exactly one failing rule (``count_nonneg``), so exactly one
# CEX LLM call is consumed.
SUBTLE_INV_CVL = """\
ghost uint256 incrementsSum;

hook Sstore currentContract.increments[KEY address who] uint256 newValue (uint256 oldValue) {
	incrementsSum = require_uint256(incrementsSum + (newValue - oldValue));
}

invariant zero_address_is_zero() currentContract.increments[0] == 0;

invariant increments_sum_is_count() currentContract.count == incrementsSum;
"""

# Two trivially-true invariants over the Counter state. Both should verify
# against Counter.sol on first try, so verify_spec stamps the prover digest
# and the author can call `result` to terminate the invariant-CVL author graph.
GOOD_INV_CVL = """\
ghost uint256 incrementsSum {
	init_state axiom incrementsSum == 0; 
}

hook Sstore currentContract.increments[KEY address who] uint256 newValue (uint256 oldValue) {
	incrementsSum = require_uint256(incrementsSum + (newValue - oldValue));
}

invariant zero_address_is_zero() currentContract.increments[0] == 0;

invariant increments_sum_is_count() currentContract.count == incrementsSum;
"""

# Component-CVL spec: three rules covering all three extracted properties.
# The first two rules verify on the first prover run. The third rule —
# ``incrementOther_credits_target_when_distinct`` — CEXes against
# ``Counter.incrementOther`` (which has a real off-target bug: it credits
# ``msg.sender`` instead of ``other``). The tape responds to that CEX by
# calling ``expect_rule_failure`` to mark the rule as surfacing a real
# implementation bug, then re-runs ``verify_spec`` with the rule excluded.
COMPONENT_CVL = """\
methods {
    function count() external returns (uint256) envfree;
    function increments(address) external returns (uint256) envfree;
    function increment() external;
    function incrementOther(address) external;
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

rule incrementOther_credits_target_when_distinct {
    env e;
    address other;
    require other != 0;
    require other != e.msg.sender;
    mathint before_other = increments(other);
    incrementOther(e, other);
    assert to_mathint(increments(other)) == before_other + 1,
        "incrementOther(other) must increase increments[other] by exactly 1 when other != msg.sender";
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
        "count and a per-caller tally of invocations via two external "
        "entry points (``increment`` and ``incrementOther``)."
    ),
    "components": [
        {
            "sort": "singleton",
            "name": "Counter",
            "path": "src/Counter.sol",
            "description": (
                "The only contract in the system; owns the count and per-"
                "caller tally state and the two increment entry points."
            ),
            "components": [
                {
                    "name": "Increment",
                    "description": (
                        "Handles all count updates through the "
                        "``increment()`` and ``incrementOther(address)`` "
                        "external entry points."
                    ),
                    "external_entry_points": [
                        "increment()", "incrementOther(address)"
                    ],
                    "state_variables": [
                        "uint256 count",
                        "mapping(address => uint256) increments",
                    ],
                    "interactions": [],
                    "requirements": [
                        "Each call to increment() increases count by exactly 1.",
                        "Each call to increment() increases increments[msg.sender] by exactly 1.",
                        "Each call to incrementOther(other) increases increments[other] by exactly 1.",
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
        "title": "count_increments_by_one",
        "methods": ["increment()"],
        "sort": "safety_property",
        "description": (
            "After calling increment(), the global count must be exactly "
            "one greater than before the call."
        ),
    },
    {
        "title": "sender_increments_by_one",
        "methods": ["increment()"],
        "sort": "safety_property",
        "description": (
            "After calling increment(), increments[msg.sender] must be "
            "exactly one greater than before the call."
        ),
    },
    {
        "title": "other_increments_by_one",
        "methods": ["incrementOther(address)"],
        "sort": "safety_property",
        "description": (
            "After calling incrementOther(other), increments[other] must "
            "be exactly one greater than before the call."
        ),
    },
]


# After the user works through the refinement conversation, the AI is asked
# to refine property 3 (incrementOther) so that it is explicit about which
# storage slot is supposed to move and which is supposed to stay put. The
# updated list is what eventually feeds the component-CVL phase.

_REFINED_BUG_ANALYSIS_PROPS = [
    {
        "title": "count_increments_by_one",
        "methods": ["increment()"],
        "sort": "safety_property",
        "description": (
            "After calling increment(), the global count must be exactly "
            "one greater than before the call."
        ),
    },
    {
        "title": "sender_increments_by_one",
        "methods": ["increment()"],
        "sort": "safety_property",
        "description": (
            "After calling increment(), increments[msg.sender] must be "
            "exactly one greater than before the call."
        ),
    },
    {
        "title": "other_increments_by_one",
        "methods": ["incrementOther(address)"],
        "sort": "safety_property",
        "description": (
            "After calling incrementOther(other) with other != msg.sender, "
            "increments[other] must increase by exactly 1 and "
            "increments[msg.sender] must be unchanged."
        ),
    },
]


# ---------------------------------------------------------------------------
# The tape
# ---------------------------------------------------------------------------
#
# Authored as one list per phase ("lane"), assembled into the per-lane
# ``_AUTOPROVE_TAPE`` dict at the bottom. HarnessFakeLLM serves each LLM call
# from its lane's cursor (keyed by run_task task_id), so the scripted responses
# stay correct even though the pipeline runs phases concurrently. Within a lane,
# entries are popped in order; if the pipeline issues a call the lane doesn't
# have, the fake raises. Editing the tape is the cheap loop.

_SYSTEM_ANALYSIS_TAPE: list[BaseMessage] = [

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
    # P2. Classifier agent (run_harness_creation → classifier_agent →
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

]

_HARNESS_TAPE: list[BaseMessage] = [

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

]

_INVARIANTS_TAPE: list[BaseMessage] = [

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
                "name": "increments_sum_is_count",
                "description": (
                    "`count` is the sum of all values in the `increments` map"
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
                "Sums can be reasoned about in CVL. Formal and inductive. Verdict: GOOD."
            ),
        ),
    ),
    _ai(
        "Judge: reading the draft.",
        _tc("read_rough_draft"),
    ),
    # F2.3 — GOOD verdict. Stamps state["invariant_data"]["count_nonneg"].
    _ai(
        "Judge: GOOD verdict on increments_sum_is_count.",
        _tc(
            "result",
            sort="GOOD",
            explanation=(
                "The invariant is inductive and formalizable."
            ),
        ),
    ),

    # P3.4 — main agent proposes second invariant. Spawns F3.{1-3}.
    _ai(
        "Proposing the second invariant.",
        _tc(
            "invariant_feedback",
            inv={
                "name": "zero_address_is_zero",
                "description": (
                    "The zero address' `increments` value is always 0."
                ),
            },
        ),
    ),

    # F3.1 — judge, third invocation.
    _ai(
        "Judge: evaluating zero_address_is_zero.",
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Trivially implied by the implementation, "
                "but formal and inductive. Verdict: GOOD."
            ),
        ),
    ),
    _ai(
        "Judge: reading the draft.",
        _tc("read_rough_draft"),
    ),
    _ai(
        "Judge: GOOD verdict on zero_address_is_zero.",
        _tc(
            "result",
            sort="GOOD",
            explanation=(
                "The invariant is trivially true"
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
                    "name": "increments_sum_is_count",
                    "description": (
                        "`count` is the sum of all values in the `increments` map"
                    ),
                },
                {
                    "name": "zero_address_is_zero",
                    "description": (
                        "The zero address' `increments` value is always 0."
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
    # 2 invariants — record_skip / unskip_property accept the property titles
    # `increments_sum_is_count` and `zero_address_is_zero`.

]

_INVARIANT_CVL_TAPE: list[BaseMessage] = [

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

    # Q7 — exercise get_cvl + record_skip. The two invariant titles are
    # `increments_sum_is_count` (1st) and `zero_address_is_zero` (2nd).
    _ai(
        "Reading back the draft + recording a tentative skip.",
        _tc("get_cvl"),
        _tc(
            "record_skip",
            property_title="increments_sum_is_count",
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
        _tc("unskip_property", property_title="increments_sum_is_count"),
    ),

    # Q9 — exercise expect_rule_failure + expect_rule_passage. The rule
    # name here needn't match any actual rule in curr_spec — both tools just
    # record a rule_skips entry. `expect_rule_passage` then removes it with
    # the DELETE_SKIP sentinel, so state["rule_skips"] returns to {}.
    _ai(
        "Marking a rule expected-to-fail...",
        _tc(
            "expect_rule_failure",
            rule_name="count_zero",
            reason=(
                "Tentative mark — about to unmark to exercise the paired "
                "expect_rule_passage tool."
            ),
        ),
    ),
    _ai(
        "Actually, just kidding",
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
                "(increments_sum_is_count, zero_address_is_zero)."
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
                "`zero_address_is_zero`. Please replace the spec with "
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
                "Second pass: the spec declares both increments_sum_is_count and "
                "zero_address_is_zero as separate invariants matching the "
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
        "Counter-example analysis for rule ``increments_sum_is_count``:\n\n"
        "The prover found a spurious starting state where incrementsSum is initialized to be"
        " non-zero in the invariant base case (constructor) which causes a trivial failure.\n\n"
        "Suggested fix: add an init_state axiom to constrain the value of the ghost in the base case."

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
                "The init state axiom is well justified given that the sum of increments is 0 on creation."
                 " Verdict: GOOD."
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
    # since Q14 / J3. PublishResultTool requires `commentary` plus a
    # `property_rules` mapping covering every (non-skipped) batch title —
    # here the two invariant titles, each verified by the invariant of the
    # same name in GOOD_INV_CVL.
    _ai(
        "Finalizing the invariant CVL.",
        _tc(
            "result",
            commentary=(
                "Formalized the two structural invariants (increments_sum_is_count, "
                "zero_address_is_zero)."
            ),
            property_rules=[
                {"property_title": "increments_sum_is_count", "rules": ["increments_sum_is_count"]},
                {"property_title": "zero_address_is_zero", "rules": ["zero_address_is_zero"]},
            ],
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

]

_BUG_TAPE: list[BaseMessage] = [

    # P5.1 — exercise source_tools + rough_draft. No did_read requirement,
    # kept for coverage.
    _ai(
        "Bug analysis: inspecting the entry point source.",
        _tc("get_file", path="src/Counter.sol"),
        _tc(
            "write_rough_draft",
            rough_draft=(
                "increment() unconditionally adds 1 to count and 1 to "
                "increments[msg.sender]. incrementOther(other) is meant "
                "to credit increments[other] but the implementation looks "
                "off — flag a property over its intended behavior. Three "
                "safety properties total: (a) increment() bumps count "
                "by 1, (b) increment() bumps increments[msg.sender] by 1, "
                "(c) incrementOther(other) bumps increments[other] by 1."
            ),
        ),
    ),

    # P5.2 — read draft before emitting result.
    _ai(
        "Bug analysis: re-reading the draft.",
        _tc("read_rough_draft"),
    ),

    # P5.3 — emit all three properties in one result call. Schema is the
    # ``_AgentRoundResult`` BaseModel (composer/spec/bug.py): ``items`` is the
    # property list and ``reasoning`` is a required narrative field — there
    # is no ``value`` wrapper here, unlike the tuple-shaped result tools.
    _ai(
        "Delivering the three extracted properties.",
        _tc(
            "result",
            items=_BUG_ANALYSIS_PROPS,
            reasoning=(
                "increment() unconditionally mutates two storage slots: it "
                "adds 1 to `count` and 1 to `increments[msg.sender]`. "
                "incrementOther(other) is documented to credit "
                "`increments[other]` by 1; whether the implementation "
                "actually does that is a question for the prover. The "
                "three pre/post equalities on those slots are the obvious "
                "safety properties; nothing else in the contract surface "
                "is worth formalizing at this stage."
            ),
        ),
    ),

    # ───────────────────────────────────────────────────────────────────
    # P5b. Interactive refinement conversation (only when --interactive)
    # ───────────────────────────────────────────────────────────────────
    # ``run_bug_analysis`` enters ``refinement_loop`` once
    # ``interactive=True``. The conversation graph has its own thread,
    # its own tools (``env.source_tools`` + ``finalize_properties`` +
    # ``update_requirements``), and is driven by stdin via
    # ``RichConsoleConversationClient`` (see
    # ``composer/ui/conversation_client.py``).
    #
    # The flow is interrupt-driven:
    #   chat_node → interrupt(HumanPrompt)  →  outer loop reads stdin
    #     →  resume with HumanMessage  →  llm_echo (TAPE ENTRY)
    #     →  (optional) tools  →  llm_echo (TAPE ENTRY)
    #     →  ... back to chat_node, repeat
    #   until the LLM calls ``finalize_properties`` (Exit) which raises
    #   ``interrupt(EndConversation())`` from inside the tools node.
    #
    # Every AI message below tags the expected human reply with
    # ``[TAPE EXPECTATION: respond '...']``. The harness operator types
    # that reply into the prompt-toolkit prompt to advance the tape. The
    # tape entries below are popped one per ``llm_echo`` invocation.

    # P5b.1 — first AI turn after the user's opening prompt. The expected
    # opening from the user is a question about the property list. The AI
    # answers conversationally with NO tool call, so control returns to
    # ``chat_node`` for the next human turn.
    _ai(
        "Sure — property 1 and property 2 cover the obvious behavior of "
        "``increment()``: it adds 1 to ``count`` and 1 to "
        "``increments[msg.sender]``. Property 3 is about "
        "``incrementOther(other)``: the function name suggests it credits "
        "a different address, so the matching post-condition is "
        "``increments[other]`` goes up by 1. I haven't tried to prove "
        "any of these yet — they're just what the surface contract "
        "promises."
    ),

    # P5b.2 — second AI turn: the user has asked for a rewording of
    # property 3. The AI calls ``update_requirements`` with the full
    # refined list. SetRequirements injects the new list into
    # ``state["extra_data"]`` via a Command and the state-diff renderer
    # fires.
    _ai(
        "Got it — I'll tighten property 3 to call out both the move and "
        "the non-move, conditioned on ``other != msg.sender``. Updating "
        "now.",
        _tc(
            "update_requirements",
            new_requirements=_REFINED_BUG_ANALYSIS_PROPS,
        ),
    ),

    # P5b.3 — after ``update_requirements`` returns, llm_echo fires again.
    # The AI yields control back to the user (no tool calls) so they can
    # approve.
    _ai(
        "Updated. Property 3 now reads: \"After calling "
        "incrementOther(other) with other != msg.sender, "
        "increments[other] must increase by exactly 1 and "
        "increments[msg.sender] must be unchanged.\" The first two are "
        "untouched."
    ),

    # P5b.4 — user has signaled finalization. The AI calls
    # ``finalize_properties`` (Exit) which raises
    # ``interrupt(EndConversation())`` from inside the tools node; the
    # outer loop catches it and exits the refinement context.
    _ai(
        "Finalizing the refined property list.",
        _tc("finalize_properties"),
    ),

    # ───────────────────────────────────────────────────────────────────
    # P6. Component CVL generation (batch_cvl_generation, component=<one>)
    # ───────────────────────────────────────────────────────────────────
    # Same author-agent shape as P4 but streamlined — we do not re-exercise
    # every tool. Tool coverage is satisfied by P4; P6 covers the
    # surface-a-real-bug path.
    #
    # 3 refined properties from P5b — record_skip would accept their titles,
    # but the tape doesn't exercise record_skip in this phase.
    #
    # The spec contains three rules: two that hold against the
    # implementation and one (``incrementOther_credits_target_when_distinct``)
    # that CEXes because ``Counter.incrementOther`` has a real bug — it
    # credits ``msg.sender`` instead of ``other``. The author marks that
    # rule as expected-to-fail with a reason explaining the surfaced bug,
    # then re-runs the prover with the rule excluded so
    # ``validations[prover]`` can be stamped.

]

_CVL_TAPE: list[BaseMessage] = [

    # R1 — put the three-rule component spec. Typechecks; covers all three
    # refined props.
    _ai(
        "Writing the component spec covering all three properties.",
        _tc("put_cvl_raw", cvl_file=COMPONENT_CVL),
    ),

    # R2 — request feedback. Spawns J3.{1-3}. Judge returns good=True on
    # first pass (the spec faithfully encodes all three properties; whether
    # the rules pass against the implementation is the prover's question).
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
                "Three rules, each asserting the exact post-condition for "
                "its respective property. The incrementOther rule "
                "constrains other != msg.sender per the refined property. "
                "Coverage is complete. Verdict: GOOD."
            ),
        ),
    ),
    _ai(
        "Judge: reading the draft.",
        _tc("read_rough_draft"),
    ),
    # J3.3 — good=True verdict. Stamps validations["feedback"] with
    # digest(COMPONENT_CVL, skipped=[]). rule_skips is NOT part of the
    # digest, so the later expect_rule_failure won't invalidate this
    # stamp.
    _ai(
        "Judge: approving the component spec.",
        _tc("result", good=True, feedback=""),
    ),

    # R3 — first prover run. The two increment() rules verify; the
    # incrementOther rule CEXes (msg.sender credited instead of other).
    # all_verified=False → validations[prover] NOT stamped, tool returns
    # raw report string. Exactly ONE failing rule → exactly ONE
    # ``analyze_cex_raw`` LLM call fires inline (CEX.2 below).
    _ai(
        "Running the prover on the component spec.",
        _tc("verify_spec", rules=None),
    ),

    # CEX.2 — inline analysis of the incrementOther CEX. Plain AIMessage,
    # no tool_calls, mirrors the CEX.1 entry in the invariant-CVL phase.
    # Critical placement: between R3 and R4 in the global tape cursor.
    _ai(
        "Counter-example analysis for rule "
        "``incrementOther_credits_target_when_distinct``:\n\n"
        "The prover constructed a state where ``msg.sender`` and "
        "``other`` are distinct nonzero addresses and ``increments[other]"
        "`` starts at 0. After the call, ``increments[other]`` is still "
        "0 — the implementation incremented ``increments[msg.sender]`` "
        "instead. This is a real bug in ``Counter.incrementOther``: it "
        "credits the caller rather than the target address. The CVL "
        "rule is correctly written; the implementation is wrong.\n\n"
        "Suggested action: leave the rule in place as a regression "
        "witness, mark it expected-to-fail with a citation back to the "
        "implementation bug, and surface this in the final commentary "
        "so a human can fix the Solidity."
    ),

    # R4 — author responds to the surfaced bug by marking the rule as
    # expected-to-fail. ``expect_rule_failure`` writes into ``rule_skips``
    # via a Command. ``rule_skips`` is NOT part of the digest used by
    # validation stamps, so the prior feedback stamp remains valid.
    _ai(
        "The CEX flags a real bug in Counter.incrementOther. Marking the "
        "rule as expected-to-fail so we can re-run the prover with it "
        "excluded.",
        _tc(
            "expect_rule_failure",
            rule_name="incrementOther_credits_target_when_distinct",
            reason=(
                "Surfaces a real implementation bug in "
                "Counter.incrementOther: the function credits "
                "increments[msg.sender] instead of increments[other]. "
                "The rule (and the property it formalizes) are correct; "
                "the Solidity needs to be fixed. Tracking the rule as "
                "expected-to-fail so the spec still verifies for the two "
                "increment() properties while the bug is open."
            ),
        ),
    ),

    # R5 — re-run prover. With the buggy rule in rule_skips, the
    # all_verified loop in verify_spec ignores it; the two increment()
    # rules pass, so all_verified=True and rules=None → stamps
    # validations[prover] at digest(COMPONENT_CVL, skipped=[]), which
    # matches the feedback stamp from J3.3.
    _ai(
        "Re-running the prover with the buggy rule excluded.",
        _tc("verify_spec", rules=None),
    ),
    _ai(
        "Counter-example analysis for rule "
        "``incrementOther_credits_target_when_distinct``:\n\n"
        "The prover constructed a state where ``msg.sender`` and "
        "``other`` are distinct nonzero addresses and ``increments[other]"
        "`` starts at 0. After the call, ``increments[other]`` is still "
        "0 — the implementation incremented ``increments[msg.sender]`` "
        "instead. This is a real bug in ``Counter.incrementOther``: it "
        "credits the caller rather than the target address. The CVL "
        "rule is correctly written; the implementation is wrong.\n\n"
        "Suggested action: leave the rule in place as a regression "
        "witness, mark it expected-to-fail with a citation back to the "
        "implementation bug, and surface this in the final commentary "
        "so a human can fix the Solidity."
    ),

    # R6 — final result. Both stamps current, curr_spec unchanged since
    # R1. Commentary documents the surfaced bug so the downstream
    # ``natspec_report`` / file-on-disk autospec output flags it for the
    # human reviewer.
    _ai(
        "Finalizing the component CVL.",
        _tc(
            "result",
            commentary=(
                "Formalized all three extracted safety properties as "
                "pre/post equalities. The two increment() rules verify. "
                "The incrementOther rule is left in place and marked "
                "expected-to-fail because the prover surfaced a real "
                "bug: Counter.incrementOther credits increments[msg."
                "sender] instead of increments[other]. The spec is "
                "correct; the implementation needs to be fixed."
            ),
            property_rules=[
                {"property_title": "count_increments_by_one", "rules": ["increment_increases_count"]},
                {"property_title": "sender_increments_by_one", "rules": ["increment_increases_sender_tally"]},
                {"property_title": "other_increments_by_one", "rules": ["incrementOther_credits_target_when_distinct"]},
            ],
        ),
    ),
]


# The tape, as a per-phase lane map keyed by run_task task_id. HarnessFakeLLM
# serves each LLM call from its lane's cursor, so the scripted responses stay
# correct even though the pipeline runs phases concurrently. The Counter
# scenario has one component, "Increment".
_AUTOPROVE_TAPE: dict[str, list[BaseMessage]] = {
    SYSTEM_ANALYSIS_TASK_ID: _SYSTEM_ANALYSIS_TAPE,
    HARNESS_TASK_ID: _HARNESS_TAPE,
    INVARIANTS_TASK_ID: _INVARIANTS_TAPE,
    INVARIANT_CVL_TASK_ID: _INVARIANT_CVL_TAPE,
    bug_analysis_task_id(0, "Increment"): _BUG_TAPE,
    cvl_gen_task_id(0, "Increment"): _CVL_TAPE,
}


# ---------------------------------------------------------------------------
# Install / configuration API
# ---------------------------------------------------------------------------
#
# The CEX analyzer's response is inlined at its position within the
# invariant-cvl / cvl-0-Increment lane (see the ``CEX.1`` entry after Q13's verify_spec).
# There is no side-channel tape — each call is routed to its phase's lane by
# ``run_task`` task_id, and within a lane responses are consumed in order.


def get_autoprove_llm() -> HarnessFakeLLM:
    """Return a fresh fake LLM loaded with the autoprove counter tape.

    Each call returns an independent instance with its own per-lane cursors, so
    tests can run multiple scenarios without cross-contamination.
    """
    return HarnessFakeLLM(lanes=_AUTOPROVE_TAPE)


def install_harness_tape() -> HarnessFakeLLM:
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
    import composer.spec.agent_index as a_ind
    a_ind._UNSAFE_DISABLE_CACHE = True

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
    "install_harness_tape",
]


# ---------------------------------------------------------------------------
# Operator notes for the interactive refinement conversation (P5b)
# ---------------------------------------------------------------------------
#
# The refinement conversation kicks in only when the auto-prove pipeline is
# invoked with ``--interactive``. The first human turn has no preceding AI
# message (state == INIT), so there is no ``[TAPE EXPECTATION]`` marker for
# it. The expected opening user prompt is:
#
#     >>> Walk me through the properties you extracted, especially
#         property 3 — I want to make sure it's right.
#
# (Or any prompt that asks the AI to discuss the properties; the AI's
# scripted P5b.1 response presupposes a question along those lines.)
#
# Subsequent human turns have ``[TAPE EXPECTATION: respond '...']`` markers
# embedded in the preceding AI message — type those verbatim to advance the
# tape.
