"""
Fake-LLM end-to-end UI harness for ``tui_pipeline.py`` (NatSpec multi-agent
pipeline).

Substitutes the real ``ChatAnthropic`` built by
``composer.workflow.services.create_llm`` with a ``FakeMessagesListChatModel``
preloaded with a hand-authored tape of responses. The rest of the pipeline
runs normally — TUI (``PipelineApp``), real tool execution (solc,
certoraTypeCheck.py, Typechecker.jar for ``put_cvl_raw``), workflow graphs,
checkpointing, store/memory/IDE bridges — so UI rendering and tool-dispatch
paths are exercised against canned responses without spending Anthropic API
credits.

Scenario inputs and wiring instructions live under
``composer/testing/scenarios/natspec_counter/``.

The tape is a single linear list of ``AIMessage`` s popped in order on every
call the pipeline makes to the LLM, across every graph:

    component_analysis  →  interface_gen  →  stub_gen  →  bug_analysis
        →  cvl-author (generate_cvl_batch)
            ├─ request_stub_field   →  registry sub-agent
            ├─ cvl_research         →  research sub-agent
            ├─ feedback_tool        →  feedback-judge sub-agent  (×2)
            └─ publish_spec         →  merge sub-agent

The scenario is deliberately constrained to a single contract with a single
component so that the per-contract / per-component concurrency in the pipeline
collapses to linear execution and the global call order is deterministic.

There is no HITL in this workflow — every turn is a plain tool_call or
text-only ``AIMessage``.
"""

from typing import Any, override, Sequence, Callable
import uuid
import asyncio
import random

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.messages.tool import ToolCall
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import AIMessage, BaseMessage


# ---------------------------------------------------------------------------
# Fake LLM plumbing
# ---------------------------------------------------------------------------


class _NatspecFakeLLM(FakeMessagesListChatModel):
    """``FakeMessagesListChatModel`` tolerant of attribute access the natspec
    pipeline performs on the bound LLM.

    The compat shims mirror ``_CodegenFakeLLM`` in ``ui_harness.py``:
    ``thinking`` and ``betas`` are declared as declared fields so pydantic-v2
    tolerates copies/updates from ``create_llm``; ``bind_tools`` is a no-op so
    the Builder can attach tool definitions without the fake raising
    ``NotImplementedError``.
    """

    thinking: Any = None
    betas: list[str] = []

    async def ainvoke(
            self,
            input: PromptValue | str | Sequence[BaseMessage | list[str] | tuple[str, str] | str | dict[str, Any]],
            config: RunnableConfig | None = None,
            *,
            stop: list[str] | None = None,
            **kwargs: Any
    ) -> AIMessage:
        delay = random.random() + 1.0
        await asyncio.sleep(delay)
        return await super().ainvoke(input, config, stop=stop, **kwargs)

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
    """Construct a tool_call dict. Unique ``id`` per call is required — LangGraph
    binds tool responses back to calls by id."""
    return {
        "id": f"toolu_{uuid.uuid4().hex[:20]}",
        "name": name,
        "args": args,
        "type": "tool_call",
    }


def _ai(text: str = "", *tool_calls: ToolCall) -> AIMessage:
    """Helper for authoring a tape entry: optional text + zero or more
    tool_calls. LangGraph's agent loop transitions to the tools node when
    ``tool_calls`` is non-empty, and to END otherwise."""
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
# These strings are emitted as argument fields of the tape's tool calls.
# The scenario is 1 contract (Counter) with 1 component (Increment) and
# 1 property. The real tools that run against these artifacts are:
#
#   - solc8.29             — validates interface + stub + registry-updated stub
#   - Typechecker.jar      — validates CVL syntax inside put_cvl_raw
#   - certoraTypeCheck.py  — validates spec+stub inside advisory_typecheck and
#                            inside publish_spec → merge-agent validator
#
# Each artifact is chosen to satisfy the validator it will hit, plus one
# deliberately broken variant (BROKEN_CVL) to exercise the typechecker-rejects
# → retry path.

INTERFACE_SOURCE = """\
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.29;

interface ICounter {
    function increment() external;
}
"""

# Stub the stub-gen agent publishes. It is a no-op override — the pipeline
# will ask the registry sub-agent to augment it with a ghost field below.
INITIAL_STUB = """\
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.29;

import "../interfaces/ICounter.sol";

contract CounterStub is ICounter {
    function increment() external override {}
}
"""

# Stub returned by the registry sub-agent in response to request_stub_field.
# Must declare the same ``contract CounterStub is ICounter`` identifier as the
# initial stub — publish_spec later runs certoraTypeCheck.py with
# ``contracts/Impl.sol:CounterStub``, so renaming the contract here would
# cause certoraTypeCheck.py to fail to find the contract by that name.
UPDATED_STUB = """\
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.29;

import "../interfaces/ICounter.sol";

contract CounterStub is ICounter {
    uint256 internal ghost_count;

    function increment() external override {
        ghost_count += 1;
    }
}
"""

# Deliberately malformed CVL — the first put_cvl_raw call emits this so
# Typechecker.jar rejects it and the author agent (per the tape) retries
# with VALID_CVL. Exercises the parse-failure rendering path.
BROKEN_CVL = """\
rule broken {
    this is definitely not valid CVL
}
"""

# Minimal CVL that typechecks against CounterStub. Referencing only
# increment() keeps the spec independent of whether the registry ran yet —
# it typechecks against both INITIAL_STUB and UPDATED_STUB.
VALID_CVL = """\
methods {
    function increment() external;
}

rule incrementAlwaysSucceeds {
    env e;
    increment(e);
    assert true;
}
"""

# Same spec with an annotated assertion — the tape uses this as the author's
# response to the first (good=False) feedback verdict. The change is
# semantically trivial; the point is to exercise another put_cvl_raw + a
# second feedback_tool round.
IMPROVED_CVL = """\
methods {
    function increment() external;
}

rule incrementAlwaysSucceeds {
    env e;
    increment(e);
    assert true, "Increment completes without reverting";
}
"""

# The merge sub-agent's result, returned as the ``value`` field of its
# result-tool call. The master spec is empty at publish time (this scenario
# runs exactly one publish), so ``merged == working_copy``. The merge agent's
# validator will run certoraTypeCheck.py on this string against the current
# stub, so it must typecheck — IMPROVED_CVL does.
MERGED_CVL = IMPROVED_CVL


# ---------------------------------------------------------------------------
# The tape
# ---------------------------------------------------------------------------
#
# Global call order (section headers mark boundaries, NOT separate tapes):
#
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │  P1. run_component_analysis                        — 2 turns          │
#   │  P2. generate_interface                            — 1 turn           │
#   │  P3. generate_stub (single contract)               — 1 turn           │
#   │  P4. run_bug_analysis (single component)           — 1 turn           │
#   │  P5. generate_cvl_batch — author                   — 15 turns         │
#   │       ├─ R. registry sub-agent (request_stub_field) — 1 turn          │
#   │       ├─ CR. cvl_research sub-agent                 — 3 turns         │
#   │       ├─ J1. feedback judge — bad verdict           — 3 turns         │
#   │       ├─ J2. feedback judge — good verdict          — 3 turns         │
#   │       └─ M.  merge sub-agent (publish_spec)         — 1 turn          │
#   └────────────────────────────────────────────────────────────────────────┘
#
# Total: 31 AIMessage entries.

_COUNTER_TAPE: list[BaseMessage] = [

    # ─────────────────────────────────────────────────────────────────
    # P1. Component analysis
    # ─────────────────────────────────────────────────────────────────
    # Tools available: memory, write_rough_draft, read_rough_draft, result.
    # No did_read validator here — the only validator is _validate_connectivity
    # which checks the Application shape (unique names, resolved references).
    # So we can go straight to `memory` (once, to exercise) and then `result`.

    # P1.1 — exercise the `memory` tool once. The memory backend constrains
    # paths to the `/memories` subtree, so `view /memories` is the no-op
    # listing here.
    _ai(
        "Cataloguing memory before analyzing the Counter system.",
        _tc("memory", command="view", path="/memories"),
    ),

    # P1.2 — emit the Application via the result tool. One ExplicitContract
    # (Counter) with one ContractComponent (Increment). No external actors,
    # no interactions — this keeps ``_validate_connectivity`` happy and the
    # per-component phases will each run exactly once.
    _ai(
        "Application model ready.",
        _tc(
            "result",
            application_type="Counter",
            description=(
                "A minimal application consisting of a single Counter contract "
                "that tracks an incrementing unsigned integer."
            ),
            components=[
                {
                    "sort": "singleton",
                    "name": "Counter",
                    "description": "Maintains and increments an unsigned integer counter.",
                    "components": [
                        {
                            "name": "Increment",
                            "description": "Handles count updates via a single external entry point.",
                            "external_entry_points": ["increment()"],
                            "state_variables": ["uint256 count"],
                            "interactions": [],
                            "requirements": [
                                "Each call to increment() must increase count by exactly 1.",
                                "increment() must not revert under normal operation.",
                            ],
                        }
                    ],
                }
            ],
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # P2. Interface generation
    # ─────────────────────────────────────────────────────────────────
    # Tools available: result (only). The validator writes the interface to
    # a tmpdir and compiles it with solc8.29 — so the content must compile.

    # P2.1 — emit a complete InterfaceResult. One entry keyed by the contract
    # name "Counter" (validator: the key must be a known contract name).
    _ai(
        "Interface drafted.",
        _tc(
            "result",
            name_to_interface={
                "Counter": {
                    "content": INTERFACE_SOURCE,
                    "solidity_identifier": "ICounter",
                }
            },
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # P3. Stub generation (single contract → single stub_gen invocation)
    # ─────────────────────────────────────────────────────────────────
    # Tools available: result (only). The validator compiles the stub with
    # solc8.29 AND requires the stub content to literally contain the
    # interface filename ("ICounter.sol") and the stub identifier
    # ("CounterStub") — both of which INITIAL_STUB satisfies.

    # P3.1 — publish the no-op stub.
    _ai(
        "Stub drafted.",
        _tc(
            "result",
            solidity_identifier="CounterStub",
            content=INITIAL_STUB,
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # P4. Bug analysis (property extraction for the Increment component)
    # ─────────────────────────────────────────────────────────────────
    # Tools available: write_rough_draft, read_rough_draft, result.
    # No did_read validator here. The result schema is `list[PropertyFormulation]`
    # wrapped via the (type, doc) overload — so args are `{"value": [...]}`.

    # P4.1 — deliver a single PropertyFormulation. Keeping the list at length
    # 1 means ``generate_cvl_batch`` runs exactly one batch with one property,
    # and record_skip / unskip_property bounds checks all accept index 1.
    _ai(
        "Property extracted.",
        _tc(
            "result",
            value=[
                {
                    "methods": ["increment()"],
                    "sort": "safety_property",
                    "description": (
                        "Each call to increment() must increase the observable "
                        "count state by exactly 1, and increment() must not revert."
                    ),
                }
            ],
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # P5. CVL batch generation (author agent)
    # ─────────────────────────────────────────────────────────────────
    # Tools available to the author:
    #   - cvl_authorship_tools (= rag_tools): cvl_manual_search,
    #     cvl_keyword_search, get_cvl_manual_section, scan_knowledge_base,
    #     get_knowledge_base_article, cvl_research, cvl_document_ref
    #   - injected_tools: read_stub, request_stub_field, advisory_typecheck,
    #     publish_spec, give_up
    #   - static_tools: put_cvl, put_cvl_raw, feedback_tool, record_skip,
    #     unskip_property, get_cvl, erc20_guidance, unresolved_call_guidance
    # The graph terminates when `output_key="result"` is written — which only
    # publish_spec or give_up can do. required_validations=["feedback"] must
    # be satisfied (digest of curr_spec+skipped matches validations["feedback"])
    # before publish_spec accepts the publish.
    #
    # cvl_document_ref is NOT exercised: it takes a `ref` string that only
    # the agent_index knows at runtime (hashed from the question), and the
    # tape can't predict it.

    # A1 — exercise read_stub + erc20_guidance in one turn.
    _ai(
        "Surveying the stub and catching up on ERC20 modelling guidance.",
        _tc("read_stub"),
        _tc("erc20_guidance"),
    ),

    # A2 — exercise the similarity-search + keyword-search paths of the CVL
    # manual RAG tools.
    _ai(
        "Searching the CVL manual for relevant rule patterns.",
        _tc(
            "cvl_manual_search",
            question="What is the syntax of a CVL rule that calls a single external function?",
            similarity_cutoff=0.5,
            max_results=5,
            manual_section=[],
        ),
        _tc("cvl_keyword_search", query="rule env", min_depth=0, limit=5),
    ),

    # A3 — exercise section retrieval + knowledge-base scan.
    _ai(
        "Reading the referenced manual section and scanning the knowledge base.",
        _tc("get_cvl_manual_section", headers=["Rules"]),
        _tc(
            "scan_knowledge_base",
            symptom="increment monotonic property",
            limit=5,
            offset=0,
        ),
    ),

    # A4 — exercise the direct-fetch KB path and the unresolved-call guidance.
    # The KB fetch is expected to miss (the title won't exist in the store) —
    # the harness cares about exercising the path, not about the result.
    _ai(
        "Checking the knowledge base for prior notes and unresolved-call guidance.",
        _tc("get_knowledge_base_article", title="Monotonic counter rule"),
        _tc("unresolved_call_guidance"),
    ),

    # A5 — request a stub field. Spawns the registry sub-agent (R1 below).
    _ai(
        "Requesting a ghost mirror for the count state variable.",
        _tc(
            "request_stub_field",
            purpose=(
                "A ghost uint256 that mirrors the Counter's count state variable "
                "so the rule can reason about the monotonic-increase property."
            ),
        ),
    ),

    # R1 — registry sub-agent. Tools: result (only). Validator re-compiles
    # ``updated_stub`` with solc8.29 — so the string must compile standalone
    # against the interface.
    _ai(
        "Registry: adding ghost_count to the stub.",
        _tc(
            "result",
            field_name="ghost_count",
            is_new=True,
            field_type="uint256",
            rejected=False,
            description="Ghost uint256 mirroring the Counter.count storage variable.",
            updated_stub=UPDATED_STUB,
        ),
    ),

    # A6 — delegate a CVL-syntax question to the research sub-agent. This
    # spawns the CVL research graph (CR1..CR3 below). The answer string and
    # a Document-Ref come back to the author, but the tape doesn't rely on
    # the ref — cvl_document_ref is not exercised.
    _ai(
        "Delegating a CVL syntax question to the researcher.",
        _tc(
            "cvl_research",
            question=(
                "How do I express that a ghost variable increases by exactly 1 "
                "after a function call in CVL?"
            ),
        ),
    ),

    # CR1 — research sub-agent turn 1. Tools: write_rough_draft,
    # read_rough_draft, base_rag_tools (cvl_manual_*, kb_*), result.
    # Validator `_did_read_draft` rejects the result tool until did_read is set.
    _ai(
        "Researcher: sketching an answer + pulling the manual.",
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Plan: express monotonicity as an assertion on a ghost that is "
                "incremented in sync with the function call. Confirm via manual."
            ),
        ),
        _tc(
            "cvl_manual_search",
            question="How does CVL express that a ghost variable is incremented?",
            similarity_cutoff=0.5,
            max_results=5,
            manual_section=[],
        ),
    ),

    # CR2 — research: read the rough draft (flips did_read=True so the
    # result-tool validator will pass on the next turn).
    _ai(
        "Researcher: reading the draft before answering.",
        _tc("read_rough_draft"),
    ),

    # CR3 — research: deliver the answer. The result schema here is
    # (str, "Your research findings"), so the tool takes a single `value` arg.
    _ai(
        "Researcher: answer ready.",
        _tc(
            "result",
            value=(
                "To express that a ghost variable g increases by exactly 1 after "
                "a call, capture its pre-state into a mathint, call the function, "
                "then assert `g == old_g + 1`. In CVL:\n\n"
                "  mathint before = ghost_count;\n"
                "  increment(e);\n"
                "  assert to_mathint(ghost_count) == before + 1;\n"
            ),
        ),
    ),

    # A7 — first put_cvl_raw with intentionally broken CVL. Typechecker.jar
    # will reject this, so no state update; the author tries again in A8.
    _ai(
        "Attempting to put an initial spec draft.",
        _tc("put_cvl_raw", cvl_file=BROKEN_CVL),
    ),

    # A8 — second put_cvl_raw with valid CVL. Accepted — state["curr_spec"]
    # and state["did_read"] (as reset_read) are mutated.
    _ai(
        "Putting a minimal valid spec after the parse error.",
        _tc("put_cvl_raw", cvl_file=VALID_CVL),
    ),

    # A9 — exercise get_cvl (read the just-written spec) + advisory_typecheck
    # (runs certoraTypeCheck.py against the current stub+spec) in one turn.
    _ai(
        "Reading back the spec and running an advisory typecheck.",
        _tc("get_cvl"),
        _tc("advisory_typecheck"),
    ),

    # A10 — exercise record_skip. num_props=1 so property_index=1 is the only
    # valid index here.
    _ai(
        "Recording a tentative skip on property 1 to exercise the tool.",
        _tc(
            "record_skip",
            property_index=1,
            reason="Temporary skip — will be undone on the next turn to exercise unskip.",
        ),
    ),

    # A11 — exercise unskip_property. The empty-reason sentinel inside
    # _merge_skips then filters the entry out of state["skipped"], so the
    # final skipped list going into feedback_tool is []. Important: the
    # feedback digest includes skipped — changing skipped between a passing
    # feedback verdict and publish_spec would invalidate the digest.
    _ai(
        "Undoing the tentative skip.",
        _tc("unskip_property", property_index=1),
    ),

    # A12 — first feedback_tool invocation. Spawns the feedback judge
    # sub-agent (J1..J3). The judge returns good=False here, which leaves
    # validations["feedback"] UNSET (digest is only stamped on good=True).
    _ai(
        "Seeking judge feedback on the current spec.",
        _tc("feedback_tool"),
    ),

    # J1 — feedback judge, first invocation, turn 1.
    # Tools available: write_rough_draft, read_rough_draft, memory, get_cvl,
    # feedback_tools (= rag_tools), result. Validator `did_rough_draft_read`
    # requires a read_rough_draft before result.
    _ai(
        "Judge: gathering state and notes.",
        _tc(
            "write_rough_draft",
            rough_draft=(
                "First pass: the rule reaches the assertion but the assertion is "
                "trivially true. Recommend adding an explanatory annotation to "
                "the assertion so the intent is captured."
            ),
        ),
        _tc("memory", command="view", path="/memories"),
        _tc("get_cvl"),
    ),

    # J2 — judge: read draft before verdict.
    _ai(
        "Judge: reading the draft before verdict.",
        _tc("read_rough_draft"),
    ),

    # J3 — judge verdict: good=False + feedback string. Leaves the digest
    # un-stamped, so the author must address and call feedback_tool again.
    _ai(
        "Judge: delivering the first verdict.",
        _tc(
            "result",
            good=False,
            feedback=(
                "The rule is syntactically valid but the assertion `assert true` "
                "has no informative failure message and does not capture the "
                "'does not revert' intent. Please add an explanatory annotation "
                "to the assertion and resubmit."
            ),
        ),
    ),

    # A13 — author addresses the feedback by publishing an improved spec.
    # put_cvl_raw resets did_read=False and curr_spec changes, so the
    # stamped digest (if any) goes stale — forcing the next feedback_tool
    # call to re-stamp.
    _ai(
        "Addressing the judge feedback with an annotated assertion.",
        _tc("put_cvl_raw", cvl_file=IMPROVED_CVL),
    ),

    # A14 — second feedback_tool invocation. Spawns the judge again (J4..J6).
    # This time the verdict is good=True, which sets
    # validations["feedback"] = digest(curr_spec, skipped). publish_spec's
    # `check_completion` then passes.
    _ai(
        "Re-running the judge on the improved spec.",
        _tc("feedback_tool"),
    ),

    # J4 — judge, second invocation, turn 1.
    _ai(
        "Judge: re-evaluating with the improved spec.",
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Second pass: the annotated assertion captures the 'does not "
                "revert' intent. Spec is accepted."
            ),
        ),
    ),

    # J5 — judge: read draft.
    _ai(
        "Judge: reading the draft before verdict.",
        _tc("read_rough_draft"),
    ),

    # J6 — judge verdict: good=True. Stamps validations["feedback"] =
    # digest(curr_spec=IMPROVED_CVL, skipped=[]).
    _ai(
        "Judge: approving the spec.",
        _tc(
            "result",
            good=True,
            feedback="",
        ),
    ),

    # A15 — publish. check_completion sees validations["feedback"] == digest
    # and dispatches to the merge sub-agent (M1). Merge succeeds → result is
    # set to the commentary and the author graph terminates via output_key.
    _ai(
        "Publishing the approved spec to the master spec.",
        _tc(
            "publish_spec",
            commentary=(
                "Formalized the Increment component's 'increment always succeeds' "
                "property as a single rule with an annotated assertion."
            ),
        ),
    ),

    # M1 — merge sub-agent. Tools: rag_tools, result. Validator runs
    # certoraTypeCheck.py on (merged_spec, current_stub). Master spec is empty
    # at this point, so the merged output is exactly the working copy.
    # The result schema is (str, "The complete merged CVL specification"), so
    # args are `{"value": "<merged CVL>"}`.
    _ai(
        "Merge: no prior master spec — working copy is the merge result.",
        _tc("result", value=MERGED_CVL),
    ),
]


# ---------------------------------------------------------------------------
# Install / configuration API
# ---------------------------------------------------------------------------


def get_counter_llm() -> _NatspecFakeLLM:
    """Return a fresh fake LLM loaded with the counter tape.

    Each call returns an independent instance (the tape list is shared but the
    internal cursor ``i`` is per-instance), so tests can run multiple scenarios
    without cross-contamination.
    """
    return _NatspecFakeLLM(responses=list(_COUNTER_TAPE))


def install_counter_tape() -> _NatspecFakeLLM:
    """Monkey-patch ``composer.workflow.services.create_llm`` so the real
    natspec pipeline receives the fake.

    Call this BEFORE importing ``tui_pipeline`` — ``tui_pipeline`` does
    ``from composer.workflow.services import create_llm`` at module load
    time, so the local binding is captured the first time the module is
    imported. Calling ``install_counter_tape()`` after that import would be
    a no-op for the call site in ``tui_pipeline.main``.

    Returns the fake instance so the caller can inspect ``.i`` / ``.responses``
    for debugging.
    """
    fake = get_counter_llm()
    import composer.workflow.services as services

    services.create_llm = lambda args: fake  # type: ignore[assignment]
    services.create_llm_base = lambda args: fake  # type: ignore[assignment]
    return fake


__all__ = [
    "BROKEN_CVL",
    "IMPROVED_CVL",
    "INITIAL_STUB",
    "INTERFACE_SOURCE",
    "MERGED_CVL",
    "UPDATED_STUB",
    "VALID_CVL",
    "get_counter_llm",
    "install_counter_tape",
]
