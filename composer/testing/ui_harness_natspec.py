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
            └─ publish              (in-process, no sub-agent)

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
#                            inside publish's typechecker call
#
# Each artifact is chosen to satisfy the validator it will hit, plus one
# deliberately broken variant (BROKEN_CVL) to exercise the typechecker-rejects
# → retry path.
#
# Contract identity convention: the contract's design-doc ``name`` and its
# ``solidity_identifier`` both happen to be ``"Counter"`` for this scenario.
# The stub validator requires ``res.solidity_identifier == c.solidity_identifier``
# (= "Counter") and that the stub's content declare ``contract Counter`` —
# so we cannot rename the stub to ``CounterStub`` even though it is a stub in
# the loose sense.

INTERFACE_SOURCE = """\
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.29;

interface ICounter {
    function increment() external;
}
"""

# Stub the stub-gen agent publishes. Must declare ``contract Counter`` with
# the same Solidity identifier the component-analysis result emitted —
# the stub validator enforces ``res.solidity_identifier == "Counter"`` and
# ``"contract Counter" in res.content``.
#
# AutoStubDeclaration auto-derives ``path = src/contracts/Counter.sol``, so
# the interface import is a sibling-relative path one directory up.
INITIAL_STUB = """\
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.29;

import "./ICounter.sol";

contract Counter is ICounter {
    function increment() external override {}
}
"""

# Stub returned by the registry sub-agent in response to request_stub_field.
# Same contract identifier as the initial stub (the registry agent only adds
# a storage field — it must not rename the contract).
UPDATED_STUB = """\
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.29;

import "./ICounter.sol";

contract Counter is ICounter {
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

# Minimal CVL that typechecks against the Counter stub.
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


# Title used for the single property; ``record_skip`` / ``unskip_property``
# and the feedback judge all key off of titles (not indices), so this string
# is referenced from multiple turns below.
_PROP_TITLE = "increment_always_succeeds"


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
#   │  P4. run_property_inference (single component)     — 1 turn           │
#   │  P5. generate_cvl_batch — author                   — 15 turns         │
#   │       ├─ R. registry sub-agent (request_stub_field) — 1 turn          │
#   │       ├─ CR. cvl_research sub-agent                 — 3 turns         │
#   │       ├─ J1. feedback judge — bad verdict           — 3 turns         │
#   │       └─ J2. feedback judge — good verdict          — 3 turns         │
#   │                                                                       │
#   │  (publish is in-process now — no separate merge sub-agent turn.)      │
#   └────────────────────────────────────────────────────────────────────────┘
#
# Total: 30 AIMessage entries.

_COUNTER_TAPE: list[BaseMessage] = [

    # ─────────────────────────────────────────────────────────────────
    # P1. Component analysis
    # ─────────────────────────────────────────────────────────────────
    # Tools available in greenfield: memory, write_rough_draft,
    # read_rough_draft, result. ``env.analysis_tools`` is empty in
    # greenfield, so no source/rag tools here.
    # Validator: ``_validate_connectivity`` — checks unique names and
    # resolved component references. No did_read gate.

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
    #
    # NOTE: ExplicitContract requires both ``name`` (ContractName) and
    # ``solidity_identifier`` (regex-validated). Here they match.
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
                    "solidity_identifier": "Counter",
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
    # Tools available: result + env.analysis_tools (empty in greenfield).
    # Validator writes the interface to a tmpdir and compiles it with
    # solc8.29 — so the content must compile.

    # P2.1 — emit a complete InterfaceResult keyed by the contract's
    # Solidity identifier ("Counter"). The interface's own
    # ``solidity_identifier`` is "ICounter" — they're different concerns:
    # the dict key is the contract being implemented, the value's
    # solidity_identifier is what the interface file declares.
    _ai(
        "Interface drafted.",
        _tc(
            "result",
            value={
                "name_to_interface": {
                    "Counter": {
                        "content": INTERFACE_SOURCE,
                        "solidity_identifier": "ICounter",
                    }
                }
            }
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # P3. Stub generation (single contract → single stub_gen invocation)
    # ─────────────────────────────────────────────────────────────────
    # Tools available: result + env.source_tools (greenfield → empty layered
    # FS, so `list_files`/`get_file` are non-useful here).
    # Validator: solc8.29 compile of the stub. Also enforces:
    #   - ``res.solidity_identifier == "Counter"``      (caller-supplied)
    #   - ``"contract Counter" in res.content``
    #   - ``ICounter.sol in res.content``               (interface_basename)
    #   - ``path.stem == "Counter"``                    (auto path → ok)

    # P3.1 — publish the no-op stub.
    _ai(
        "Stub drafted.",
        _tc(
            "result",
            value=dict(
                solidity_identifier="Counter",
                content=INITIAL_STUB,
            )
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # P4. Property inference (extraction for the Increment component)
    # ─────────────────────────────────────────────────────────────────
    # Tools available: write_rough_draft, read_rough_draft, result.
    # ``env.analysis_tools`` is empty in greenfield. No did_read gate.
    # Result schema: ``_AgentRoundResult`` = ``{items, reasoning}``.
    # Validator: ``_unique_titles_validator`` — every property title must
    # be unique within the batch (and across prior rounds, but this run
    # has none).

    # P4.1 — deliver a single PropertyFormulation. Title is fixed (see
    # ``_PROP_TITLE``) because record_skip / unskip_property / the
    # feedback judge all key off of titles, not indices.
    _ai(
        "Property extracted.",
        _tc(
            "result",
            items=[
                {
                    "title": _PROP_TITLE,
                    "methods": ["increment()"],
                    "sort": "safety_property",
                    "description": (
                        "Each call to increment() must increase the observable "
                        "count state by exactly 1, and increment() must not revert."
                    ),
                }
            ],
            reasoning=(
                "Counter has a single observable method and a single state "
                "transition; the only meaningful behavioral assertion is the "
                "monotonic-by-one increment and no-revert pair, captured as one "
                "property here."
            ),
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # P5. CVL batch generation (author agent)
    # ─────────────────────────────────────────────────────────────────
    # Tools available to the author:
    #   - env.all_tools = source_tools (fs) + rag_tools (cvl_manual_search,
    #     cvl_keyword_search, get_cvl_manual_section, scan_knowledge_base,
    #     get_knowledge_base_article, cvl_research, cvl_document_ref)
    #   - injected_tools: request_stub_field, register_verification_file,
    #     list_verification_files
    #   - static_tools: put_cvl, put_cvl_raw, feedback_tool, record_skip,
    #     unskip_property, get_cvl, erc20_guidance, unresolved_call_guidance
    #   - give_up, advisory_typecheck, publish, memory
    # The graph terminates when ``output_key="result"`` is written — only
    # ``publish`` or ``give_up`` can do that. ``required_validations=["feedback"]``
    # must be satisfied (digest of curr_spec+skipped matches
    # ``validations["feedback"]``) before publish accepts the call.
    #
    # cvl_document_ref is NOT exercised: it takes a `ref` string that only
    # the agent_index knows at runtime (hashed from the question), and the
    # tape can't predict it.

    # A1 — exercise erc20_guidance + list_verification_files in one turn.
    # (``read_stub`` is no longer a tool; the stub source is supplied via
    # the initial prompt, and the layered FS can be inspected via the
    # standard source tools if needed.)
    _ai(
        "Catching up on ERC20 modelling guidance and the file registry.",
        _tc("erc20_guidance"),
        _tc("list_verification_files"),
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
    # ``request_stub_field`` takes ``contract_identifier`` (the Solidity
    # identifier of the stub to grow) AND ``purpose``.
    _ai(
        "Requesting a ghost mirror for the count state variable.",
        _tc(
            "request_stub_field",
            contract_identifier="Counter",
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

    # A10 — exercise record_skip. Properties are now keyed by ``title``
    # (not index); the only valid title here is ``_PROP_TITLE``.
    _ai(
        "Recording a tentative skip on the increment property to exercise the tool.",
        _tc(
            "record_skip",
            property_title=_PROP_TITLE,
            reason="Temporary skip — will be undone on the next turn to exercise unskip.",
        ),
    ),

    # A11 — exercise unskip_property. The empty-reason sentinel inside
    # _merge_skips then filters the entry out of state["skipped"], so the
    # final skipped list going into feedback_tool is []. Important: the
    # feedback digest includes skipped — changing skipped between a passing
    # feedback verdict and publish would invalidate the digest.
    _ai(
        "Undoing the tentative skip.",
        _tc("unskip_property", property_title=_PROP_TITLE),
    ),

    # A12 — first feedback_tool invocation. Spawns the feedback judge
    # sub-agent (J1..J3). The judge returns good=False here, which leaves
    # validations["feedback"] UNSET (digest is only stamped on good=True).
    # ``rebuttals`` defaults to []; no need to pass it explicitly.
    _ai(
        "Seeking judge feedback on the current spec.",
        _tc("feedback_tool"),
    ),

    # J1 — feedback judge, first invocation, turn 1.
    # Tools available: write_rough_draft, read_rough_draft, memory, get_cvl,
    # env.all_tools (source + rag), result. Validator `did_rough_draft_read`
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
    # validations["feedback"] = digest(curr_spec, skipped). publish's
    # ``check_completion`` then passes.
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

    # A15 — publish. ``check_completion`` sees validations["feedback"] ==
    # digest and dispatches; ``PublishTool`` runs ``certoraTypeCheck.py``
    # in-process against the current stub+spec. There's no merge sub-agent
    # anymore — the typechecker is the only validator and the result is
    # the commentary string. ``suggested_spec_name`` (without the ``.spec``
    # extension) is required.
    _ai(
        "Publishing the approved spec.",
        _tc(
            "publish",
            commentary=(
                "Formalized the Increment component's 'increment always succeeds' "
                "property as a single rule with an annotated assertion."
            ),
            suggested_spec_name="counter_increment",
        ),
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


def install_harness_tape() -> _NatspecFakeLLM:
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
    "UPDATED_STUB",
    "VALID_CVL",
    "get_counter_llm",
    "install_harness_tape",
]
