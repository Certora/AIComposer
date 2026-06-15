"""
AUTO-GENERATED fake-LLM tape for the 'autoprove_Answer' AutoProve scenario.

Recorded by composer.testing.record_tape from a real run. Each lane is the
ordered list of AIMessage responses for one run_task task_id; HarnessFakeLLM
replays one per llm.ainvoke. This is a faithful, runnable starting point —
edit freely (add comments, hoist artifacts) the way ui_harness_autoprove_Counter.py is
hand-curated.

Replay with the SAME CLI flags used to record:

    COMPOSER_TEST_TAPE=autoprove_Answer console-autoprove <project> <Contract.sol:Contract> \
        <system.md> --max-bug-rounds 1 [--interactive]

Lanes captured: system-analysis(4), harness(4), bug-0-Answer(3), invariants(9), cvl-0-Answer(7)
"""

from typing import Any
import uuid

from composer.testing.harness_tape import HarnessFakeLLM
from composer.spec.source.task_ids import (
    SYSTEM_ANALYSIS_TASK_ID, HARNESS_TASK_ID, INVARIANTS_TASK_ID,
    bug_analysis_task_id, cvl_gen_task_id,
)

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import ToolCall


def _tc(name: str, **args: Any) -> ToolCall:
    """Tool-call dict with a unique id (LangGraph binds tool responses back to
    calls by id, so every entry needs its own)."""
    return {
        "id": f"toolu_{uuid.uuid4().hex[:20]}",
        "name": name,
        "args": args,
        "type": "tool_call",
    }


def _ai(text: str = "", *tool_calls: ToolCall) -> AIMessage:
    """Tape entry: optional text + zero or more tool_calls."""
    content: list[str | dict] = []
    if text:
        content.append(text)
    content.extend(
        {"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["args"]}
        for t in tool_calls
    )
    return AIMessage(content=content, tool_calls=list(tool_calls))


# Hoisted string artifacts (CVL specs, long messages).
_T0 = """\


Let me start by checking my memory for any prior progress on this task."""

_T1 = """\

Application Type: Smoke Test Contract
Description: A deliberately trivial smoke-test system consisting of a single contract with a single pure function. It is designed to exercise an auto-prove pipeline end-to-end on the smallest possible input.

Components:
1. Explicit Contract: Answer (singleton)
   - Path: src/Answer.sol
   - Description: A stateless singleton contract that exposes a single pure function returning the constant value 42.
   - Component: "Answer"
     - Description: Provides a single external entry point that always returns the constant value 42.
     - Entry points: theAnswer()
     - State variables: none
     - Interactions: none
     - Requirements: theAnswer() always returns 42.

No external actors.
"""

_T2 = 'The draft is accurate and aligns with both the design document and the source code. Let me submit the final result.'

_T3 = 'A deliberately trivial smoke-test system consisting of a single contract with a single pure function. It is designed to exercise an auto-prove pipeline end-to-end on the smallest possible input, not to demonstrate creative specification work.'

_T4 = """\


I'll start by checking my memory directory for any earlier progress, then work through the task systematically."""

_T5 = 'No earlier progress. Let me start fresh. First, let me examine the source code to confirm the structure.'

_T6 = 'This is indeed a trivially simple contract. Let me now work through all steps and record my findings.'

_T7 = 'The Answer contract is completely stateless — it has no storage variables, no primary entity, and no managed state of any kind. It consists of a single pure function theAnswer() that always returns the constant 42. Since there is no state, the concept of a "non-trivial state" is vacuous/degenerate. A single instance of Answer with no additional setup fully captures all possible behaviors of the contract.'

_T8 = """\


I'll start by examining the implementation of the Answer contract."""

_T9 = """\

Properties for Answer contract:

1. **theAnswer_returns_42** (safety_property)
   - Methods: theAnswer()
   - Description: theAnswer() must always return exactly 42.
   - Rationale: This is the sole requirement for the contract. A correct implementation must satisfy this.
"""

_T10 = 'The contract is trivial: a single pure function with no state, no interactions, no parameters. The only meaningful property is the one stated requirement — that `theAnswer()` returns 42. There are no state variables to form invariants over, no access control, no external calls, no arithmetic that could overflow, no oracle interactions, no reentrancy surface, no rounding. Proposing anything beyond the core requirement would be padding.'

_T11 = 'The Answer contract contains a single external pure function `theAnswer()` that takes no arguments and returns a uint256. There is no state, no constructor logic, no modifiers, no external calls, no other functions, and no inheritance. The only requirement is that `theAnswer()` returns 42. This is the sole meaningful, verifiable property. I considered whether there were any invariants to state (there are none — no state variables exist), any attack vectors (there are none — no state mutations, no external interactions, no parameters to manipulate), or any other safety properties (there are none — pure function with a single behavior). Returning a single property is the honest, correct result for this trivially simple contract.'

_T12 = """\


I'll start by checking my memory directory for any earlier progress, then examine the contract."""

_T13 = """\
This is an extremely simple contract. Let me analyze it:

- **No state variables** — the contract has zero storage slots
- **One function** — `theAnswer()` which is `pure` (no state reads or writes)
- **No access control** — no owner, no roles
- **No data structures** — no mappings, arrays, or structs

For a stateless contract with a single pure function, the only structural invariant one could state is about the absence of state — but that's inherent to the Solidity compilation (a contract with no state variables has nothing in storage).

However, there is one meaningful invariant: the function always returns 42. But that's more of a functional correctness property than a structural invariant. Let me think about what could qualify as structural here...

The only thing that could serve as a structural invariant for a stateless contract is that all storage slots are zero (since no state variables are declared). Let me try this."""

_T14 = """\


I'll start by checking my memory for any earlier progress on this task."""

_T15 = 'The contract is extremely simple - it has no state variables, no constructor that writes storage, and only a single `pure` function that returns 42. Let me save my findings and draft my feedback.'

_T16 = """\
Now let me think more carefully about formalizability. The invariant says "all storage slots are zero." In CVL, you can write invariants about specific state variables, but the contract has none. You could potentially use a `hook` on `Sstore` to track writes, but quantifying over all 2^256 storage slots is not standard CVL practice.

However, there's a subtlety: since there are no state variables, there's literally nothing to assert about. The invariant is vacuously true but also vacuously unformalizable in a meaningful way in CVL.

Let me also consider: is this really a useful structural invariant? Even if it's true, it doesn't help prove any other property since the contract has no meaningful state."""

_T17 = """\
The invariant "all_storage_zero" states that all storage slots of the Answer contract are zero since the contract declares no state variables and no function modifies storage.

While this statement is factually correct about the contract's behavior, it has significant formalizability issues:

1. **Formalizability**: The invariant quantifies over "all storage slots," which means all 2^256 possible storage locations. CVL invariants typically operate on named state variables, but this contract declares none. There is no straightforward way to express "all storage slots are zero" as a CVL invariant. You could use a `hook Sstore` with a ghost variable to detect any storage write, but even that doesn't directly verify that pre-existing storage is zero — it only checks that no writes occur. The universal quantification over an unbounded storage space is not directly expressible in standard CVL.

2. **Structural value**: While technically true, this invariant is vacuously structural. The contract has no state, so there are no other security/safety properties that would depend on this invariant. It provides no practical value as a "structural invariant" for ruling out infeasible starting states.

Recommendation: Reject on formalizability grounds. The invariant cannot be meaningfully formalized as a CVL specification because (a) universal quantification over all 2^256 storage slots is not expressible in CVL, and (b) there are no named state variables to write an invariant about."""

_T18 = """\
The draft looks accurate. Let me review against the criteria once more:

- **Plausibility**: It's true. ✓
- **Formalizability**: Universal quantification over all 2^256 storage slots is not expressible in CVL. No named state variables exist to write an invariant about. This is a significant issue. ✗
- **Inductive**: Trivially yes. ✓
- **Structural**: Vacuously true but not meaningfully structural. Borderline, but the formalizability issue alone is sufficient.

The formalizability concern is strong enough (well above 60% confidence) to reject."""

_T19 = """\
The invariant "all_storage_zero" states that all storage slots of the Answer contract are zero since the contract declares no state variables and no function modifies storage.

While this statement is factually correct about the contract's behavior, it has significant formalizability issues:

1. **Formalizability**: The invariant quantifies over "all storage slots," which means all 2^256 possible storage locations. CVL invariants typically operate on named state variables, but this contract declares none. There is no straightforward way to express "all storage slots are zero" as a CVL invariant. You could use a `hook Sstore` with a ghost variable to detect any storage write, but even that doesn't directly verify that pre-existing storage is zero — it only checks that no writes occur. The universal quantification over an unbounded storage space is not directly expressible in standard CVL.

2. **Structural value**: While technically true, this invariant is vacuously structural. The contract has no state, so there are no other security/safety properties that would depend on this invariant. It provides no practical value as a "structural invariant" for ruling out infeasible starting states.

**Suggestion**: Since the Answer contract has no state variables and only a single `pure` function, there are no meaningful structural invariants to formulate for it. If the goal is to verify the contract's behavior, a simple rule asserting that `theAnswer()` returns 42 would be more appropriate and directly formalizable."""

_T20 = """\
The feedback confirms what I suspected — this contract is so trivial that there are no meaningful structural invariants to formulate. It has no state variables, no storage, and only a pure function.

Let me save my analysis and submit the result."""

_T21 = """\
Given that the Answer contract is completely stateless with a single pure function, there are genuinely no structural invariants to identify. Structural invariants relate to relationships between state variables, storage layout consistency, and data structure well-formedness — none of which exist in this contract.

I'll submit an empty invariant set, as this is the correct answer for a stateless contract."""

_T22 = """\
import "summaries/Answer_base_summaries.spec";

methods {
    function theAnswer() external returns (uint256) envfree;
}

/// @title theAnswer_returns_42
/// Property: theAnswer() must always return exactly 42.
rule theAnswer_returns_42 {
    mathint result = theAnswer();
    assert result == 42, "theAnswer() must return 42";
}
"""

_T23 = 'The spec declares theAnswer() envfree and asserts it returns exactly 42 via rule theAnswer_returns_42 — it faithfully encodes the sole extracted property. Coverage is complete. Verdict: GOOD.'

_T24 = 'Formalized the single safety property: theAnswer() always returns 42. The rule verifies against Answer.sol.'

_TAPE: dict[str, list[BaseMessage]] = {
    # lane: system-analysis (4 entries)
    SYSTEM_ANALYSIS_TASK_ID: [
        _ai(
            _T0,
        ),
        _ai(
            'No prior progress. Let me read the source file directly.',
            _tc(
                'get_file',
                path='src/Answer.sol',
            ),
        ),
        _ai(
            'The implementation matches the design document exactly. Let me draft the result.',
            _tc(
                'write_rough_draft',
                rough_draft=_T1,
            ),
        ),
        _ai(
            _T2,
            _tc(
                'result',
                application_type='Smoke Test Contract',
                description=_T3,
                components=[{'sort': 'singleton',
  'name': 'Answer',
  'description': 'A stateless singleton contract that exposes a single pure function '
                 'returning the constant value 42. It holds no state and has no '
                 'interactions with any other contracts or external actors.',
  'path': 'src/Answer.sol',
  'components': [{'name': 'Answer',
                  'description': 'Provides a single external entry point, theAnswer(), '
                                 'which is a pure function that always returns the '
                                 'constant value 42. The component holds no state and '
                                 'has no interactions with other contracts or actors.',
                  'external_entry_points': ['theAnswer()'],
                  'state_variables': [],
                  'interactions': [],
                  'requirements': ['theAnswer() always returns 42.']}]}],
            ),
        ),
    ],
    # lane: harness (4 entries)
    HARNESS_TASK_ID: [
        _ai(
            _T4,
        ),
        _ai(
            _T5,
            _tc('list_files'),
        ),
        _ai(
            _T6,
        ),
        _ai(
            "Now I'll output the final results:",
            _tc(
                'result',
                erc20_contracts=[],
                external_interfaces=[],
                non_trivial_state=_T7,
                transitive_closure=[{'name': 'Answer', 'link_fields': [], 'num_instances': None}],
            ),
        ),
    ],
    # lane: bug-0-Answer (3 entries)
    bug_analysis_task_id(0, "Answer"): [
        _ai(
            _T8,
            _tc('list_files'),
        ),
        _ai(
            'The contract is extremely simple. Let me draft my properties.',
            _tc(
                'write_rough_draft',
                rough_draft=_T9,
            ),
        ),
        _ai(
            _T10,
            _tc(
                'result',
                items=[{'title': 'theAnswer_returns_42',
  'methods': ['theAnswer()'],
  'sort': 'safety_property',
  'description': 'theAnswer() must always return exactly 42.'}],
                reasoning=_T11,
            ),
        ),
    ],
    # lane: invariants (9 entries)
    INVARIANTS_TASK_ID: [
        _ai(
            _T12,
            _tc(
                'get_file',
                path='src/Answer.sol',
            ),
        ),
        _ai(
            _T13,
            _tc(
                'invariant_feedback',
                inv={'name': 'all_storage_zero',
 'description': 'All storage slots of the Answer contract are zero, since the contract '
                'declares no state variables and no function modifies storage.'},
            ),
        ),
        _ai(
            _T14,
        ),
        _ai(
            'No prior memory. Let me examine the Answer contract to understand its structure.',
            _tc(
                'get_file',
                path='src/Answer.sol',
            ),
            _tc('list_files'),
        ),
        _ai(
            _T15,
        ),
        _ai(
            _T16,
            _tc(
                'write_rough_draft',
                rough_draft=_T17,
            ),
        ),
        _ai(
            _T18,
            _tc(
                'result',
                sort='NOT_FORMAL',
                explanation=_T19,
            ),
        ),
        _ai(
            _T20,
        ),
        _ai(
            _T21,
            _tc(
                'result',
                inv=[],
            ),
        ),
    ],
    # lane: cvl-0-Answer (7 entries)
    cvl_gen_task_id(0, "Answer"): [
        _ai(
            'Writing the component spec for the sole property.',
            _tc(
                'put_cvl_raw',
                cvl_file=_T22,
            ),
        ),
        _ai(
            'Requesting judge feedback on the spec.',
            _tc('feedback_tool'),
        ),
        _ai(
            'Judge: inspecting the spec and drafting a verdict.',
            _tc('get_cvl'),
            _tc(
                'write_rough_draft',
                rough_draft=_T23,
            ),
        ),
        _ai(
            'Judge: reading the draft before the verdict.',
            _tc('read_rough_draft'),
        ),
        _ai(
            'Judge: approving the spec.',
            _tc(
                'result',
                good=True,
                feedback='',
            ),
        ),
        _ai(
            'Feedback is positive. Running the prover.',
            _tc(
                'verify_spec',
                rules=None,
            ),
        ),
        _ai(
            'All rules verified. Finalizing the component CVL.',
            _tc(
                'result',
                commentary=_T24,
                property_rules=[{'property_title': 'theAnswer_returns_42', 'rules': ['theAnswer_returns_42']}],
            ),
        ),
    ],
}


def get_autoprove_Answer_llm() -> HarnessFakeLLM:
    """Return a fresh fake LLM loaded with the 'autoprove_Answer' tape."""
    return HarnessFakeLLM(lanes=_TAPE)


def install_harness_tape() -> HarnessFakeLLM:
    """Monkeypatch create_llm / create_llm_base so the pipeline receives the
    fake. Call before importing the autoprove entry path (composer/bind.py
    does this when COMPOSER_TEST_TAPE=autoprove_Answer is set)."""
    fake = get_autoprove_Answer_llm()
    import composer.spec.agent_index as a_ind
    a_ind._UNSAFE_DISABLE_CACHE = True
    import composer.workflow.services as services
    services.create_llm = lambda args: fake  # type: ignore[assignment]
    services.create_llm_base = lambda args: fake  # type: ignore[assignment]
    return fake


__all__ = ["get_autoprove_Answer_llm", "install_harness_tape"]
