"""
Fake-LLM end-to-end UI harness for ``tui_main.py``.

This module substitutes the real ``ChatAnthropic`` built by
``composer.workflow.services.create_llm`` with a
``FakeMessagesListChatModel`` preloaded with a hand-authored tape of
responses. The rest of ``tui_main.py`` runs normally — TUI, real tool
execution (prover, RAG, HITL), workflow graphs, checkpointing, IDE
bridge — so UI rendering and tool-dispatch paths are exercised against
canned responses without spending Anthropic API credits.

Scenario inputs and wiring instructions live under
``composer/testing/scenarios/vault/``.

The tape is a single linear list of ``AIMessage`` s that is popped in
order on every call the codegen workflow makes to the LLM, across
every graph:

    natreq extractor  →  main codegen agent  →  CVL research sub-agent
        (when called)  →  judge sub-agent (each invocation)  →  final
        ``create_resume_commentary`` structured-output call.

It is the author's responsibility to keep the tape's global ordering in
sync with the real dispatch sequence. When the trace drifts, edit the
tape — it is cheap.

Every HITL tool call in the tape embeds its expected human response
directly inside the AI-authored argument field (usually ``question``,
``explanation``, or ``context``), tagged ``[TAPE EXPECTATION: ...]``.
When the TUI pauses for human input, read the bracket and respond
accordingly.
"""

from typing import Any, override, Sequence, Callable
from langchain_core.tools import BaseTool
import uuid

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage


# ---------------------------------------------------------------------------
# Fake LLM plumbing
# ---------------------------------------------------------------------------


class _CodegenFakeLLM(FakeMessagesListChatModel):
    """``FakeMessagesListChatModel`` tolerant of the specific shape of attribute
    access the codegen workflow performs on the bound LLM.

    Two compatibility shims:

    * ``thinking`` — ``composer.workflow.meta.create_resume_commentary``
      calls ``llm.copy(update={"thinking": None})``. Pydantic v2 tolerates
      unknown keys but prints less predictably; declaring the field makes
      the copy a no-op explicitly.
    * ``betas`` — ``composer.workflow.executor`` does
      ``getattr(llm, "betas")``. An empty list keeps the memory-tool
      beta branch off, so the main codegen agent's tool list matches
      what the tape expects.
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


def _tc(name: str, **args: Any) -> dict[str, Any]:
    """Construct a tool_call dict. The ``id`` is generated per call so
    every tape entry has a unique id (LangGraph requires this to bind
    tool responses back to their calls)."""
    return {
        "id": f"toolu_{uuid.uuid4().hex[:20]}",
        "name": name,
        "args": args,
        "type": "tool_call",
    }


def _ai(text: str = "", *tool_calls: dict[str, Any]) -> AIMessage:
    """Helper for authoring a tape entry: optional text + zero or more
    tool_calls. LangGraph's ReAct loop transitions to the tools node when
    ``tool_calls`` is non-empty, and to END otherwise."""
    return AIMessage(content=text, tool_calls=list(tool_calls))


# ---------------------------------------------------------------------------
# Buggy and fixed Solidity implementations
# ---------------------------------------------------------------------------
#
# The first ``put_file`` lands BUGGY_VAULT so the prover produces a
# genuine CEX when it runs the ``depositIncreasesBalance`` rule. Later,
# the tape corrects this via two subsequent ``put_file`` calls
# (FIXED_VAULT_V1 and FIXED_VAULT_V2). These strings are literal VFS
# contents — do not reformat them.

BUGGY_VAULT = """\
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Vault {
    mapping(address => uint256) public bal;
    uint256 public total;

    // BUG: doubles the deposit credit, so depositIncreasesBalance CEXes.
    function deposit(uint256 amount) external {
        bal[msg.sender] += amount * 2;
        total += amount;
    }

    function withdraw(uint256 amount) external {
        require(bal[msg.sender] >= amount);
        bal[msg.sender] -= amount;
        total -= amount;
    }
}
"""

# After the first CEX and propose_spec_change round, the LLM lands a
# nearly-correct implementation. One requirement is still "violated"
# per the scripted judge verdict; that's the req whose relaxation is
# REJECTED and which then motivates FIXED_VAULT_V2.
FIXED_VAULT_V1 = """\
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Vault {
    mapping(address => uint256) public bal;
    uint256 public total;

    function deposit(uint256 amount) external {
        bal[msg.sender] += amount;
        total += amount;
    }

    // Intentionally missing a no-op early-return for amount == 0 so that
    // the scripted judge verdict has something to flag on the first pass.
    function withdraw(uint256 amount) external {
        require(bal[msg.sender] >= amount);
        bal[msg.sender] -= amount;
        total -= amount;
    }
}
"""

# Final corrected implementation that satisfies every prover rule AND
# every requirement the judge will not let be relaxed.
FIXED_VAULT_V2 = """\
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Vault {
    mapping(address => uint256) public bal;
    uint256 public total;

    function deposit(uint256 amount) external {
        if (amount == 0) return;
        bal[msg.sender] += amount;
        total += amount;
    }

    function withdraw(uint256 amount) external {
        if (amount == 0) return;
        require(bal[msg.sender] >= amount);
        bal[msg.sender] -= amount;
        total -= amount;
    }
}
"""


# ---------------------------------------------------------------------------
# Spec variants for propose_spec_change / working_spec rounds
# ---------------------------------------------------------------------------
#
# These are NOT intended to be semantically meaningful spec changes —
# they're plausible-looking proposals the LLM might author to exercise
# the three propose_spec_change outcomes and the two commit_working_spec
# outcomes. REFINE and REJECTED responses discard the proposal without
# mutating the VFS; ACCEPTED replaces ``rules.spec`` in the VFS with the
# proposal's text, so the final prover run must be against the accepted
# version.

_ORIGINAL_SPEC_TEXT = """\
methods {
    function bal(address) external returns uint256 envfree;
    function total() external returns uint256 envfree;
    function deposit(uint256) external;
    function withdraw(uint256) external;
}

rule depositIncreasesBalance(uint256 amount) {
    env e;
    require e.msg.sender != 0;
    mathint before = bal(e.msg.sender);
    deposit(e, amount);
    assert bal(e.msg.sender) == assert_uint256(before + to_mathint(amount));
}

rule withdrawDecreasesBalance(uint256 amount) {
    env e;
    require e.msg.sender != 0;
    mathint before = bal(e.msg.sender);
    require before >= to_mathint(amount);
    withdraw(e, amount);
    assert bal(e.msg.sender) == assert_uint256(before - to_mathint(amount));
}
"""

# Variant 1 — the LLM's first attempt: adds an unnecessary summary for
# bal(). The expected human response is REFINE.
SPEC_PROPOSAL_V1 = _ORIGINAL_SPEC_TEXT.replace(
    "methods {",
    "methods {\n    function bal(address) external returns uint256 envfree => AUTO;",
)

# Variant 2 — LLM tries again, over-corrects and weakens the rule.
# Expected human response: REJECTED.
SPEC_PROPOSAL_V2 = _ORIGINAL_SPEC_TEXT.replace(
    "assert bal(e.msg.sender) == assert_uint256(before + to_mathint(amount));",
    "assert bal(e.msg.sender) >= before;",
)

# Variant 3 — only adds a clarifying comment. Expected: ACCEPTED.
# This text replaces rules.spec in the VFS, so the final prover run
# verifies against this version (which is semantically equivalent to the
# original — the comment is a trivially-safe edit).
SPEC_PROPOSAL_V3 = (
    "// Vault spec — annotated during fake-LLM scenario.\n" + _ORIGINAL_SPEC_TEXT
)


# Working-spec drafts. First one is rejected at commit time; second one
# is accepted.
WORKING_SPEC_DRAFT_V1 = SPEC_PROPOSAL_V3 + """
// Adds a (redundant) helper rule for commit_working_spec REJECTED path.
rule totalIsNonNegative() {
    assert total() >= 0;
}
"""

WORKING_SPEC_DRAFT_V2 = SPEC_PROPOSAL_V3 + """
// Minimal helper kept after the first commit was rejected.
"""


# ---------------------------------------------------------------------------
# The tape
# ---------------------------------------------------------------------------
#
# Global call order (section headers mark boundaries, NOT separate tapes):
#
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  A. Natreq extractor sub-graph (run by get_requirements)         │
#   │  B. Main codegen agent — initial exploration                     │
#   │  C. Main codegen — first buggy impl + prover CEX                 │
#   │  D. Main codegen calls cvl_research — sub-agent runs             │
#   │  E. Main codegen — human_in_the_loop (FOLLOWUP, then plain)      │
#   │  F. Main codegen — propose_spec_change × 3 (REFINE/REJ/ACCEPT)   │
#   │  G. Main codegen — working_spec tools (REJ then ACCEPT)          │
#   │  H. Main codegen — apply FIXED_VAULT_V1, first full prover pass  │
#   │  I. Main codegen calls requirements_evaluation — judge sub-agent │
#   │  J. Main codegen — requirements_relaxation REJECTED              │
#   │  K. Main codegen — apply FIXED_VAULT_V2, re-run prover           │
#   │  L. Main codegen — judge again, requirements_relaxation ACCEPTED │
#   │  M. Main codegen — judge final (all satisfied)                   │
#   │  N. Main codegen — code_result                                   │
#   │  O. Post: create_resume_commentary (structured output)           │
#   └──────────────────────────────────────────────────────────────────┘
#
# Turn counts (approximate, as authored; re-number if you edit):
#   A: 5   B: 2   C: 2   D: 3   E: 2   F: 3   G: 4   H: 2
#   I: 4   J: 1   K: 2   L: 5   M: 4   N: 1   O: 1
#   ── total: 41 responses ──

_VAULT_TAPE: list[BaseMessage] = [

    # ─────────────────────────────────────────────────────────────────
    # A. Natreq extractor sub-graph
    # ─────────────────────────────────────────────────────────────────
    # Runs inside get_requirements() BEFORE the main codegen graph even
    # builds. Tools available in this graph: memory, cvl_manual_search,
    # human_in_the_loop (ExtractionQuestionType), read_rough_draft,
    # write_rough_draft, result (reqs list).
    # The validator rejects the result tool unless `did_read` is true when
    # `memory` (rough-draft memory) is non-None — so the last action
    # before calling `result` must be a read_rough_draft.

    # A.1 — initial turn: look up manual context + open a rough-draft
    # note. (Parallel tool calls in one turn exercise the
    # multi-tool-per-turn rendering path.)
    _ai(
        "Starting requirements extraction.",
        _tc(
            "cvl_manual_search",
            question="What does a methods block envfree declaration mean?",
            similarity_cutoff=0.5,
            max_results=5,
            manual_section=[],
        ),
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Initial notes on the Vault requirements:\n"
                "- deposit increments bal[msg.sender] by the deposit amount\n"
                "- withdraw decrements bal[msg.sender] by the withdrawn amount (when allowed)\n"
                "- total tracks sum of balances\n"
                "- system.md is ambiguous about deposit(0) and total underflow"
            ),
        ),
    ),

    # A.2 — after tool results come back, the LLM wants a clarification
    # from the human. Only ExtractionQuestionType is supported here; the
    # tape author encodes the expected answer in the question field.
    _ai(
        "I need to clarify one ambiguity before finalizing requirements.",
        _tc(
            "human_in_the_loop",
            question=(
                "Should deposit(0) revert or silently no-op? "
                "[TAPE EXPECTATION: respond 'Silently no-op — a zero deposit is a valid but trivial call']"
            ),
            context=(
                "system.md describes deposit credit but does not explicitly address zero-amount calls. "
                "Clarifying this informs a requirement that enforces the chosen behavior."
            ),
        ),
    ),

    # A.3 — persist the clarification to the agent's file-like memory so
    # future codegen-agent runs can recall it. This exercises the
    # `memory` tool (which is the file-system-shaped MemoryBackend, not
    # the rough-draft `write_rough_draft`).
    _ai(
        "Storing the clarification for future reference.",
        _tc(
            "memory",
            command="create",
            path="/memories/vault_assumptions.md",
            file_text=(
                "# Vault assumptions recorded during requirements extraction\n\n"
                "- deposit(0) is a silent no-op per user clarification\n"
                "- total underflow is prevented by Solidity 0.8 checked arithmetic\n"
            ),
        ),
    ),

    # A.4 — read back the rough draft (flips `did_read` to True so the
    # result tool's validator will pass).
    _ai(
        "Re-reading rough draft before finalizing.",
        _tc("read_rough_draft"),
    ),

    # A.5 — emit the final requirements list. Tool name is "result"
    # (the decorator default inside result_tool_generator), and the list
    # payload goes through the auto-generated "value" field because the
    # schema was built from `(list[str], "...")`.
    _ai(
        "Requirements finalized.",
        _tc(
            "result",
            value=[
                "deposit(amount) must credit bal[msg.sender] by exactly amount and add amount to total",
                "withdraw(amount) must debit bal[msg.sender] by exactly amount and subtract amount from total when amount <= bal[msg.sender]",
                "withdraw must revert when amount > bal[msg.sender]",
                "deposit(0) must be a silent no-op",
                "total must always equal the sum of bal[x] over all x",
            ],
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # B. Main codegen — initial exploration
    # ─────────────────────────────────────────────────────────────────
    # From here until section O, every tape entry is produced by the
    # main codegen agent graph (except the sub-agent slices D/I/L/M which
    # pop while control is inside a sub-graph launched by a tool call).

    # B.1 — survey the initial VFS. Parallel calls to list_files and
    # grep_files exercise multi-tool rendering on the codegen UI.
    _ai(
        "Surveying the initial VFS to plan the implementation.",
        _tc("list_files"),
        _tc(
            "grep_files",
            search_string="deposit",
            matching_lines=True,
            match_in=None,
        ),
    ),

    # B.2 — read the spec in full so the rendering exercises get_file.
    _ai(
        "Reading the spec to identify rules.",
        _tc("get_file", path="rules.spec"),
    ),

    # ─────────────────────────────────────────────────────────────────
    # C. First buggy implementation + prover CEX
    # ─────────────────────────────────────────────────────────────────

    # C.1 — write an implementation whose deposit function is broken
    # (credits 2*amount). The prover WILL CEX this on the
    # depositIncreasesBalance rule. This exercises the prover failure
    # rendering path (collapsible pane, spinner, CEX trace).
    _ai(
        "Drafting an initial Vault implementation to test against the spec.",
        _tc(
            "put_file",
            files={"src/Vault.sol": BUGGY_VAULT},
        ),
    ),

    # C.2 — focus the prover on the specific rule we know is broken so
    # the CEX output is small and readable.
    _ai(
        "Running the prover against the first rule.",
        _tc(
            "certora_prover",
            source_files=["src/Vault.sol"],
            target_contract="Vault",
            compiler_version="solc8.29",
            loop_iter=1,
            rule="depositIncreasesBalance",
            use_working_spec=False,
        ),
    ),

    _ai(
        "Your implementation incorrectly doubles the amount on deposit"
    ),

    # ─────────────────────────────────────────────────────────────────
    # D. Sub-agent: cvl_research (exercises the research sub-graph)
    # ─────────────────────────────────────────────────────────────────
    # The next main-codegen turn issues a cvl_research tool call. That
    # tool launches the CVL research sub-graph, which then pops D.1/D.2/D.3
    # from this tape BEFORE the main agent's next turn (E.1) is popped.

    # Main codegen — dispatch to research.
    _ai(
        "Consulting the CVL manual and the research sub-agent to understand "
        "the counterexample.",
        _tc(
            "cvl_manual_search",
            question="How is a CVL counterexample interpreted for arithmetic rules?",
            similarity_cutoff=0.5,
            max_results=5,
            manual_section=[],
        ),
        _tc(
            "cvl_research",
            question=(
                "What preconditions does CVL assume about env.msg.sender "
                "when it is not explicitly required?"
            ),
        ),
    ),

    # D.1 — CVL researcher sub-agent turn 1: sketch a rough draft and
    # use keyword search + section fetch in parallel.
    _ai(
        "Starting research on env.msg.sender semantics.",
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Hypothesis: env.msg.sender is a universally-quantified "
                "address unless constrained. Need to confirm from the manual."
            ),
        ),
        _tc(
            "cvl_keyword_search",
            query="env sender",
            min_depth=0,
            limit=5,
        ),
    ),

    # D.2 — read the rough draft so the result validator passes.
    _ai(
        "Reading the draft back before answering.",
        _tc("read_rough_draft"),
    ),

    # D.3 — deliver the research answer via the sub-graph's result tool.
    _ai(
        "Answer ready.",
        _tc(
            "result",
            value=(
                "env.msg.sender is universally quantified across rule inputs "
                "unless the rule issues `require e.msg.sender == X` or similar. "
                "Since the depositIncreasesBalance rule does not constrain the "
                "sender beyond non-zero, any counterexample that assumes an "
                "arbitrary sender is admissible."
            ),
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # E. Main codegen — human_in_the_loop, both outcomes
    # ─────────────────────────────────────────────────────────────────

    # E.1 — ask a question whose expected answer is a FOLLOWUP (exercises
    # the re-pose branch in question.py's docstring).
    _ai(
        "Asking the user for confirmation on the arithmetic model assumption.",
        _tc(
            "human_in_the_loop",
            question=(
                "The counterexample suggests our deposit formula is off by a factor. "
                "Should I assume solc 0.8 checked arithmetic for overflow protection? "
                "[TAPE EXPECTATION: respond 'FOLLOWUP: which specific overflow are you worried about?']"
            ),
            context=(
                "The depositIncreasesBalance rule CEX shows bal differs from before+amount. "
                "I suspect my implementation doubled the amount, but want to rule out "
                "compiler-level subtleties before rewriting."
            ),
            code=None,
        ),
    ),

    # E.2 — rephrase and re-pose per the FOLLOWUP convention. Expected
    # answer is a plain string (no FOLLOWUP prefix).
    _ai(
        "Refining the question per the user's follow-up.",
        _tc(
            "human_in_the_loop",
            question=(
                "Concretely: will `bal[msg.sender] += amount` overflow silently or revert, "
                "given pragma ^0.8.0? "
                "[TAPE EXPECTATION: respond 'Revert — solc 0.8 checked arithmetic on += overflow']"
            ),
            context=(
                "Narrowing the earlier question: the concern is specifically the += "
                "operator on a uint256 with a user-supplied operand."
            ),
            code=None,
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # F. propose_spec_change — exercise REFINE, REJECTED, ACCEPTED
    # ─────────────────────────────────────────────────────────────────

    # F.1 — first proposal: adds an unnecessary summary for bal(). Expected
    # response is REFINE, so the tool rejects but keeps the door open.
    _ai(
        "Proposing a spec change that adds a summary for bal().",
        _tc(
            "propose_spec_change",
            proposed_spec=SPEC_PROPOSAL_V1,
            explanation=(
                "Adding `=> AUTO` summary for bal() to improve prover performance on "
                "large call graphs. "
                "[TAPE EXPECTATION: respond 'REFINE: the AUTO summary is unnecessary for envfree "
                "getters; drop it and resubmit']"
            ),
        ),
    ),

    # F.2 — second proposal: over-corrected, weakens the rule. Expected
    # REJECTED.
    _ai(
        "Revised proposal: weaker assertion on deposit result.",
        _tc(
            "propose_spec_change",
            proposed_spec=SPEC_PROPOSAL_V2,
            explanation=(
                "Weakening depositIncreasesBalance to '>= before' to avoid overflow "
                "concerns. "
                "[TAPE EXPECTATION: respond 'REJECTED: weakening the postcondition defeats the "
                "rule; keep the exact-equality form']"
            ),
        ),
    ),

    # F.3 — third proposal: harmless comment only. Expected ACCEPTED —
    # this WILL replace rules.spec in the VFS, so the final prover run
    # must target this version (which is semantically identical to the
    # original minus the leading comment).
    _ai(
        "Final proposal: comment-only annotation.",
        _tc(
            "propose_spec_change",
            proposed_spec=SPEC_PROPOSAL_V3,
            explanation=(
                "Just adding a leading comment to mark the spec as annotated during "
                "this session. No semantic change. "
                "[TAPE EXPECTATION: respond 'ACCEPTED']"
            ),
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # G. Working spec tools — exercise REJECTED and ACCEPTED
    # ─────────────────────────────────────────────────────────────────

    # G.1 — draft a working spec that adds a redundant rule.
    _ai(
        "Drafting a working spec with an extra helper rule.",
        _tc(
            "write_working_spec",
            new_cvl=WORKING_SPEC_DRAFT_V1,
        ),
    ),

    # G.2 — read back the working spec, then run the prover against it.
    # certora_prover with use_working_spec=True does NOT set
    # validation[prover] (even on full success), so this is safe to
    # interleave.
    _ai(
        "Reading back and probing the working spec with the prover.",
        _tc("read_working_spec"),
        _tc(
            "certora_prover",
            source_files=["src/Vault.sol"],
            target_contract="Vault",
            compiler_version="solc8.29",
            loop_iter=1,
            rule="depositIncreasesBalance",
            use_working_spec=True,
        ),
    ),

    # G.3 — ask to commit. Expected human response is a rejection string.
    # commit_working_spec treats anything not starting with "ACCEPTED" as
    # returned-to-LLM feedback, so the working spec stays in state.
    _ai(
        "Asking to commit the working spec.",
        _tc(
            "commit_working_spec",
            explanation=(
                "Committing the working spec with the extra helper rule. "
                "[TAPE EXPECTATION: respond 'REJECTED: the extra rule is redundant "
                "and blocks simpler proofs; drop it and resubmit']"
            ),
        ),
    ),

    # G.4 — revise the working spec (drop the redundant rule) and try
    # commit again. Expected response ACCEPTED.
    #
    # KNOWN ISSUE: ``composer/tools/working_spec.py`` line 74 builds the
    # state update as ``vfs={"rules.spec", work_spec}`` which is a SET
    # literal, not a dict — so the ACCEPTED path may error at state
    # application. Flag this to the user if it surfaces during execution.
    _ai(
        "Replacing the working spec with a minimal version, then committing.",
        _tc(
            "write_working_spec",
            new_cvl=WORKING_SPEC_DRAFT_V2,
        ),
        _tc(
            "commit_working_spec",
            explanation=(
                "Committing the minimal helper-free working spec. "
                "[TAPE EXPECTATION: respond 'ACCEPTED']"
            ),
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # H. Apply FIXED_VAULT_V1, first full prover pass
    # ─────────────────────────────────────────────────────────────────

    # H.1 — write the corrected implementation. This is the FINAL VFS
    # mutation before the run's validation-gated tail; after this, only
    # the prover / judge / code_result dance (which either reads state or
    # mutates non-VFS fields) should run until FIXED_VAULT_V2 in step K.
    _ai(
        "Writing the corrected Vault implementation.",
        _tc(
            "put_file",
            files={"src/Vault.sol": FIXED_VAULT_V1},
        ),
    ),

    # H.2 — full prover run with no rule filter and no working-spec
    # override. Per composer/tools/prover.py, this is the exact condition
    # that sets ``validation[prover] = digest(VFS)`` when ALL rules pass.
    _ai(
        "Running the prover against the full rule set.",
        _tc(
            "certora_prover",
            source_files=["src/Vault.sol"],
            target_contract="Vault",
            compiler_version="solc8.29",
            loop_iter=1,
            rule=None,
            use_working_spec=False,
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # I. Judge sub-agent — first invocation
    # ─────────────────────────────────────────────────────────────────
    # The main agent calls requirements_evaluation. That tool spins up
    # the judge sub-graph (which has its own rough_draft + memory + vfs
    # + result tools) and runs it to completion, popping I.1..I.4 from
    # this tape. The return is a formatted string showing per-req
    # classification; the "LLM reacts" turn (J.1) comes after.

    _ai(
        "Invoking the requirements judge to evaluate the implementation.",
        _tc("requirements_evaluation"),
    ),

    # I.1 — judge sub-agent: grab the code, drop a rough draft, and
    # record some notes in sub-agent memory in one turn.
    _ai(
        "Judge: gathering code and writing initial analysis.",
        _tc("get_file", path="src/Vault.sol"),
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Judge analysis pass 1:\n"
                "- deposit and withdraw arithmetic match reqs 1–3\n"
                "- total invariant req 5 looks SATISFIED given the checked arithmetic\n"
                "- deposit(0) no-op req 4 is VIOLATED — impl does not early-return on zero"
            ),
        ),
        _tc(
            "memory",
            command="create",
            path="/memories/judge_pass1.md",
            file_text="Pass 1: 4 of 5 reqs satisfied; req 4 violated (no zero-amount short-circuit).",
        ),
    ),

    # I.2 — judge: read draft back.
    _ai(
        "Judge: reading draft before returning a verdict.",
        _tc("read_rough_draft"),
    ),

    # I.3 — judge result. Requirement text MUST match the natreq result
    # output (see A.5); any mismatch raises the sub-graph's validator.
    _ai(
        "Judge: delivering verdict.",
        _tc(
            "result",
            judgement_result=[
                {
                    "requirement_number": 1,
                    "requirement": "deposit(amount) must credit bal[msg.sender] by exactly amount and add amount to total",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 2,
                    "requirement": "withdraw(amount) must debit bal[msg.sender] by exactly amount and subtract amount from total when amount <= bal[msg.sender]",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 3,
                    "requirement": "withdraw must revert when amount > bal[msg.sender]",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 4,
                    "requirement": "deposit(0) must be a silent no-op",
                    "classification": "VIOLATED",
                    "commentary": (
                        "The implementation has no early-return for amount == 0; while the "
                        "operations remain total-preserving for a zero amount, the req asks "
                        "for an explicit no-op path."
                    ),
                },
                {
                    "requirement_number": 5,
                    "requirement": "total must always equal the sum of bal[x] over all x",
                    "classification": "LIKELY",
                    "commentary": None,
                },
            ],
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # J. requirements_relaxation — REJECTED outcome
    # ─────────────────────────────────────────────────────────────────

    # J.1 — argue that req 4 should be relaxed. Expected REJECTED so the
    # LLM is forced to go fix the code.
    _ai(
        "Attempting to argue req 4 (deposit(0) no-op) is too strict.",
        _tc(
            "requirement_relaxation_request",
            req_number=4,
            req_text="deposit(0) must be a silent no-op",
            judgment=(
                "The implementation has no early-return for amount == 0; while the "
                "operations remain total-preserving for a zero amount, the req asks "
                "for an explicit no-op path."
            ),
            context=(
                "The implementation remains observably total-preserving for amount == 0 "
                "because both bal[x] += 0 and total += 0 are no-ops."
            ),
            explanation=(
                "A zero-amount deposit is already a no-op in practice; adding an explicit "
                "early-return is cosmetic. "
                "[TAPE EXPECTATION: respond 'REJECTED: the requirement is explicit about "
                "a no-op short-circuit; please implement it rather than rely on arithmetic']"
            ),
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # K. Apply FIXED_VAULT_V2, re-run prover
    # ─────────────────────────────────────────────────────────────────

    # K.1 — fix the impl per the relaxation rejection.
    _ai(
        "Addressing the rejected relaxation by adding the explicit zero-amount early-return.",
        _tc(
            "put_file",
            files={"src/Vault.sol": FIXED_VAULT_V2},
        ),
    ),

    # K.2 — re-run the prover in full-verify mode. Resets
    # validation[prover] to the digest of the newly-written VFS.
    _ai(
        "Re-running the prover against the fixed implementation.",
        _tc(
            "certora_prover",
            source_files=["src/Vault.sol"],
            target_contract="Vault",
            compiler_version="solc8.29",
            loop_iter=1,
            rule=None,
            use_working_spec=False,
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # L. Judge second pass + requirements_relaxation ACCEPTED
    # ─────────────────────────────────────────────────────────────────

    # L.1 — re-invoke the judge. Will produce a VIOLATED on req 5 (the
    # LIKELY-under-uncertainty requirement) this pass, so the tape can
    # exercise the ACCEPTED branch of requirements_relaxation.
    _ai(
        "Re-running the judge after the fix.",
        _tc("requirements_evaluation"),
    ),

    # L.2 — judge second invocation, turn 1.
    _ai(
        "Judge (pass 2): analysis setup.",
        _tc("get_file", path="src/Vault.sol"),
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Judge pass 2: req 4 now addressed by the explicit amount==0 early-return. "
                "On re-examination, req 5 (total == sum of bal) requires a ghost-variable "
                "proof that isn't present in the current spec, so classify as VIOLATED and "
                "let the human decide whether to relax."
            ),
        ),
    ),

    # L.3 — judge pass 2: read draft.
    _ai(
        "Judge (pass 2): reading draft before verdict.",
        _tc("read_rough_draft"),
    ),

    # L.4 — judge pass 2 result.
    _ai(
        "Judge (pass 2): verdict.",
        _tc(
            "result",
            judgement_result=[
                {
                    "requirement_number": 1,
                    "requirement": "deposit(amount) must credit bal[msg.sender] by exactly amount and add amount to total",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 2,
                    "requirement": "withdraw(amount) must debit bal[msg.sender] by exactly amount and subtract amount from total when amount <= bal[msg.sender]",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 3,
                    "requirement": "withdraw must revert when amount > bal[msg.sender]",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 4,
                    "requirement": "deposit(0) must be a silent no-op",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 5,
                    "requirement": "total must always equal the sum of bal[x] over all x",
                    "classification": "VIOLATED",
                    "commentary": (
                        "No ghost-variable or invariant formalizes this in the spec; the judge "
                        "cannot confirm the sum-of-balances invariant from local assertions alone."
                    ),
                },
            ],
        ),
    ),

    # L.5 — request relaxation on req 5; expected ACCEPTED so it gets
    # added to skipped_reqs and drops out of the next judge pass.
    _ai(
        "Requesting relaxation of req 5 on the grounds that formalizing it requires "
        "a ghost invariant outside the current spec scope.",
        _tc(
            "requirement_relaxation_request",
            req_number=5,
            req_text="total must always equal the sum of bal[x] over all x",
            judgment=(
                "No ghost-variable or invariant formalizes this in the spec; the judge "
                "cannot confirm the sum-of-balances invariant from local assertions alone."
            ),
            context=(
                "The implementation maintains the property by construction (every mutation to "
                "a balance is mirrored in total), but expressing it in CVL requires a ghost "
                "sum that isn't in scope for this iteration."
            ),
            explanation=(
                "The property holds by construction; formalizing it is a spec evolution task "
                "orthogonal to the implementation. "
                "[TAPE EXPECTATION: respond 'ACCEPTED']"
            ),
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # M. Judge final pass (all satisfied) — sets validation[reqs]
    # ─────────────────────────────────────────────────────────────────

    # M.1 — judge third invocation. With req 5 in skipped_reqs, this pass
    # should report it as IGNORED (the formatter marks skipped as such)
    # but the judge itself only needs every remaining classification to
    # be SATISFIED/LIKELY for validation[reqs] to be set.
    _ai(
        "Running the judge one more time to confirm all remaining requirements are satisfied.",
        _tc("requirements_evaluation"),
    ),

    # M.2 — judge pass 3: trivial analysis turn.
    _ai(
        "Judge (pass 3): confirming all non-skipped requirements are satisfied.",
        _tc(
            "write_rough_draft",
            rough_draft=(
                "Judge pass 3: reqs 1–4 are SATISFIED as before; req 5 is user-relaxed "
                "and will be marked IGNORED by the formatter."
            ),
        ),
    ),

    # M.3 — read draft.
    _ai(
        "Judge (pass 3): reading draft before verdict.",
        _tc("read_rough_draft"),
    ),

    # M.4 — judge pass 3 result. Crucially, every non-skipped
    # requirement must be SATISFIED or LIKELY for validation[reqs] to be
    # set. Req 5's classification value is immaterial since it's in
    # skipped_reqs; the judge formatter will mark it IGNORED.
    _ai(
        "Judge (pass 3): final verdict.",
        _tc(
            "result",
            judgement_result=[
                {
                    "requirement_number": 1,
                    "requirement": "deposit(amount) must credit bal[msg.sender] by exactly amount and add amount to total",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 2,
                    "requirement": "withdraw(amount) must debit bal[msg.sender] by exactly amount and subtract amount from total when amount <= bal[msg.sender]",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 3,
                    "requirement": "withdraw must revert when amount > bal[msg.sender]",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 4,
                    "requirement": "deposit(0) must be a silent no-op",
                    "classification": "SATISFIED",
                    "commentary": None,
                },
                {
                    "requirement_number": 5,
                    "requirement": "total must always equal the sum of bal[x] over all x",
                    "classification": "LIKELY",
                    "commentary": "User-relaxed; will be reported as IGNORED.",
                },
            ],
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # N. code_result — finalize the workflow
    # ─────────────────────────────────────────────────────────────────

    # N.1 — deliver the result. code_result's validator will
    # re-compute the VFS digest and match it against validation[prover]
    # and validation[reqs]; both should match because VFS has not been
    # mutated since K.1 (and since the judge sub-agent has only read).
    _ai(
        "Delivering the verified implementation.",
        _tc(
            "result",
            source=["src/Vault.sol"],
            comments=(
                "Final Vault implementation:\n"
                "- deposit/withdraw arithmetic matches spec exactly;\n"
                "- zero-amount deposits short-circuit;\n"
                "- the sum-of-balances invariant (req 5) was user-relaxed pending a "
                "ghost-variable spec evolution."
            ),
        ),
    ),

    # ─────────────────────────────────────────────────────────────────
    # O. create_resume_commentary — final structured-output call
    # ─────────────────────────────────────────────────────────────────
    # After code_result ends the graph, execute_ai_composer_workflow
    # calls create_resume_commentary, which does
    #   llm.copy(update={"thinking": None}).with_structured_output(ResumeCommentary).ainvoke(...)
    # The structured-output wrapper binds a tool whose name is the
    # schema's class name — here, "ResumeCommentary" — and parses the
    # tool_call args into the Pydantic model.

    _ai(
        "",
        _tc(
            "ResumeCommentary",
            commentary=(
                "Implemented a minimal Vault with deposit/withdraw and a sum-of-balances "
                "total. Caught and corrected an initial deposit-doubling bug via prover "
                "CEX on depositIncreasesBalance. Accepted a comment-only spec annotation "
                "and committed a simplified working spec. Relaxed the total-sum "
                "requirement with user approval pending ghost-variable spec evolution. "
                "Resume-friendly: the accepted spec diff and the FIXED_VAULT_V2 "
                "implementation are both in the VFS."
            ),
            interface_path="src/IVault.sol",
        ),
    ),
]


# ---------------------------------------------------------------------------
# Install / configuration API
# ---------------------------------------------------------------------------


def get_vault_llm() -> _CodegenFakeLLM:
    """Return a fresh fake LLM loaded with the vault tape.

    Each call returns an independent instance (the tape list is shared
    but the internal cursor ``i`` is per-instance), so tests can run
    multiple scenarios without cross-contamination.
    """
    return _CodegenFakeLLM(responses=list(_VAULT_TAPE))


def install_vault_tape() -> _CodegenFakeLLM:
    """Monkey-patch ``composer.workflow.services.create_llm`` so the
    real codegen workflow receives the fake.

    Call this once, BEFORE invoking ``tui_main.main()``. Returns the
    fake instance so the caller can inspect ``.i`` or ``.responses``
    if needed.
    """
    fake = get_vault_llm()
    import composer.workflow.services as services

    services.create_llm = lambda args: fake  # type: ignore[assignment]
    services.create_llm_base = lambda args: fake  # type: ignore[assignment]
    return fake


__all__ = [
    "BUGGY_VAULT",
    "FIXED_VAULT_V1",
    "FIXED_VAULT_V2",
    "SPEC_PROPOSAL_V1",
    "SPEC_PROPOSAL_V2",
    "SPEC_PROPOSAL_V3",
    "WORKING_SPEC_DRAFT_V1",
    "WORKING_SPEC_DRAFT_V2",
    "get_vault_llm",
    "install_vault_tape",
]
