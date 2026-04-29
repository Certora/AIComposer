"""Fake driver for Phase 3.

Drives the exact protocol surface ``run_autoprove_pipeline`` will hit:
``HandlerFactory.make_handler`` → ``TaskHandle.on_start`` →
``IOHandler.log_start`` / ``log_state_update`` / ``log_end`` (plus
nested ``log_start`` / ``log_end`` pairs to exercise the EmitTree
nesting machinery) → ``TaskHandle.on_done`` (or ``on_error``).

If anything in the wiring between the IOHandler protocol and our SSE
output is wrong, this driver surfaces it before Phase 4 introduces
real-pipeline noise.

The scenario mirrors Phase 2's mock for visual continuity:

  HARNESS → SUMMARIES → INVARIANTS → COMPONENT_ANALYSIS  (sequential)
  BUG_ANALYSIS                                            (parallel × 3)
  CVL_GEN                                                 (parallel × 3, mixed)

CVL_GEN deliberately exercises every nesting case:

  - ``Vault``  spawns a depth-1 CVL-research nested workflow
  - ``Token``  spawns nested orchestrator → verifier (depth 2)
  - ``Oracle`` raises mid-task (exercises ``on_error``)
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from composer.io.multi_job import TaskInfo
from composer.io.protocol import IOHandler
from composer.ui.autoprove_app import AutoProvePhase
from composer.web.handler import AutoProveWebHandler, Task
from composer.web.runs import RunState


# Tunable: scale all sleeps. Default keeps the demo snappy (~13s);
# bump it up if you want to inspect transitions in the browser.
TIME_SCALE = 5.0


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds * TIME_SCALE)


# ---------------------------------------------------------------------------
# Synthetic state-update payloads — real langchain message types
#
# Use the actual ``AIMessage`` / ``ToolMessage`` classes so the handler's
# ``isinstance(msg, ToolMessage)`` and ``msg.tool_calls`` accessors hit the
# same code paths the real pipeline drives. ``tool_calls`` entries match
# the dict shape langgraph emits (``{"name", "args", "id"}``); args are
# realistic for the registered ``CommonTools`` extractors so the
# grouping / display logic gets exercised honestly.
# ---------------------------------------------------------------------------

_call_counter = 0


def _next_call_id() -> str:
    global _call_counter
    _call_counter += 1
    return f"call_{_call_counter:04d}"


def _state_with_tools(node_name: str, *calls: tuple[str, dict]) -> dict:
    """Build a state update with one AIMessage carrying tool_calls.

    *calls* is a sequence of ``(tool_name, args_dict)`` pairs. Args
    must match the registered display extractor's expected keys —
    e.g. ``("get_file", {"path": "src/Vault.sol"})``."""
    return {
        node_name: {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": name, "args": args, "id": _next_call_id()}
                        for name, args in calls
                    ],
                ),
            ],
        },
    }


def _state_with_text(node_name: str, text: str) -> dict:
    """Build a state update with one AIMessage carrying plain text."""
    return {node_name: {"messages": [AIMessage(content=text)]}}


def _state_with_thinking(
    node_name: str, thinking: str, text: str | None = None,
) -> dict:
    """Build a state update with an AIMessage in extended-thinking
    shape: content is a list of typed blocks. Optional *text* tacks on
    a follow-up text block in the same message."""
    blocks: list[dict | str] = [{"type": "thinking", "thinking": thinking}]
    if text is not None:
        blocks.append({"type": "text", "text": text})
    return {node_name: {"messages": [AIMessage(content=blocks)]}}


def _state_with_result(
    node_name: str, tool_name: str, content: str,
) -> dict:
    """Build a state update with one ToolMessage."""
    return {
        node_name: {
            "messages": [
                ToolMessage(
                    content=content,
                    name=tool_name,
                    tool_call_id=_next_call_id(),
                ),
            ],
        },
    }


def _state_with_prompts(
    node_name: str,
    *,
    system_text: str | None = None,
    human_text: str | None = None,
) -> dict:
    """Build a state update carrying SystemMessage and/or HumanMessage —
    used at the start of each task to demo the prompt-block rendering
    (closed-by-default ``<details>`` for set-once context)."""
    messages: list = []
    if system_text is not None:
        messages.append(SystemMessage(content=system_text))
    if human_text is not None:
        messages.append(HumanMessage(content=human_text))
    return {node_name: {"messages": messages}}


# ---------------------------------------------------------------------------
# Single-task drivers
# ---------------------------------------------------------------------------

@dataclass
class _Nest:
    """One nested-workflow node in a task's nesting tree.

    Tree shape (instead of a flat list) because the IOHandler protocol's
    ``log_start`` / ``log_end`` pairing is genuinely nested: a child must
    be opened *and closed* while its parent is still open. A flat list
    can only express open-do-close-next-sibling, which fails as soon as
    you want a depth-2+ scenario like orchestrator-spawns-verifier.
    """
    segment: str                           # path segment relative to parent
    label: str                             # <summary> label
    events: list[dict] = field(default_factory=list)
    children: list[_Nest] = field(default_factory=list)


async def _drive_nest(
    task: IOHandler[None], parent_path: list[str], nest: _Nest, slice_dur: float,
) -> None:
    """Recursively drive one nest: open, emit events, descend into
    children, close. Pre-order on entry, post-order on close — the
    standard nested-workflow protocol."""
    sub_path = parent_path + [nest.segment]
    await task.log_start(path=sub_path, description=nest.label, tool_id=None)
    for payload in nest.events:
        await _sleep(slice_dur)
        await task.log_state_update(sub_path, payload)
    for child in nest.children:
        await _drive_nest(task, sub_path, child, slice_dur)
    await task.log_end(sub_path)


async def _drive_task(
    handler: AutoProveWebHandler,
    info: TaskInfo[AutoProvePhase],
    *,
    duration: float,
    nested: list[_Nest] | None = None,
    final_text: str = "completed",
    raise_at_end: Exception | None = None,
) -> None:
    """Drive one task end-to-end through the IOHandler protocol.

    The fake-driver contract: every IOHandler / EventHandler call goes
    through ``handle.handler`` (the per-task :class:`Task` instance the
    framework would install via ``with_handler``). The factory's
    ``handler`` attribute is what the real pipeline routes events to,
    so going through it keeps us honest.
    """
    handle = await handler.make_handler(info)
    task = handle.handler  # the per-task IOHandler installed by with_handler
    handle.on_start()

    try:
        root = [info.task_id]
        await task.log_start(path=root, description=info.label, tool_id=None)

        # System + initial human prompts — closed-by-default <details>.
        # Every real autoprove task starts with both; the mock mirrors
        # that so the demo exercises the prompt-block rendering on
        # every panel.
        await task.log_state_update(root, _state_with_prompts(
            "init",
            system_text=(
                f"You are an expert smart-contract verification engineer "
                f"working on the **{info.phase.value}** phase of the autoprove "
                f"pipeline.\n\n"
                f"## Your role\n\n"
                f"Analyse the contract source code and the system documentation "
                f"provided to extract candidate properties for formal verification "
                f"in CVL (Certora Verification Language).\n\n"
                f"## Output expectations\n\n"
                f"- Properties must be expressible as CVL rules / invariants.\n"
                f"- Prefer specificity over creativity — if the design doc says "
                f"  the function returns 42, write a property that asserts "
                f"  `theAnswer() == 42`, nothing more.\n"
                f"- Do not invent edge cases that aren't explicitly mentioned "
                f"  in the design document.\n\n"
                f"## Tools\n\n"
                f"You have access to the standard file-reading tools "
                f"(`get_file`, `list_files`, `grep_files`) plus the CVL manual "
                f"search tools. Use them to ground your analysis."
            ),
            human_text=(
                f"Please run the {info.phase.value} phase for the contract under "
                f"review. The system document and source files are available via "
                f"the standard file tools.\n\n"
                f"Begin by surveying the source, then proceed phase-specifically."
            ),
        ))

        # Thinking block — model's stream-of-consciousness before tools
        # fire. Renders as a closed-by-default <details>.
        await _sleep(duration / 12)
        await task.log_state_update(root, _state_with_thinking(
            "planner",
            f"Approaching {info.label}. The primary concerns here are "
            f"balance accounting and access-control invariants. Let me "
            f"start by surveying the source — `Vault.sol` is the main "
            f"contract; `IVault.sol` defines the public surface.",
        ))

        # A run of grouped get_file calls — emitted one per state update
        # (matching how langgraph actually drives them) so the user
        # sees the row's item list grow live, not all at once.
        for fname in ("src/Vault.sol", "src/IVault.sol", "src/Token.sol"):
            await _sleep(duration / 14)
            await task.log_state_update(root, _state_with_tools(
                "loader", ("get_file", {"path": fname}),
            ))

        # AI markdown — resets grouping by virtue of being non-tool-call.
        await _sleep(duration / 10)
        await task.log_state_update(root, _state_with_text(
            "synthesizer",
            "## Initial findings\n\n"
            "Three contracts in scope:\n\n"
            "1. **`Vault`** — holds user deposits, mints shares\n"
            "2. **`IVault`** — interface\n"
            "3. **`Token`** — underlying ERC20\n\n"
            "Candidate invariants:\n\n"
            "- `totalSupply` equals the sum of all `balances[*]`\n"
            "- `transfer(to, ...)` reverts when `to == address(0)`\n"
            "- Share-to-asset ratio stays monotonic across `deposit`s\n\n"
            "Will follow up with a `grep_files` to locate existing "
            "invariant assertions before deciding what to formalise.",
        ))

        # Non-grouped tool + result — shows tool-call rendering AND
        # tool-result <details>. (get_file's results are intentionally
        # suppressed by ToolDisplayConfig because file dumps would be
        # spam; grep_files is a good showcase instead.)
        await _sleep(duration / 14)
        await task.log_state_update(root, _state_with_tools(
            "search", ("grep_files", {"search_string": "totalSupply"}),
        ))
        await _sleep(duration / 14)
        await task.log_state_update(root, _state_with_result(
            "search", "grep_files",
            "src/Vault.sol:23: uint256 public totalSupply;\n"
            "src/Vault.sol:67: emit TotalSupplyUpdated(totalSupply);\n"
            "src/Token.sol:45: function totalSupply() external view returns (uint256);\n"
            "src/Token.sol:128: balances[to] = balances[to] + amount;\n"
            "src/Token.sol:129: totalSupply = totalSupply + amount;",
        ))

        # Per-event slice for nested workflows. A coarse approximation —
        # the visual cadence doesn't have to match real pipeline timing.
        nested = nested or []
        total_events = sum(_count_events(n) for n in nested)
        slice_dur = (duration / 2) / max(1, total_events)
        for n in nested:
            await _drive_nest(task, root, n, slice_dur)

        await _sleep(duration / 12)
        if raise_at_end is not None:
            raise raise_at_end
        await task.log_state_update(root, _state_with_text("finisher", final_text))
        await task.log_end(root)
    except Exception as exc:
        await handle.on_error(exc, _format_traceback(exc))
        return

    handle.on_done()


def _count_events(n: _Nest) -> int:
    """Total event count across *n* and its descendants — used by
    ``_drive_task`` to spread the per-event sleep budget evenly."""
    return len(n.events) + sum(_count_events(c) for c in n.children)


def _format_traceback(exc: Exception) -> str:
    """Format an exception's traceback the way ``run_task`` would."""
    import traceback
    return "".join(traceback.format_exception(exc))


# ---------------------------------------------------------------------------
# Whole-run driver
# ---------------------------------------------------------------------------

async def run_mock_pipeline(run: RunState) -> None:
    """Drive a complete Phase-3 mock run.

    Catches its own exceptions: a bug in the script renders as a
    pipeline-crashed status, not a stuck SSE stream that never finishes."""
    handler = AutoProveWebHandler(run)

    try:
        # Sequential phases — one task each.
        await _drive_task(
            handler,
            TaskInfo(task_id=_tid(), label="Compile + harness generation",
                     phase=AutoProvePhase.HARNESS),
            duration=1.4,
        )
        await _drive_task(
            handler,
            TaskInfo(task_id=_tid(), label="ERC20 summaries",
                     phase=AutoProvePhase.SUMMARIES),
            duration=1.0,
        )
        await _drive_task(
            handler,
            TaskInfo(task_id=_tid(), label="Total-supply invariant",
                     phase=AutoProvePhase.INVARIANTS),
            duration=1.6,
        )
        await _drive_task(
            handler,
            TaskInfo(task_id=_tid(), label="System decomposition",
                     phase=AutoProvePhase.COMPONENT_ANALYSIS),
            duration=1.2,
        )

        # Parallel BUG_ANALYSIS — three components.
        components = ["Vault", "Token", "Oracle"]
        await asyncio.gather(*[
            _drive_task(
                handler,
                TaskInfo(task_id=_tid(), label=f"Properties: {c}",
                         phase=AutoProvePhase.BUG_ANALYSIS),
                duration=2.0 + i * 0.3,
            )
            for i, c in enumerate(components)
        ])

        # Parallel CVL_GEN — three components, each exercising a
        # different nesting scenario; one fails.
        await asyncio.gather(
            _drive_task(
                handler,
                TaskInfo(task_id=_tid(), label="CVL: Vault",
                         phase=AutoProvePhase.CVL_GEN),
                duration=3.0,
                nested=[
                    _Nest(
                        segment="cvl-research",
                        label="CVL Research",
                        events=[
                            _state_with_thinking(
                                "researcher",
                                "I need to find the CVL idiom for tracking "
                                "cumulative state across calls — that's "
                                "what ghost variables are for.",
                            ),
                            _state_with_tools(
                                "search",
                                ("cvl_manual_search",
                                 {"question": "ghost variable patterns"}),
                            ),
                            _state_with_result(
                                "search", "cvl_manual_search",
                                "## Ghost Variables (CVL Manual §4.2)\n\n"
                                "Ghost variables let you track program state "
                                "for properties that aren't directly "
                                "observable from the Solidity source. "
                                "Common pattern:\n\n"
                                "```cvl\n"
                                "ghost mathint sumOfBalances {\n"
                                "    init_state axiom sumOfBalances == 0;\n"
                                "}\n"
                                "```\n",
                            ),
                            _state_with_tools(
                                "search",
                                ("cvl_manual_search",
                                 {"question": "preserved blocks syntax"}),
                            ),
                            _state_with_text(
                                "synth",
                                "Found the pattern. Will use "
                                "`init_state preserved` to bootstrap the "
                                "ghost when the contract is first deployed.",
                            ),
                        ],
                    ),
                ],
            ),
            _drive_task(
                handler,
                TaskInfo(task_id=_tid(), label="CVL: Token",
                         phase=AutoProvePhase.CVL_GEN),
                duration=3.5,
                nested=[
                    _Nest(
                        segment="orchestrator",
                        label="Sub-orchestrator",
                        events=[
                            _state_with_thinking(
                                "orch",
                                "Decomposing the problem into compile + "
                                "verify phases. The verifier sub-agent "
                                "owns the actual prover invocation.",
                            ),
                            _state_with_text(
                                "orch",
                                "Delegating spec compilation to verifier.",
                            ),
                        ],
                        children=[
                            _Nest(
                                segment="verifier",
                                label="Verifier sub-agent",
                                events=[
                                    _state_with_tools(
                                        "verify",
                                        ("verify_spec",
                                         {"spec_path": "certora/Token.spec"}),
                                    ),
                                    _state_with_text(
                                        "verify", "Rule `total_supply_ok` passes.",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            _drive_task(
                handler,
                TaskInfo(task_id=_tid(), label="CVL: Oracle",
                         phase=AutoProvePhase.CVL_GEN),
                duration=2.5,
                raise_at_end=RuntimeError("prover returned UNKNOWN; spec not provable as written"),
            ),
        )

        # Materialise mock output files into the run workspace.
        certora_dir = run.workspace / "certora"
        certora_dir.mkdir(parents=True, exist_ok=True)
        for c in ("Vault", "Token"):
            f = certora_dir / f"autospec_{c.lower()}.spec"
            f.write_text(
                f"// Mock CVL spec for {c}\n"
                f"rule sanity_{c.lower()} {{\n    assert true;\n}}\n"
            )
            rel = str(f.relative_to(run.workspace))
            run.output_files.append({
                "url":   f"/runs/{run.run_id}/files/{rel}",
                "label": rel,
            })

        handler.finish()
    except Exception as exc:
        handler.crashed(exc)


def _tid() -> str:
    """Generate a fake thread id (12 hex chars matches the autoprove
    pipeline's ``f"autoprove_{uuid.uuid4().hex[:12]}"`` shape closely
    enough to spot in logs)."""
    return uuid.uuid4().hex[:12]
