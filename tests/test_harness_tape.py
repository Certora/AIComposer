"""Unit tests for the lane-routed fake LLM used by the autoprove UI harness.

These cover the routing mechanism in isolation (no pipeline, no DB) — the part
that makes the deterministic tape survive the pipeline's concurrent phases. The
end-to-end tape *content* is still validated by running the real smoke harness.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from composer.diagnostics.timing import set_current_task_id
from composer.testing.harness_tape import HarnessFakeLLM, LaneMarker, partition_tape


def _msg(text: str) -> AIMessage:
    return AIMessage(text)


def _prompt() -> list[HumanMessage]:
    return [HumanMessage("hello")]


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# partition_tape
# --------------------------------------------------------------------------

def test_partition_preserves_order_and_is_lossless():
    a0, a1, b0 = _msg("a0"), _msg("a1"), _msg("b0")
    lanes = partition_tape([LaneMarker("A"), a0, a1, LaneMarker("B"), b0])
    assert list(lanes.keys()) == ["A", "B"]
    assert lanes["A"] == [a0, a1]
    assert lanes["B"] == [b0]
    assert sum(len(v) for v in lanes.values()) == 3


def test_partition_allows_empty_lane():
    b0 = _msg("b0")
    lanes = partition_tape([LaneMarker("A"), LaneMarker("B"), b0])
    assert lanes["A"] == []
    assert lanes["B"] == [b0]


def test_partition_rejects_response_before_first_marker():
    with pytest.raises(ValueError):
        partition_tape([_msg("x")])


# --------------------------------------------------------------------------
# HarnessFakeLLM lane routing
# --------------------------------------------------------------------------

def test_per_lane_cursors_are_independent_under_interleaving():
    a0, a1, a2 = _msg("a0"), _msg("a1"), _msg("a2")
    b0, b1 = _msg("b0"), _msg("b1")
    llm = HarnessFakeLLM(
        responses=[a0, a1, a2, b0, b1],
        lanes={"A": [a0, a1, a2], "B": [b0, b1]},
        jitter=False,
    )

    async def go():
        out = []
        with set_current_task_id("A"):
            out.append(await llm.ainvoke(_prompt()))  # a0
            out.append(await llm.ainvoke(_prompt()))  # a1
        with set_current_task_id("B"):
            out.append(await llm.ainvoke(_prompt()))  # b0
        with set_current_task_id("A"):
            out.append(await llm.ainvoke(_prompt()))  # a2 — A cursor unaffected by B
        with set_current_task_id("B"):
            out.append(await llm.ainvoke(_prompt()))  # b1
        return out

    assert _run(go()) == [a0, a1, b0, a2, b1]


def test_truly_concurrent_lanes_do_not_steal_each_others_responses():
    a0, a1 = _msg("a0"), _msg("a1")
    b0, b1 = _msg("b0"), _msg("b1")
    llm = HarnessFakeLLM(
        responses=[a0, a1, b0, b1],
        lanes={"A": [a0, a1], "B": [b0, b1]},
        jitter=False,
    )

    async def lane(task_id):
        with set_current_task_id(task_id):
            first = await llm.ainvoke(_prompt())
            await asyncio.sleep(0)  # yield, let the other lane interleave
            second = await llm.ainvoke(_prompt())
        return [first, second]

    async def go():
        return await asyncio.gather(lane("A"), lane("B"))

    res_a, res_b = _run(go())
    assert res_a == [a0, a1]
    assert res_b == [b0, b1]


def test_exhausted_lane_raises():
    a0 = _msg("a0")
    llm = HarnessFakeLLM(responses=[a0], lanes={"A": [a0]}, jitter=False)

    async def go():
        with set_current_task_id("A"):
            await llm.ainvoke(_prompt())
            await llm.ainvoke(_prompt())

    with pytest.raises(RuntimeError, match="exhausted"):
        _run(go())


def test_unknown_lane_raises():
    llm = HarnessFakeLLM(responses=[], lanes={"A": [_msg("a0")]}, jitter=False)

    async def go():
        with set_current_task_id("does-not-exist"):
            await llm.ainvoke(_prompt())

    with pytest.raises(RuntimeError, match="no tape lane"):
        _run(go())


def test_missing_task_id_raises():
    llm = HarnessFakeLLM(responses=[], lanes={"A": [_msg("a0")]}, jitter=False)

    async def go():
        await llm.ainvoke(_prompt())  # no set_current_task_id scope

    with pytest.raises(RuntimeError, match="outside any run_task"):
        _run(go())


def test_legacy_positional_fallback_when_no_lanes():
    a0, a1 = _msg("a0"), _msg("a1")
    llm = HarnessFakeLLM(responses=[a0, a1], jitter=False)  # lanes empty

    async def go():
        return [await llm.ainvoke(_prompt()), await llm.ainvoke(_prompt())]

    assert _run(go()) == [a0, a1]


# --------------------------------------------------------------------------
# Regression guard on the real autoprove tape's lane structure
# --------------------------------------------------------------------------

def test_autoprove_tape_partitions_into_expected_lanes():
    from composer.testing.ui_harness_autoprove import _AUTOPROVE_TAPE

    lanes = partition_tape(_AUTOPROVE_TAPE)
    assert set(lanes) == {
        "system-analysis",
        "harness",
        "invariants",
        "invariant-cvl",
        "bug-0",
        "cvl-0",
    }
    n_msgs = sum(1 for x in _AUTOPROVE_TAPE if not isinstance(x, LaneMarker))
    assert sum(len(v) for v in lanes.values()) == n_msgs
    assert all(len(v) > 0 for v in lanes.values())
