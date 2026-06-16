"""Tests for end-of-run LLM token-usage tracking (raw counts, no pricing).

Covers the three layers of the feature:

  * **Accumulation** — :class:`RunSummary` per-model totals, per-task attribution
    folded into :class:`PhaseRecord`, ``total_tokens``, and the ``_format_summary``
    token block.
  * **Serialization** — ``dump_token_usage`` JSON shape.
  * **Wiring** — the ``UsageCallback`` attached at model construction fires on both
    ``.invoke`` and ``.ainvoke`` (including through a bound derivative), records into
    the active ``RunSummary``, and attributes to the active task. Driven by a fake
    chat model so no API key / network is needed.
"""

import json
from typing import cast

import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

import composer.diagnostics.timing as timing_mod
from composer.diagnostics.timing import (
    RunSummary,
    TokenTotals,
    install_run_summary,
    set_current_task_id,
)
from composer.diagnostics.usage_callback import UsageCallback
from composer.spec.source.common_pipeline import dump_token_usage
from graphcore.utils import TokenUsageDict


@pytest.fixture(autouse=True)
def _isolate_run_summary():
    """Keep the run-summary context var from leaking between tests."""
    tok = timing_mod._run_summary.set(None)
    try:
        yield
    finally:
        timing_mod._run_summary.reset(tok)


def _usage(model: str, i: int, o: int, cr: int, cw: int) -> TokenUsageDict:
    return {
        "input_tokens": i,
        "output_tokens": o,
        "cache_read_input_tokens": cr,
        "cache_creation_input_tokens": cw,
        "model_name": model,
    }


# --------------------------------------------------------------------------- #
# Accumulation / attribution
# --------------------------------------------------------------------------- #

def test_accumulation_and_per_task_attribution():
    s = RunSummary()
    with set_current_task_id("system-analysis"):
        s.record_token_usage(_usage("opus", 100, 10, 1000, 50))
        s.record_token_usage(_usage("opus", 200, 20, 2000, 0))
    with set_current_task_id("cvl-0-vault"):
        s.record_token_usage(_usage("sonnet", 300, 30, 0, 5))
    # A call with no active task: counts in run totals but in no phase.
    s.record_token_usage(_usage("opus", 7, 1, 0, 0))

    s.record_phase(task_id="system-analysis", label="system-analysis",
                   phase="component_analysis", wall_s=1.0, queue_wait_s=0.0)
    s.record_phase(task_id="cvl-0-vault", label="vault",
                   phase="cvl_gen", wall_s=2.0, queue_wait_s=0.0)

    tot = s.total_tokens()
    assert (tot.input, tot.output, tot.cache_read, tot.cache_write) == (607, 61, 3000, 55)
    assert s.token_usage_by_model["opus"].input == 307
    assert s.token_usage_by_model["sonnet"].input == 300

    by_phase = {p.task_id: p for p in s.phases}
    assert sum(by_phase["system-analysis"].token_usage_by_model.values(), TokenTotals()).input == 300
    assert sum(by_phase["cvl-0-vault"].token_usage_by_model.values(), TokenTotals()).cache_write == 5


def test_format_summary_includes_token_block():
    s = RunSummary()
    with set_current_task_id("t1"):
        s.record_token_usage(_usage("opus", 100, 10, 0, 0))
    s.record_phase(task_id="t1", label="t1", phase="p", wall_s=1.0, queue_wait_s=0.0)
    out = s.format()
    assert "Tokens: in 100" in out
    assert "opus" in out


def test_format_summary_omits_token_block_when_no_usage():
    s = RunSummary()
    s.record_phase(task_id="t1", label="t1", phase="p", wall_s=1.0, queue_wait_s=0.0)
    assert "Tokens:" not in s.format()


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #

def test_dump_token_usage_shape(tmp_path):
    s = RunSummary()
    with set_current_task_id("t1"):
        s.record_token_usage(_usage("opus", 100, 10, 5, 2))
    s.record_phase(task_id="t1", label="t1", phase="p", wall_s=1.0, queue_wait_s=0.0)

    dump_token_usage(str(tmp_path), s)
    out = tmp_path / ".certora_internal" / "autoProve" / "token_usage.json"
    data = json.loads(out.read_text())

    assert data["run_id"] == s.run_id
    assert data["totals"] == {"input": 100, "output": 10, "cache_read": 5, "cache_write": 2}
    assert data["by_model"]["opus"]["input"] == 100
    assert data["by_phase"][0]["task_id"] == "t1"
    assert data["by_phase"][0]["phase"] == "p"


def test_token_usage_summary_shape():
    s = RunSummary()
    with set_current_task_id("t1"):
        s.record_token_usage(_usage("opus", 100, 10, 5, 2))
    s.record_phase(task_id="t1", label="t1", phase="p", wall_s=1.0, queue_wait_s=0.0)
    counts = {"input": 100, "output": 10, "cache_read": 5, "cache_write": 2}
    summary = s.token_usage_summary()
    assert summary["totals"] == counts
    assert summary["by_model"] == {"opus": counts}
    assert summary["by_phase"] == [{"task_id": "t1", "phase": "p", **counts}]


@pytest.mark.asyncio
async def test_token_usage_persisted_to_run_meta_tags():
    """finalize_tags wires the run's token totals into the stored RunMeta.tags."""
    from langgraph.store.memory import InMemoryStore
    from composer.io.thread_logging import thread_logger, runs_ns, DEFAULT_META_NS

    store = InMemoryStore()
    s = RunSummary()
    with set_current_task_id("t1"):
        s.record_token_usage(_usage("opus", 100, 10, 5, 2))

    async with thread_logger(
        store, {"root_thread_id": "x"}, DEFAULT_META_NS, run_id=s.run_id,
        finalize_tags=lambda: {"token_usage": s.token_usage_summary()},
    ):
        pass  # pipeline body would run here; totals already recorded above

    item = await store.aget(runs_ns(DEFAULT_META_NS), s.run_id)
    assert item is not None
    tags = cast(dict, item.value["tags"])
    assert tags["root_thread_id"] == "x"  # original tag preserved
    token_usage = cast(dict, tags["token_usage"])
    assert token_usage["totals"] == {"input": 100, "output": 10, "cache_read": 5, "cache_write": 2}
    assert token_usage["by_model"] == {"opus": {"input": 100, "output": 10, "cache_read": 5, "cache_write": 2}}


def test_dump_token_usage_empty_run(tmp_path):
    """A run with no LLM calls still writes a well-formed (zeroed) file."""
    s = RunSummary()
    dump_token_usage(str(tmp_path), s)
    data = json.loads((tmp_path / ".certora_internal" / "autoProve" / "token_usage.json").read_text())
    assert data["totals"] == {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    assert data["by_model"] == {}
    assert data["by_phase"] == []


# --------------------------------------------------------------------------- #
# Callback wiring (fake model — no network)
# --------------------------------------------------------------------------- #

def _fake_model(callbacks):
    resp = AIMessage(
        content="ok",
        response_metadata={
            "model_name": "claude-test",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 2,
            },
        },
    )
    return FakeMessagesListChatModel(responses=[resp, resp], callbacks=callbacks)


def test_callback_records_on_sync_invoke():
    s = RunSummary()
    install_run_summary(s)
    model = _fake_model([UsageCallback()])

    with set_current_task_id("sync-task"):
        model.invoke("hi")

    assert s.token_usage_by_model["claude-test"].input == 100
    s.record_phase(task_id="sync-task", label="x", phase="p", wall_s=0.1, queue_wait_s=0.0)
    assert sum(s.phases[0].token_usage_by_model.values(), TokenTotals()).input == 100


@pytest.mark.asyncio
async def test_callback_records_on_async_invoke_through_binding():
    """Constructor-attached callback must fire through a bound (.bind) derivative
    on ``ainvoke`` — the same propagation the graph relies on for ``.bind_tools()`` —
    and attribute to the active task (run_inline keeps it in-context)."""
    s = RunSummary()
    install_run_summary(s)
    model = _fake_model([UsageCallback()]).bind(stop=None)

    with set_current_task_id("async-task"):
        await model.ainvoke("hi")

    assert s.token_usage_by_model["claude-test"].input == 100
    s.record_phase(task_id="async-task", label="x", phase="p", wall_s=0.1, queue_wait_s=0.0)
    assert sum(s.phases[0].token_usage_by_model.values(), TokenTotals()).input == 100
