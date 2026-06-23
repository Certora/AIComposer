"""
Record a real AutoProve run into a replayable fake-LLM tape.

This is the exact inverse of :class:`composer.testing.harness_tape.HarnessFakeLLM`.
``HarnessFakeLLM`` *serves* scripted ``AIMessage`` responses routed by the
active ``run_task`` task_id (``get_current_task_id()``); this module *captures*
every real LLM response keyed by the same task_id, in call order, and
serializes them to a ``composer/testing/ui_harness_<name>.py`` module that
``HarnessFakeLLM`` can replay with no real LLM calls.

Why record instead of reconstruct from logs
--------------------------------------------
A faithful tape is, per task_id lane, the ordered sequence of ``AIMessage``
(text + tool_calls) that the pipeline's ``llm.ainvoke`` calls returned. Two of
those entries cannot be recovered from the persisted LangGraph checkpoints:

* **Inline counter-example analysis** — ``composer.prover.analysis.analyze_cex_raw``
  does a bare ``await llm.ainvoke(...)`` *outside* the LangGraph agent loop, so
  its ``AIMessage`` is never checkpointed to any thread. It is invisible to
  post-hoc reconstruction, but it flows through the *same* llm object, so
  recording captures it for free — in the correct lane and position.
* **Subagent interleaving** — code_explorer / feedback / cvl_research /
  invariant_feedback subagents run inside the parent phase's task scope, so
  ``get_current_task_id()`` returns the parent task_id for their calls. Recording
  therefore lands them in the parent lane in exact call order, with no
  thread-stitching heuristics.

How it works
------------
``create_llm`` / ``create_llm_base`` (``composer.workflow.services``) build a
``ChatAnthropic`` with a ``callbacks=[UsageCallback()]`` list. ``install_recorder``
patches ``create_llm_base`` (which ``create_llm`` delegates to, so one patch
covers every build) to append a :class:`RecordingCallback` to that list. Its
``on_llm_end`` fires for every generation — agent-loop turns through
``bind_tools`` / ``model_copy`` / ``copy`` derivatives (``create_resume_commentary``,
the prover summarizer) AND the out-of-graph ``analyze_cex_raw`` side-call —
capturing each response into the lane for the active ``get_current_task_id()``.
This is the same stable callback surface
``composer.diagnostics.usage_callback.UsageCallback`` already uses to observe
every call, so the recorder is its mirror image rather than a bespoke
interception layer.

Usage
-----
Record (one real, paid run)::

    COMPOSER_RECORD_TAPE=<name> [COMPOSER_RECORD_OUT=<path>] \\
        console-autoprove <project> <Contract.sol:Contract> <system.md> \\
        --max-bug-rounds 1 [--interactive]

The recorder installs itself from ``composer/bind.py`` (the same hook point as
``COMPOSER_TEST_TAPE``) and writes the tape at interpreter exit. Replay (free,
no LLM) with the *same* CLI flags::

    COMPOSER_TEST_TAPE=<name> console-autoprove <project> ... --max-bug-rounds 1

The generated module embeds the tape as JSON and validates it back into
``AIMessage`` objects on import. Curate it by editing the embedded ``_TAPE_JSON``
(drop diverging turns, fix a phase's tail, …).
"""

import atexit
import json
import sys
from pathlib import Path
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from composer.diagnostics.timing import get_current_task_id

# task_id used for LLM calls that fire outside any run_task scope. HarnessFakeLLM
# raises on such calls, so anything landing here needs manual attention before
# the tape can replay.
NO_TASK_LANE = "__no_task__"


def _entries(n: int) -> str:
    """``'1 entry'`` / ``'N entries'`` for log lines."""
    return f"{n} entr{'y' if n == 1 else 'ies'}"


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class TapeRecorder:
    """Accumulates real ``AIMessage`` responses per task_id lane, in call order."""

    def __init__(self, name: str, out_path: Path) -> None:
        self.name = name
        self.out_path = out_path
        # task_id -> ordered list of recorded AIMessages.
        self.lanes: dict[str, list[AIMessage]] = {}

    def record(self, message: AIMessage) -> None:
        if not message.text and not (message.tool_calls or []):
            # Content-less turn (no text, no tool_calls): a transient no-tool-call
            # turn the agent loop rejects and retries — never kept in thread state.
            # Replaying it would trigger a spurious "every AI turn must end with a
            # tool call" retry and exhaust the lane, so drop it.
            return
        task_id = get_current_task_id() or NO_TASK_LANE
        self.lanes.setdefault(task_id, []).append(message)

    def dump(self) -> None:
        total = sum(len(v) for v in self.lanes.values())
        if total == 0:
            print(
                "[record_tape] no LLM responses captured — nothing written. "
                "Was the recorder installed before the pipeline imported "
                "create_llm? (COMPOSER_RECORD_TAPE must be set before launch.)",
                file=sys.stderr,
            )
            return
        src = render_tape_module(self.name, self.lanes)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(src)
        counts = ", ".join(f"{k}={len(v)}" for k, v in self.lanes.items())
        print(
            f"[record_tape] wrote {_entries(total)} "
            f"across {len(self.lanes)} lane(s) to {self.out_path}\n"
            f"[record_tape]   lanes: {counts}",
            file=sys.stderr,
        )
        if NO_TASK_LANE in self.lanes:
            print(
                f"[record_tape] WARNING: {len(self.lanes[NO_TASK_LANE])} call(s) "
                f"fired outside any run_task scope and were parked in the "
                f"{NO_TASK_LANE!r} lane. HarnessFakeLLM cannot route these — "
                f"move or drop them before replaying.",
                file=sys.stderr,
            )


_RECORDER: TapeRecorder | None = None


class RecordingCallback(BaseCallbackHandler):
    """Captures each LLM response into the active recorder's lane.

    ``install_recorder`` appends one of these to the ``callbacks`` list of every
    model the pipeline builds (next to ``UsageCallback``). ``on_llm_end`` fires
    for every generation — agent-loop turns AND the out-of-graph
    ``analyze_cex_raw`` side-call — so all are captured with no special-casing.
    ``run_inline = True`` keeps the handler on the event-loop thread so the
    ``get_current_task_id()`` ContextVar that ``run_task`` set is visible (the
    same reason ``UsageCallback`` sets it). The extraction mirrors
    ``UsageCallback.on_llm_end``.
    """

    run_inline = True

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        rec = _RECORDER
        if rec is None:
            return
        try:
            generation = response.generations[0][0]
        except IndexError:
            return
        # on_llm_end for a chat model always carries a ChatGeneration whose
        # `.message` is the AIMessage (same access UsageCallback makes).
        if isinstance(generation, ChatGeneration):
            rec.record(cast(AIMessage, generation.message))


def default_out_path(name: str) -> Path:
    """``composer/testing/ui_harness_<name>.py`` next to this module."""
    return Path(__file__).resolve().parent / f"ui_harness_{name}.py"


def install_recorder(name: str, out_path: str | None = None, *, no_thinking: bool = False) -> TapeRecorder:
    """Append a :class:`RecordingCallback` to every LLM the pipeline builds, so
    each response is captured, and arrange for the tape to be written at
    interpreter exit.

    Patches only ``create_llm_base`` — ``create_llm`` delegates to it, so one
    patch covers every construction path. Must run before the entry path imports
    ``create_llm`` — ``composer/bind.py`` is that hook.

    ``no_thinking`` (env ``COMPOSER_RECORD_NO_THINKING``) disables thinking on every
    built model (``model_copy(update={"thinking": None})``, the same move
    ``composer.prover.core`` uses for the summarizer). Recording with thinking on can
    capture max-tokens-truncated thinking-only turns (empty AIMessages) that make the
    tape hard to replay deterministically; disabling it yields a cleaner, more
    replay-friendly tape.
    """
    global _RECORDER

    resolved = Path(out_path).expanduser() if out_path else default_out_path(name)
    recorder = TapeRecorder(name, resolved)
    _RECORDER = recorder

    # Match the replay harness: disable the agent_index cache so cached
    # code_explorer answers don't silently skip an LLM call during recording
    # while replay (which also disables the cache) issues it and exhausts the lane.
    import composer.spec.agent_index as a_ind
    a_ind._UNSAFE_DISABLE_CACHE = True

    import composer.workflow.services as services
    orig_base = services.create_llm_base

    def _build_with_recording(args: Any) -> Any:
        llm = orig_base(args)
        if no_thinking:
            llm = llm.model_copy(update={"thinking": None})
        # create_llm_base builds the model with `callbacks=[UsageCallback()]` (a
        # list), so append our recorder in place — no reassignment, no type juggling.
        assert isinstance(llm.callbacks, list), \
            "record_tape: expected create_llm_base to build a list of callbacks"
        llm.callbacks.append(RecordingCallback())
        return llm

    services.create_llm_base = _build_with_recording
    if no_thinking:
        print("[record_tape] thinking disabled for recording (COMPOSER_RECORD_NO_THINKING)", file=sys.stderr)

    atexit.register(recorder.dump)
    print(
        f"[record_tape] recording enabled (name={name!r}); tape will be written "
        f"to {resolved} at exit.",
        file=sys.stderr,
    )
    return recorder


# ---------------------------------------------------------------------------
# Serialization — emit a ui_harness_<name>.py module
# ---------------------------------------------------------------------------

def render_tape_module(name: str, lanes: dict[str, list[AIMessage]]) -> str:
    """Render the ``ui_harness_<name>.py`` source: the lanes serialized as embedded
    JSON (each ``AIMessage`` via ``model_dump``) plus the boilerplate that loads it
    back with ``AIMessage.model_validate`` and installs the fake."""
    payload = {
        task_id: [m.model_dump(mode="json", exclude_none=True) for m in msgs]
        for task_id, msgs in lanes.items()
    }
    tape_json = json.dumps(payload, indent=2)
    # JSON escapes its own double quotes, so the dumped text can't contain a
    # literal triple-quote — guard the r"""...""" embedding just in case.
    assert '"""' not in tape_json
    lane_summary = ", ".join(f"{k}={len(v)}" for k, v in lanes.items())
    return f'''\
"""
AUTO-GENERATED fake-LLM tape for the {name!r} AutoProve scenario.

Recorded by composer.testing.record_tape from a real run: each lane is the ordered
list of a phase's AIMessage responses (keyed by run_task task_id), serialized as
JSON and validated back into AIMessages on import. HarnessFakeLLM replays one per
llm.ainvoke. To curate, edit `_TAPE_JSON` (drop diverging turns, fix the tail, etc.).

Replay with the SAME CLI flags used to record:

    COMPOSER_TEST_TAPE={name} console-autoprove <project> <Contract.sol:Contract> \\
        <system.md> --max-bug-rounds 1 [--interactive]

Lanes captured: {lane_summary}
"""

import json

from langchain_core.messages import AIMessage, BaseMessage

from composer.testing.harness_tape import HarnessFakeLLM

# task_id -> ordered list of recorded AIMessage responses (pydantic model_dump JSON).
_TAPE_JSON = r"""
{tape_json}
"""

_TAPE: dict[str, list[BaseMessage]] = {{
    task_id: [AIMessage.model_validate(m) for m in messages]
    for task_id, messages in json.loads(_TAPE_JSON).items()
}}


def get_{name}_llm() -> HarnessFakeLLM:
    """Return a fresh fake LLM loaded with the {name!r} tape."""
    return HarnessFakeLLM(lanes=_TAPE)


def install_harness_tape() -> HarnessFakeLLM:
    """Monkeypatch create_llm / create_llm_base so the pipeline receives the fake.
    composer/bind.py calls this when COMPOSER_TEST_TAPE={name} is set."""
    fake = get_{name}_llm()
    import composer.spec.agent_index as a_ind
    a_ind._UNSAFE_DISABLE_CACHE = True
    import composer.workflow.services as services
    services.create_llm = lambda args: fake  # type: ignore[assignment]
    services.create_llm_base = lambda args: fake  # type: ignore[assignment]
    return fake


__all__ = ["get_{name}_llm", "install_harness_tape"]
'''
