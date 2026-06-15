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

The generated module is a faithful, runnable *starting point*: hand-edit it to
add explanatory comments or hoist artifacts, exactly like
``ui_harness_autoprove_Counter.py``.
"""

import atexit
import pprint
import sys
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
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
        self._dumped = False

    def record(self, response: BaseMessage) -> None:
        if not isinstance(response, AIMessage):
            # Chat models always return AIMessage; anything else is not
            # replayable by HarnessFakeLLM, so skip it.
            return
        if not response.text and not (response.tool_calls or []):
            # Content-less response (no text, no tool_calls): a transient
            # no-tool-call turn the agent loop rejects and retries — it is never
            # kept in thread state. Replaying it would trigger a spurious "every
            # AI turn must end with a tool call" retry and exhaust the lane, so
            # drop it (the next real response is what the replay needs).
            return
        task_id = get_current_task_id() or NO_TASK_LANE
        self.lanes.setdefault(task_id, []).append(response)

    def dump(self) -> None:
        if self._dumped:
            return
        self._dumped = True
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
    ``UsageCallback.on_llm_end`` exactly.
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
        if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
            rec.record(generation.message)


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

    services.create_llm_base = _build_with_recording  # type: ignore[assignment]
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

def _py_str(s: str) -> str:
    """A valid Python string literal — triple-quoted when that stays faithful,
    otherwise ``repr``."""
    if "\n" in s and '"""' not in s and "\\" not in s and not s.endswith('"'):
        return '"""\\\n' + s + '"""'
    return repr(s)


class _Hoister:
    """Hoists large/multiline string values into module-level constants so the
    lane lists stay readable (mirrors how ui_harness_autoprove_Counter.py keeps CVL
    blobs as top-level constants)."""

    def __init__(self) -> None:
        self._by_value: dict[str, str] = {}
        self.consts: list[tuple[str, str]] = []  # (name, literal)

    def ref(self, s: str) -> str:
        """Return an expression for string ``s`` — a hoisted constant name when
        the string is multiline or long, otherwise an inline literal."""
        if "\n" not in s and len(s) <= 88:
            return repr(s)
        if s in self._by_value:
            return self._by_value[s]
        name = f"_T{len(self.consts)}"
        self._by_value[s] = name
        self.consts.append((name, _py_str(s)))
        return name


def _py_value(v: Any, hoist: _Hoister) -> str:
    """A valid Python expression for a tool-call argument value."""
    if isinstance(v, str):
        return hoist.ref(v)
    # Non-str (None/bool/int/float/dict/list): pformat emits a valid Python
    # literal and preserves dict order.
    return pprint.pformat(v, width=88, sort_dicts=False)


def _emit_tc(tc: Any, hoist: _Hoister, indent: str) -> str:
    name = tc.get("name", "")
    args = tc.get("args") or {}
    if not args:
        return f"{indent}_tc({name!r})"
    if all(str(k).isidentifier() for k in args):
        kvs = [f"{k}={_py_value(v, hoist)}" for k, v in args.items()]
    else:
        kvs = [f"**{_py_value(args, hoist)}"]
    inner_indent = indent + "    "
    body = (",\n" + inner_indent).join(kvs)
    return f"{indent}_tc(\n{inner_indent}{name!r},\n{inner_indent}{body},\n{indent})"


def _emit_ai(msg: AIMessage, hoist: _Hoister, indent: str) -> str:
    text = msg.text  # langchain concatenates text blocks, drops thinking/tool_use
    tool_calls = list(msg.tool_calls or [])
    inner_indent = indent + "    "
    arg_exprs: list[str] = []
    if text:
        arg_exprs.append(f"{inner_indent}{hoist.ref(text)}")
    for tc in tool_calls:
        arg_exprs.append(_emit_tc(tc, hoist, inner_indent))
    if not arg_exprs:
        return f"{indent}_ai()"
    body = ",\n".join(arg_exprs)
    return f"{indent}_ai(\n{body},\n{indent})"


_MODULE_HEADER = '''\
"""
AUTO-GENERATED fake-LLM tape for the {name!r} AutoProve scenario.

Recorded by composer.testing.record_tape from a real run. Each lane is the
ordered list of AIMessage responses for one run_task task_id; HarnessFakeLLM
replays one per llm.ainvoke. This is a faithful, runnable starting point —
edit freely (add comments, hoist artifacts) the way ui_harness_autoprove_Counter.py is
hand-curated.

Replay with the SAME CLI flags used to record:

    COMPOSER_TEST_TAPE={name} console-autoprove <project> <Contract.sol:Contract> \\
        <system.md> --max-bug-rounds 1 [--interactive]

Lanes captured: {lane_summary}
"""

from typing import Any
import uuid

from composer.testing.harness_tape import HarnessFakeLLM

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import ToolCall


def _tc(name: str, **args: Any) -> ToolCall:
    """Tool-call dict with a unique id (LangGraph binds tool responses back to
    calls by id, so every entry needs its own)."""
    return {{
        "id": f"toolu_{{uuid.uuid4().hex[:20]}}",
        "name": name,
        "args": args,
        "type": "tool_call",
    }}


def _ai(text: str = "", *tool_calls: ToolCall) -> AIMessage:
    """Tape entry: optional text + zero or more tool_calls."""
    content: list[str | dict] = []
    if text:
        content.append(text)
    content.extend(
        {{"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["args"]}}
        for t in tool_calls
    )
    return AIMessage(content=content, tool_calls=list(tool_calls))
'''


_MODULE_FOOTER = '''\

_TAPE: dict[str, list[BaseMessage]] = {{
{lane_entries}
}}


def get_{name}_llm() -> HarnessFakeLLM:
    """Return a fresh fake LLM loaded with the {name!r} tape."""
    return HarnessFakeLLM(lanes=_TAPE)


def install_harness_tape() -> HarnessFakeLLM:
    """Monkeypatch create_llm / create_llm_base so the pipeline receives the
    fake. Call before importing the autoprove entry path (composer/bind.py
    does this when COMPOSER_TEST_TAPE={name} is set)."""
    fake = get_{name}_llm()
    import composer.spec.agent_index as a_ind
    a_ind._UNSAFE_DISABLE_CACHE = True
    import composer.workflow.services as services
    services.create_llm = lambda args: fake  # type: ignore[assignment]
    services.create_llm_base = lambda args: fake  # type: ignore[assignment]
    return fake


__all__ = ["get_{name}_llm", "install_harness_tape"]
'''


def render_tape_module(name: str, lanes: dict[str, list[AIMessage]]) -> str:
    """Render the full ``ui_harness_<name>.py`` source for ``lanes``."""
    hoist = _Hoister()
    lane_blocks: list[str] = []
    for task_id, msgs in lanes.items():
        key_expr = repr(task_id)
        entries = ",\n".join(_emit_ai(m, hoist, "        ") for m in msgs)
        lane_blocks.append(
            f"    # lane: {task_id} ({_entries(len(msgs))})\n"
            f"    {key_expr}: [\n{entries},\n    ],"
        )

    lane_summary = ", ".join(f"{k}({len(v)})" for k, v in lanes.items())
    header = _MODULE_HEADER.format(name=name, lane_summary=lane_summary)
    consts_block = ""
    if hoist.consts:
        consts_block = "\n\n# Hoisted string artifacts (CVL specs, long messages).\n" + "\n".join(
            f"{cname} = {literal}\n" for cname, literal in hoist.consts
        )
    footer = _MODULE_FOOTER.format(name=name, lane_entries="\n".join(lane_blocks))
    return header + consts_block + footer
