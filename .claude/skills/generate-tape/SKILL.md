---
name: generate-tape
description: Generate a fake-LLM replay tape (composer/testing/ui_harness_<name>.py) for the AutoProve smoke harness by RECORDING a real run and curating it. Use this whenever the user wants to create, record, or regenerate a tape / ui_harness script for a scenario so the pipeline can be replayed end-to-end with no real LLM calls — e.g. "make a tape for the Answer smoketest", "record a tape from this run", "generate a ui_harness for scenario X", "I need a deterministic replay of the autoprove pipeline". The tape is keyed by run_task task_id and replayed by composer.testing.harness_tape.HarnessFakeLLM. Recording yields a draft; a clean replay needs a hand-clean pass. Pairs with the inspect-run skill for debugging a recorded run.
---

# generate-tape

## What a tape is

`composer/testing/ui_harness_<name>.py` is a hand-/auto-authored **tape**: a
`dict[task_id -> list[AIMessage]]` (each task_id's list is a **lane**) that
`HarnessFakeLLM` (`composer/testing/harness_tape.py`) replays one entry per
`llm.ainvoke`, routed by the active `run_task` task_id (`get_current_task_id()`).
Within a lane, entries are served in order; subagents inherit their parent
phase's task_id, so their calls land in the parent's lane. It lets the
*entire* AutoProve pipeline run end-to-end with **zero real LLM calls** — every
other tool (solc, the Certora prover, PreAudit, Postgres, RAG) runs for real.
`COMPOSER_TEST_TAPE=<name>` (handled in `composer/bind.py`) installs it.

`ui_harness_autoprove_Counter.py` (the Counter scenario) is the curated reference.

## How tapes are made: RECORD a real run — don't reconstruct from logs

The only faithful way to capture a tape is to **record one real run** with the
recorder in `composer/testing/record_tape.py` — the inverse of `HarnessFakeLLM`.
It appends a callback (`RecordingCallback`, alongside the existing `UsageCallback`)
to every model the pipeline builds; the callback files each LLM response into the
lane for the active task_id, in call order, and the recorder dumps a runnable
`ui_harness_<name>.py` at exit.

Why not reconstruct from a past run's Postgres checkpoints (e.g. via inspect-run
/ `ap-trail`)? Two tape entries are **unrecoverable** that way:

- **Inline counter-example analysis** (`composer.prover.analysis.analyze_cex_raw`)
  is a bare `llm.ainvoke` *outside* the LangGraph agent loop, so it is never
  checkpointed to any thread. Recording captures it for free, in the right lane
  and position; reconstruction can at best regex it out of the `verify_spec`
  tool output.
- **Subagent interleaving** (code_explorer / feedback / cvl_research /
  invariant_feedback) — these inherit the parent phase's task_id, so recording
  lands them in the parent lane in exact call order; reconstruction has to
  stitch separate threads back together heuristically.

Recording is faithful by construction. It costs one real (paid) LLM run, which
you need anyway for a new scenario.

### Record produces a DRAFT — then hand-clean it

A recorded tape captures *exactly* what one real run did, which is **not the same
as a tape that replays cleanly**. The curated reference (`ui_harness_autoprove_Counter.py`)
is hand-authored for determinism. So the real workflow is **record → hand-clean →
verify (iterate)**, not record-and-done:

- The recorder **auto-drops content-less turns** (no text, no tool_calls) — those
  are transient no-tool-call retries the agent loop discards; replaying them would
  trigger spurious "every AI turn must end with a tool call" retries.
- *You* then remove the entries that depend on **per-run / stateful / non-deterministic
  tool results**, which diverge on replay (see step 3). The big one is the `memory`
  tool: its per-run namespace means a recorded `memory` read returns different content
  on replay (`File not found: /memories/progress.md`), and the agent's next call
  diverges → lane exhausted.

This is why a trivial scenario records poorly: with little real work to do, the agent
flails (lots of `memory`/draft churn) and the draft needs heavy cleaning. A scenario
with genuine work (like Counter) yields a cleaner draft.

## Inputs you need from the user

1. **Tape name** `<name>` — by convention `<flow>_<scenario>`, e.g. `autoprove_Answer`
   (the existing tapes are `autoprove_Counter` and `autoprove_Answer`). Must be a valid
   Python identifier suffix; the tape lands at `composer/testing/ui_harness_<name>.py`
   and is replayed with `COMPOSER_TEST_TAPE=<name>`. The flow is `autoprove` for the
   auto-prove pipeline — the harness (`HarnessFakeLLM` / `record_tape`) is itself
   flow-agnostic, so the prefix leaves room for other flows.
2. **Scenario project** — a directory with the contract, a system doc, and a
   `foundry.toml`, laid out like `test_scenarios/autoprove_counter/`
   (`src/<Contract>.sol`, `system.md`, `foundry.toml`). The `certora/` config is
   auto-generated on first run.
3. **Run flags** — `--max-bug-rounds N` (use the smallest that covers the
   scenario; `1` for trivial ones) and whether to exercise `--interactive`
   (the post-bug refinement conversation). **The replay must use the same
   flags**, so record with exactly what you intend to replay.

## Procedure

### 1. Lay out the scenario project (if it doesn't exist)

Model it on `test_scenarios/autoprove_counter/`. Minimum:

```
test_scenarios/autoprove_<name>/
  foundry.toml
  system.md            # describes the system; passed as the system_doc arg
  src/<Contract>.sol
```

### 2. Record

Set `COMPOSER_RECORD_TAPE=<name>` (and optionally `COMPOSER_RECORD_OUT=<path>`
to override the default output) and run the pipeline once for real:

```bash
COMPOSER_RECORD_TAPE=<name> [COMPOSER_RECORD_NO_THINKING=1] \
  console-autoprove \
  <repo>/test_scenarios/autoprove_<name> \
  <repo>/test_scenarios/autoprove_<name>/src/<Contract>.sol:<Contract> \
  <repo>/test_scenarios/autoprove_<name>/system.md \
  --max-bug-rounds 1            # [--interactive] for the refinement turns
```

Recording knob worth setting:
- **`COMPOSER_RECORD_NO_THINKING=1`** — disables thinking on the recorded models
  (`model_copy(thinking=None)`, what the prover summarizer already does). Yields a
  cleaner draft and a smaller tape; harmless for replay (the fake doesn't think).

**Prerequisites.** Record in an environment where a plain
`console-autoprove … --max-bug-rounds 1` already completes the full pipeline
end-to-end — i.e. the usual infra up (Postgres via `scripts/docker-compose.yml`,
solc, the Certora prover, RAG, `AUTOSETUP_PATH`) and the pipeline importing a
`composer` that includes the recorder (your dev/editable install). The recorder
just rides along on whatever runs; if the exit summary says *"no LLM responses
captured,"* the recorder wasn't active in that `composer`.

The recorder prints, at exit:

```
[record_tape] wrote N entries across K lane(s) to .../ui_harness_<name>.py
[record_tape]   lanes: system-analysis=.., harness=.., invariants=.., ...
```

A `__no_task__` lane in that summary means some LLM call fired outside any
`run_task` scope — `HarnessFakeLLM` can't route those, so move or drop them
before replaying.

### 3. Hand-clean the recorded draft (the load-bearing step)

This is where a draft becomes a replayable tape. Edit
`composer/testing/ui_harness_<name>.py` to remove entries whose replay depends on
**per-run / stateful / non-deterministic** tool results — those make the agent's
next call diverge from the recording. In order of impact:

- **`memory` tool calls** — the agent uses `memory` (per-run namespace) to stash and
  re-read "progress" notes; replay can't reproduce that store, so a recorded
  `memory` read returns different content and the agent veers off-tape. Drop the
  `memory` tool-call from each `_ai(...)` entry (and drop the whole entry if `memory`
  was its only tool call). The pipeline tolerates the agent not using memory.
- **Flailing / redundant turns** — on trivial scenarios the agent loops (re-reads
  files, re-drafts). Trim each lane to its canonical path. For a CVL lane that is:
  `put_cvl_raw → feedback_tool → verify_spec → result`, where `feedback_tool` spawns
  the **feedback-judge subagent** whose turns are served from the *same* lane,
  interleaved right after the `feedback_tool` call (`get_cvl` + `write_rough_draft`
  → `read_rough_draft` → `result(good=True)`), before the author's `verify_spec`.
- **Missing terminal turn** — if the recording flailed *past* its terminal `result`
  (e.g. hit the recursion limit before publishing), the lane has no clean ending and
  replay exhausts. **Hand-author the tail**, and reuse the recording's *real*
  artifacts so it stays faithful: the CVL spec the run already wrote (it typechecked
  and the prover verified it), and the property/rule names. The final publish is
  `result(commentary=..., property_rules=[{"property_title": "<from bug lane>", "rules": ["<rule in the spec>"]}])`.
- **Anything else that reads mutable side state** you spot diverging in step 4.

`ui_harness_autoprove_Counter.py` is the structural template for every lane (its `_CVL_TAPE`
is the exact author+judge+prover flow above). Also do the cosmetic curation: add
comments, hoist big artifacts to module constants.

The generator emits **literal** `run_task` task_id keys (`"system-analysis"`,
`"bug-0-<slug>"`, …), which must match the task_ids the pipeline sets in
`composer/spec/source/task_ids.py`. For the committed tape, optionally swap them to
the `task_ids.py` constants — rename-safe and matching the
`ui_harness_autoprove_Counter.py` reference (add
`from composer.spec.source.task_ids import (...)`):

- `"system-analysis"` → `SYSTEM_ANALYSIS_TASK_ID` (likewise `HARNESS_TASK_ID`,
  `AUTOSETUP_TASK_ID`, `SUMMARIES_TASK_ID`, `INVARIANTS_TASK_ID`, `INVARIANT_CVL_TASK_ID`)
- `"bug-<i>-<slug>"` → `bug_analysis_task_id(<i>, "<slug>")`
- `"cvl-<i>-<slug>"` → `cvl_gen_task_id(<i>, "<slug>")`

(Tip: `inspect-run` on the recording shows each phase's real messages — useful for
deciding what's load-bearing vs. flailing.)

### 4. Verify by replaying — and iterate

Replay with `COMPOSER_TEST_TAPE` and the **same CLI flags** you recorded with:

```bash
COMPOSER_TEST_TAPE=<name> \
  console-autoprove <same project> <same Contract.sol:Contract> <same system.md> \
  --max-bug-rounds 1            # match the record flags exactly
```

A clean replay reaches the end with no real LLM calls. Otherwise read the
`HarnessFakeLLM` error, fix that one divergence in the tape (step 3), and replay
again — it's an iterate-to-green loop:

- `tape lane '<x>' exhausted … Prompt -> ToolMessage: <something>` — replay issued
  an extra call after a tool result that differs from the recording. The
  `<something>` names the culprit (e.g. `File not found: /memories/progress.md` →
  remove that lane's `memory` calls). Fix and re-replay.
- `tape lane '<x>' exhausted … Prompt -> HumanMessage: Every AI turn must end with a
  tool call` — a content-less turn slipped through (the recorder normally drops
  these); remove the trailing/empty `_ai()` entries in that lane.
- `no tape lane for task_id '<x>'` — replay took a phase the recording never hit
  (usually a flag mismatch, or a non-deterministic branch). Match flags; if a phase
  legitimately makes no LLM calls (e.g. `invariant-cvl` on a stateless contract) it
  correctly has no lane.
- `LLM call outside any run_task scope` — a `__no_task__` lane entry; move or drop it.

## How it wires together (for debugging)

- `composer/bind.py`: `COMPOSER_TEST_TAPE` → install replay tape;
  `COMPOSER_RECORD_TAPE` → `composer.testing.record_tape.install_recorder`.
  They are mutually exclusive (replay wins if both set).
- `install_recorder` patches `composer.workflow.services.create_llm_base` (which
  `create_llm` delegates to) to append a `RecordingCallback` to every model's
  `callbacks` list — the same stable callback surface as `UsageCallback`, so it
  captures agent-loop turns and out-of-graph calls (CEX analysis) alike. It
  auto-drops content-less responses and registers an `atexit` dump. With
  `COMPOSER_RECORD_NO_THINKING=1` it also disables thinking on the built models.
- Lane task_ids are centralized in `composer/spec/source/task_ids.py`.
