# Natspec Counter scenario

Inputs for the fake-LLM UI harness defined in
`composer/testing/ui_harness_natspec.py`. The tape was hand-traced against
these exact files — changing them will drift the trace and require tape edits.

## Files

| File | Wiring |
|------|--------|
| `counter_design.txt` | Pass as the positional `input_file` to `tui_pipeline.py`. |

The scenario is deliberately 1 contract × 1 component × 1 property so that
the pipeline's per-contract and per-component `asyncio.gather` fan-outs
collapse to linear execution. The tape is a single flat list of
`AIMessage`s popped in the order the pipeline makes LLM calls.

There is **no HITL** in this workflow — no `[TAPE EXPECTATION: ...]`
annotations to respond to.

## Running

The harness monkey-patches `composer.workflow.services.create_llm` to return
the fake model. `install_counter_tape()` **must** be called before
`tui_pipeline` is imported, because `tui_pipeline` does
`from composer.workflow.services import create_llm` at module load time and
captures the local binding once.

```bash
python -c "
import composer.certora as _
from composer.testing.ui_harness_natspec import install_counter_tape
install_counter_tape()
import asyncio
import tui_pipeline
asyncio.run(tui_pipeline.main())
" composer/testing/scenarios/natspec_counter/counter_design.txt --max-concurrent 1
```

Set `--max-concurrent 1` for determinism — the scenario is already linear
(1 component) but explicit is better.

## What the tape exercises

### Main pipeline phases

- `run_component_analysis` — `memory` + `result` (Application with 1 contract
  / 1 component).
- `generate_interface` — `result` (single `ICounter` interface; validator
  compiles it with `solc8.29`).
- `generate_stub` — `result` (`CounterStub` no-op override; validator compiles
  it and enforces the `ICounter.sol` import / `CounterStub` identifier string
  checks).
- `run_bug_analysis` — `result` (single `PropertyFormulation`).

### CVL generation author

Exercises every non-cvl\_document\_ref author-visible tool:

- `read_stub`, `erc20_guidance`, `unresolved_call_guidance`
- `cvl_manual_search`, `cvl_keyword_search`, `get_cvl_manual_section`
- `scan_knowledge_base`, `get_knowledge_base_article`
- `request_stub_field` → registry sub-agent (adds `ghost_count`)
- `cvl_research` → research sub-agent
- `put_cvl_raw` (deliberately broken first → valid second — exercises the
  Typechecker.jar parse-failure recovery path)
- `get_cvl`, `advisory_typecheck`
- `record_skip`, `unskip_property`
- `feedback_tool` — spawns the feedback judge sub-agent twice (bad verdict
  then good verdict)
- `publish_spec` — spawns the merge sub-agent (success path)

`cvl_document_ref` is **not** exercised because its `ref` argument is a hash
the `AgentIndex` computes from the research-sub-agent's question; the tape
cannot predict the ref string at authoring time.

`put_cvl` (the structured AST variant) is **not** exercised — `put_cvl_raw`
covers the same state-mutation path and is easier to author literally.

`give_up` is **not** exercised — it terminates the author graph early and
would short-circuit the rest of the tape.

### Sub-agents

- **Registry sub-agent** — `result` with `is_new=True` and a `updated_stub`
  that the real solc validator recompiles.
- **CVL research sub-agent** — `write_rough_draft` + `cvl_manual_search`,
  `read_rough_draft`, `result`.
- **Feedback judge** (×2) — `write_rough_draft` + `memory` + `get_cvl`,
  `read_rough_draft`, `result`.
- **Merge sub-agent** — `result` with a merged CVL string that
  `certoraTypeCheck.py` accepts against the current stub.

## Failure-recovery paths

Per the harness methodology, one artifact is deliberately broken so the
failure-rendering / retry path is exercised:

- `BROKEN_CVL` is emitted as the first `put_cvl_raw` — `Typechecker.jar`
  rejects it, the tool returns the error text, and the author's next turn
  (`VALID_CVL`) succeeds.

The first `feedback_tool` invocation also receives `good=False` from the
judge, forcing the author to re-submit an improved spec before publishing.

## Known gaps / not exercised

- HITL tools (none in this workflow).
- `put_cvl` (AST form — covered by `put_cvl_raw`).
- `cvl_document_ref` (requires knowing the `AgentIndex` ref at tape-author time).
- `give_up` (would end the author graph mid-tape).
- Summarization (the author's `_CVLConfig` triggers at 50 messages; this
  tape stays well under).
