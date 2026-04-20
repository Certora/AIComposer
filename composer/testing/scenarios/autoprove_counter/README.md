# AutoProve Counter scenario

Inputs for the fake-LLM UI harness defined in
`composer/testing/ui_harness_autoprove.py`. The tape was hand-traced against
these exact files — changing them will drift the trace and require tape edits.

## Files

| File | Wiring |
|------|--------|
| `src/Counter.sol` | The main (and only) contract. Real `solc` and the real Certora prover run against this during the scenario. |
| `system.md` | Pass as the positional `system_doc` to `tui_autoprove.py`. |

The project root is this scenario directory. You may need to add a
`foundry.toml` / remappings / compiler setup matching your local PreAudit
expectations — the harness does not generate those.

## Layout for the CLI arguments

```
project_root  = composer/testing/scenarios/autoprove_counter
main_contract = src/Counter.sol:Counter
system_doc    = composer/testing/scenarios/autoprove_counter/system.md
```

## Design constraints baked into the tape

- **One contract, one component.** The per-contract fan-outs
  (`asyncio.gather` in `run_generation_pipeline` phases 5 & 6) collapse to
  linear execution. One feedback-judge / cvl-research / code-explorer
  sub-agent invocation happens at a time, so the single linear
  `FakeMessagesListChatModel` tape serves them sequentially.
- **No external actors, no ERC20 contracts.** The classifier agent returns
  `external_interfaces=[]` and `erc20_contracts=[]`, so the summaries phase
  is skipped entirely.
- **No harnessing.** The classifier returns `num_instances=None` for the
  single contract, so `generate_harnesses` is skipped.
- **Multiple invariants + multiple bugs.** The invariant-formulation agent
  delivers 2 invariants and the bug-analysis agent delivers 2 properties —
  both serviced by a single authoring agent per-phase, so the tape stays
  linear.

## No HITL

`AutoProveTaskHandler.format_hitl_prompt` raises `NotImplementedError` — the
auto-prove pipeline never prompts for human input. There are no
`[TAPE EXPECTATION: ...]` annotations to respond to.

## Running

The harness monkey-patches `composer.workflow.services.create_llm` to return
the fake model. `install_autoprove_tape()` **must** be called before
`tui_autoprove` is imported, because `tui_autoprove`'s entry path
(`composer.cli.tui_autoprove._main` →
`composer.spec.source.autoprove_common._entry_point`) imports `create_llm`
at module load time.

```bash
python -c "
import composer.bind as _
from composer.testing.ui_harness_autoprove import install_autoprove_tape
install_autoprove_tape()
import asyncio
import tui_autoprove
asyncio.run(tui_autoprove._main() if hasattr(tui_autoprove, '_main') else tui_autoprove.main())
" \
  composer/testing/scenarios/autoprove_counter \
  src/Counter.sol:Counter \
  composer/testing/scenarios/autoprove_counter/system.md \
  --max-concurrent 1
```

(The exact launcher is indicative — the user's real invocation may differ.)

Required environment:

- `PREAUDIT_PATH` — points at the PreAudit source tree. `run_preaudit_setup`
  shells out to `python -m orchestrator` and uses this.
- `CERTORA` — points at the Certora install (Typechecker.jar lives under
  `$CERTORA/certora_jars`).
- Postgres + pgvector available to `standard_connections`.
- Certora prover credentials (or local prover) — `verify_spec` runs the real
  prover against `src/Counter.sol` + generated spec.

## What the tape exercises

### Main pipeline phases

- **Component analysis** — `memory`, `write_rough_draft`/`read_rough_draft`,
  `list_files`, `get_file`, `explore_code`, `code_document_ref`, `result`
  (SourceApplication).
- **Harness classifier** — `memory`, `list_files`, `get_file`, `result`
  (AgentSystemDescription with empty erc20 + empty external_interfaces and
  `num_instances=None`).
- **Summaries phase** — skipped by design.
- **Harness generation** — skipped by design.
- **Structural invariant formulation** — `memory`, `invariant_feedback` ×3
  (1 bad verdict + 2 good verdicts).
- **Invariant CVL** — full author-agent tool coverage (see below). Includes a
  deliberate prover-failure round-trip (`SUBTLE_INV_CVL` → CEX →
  `GOOD_INV_CVL`) to exercise `analyze_cex_raw` and the UI's prover-output
  rendering.
- **Bug analysis** — `write_rough_draft`, `read_rough_draft`, `result`
  (list[PropertyFormulation] with 2 properties).
- **Component CVL** — streamlined tape (no re-exercise of every tool).

### CVL authorship (invariant CVL pass)

Every non-`cvl_document_ref` author tool:

- `cvl_manual_search`, `cvl_keyword_search`, `get_cvl_manual_section`,
  `scan_knowledge_base`, `get_knowledge_base_article`
- `cvl_research` → CVL research sub-agent
- `explore_code` → code-explorer sub-agent
- `erc20_guidance`, `unresolved_call_guidance`
- `put_cvl_raw` (parse-fail first, then valid — exercises `Typechecker.jar`
  rejection rendering)
- `get_cvl`
- `record_skip` + `unskip_property`
- `expect_rule_failure` + `expect_rule_passage`
- `feedback_tool` — feedback judge sub-agent, exercised three times:
  - First on `BAD_INV_CVL` (``count == 0``) with `good=False` — judge
    catches the obvious bug.
  - Second on `SUBTLE_INV_CVL` with `good=True` — judge approves by
    name-coverage and misses the `count > 0` operator typo.
  - Third on `GOOD_INV_CVL` after the prover CEX forces a fix — `good=True`
    stamps a fresh feedback digest matching the new `curr_spec`.
- `verify_spec` — real prover, exercised twice:
  - First on `SUBTLE_INV_CVL` (count_nonneg body is ``count > 0`` — a
    one-character operator typo the feedback judge approved). One rule
    violated at the base case → one `analyze_cex_raw` LLM call fires
    inline (plain-text AIMessage, no tool_calls) between the
    `verify_spec` tape entry and the author's next turn.
  - Second on `GOOD_INV_CVL`: all rules pass → stamps prover digest.
- `result` (str commentary) — terminates author graph.

`put_cvl` (AST variant) and `cvl_document_ref` are intentionally
**not** exercised; same rationale as the natspec harness.

### Sub-agents in the tape

- **Invariant feedback judge** (×3 invocations): `write_rough_draft` +
  source exploration → `read_rough_draft` → `result` (InvariantFeedback).
- **CVL research** (×1): `write_rough_draft` + `cvl_manual_search` →
  `read_rough_draft` → `result(value=...)`.
- **Code explorer** (×1): `write_rough_draft` not available here — just
  `list_files` / `get_file` → `result`.
- **Property feedback judge** (×3 in invariant CVL, ×1 in component CVL):
  `write_rough_draft` + `get_cvl` + `memory` → `read_rough_draft` →
  `result(good, feedback)`.
- **CEX analyzer** (×1): plain AIMessage, no tool_calls — response to
  `analyze_cex_raw`'s single `llm.ainvoke`.

## Failure-recovery paths

- `BROKEN_PARSE_CVL` on the first `put_cvl_raw` — `Typechecker.jar` rejects
  the parse and the author retries with valid surface syntax.
- `BAD_INV_CVL` (invariant `currentContract.count == 0`) is rejected by the
  feedback judge on its first pass (good=False) — exercises the author's
  response to a rejecting feedback verdict.
- `SUBTLE_INV_CVL` (body `count > 0` instead of `>= 0`) passes the judge but
  fails the prover at the base case — exercises the `verify_spec` CEX
  recovery path and `DefaultCexHandler` → `analyze_cex_raw` LLM round-trip
  (one failing rule, one inline CEX tape entry).
- First structural-invariant candidate judged `NOT_INDUCTIVE` — main
  invariant agent resubmits a stronger candidate before delivering the final
  invariant list.

## Known gaps / not exercised

- Summaries agent and harness-generation agent (skipped by scenario design).
- HITL tools (none in this workflow).
- `put_cvl` (AST variant).
- `cvl_document_ref` (ref strings are `AgentIndex`-hashed at runtime and
  cannot be predicted in a fixed tape).
- `knowledge_base_contribute` — kb tools are attached `read_only=True` in
  `build_basic_rag_tools`, so this tool isn't bound.
