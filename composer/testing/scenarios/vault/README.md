# Vault scenario

Inputs for the fake-LLM UI harness defined in `composer/testing/ui_harness.py`.
The tape was hand-traced against these exact files — changing them will drift
the trace and require tape edits.

## Files

| File | Wiring |
|------|--------|
| `IVault.sol` | Pass as the positional `interface_file` to `tui_main.py`. |
| `Vault.spec` | Pass as the positional `spec_file`. |
| `system.md`  | Pass as the positional `system_doc`. |

The tape expects both requirements extraction and the judge sub-agent to run,
so do **not** pass `--skip-reqs`. Pass no `--set-reqs` either; the natreq
extractor is exercised as part of the tape.

## Running

The harness monkey-patches `composer.workflow.services.create_llm` to return
the fake model. To drive the scenario, start `tui_main.py` as you normally
would with the three positional files above, but first import the harness so
the substitution is installed:

```bash
python -c "
import composer.testing.ui_harness as h
h.install_vault_tape()
import asyncio
import tui_main
asyncio.run(tui_main.main())
" \
  composer/testing/scenarios/vault/Vault.spec \
  composer/testing/scenarios/vault/IVault.sol \
  composer/testing/scenarios/vault/system.md
```

(See `ui_harness.py` for the exact public API — the above is indicative; the
user may prefer a different launcher.)

## HITL expectations

For every HITL tool call the tape emits, the expected human response is
embedded **directly in the AI-authored argument fields** of the tool call
(typically the `question` or `explanation` field, in square brackets with the
prefix `[TAPE EXPECTATION: ...]`). When the TUI pauses for human input, read
the brackets and respond accordingly.

## What the tape exercises

- Natreq extraction sub-graph (rough_draft tools, memory, cvl_manual_search,
  human_in_the_loop extraction, results tool).
- Main codegen tools: vfs (put_file, get_file, list_files, grep_files),
  cvl_manual_search, cvl_research (sub-agent), read/write/commit working_spec,
  propose_spec_change, requirements_evaluation (judge sub-agent, multiple
  times), requirements_relaxation, human_in_the_loop, certora_prover (real
  prover runs), code_result.
- Every HITL tool outcome: ACCEPTED / REJECTED / REFINE for
  `propose_spec_change`; ACCEPTED / REJECTED for `requirements_relaxation` and
  `commit_working_spec`; plain response + `FOLLOWUP` for `human_in_the_loop`.
- A deliberately buggy Solidity first draft that the real prover CEXes,
  followed by a fix and a passing run, to exercise the prover failure-recovery
  UI path.
- `create_resume_commentary` final structured-output call.
