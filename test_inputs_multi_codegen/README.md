# Multi-spec codegen test inputs

Six live scenarios for smoke-testing the multi-spec codegen changes. Each
scenario directory carries its own `input.json` plus the referenced files
(spec, interface, system doc, optional workspace). Run from the repo root:

```
python main.py --input-json test_inputs_multi_codegen/<scenario>/input.json
```

The scenarios split into three happy-path cases (should complete end-to-end)
and three input-validation cases (should fail at `upload_input` time with a
specific, expected error and never reach the agent).

| # | Scenario                                  | Mode       | Expected outcome |
| - | ----------------------------------------- | ---------- | ---------------- |
| 1 | `01_greenfield_single_spec/`              | greenfield | Happy path, one spec |
| 2 | `02_greenfield_multi_spec/`               | greenfield | Happy path, two specs → `certora/<basename>.spec` each |
| 3 | `03_greenfield_collision/`                | greenfield | **Upload error:** two specs with the same basename → VFS-path collision |
| 4 | `04_from_source/`                         | from-source | Happy path, two specs living at workspace-relative paths |
| 5 | `05_from_source_spec_outside_root/`       | from-source | **Upload error:** spec path does not resolve under `source_root` |
| 6 | `06_prover_conf/`                         | greenfield | Happy path, per-task `prover_conf` carried via JSON |

The contracts under test are deliberately trivial — single-contract
"Counter"-style components with one or two rules — so the LLM work fits in
a small window. They exist to exercise the plumbing (input parsing, VFS
layout, per-spec stamping, error paths), not to test agent capability.

## Edge cases covered

- VFS layout computation for greenfield (`certora/<basename>`) and
  from-source (workspace-relative).
- Basename collision detection at upload time.
- From-source descendant check.
- Multi-spec registration: all specs land in the VFS at upload; executor
  sees them in `InputData.specs`.
- Per-spec prover stamping: `validation["prover:<vfs_path>"]`. Task is done
  only when every spec has its own stamp.
- Per-task prover_conf via JSON (not CLI `--prover-conf`).
- JSON relative-path resolution against the JSON file's directory.

## What these fixtures do *not* cover

- Legacy CLI triad (`main.py spec intf sys`) — existing callers can smoke
  that independently; it's not a multi-spec concern.
- Resume paths — out of scope for these fixtures.
- The prover itself — these scenarios don't exercise Certora cloud, just
  the codegen pipeline around it.
