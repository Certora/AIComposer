# Getting Started — The Complete Checklist

One ordered, verifiable path from a clean machine to a working install. Every other doc in this repo (`README.md`, `AUTOPROVE.md`, `SETUP_GUIDE.md`, `CEX_AND_REQUIREMENTS_GUIDE.md`) explains a piece of this in more depth — this file's job is just to make sure you do the pieces **in order** and **check each one** before moving to the next, so nothing silently doesn't work three steps later.

There are two tracks. Pick based on what you actually need:

- **Track A — CEX tool only.** Extract and explain Prover counterexamples with Claude. No Docker, no databases, no Prover build. ~5 minutes.
- **Track B — Full install.** Everything in Track A, plus AutoProve (auto-generate & verify CVL specs), AI Composer (generate implementations from specs), Foundry test generation, and the Sanity Analyzer. Needs Docker, a Postgres-backed RAG index, and either a local Prover build or Certora cloud access.

Track B *includes* everything Track A gives you — you don't need to do both.

---

## Track A — CEX Extraction & Analysis Only

- [ ] **1. Python 3.9+ and an Anthropic API key.** Get a key at [console.anthropic.com](https://console.anthropic.com/account/keys).

- [ ] **2. Download and run the setup script.**
  ```bash
  curl -o setup_certora_tools.sh \
    https://raw.githubusercontent.com/0xbrett8571/CertoraAIComposer/master/setup_certora_tools.sh
  chmod +x setup_certora_tools.sh
  ./setup_certora_tools.sh
  ```
  Enter your API key when prompted. This creates `~/certora-tools` (isolated venv), installs `anthropic`/`python-dotenv`, copies in `extract_and_analyze_cex.py`, and adds shell aliases.

- [ ] **3. Reload your shell.**
  ```bash
  source ~/.bashrc   # or ~/.zshrc
  ```

- [ ] **4. Verify.**
  ```bash
  certora-verify
  ```
  Should print your tools home, python env, tool path, and a masked API key. If it doesn't, see the troubleshooting table in [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md).

- [ ] **5. Smoke-test against a real Prover run.** Run your Prover job with `--output_dir` pointed somewhere local, then:
  ```bash
  cex-list ./prover_results
  ```
  If you see a rule table (even with zero VIOLATED rules), you're done — **Track A complete.**

---

## Track B — Full Install

Everything below is additive on top of a normal clone; do it in this order, since later steps depend on earlier ones (e.g. `uv sync` will fail if `graphcore` isn't checked out yet).

### B0. Prerequisites

- [ ] Python **3.12+**
- [ ] [`uv`](https://docs.astral.sh/uv/)
- [ ] Docker with Compose
- [ ] Anthropic API key
- [ ] SSH access to `git@github.com:Certora/graphcore.git` (private Certora repo)
- [ ] Either: a local build of the [Certora Prover](https://github.com/Certora/CertoraProver), **or** a `CERTORAKEY` for cloud mode
- [ ] Solidity compiler(s) on `$PATH`, named `solcX.Y` (e.g. `solc8.29` for 0.8.29)
- [ ] *(AutoProve only)* access to Certora's private `AutoSetup` repo

### B1. Clone with submodules

- [ ] 
  ```bash
  git clone --recurse-submodules https://github.com/0xbrett8571/CertoraAIComposer.git
  cd CertoraAIComposer
  ```
  Already cloned without `--recurse-submodules`? Fix it in place instead of re-cloning:
  ```bash
  git submodule update --init
  ```
- [ ] **Verify:** `ls graphcore` should show real files, not an empty directory.

### B2. Provision the databases

- [ ] ```bash
  cd scripts/
  docker compose create && docker compose start
  cd ..
  ```
- [ ] **Verify the container is healthy:**
  ```bash
  docker compose -f scripts/docker-compose.yml ps
  ```
  Status should show `healthy`, not `starting` or `unhealthy`.
- [ ] **Verify the databases actually got created** (via `init-db.sql`, run automatically on first start):
  ```bash
  docker exec -it composer_db-postgres-1 psql -U postgres -c "\l" | grep -E "rag_db|audit_db|langgraph_store_db|langgraph_checkpoint_db|memory_tool_db"
  ```
  You should see all five. If the container was already running from a previous partial setup and these are missing, remove the volume and recreate: `docker compose down -v && docker compose create && docker compose start`.
- [ ] ⚠️ No attempt has been made to secure this database — local dev only. Restart it after every host reboot (or set a `restart` policy in `docker-compose.yml` — it's currently `unless-stopped`, which should survive most cases, but confirm with `docker ps` after a reboot).

### B3. Build the RAG index

- [ ] ```bash
  ./gen_docs.sh          # builds the CVL manual HTML into prover-docs/
  ./populate_rag.sh       # populates the rag schema in rag_db — used by AutoProve, AI Composer, cex-analyzer
  ```
- [ ] *(Only if you'll use the Sanity Analyzer)*
  ```bash
  ./populate_extended_rag.sh   # populates the extended_rag schema, same rag_db — CVL + Prover docs
  ```
- [ ] **Verify:**
  ```bash
  docker exec -it composer_db-postgres-1 psql -U rag_user -d rag_db -c "SELECT count(*) FROM rag.langchain_pg_embedding;"
  ```
  (Password is `rag_password`, from `composer/scripts/init-db.sql` — table name may differ slightly by embedding-store version; if this exact query errors, `\dt rag.*` first to see what's actually there.) A non-zero count means the RAG index is populated.
- [ ] Documentation changes later? Don't repeat all of the above by hand — use `./refresh_rag.sh` (see its `--help` for `--all` / `--skip-gen-docs`). Run it offline; it wipes before rebuilding.

### B4. Get a Certora Prover

Pick one:

- [ ] **Local mode:** from the root of your Certora Prover repo clone, run `./gradlew copy-assets`, then set:
  ```bash
  export CERTORA=/path/to/CertoraProver/target
  ```
- [ ] **Cloud mode:** just set `CERTORAKEY` — no local build needed. AutoProve supports this via `--cloud`.
- [ ] **Verify:** `echo $CERTORA` (local) or `echo $CERTORAKEY` (cloud) prints something non-empty.

### B5. Solidity compilers

- [ ] Install whatever `solc` versions your target projects need, named `solcX.Y` on `$PATH` (e.g. `solc8.29` → Solidity `0.8.29`).
- [ ] **Verify:** `which solc8.29` (or whatever version you need) resolves.

### B6. Install Python dependencies

- [ ] Choose **exactly one** `certora-cli` extra (they're mutually exclusive — the build will fail if you pick more than one):
  ```bash
  uv sync --extra ml --extra certora-cli          # stable channel (recommended default)
  # or: --extra certora-cli-beta / --extra certora-cli-beta-mirror
  ```
- [ ] *(Only if you're in your own venv instead of `uv`'s managed one)* also install the certora-cli requirements from the Prover repo, and remember to activate this environment every time:
  ```bash
  uv pip install -r certora_cli_requirements.txt   # run from CertoraProver/scripts
  ```
- [ ] **Verify:** `uv run python -c "import composer, analyzer, sanity_analyzer; print('imports OK')"`

### B7. Install the console scripts

- [ ] ```bash
  uv tool install '.[ml,certora-cli]'
  ```
  > ⚠️ **`AUTOPROVE.md` documents this as `uv tool install '.[ml,certora-cli,pou]'`.** The `pou` extra (needed by the AutoSetup-backed auto-setup component) is **not currently defined in `pyproject.toml`** — that install command will fail as written. If you only need AI Composer, the CEX tools, Foundry generation, or the Sanity Analyzer, omit `pou` (as above) — none of those depend on it. If you specifically need **AutoProve**, you'll need `AUTOSETUP_PATH` set to a working `AutoSetup` checkout regardless (see B8) and may need to resolve the missing `pou` extra with whoever maintains that dependency internally before `tui-autoprove`/`console-autoprove` will run cleanly.
- [ ] **Verify each command resolves** (doesn't need to succeed yet, just be found):
  ```bash
  which cex-analyzer sanity-analyzer ap-trail console-foundry tui-foundry \
        console-autoprove tui-autoprove tui-natspec cache-natspec autoprove-report-render
  ```
  All ten should print a path — all ten are registered in `[project.scripts]`. If any are missing, re-run `uv tool install`.

### B8. *(AutoProve only)* AutoSetup

- [ ] Clone Certora's private `AutoSetup` repo, then:
  ```bash
  export AUTOSETUP_PATH=/path/to/autosetup
  ```
- [ ] AutoProve will fail at import time if this isn't set — this only applies to `tui-autoprove`/`console-autoprove`, nothing else.

### B9. API key (same as Track A)

- [ ] ```bash
  export ANTHROPIC_API_KEY="sk-ant-..."
  ```

### B10. End-to-end smoke test

Run each tool you actually plan to use against a trivial project — `examples/trivial` exists exactly for this:

- [ ] **CEX tools:** already covered by Track A above.
- [ ] **Foundry generation:**
  ```bash
  console-foundry examples/trivial path/to/Contract.sol:ContractName
  ```
- [ ] **Sanity Analyzer:** point it at any `UnsatCoreTAC-*.txt` report file you have (needs the extended_rag schema populated in B3).
- [ ] **AutoProve** *(if B8 is done)*:
  ```bash
  console-autoprove examples/trivial path/to/Contract.sol:ContractName examples/trivial/system_doc_simple.txt
  ```
- [ ] **AI Composer core:** rough edges expected — check `composer/console/app.py` and [`TOOL_STATUS_AND_USAGE.md`](TOOL_STATUS_AND_USAGE.md), this workflow predates AutoProve and is being consolidated with it.

If a smoke test fails, it's almost always one of: B1 (submodule not checked out), B2 (DB not running/healthy), B3 (RAG not populated), B4 (`CERTORA`/`CERTORAKEY` not set), or B9 (API key not set) — check those four first before digging further.

---

## If you get stuck

1. Re-check the four most common blockers listed at the end of **B10**.
2. [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) — CEX tool troubleshooting table.
3. [`SETUP_GUIDE.md`](SETUP_GUIDE.md) — full step-by-step for Track A.
4. [`AUTOPROVE.md`](AUTOPROVE.md) — AutoProve-specific detail, including pipeline phases and caching.
5. [`CEX_AND_REQUIREMENTS_GUIDE.md`](CEX_AND_REQUIREMENTS_GUIDE.md) — how the prompt templates and CEX pipeline work internally, if something's misbehaving rather than just failing to start.