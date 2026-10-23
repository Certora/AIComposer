<div align="center">

# Certora AI Composer

**LLM-driven tooling for formal verification with the Certora Prover.**
Generate CVL specifications and verified implementations, explain failed proofs, and close the loop between Claude and the Prover.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#project-status)
[![Built by Certora Labs](https://img.shields.io/badge/built%20by-Certora%20Labs-6E56CF.svg)](https://www.certora.com/)

</div>

---

> [!TIP]
> **New here?** Go straight to [`GETTING_STARTED.md`](GETTING_STARTED.md) — a single ordered, verifiable checklist covering both the lightweight CEX tool and the full install (AutoProve, AI Composer, Foundry generation, Sanity Analyzer), with a check after every step so nothing silently fails three steps later. The [Quick Start](#-quick-start) and [Full Installation](#-full-installation) sections below cover the same ground in narrative form.

## What is AI Composer?

AI Composer is a collection of **Claude-powered agents and utilities**, built by Certora Labs, that sit on top of the [Certora Prover](https://docs.certora.com/) and the CVL specification language. It aims to close the loop between *writing formal specs* and *getting them to verify* — using an LLM to draft, diagnose, and iterate, while the Prover remains the ground truth.

Depending on what you're trying to do, it can:

| Capability | What it means in practice |
|---|---|
| 🧩 **Generate a Solidity implementation** | ...that provably satisfies a given CVL spec and interface (the original "AI Composer" workflow). |
| 📜 **Generate & verify CVL specs** | ...for an *existing* Solidity project from a design document (the **AutoProve** pipeline). |
| 🧪 **Generate Foundry tests** | ...for a contract, from its source alone. |
| 🔍 **Explain a failed Prover run** | ...by extracting the counterexample (CEX) and asking Claude to diagnose the root cause and propose a fix. |
| 🧭 **Diagnose unsatisfiable rules** | ...by analyzing the Prover's unsat core and explaining why the constraints conflict. |

All of this is backed by a **local, documentation-derived RAG index**, so the LLM grounds its CVL usage in the actual Prover manual rather than guessing at syntax.

```mermaid
flowchart LR
    A["Solidity project<br/>+ design doc"] -->|AutoProve| B["CVL rules"]
    C["CVL spec<br/>+ interface"] -->|AI Composer| D["Solidity implementation"]
    B --> E["Certora Prover"]
    D --> E
    E -->|"✅ verified"| F["Done"]
    E -->|"❌ counterexample"| G["CEX Analyzer"]
    E -->|"❓ unsat"| H["Sanity Analyzer"]
    G -->|"diagnosis + fix"| B
    H -->|"diagnosis + fix"| B
    RAG[("RAG index<br/>(CVL manual)")] -.grounds.-> B
    RAG -.grounds.-> D
    RAG -.grounds.-> G
    RAG -.grounds.-> H
```

If you only care about *why a rule failed*, you want the **CEX Analyzer** — a single dependency-light script that's the fastest thing to get running. Everything else generates or verifies code and needs the fuller setup described below.

---

## Table of Contents

- [What is AI Composer?](#what-is-ai-composer)
- [What's in this repo](#-whats-in-this-repo)
- [Quick Start: CEX Extraction & Analysis](#-quick-start)
- [Full Installation](#-full-installation)
- [Usage](#-usage)
  - [AutoProve — generate & verify CVL specs](#autoprove--generate--verify-cvl-specs)
  - [AI Composer — generate code from a spec](#ai-composer--generate-code-from-a-spec)
  - [Foundry test generation](#foundry-test-generation)
  - [CEX extraction & analysis](#cex-extraction--analysis-1)
  - [Sanity Analyzer — diagnosing unsat rules](#sanity-analyzer--diagnosing-unsat-rules)
  - [Inspecting past runs](#inspecting-past-runs)
- [Repository layout](#-repository-layout)
- [Examples](#-examples)
- [Documentation map](#-documentation-map)
- [Development](#-development)
- [Known issues](#-known-issues)
- [Project status](#-project-status)
- [License](#license)

---

## 🧰 What's in this repo

| Tool | Entry point | What it does |
|---|---|---|
| **AutoProve** | `tui-autoprove` / `console-autoprove` | Multi-agent pipeline that reads a Solidity project + design doc, derives properties, generates CVL rules, and iterates against the Prover until they verify. |
| **AI Composer (core)** | `composer.console.app` | The original workflow: given a CVL spec, an interface, and a system description, generates a Solidity implementation that satisfies the spec. |
| **NatSpec pipeline** | `tui-natspec` | Greenfield and update workflows for generating/refreshing NatSpec comments and CVL from source. |
| **Foundry generator** | `tui-foundry` / `console-foundry` | Generates Foundry test suites for a contract. |
| **CEX Analyzer** | `extract_and_analyze_cex.py`, `cex-analyzer` | Standalone tool: extracts counterexamples from Prover JSON output and asks Claude to explain the failure and suggest a fix. No Docker/DB setup required. |
| **Sanity Analyzer** | `sanity-analyzer` | Explains *unsatisfiable* rules by analyzing the Prover's unsat core. |
| **ap-trail** | `ap-trail` | Lists and inspects past AutoProve run "trails" stored in the audit database. |
| **cache-natspec** | `cache-natspec` | Browses the cache/memory namespaces produced by the NatSpec pipeline. |

---

## 🚀 Quick Start

Gets the standalone **CEX (counterexample) tool** running in a few minutes. It only needs Python and an Anthropic API key — no Docker, no databases, no Prover build.

### Prerequisites

- Python 3.9+
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com/account/keys))

### 1. Download the setup script

```bash
curl -o setup_certora_tools.sh \
  https://raw.githubusercontent.com/0xbrett8571/CertoraAIComposer/master/setup_certora_tools.sh
chmod +x setup_certora_tools.sh
```

### 2. Run it

```bash
./setup_certora_tools.sh
```

This creates an isolated virtual environment under `~/certora-tools`, installs `anthropic`/`python-dotenv`, copies `extract_and_analyze_cex.py` into it, prompts for your API key, and adds shell aliases (`cex-list`, `cex-extract`, `cex-analyze`, `certora-verify`).

### 3. Reload your shell and verify

```bash
source ~/.bashrc   # or ~/.zshrc
certora-verify
```

### 4. Use it against a Prover run

```bash
# See which rules passed/failed and which have a counterexample
cex-list ./prover_results

# Pretty-print the raw counterexample trace for one rule
cex-extract ./prover_results transfer_succeeds

# Ask Claude to diagnose the failure and propose a fix
cex-analyze ./prover_results transfer_succeeds \
  --spec ./spec/Token.spec \
  --output analysis.md
```

`cex-analyze` writes a structured Markdown report (scenario, root cause, ranked fix suggestions, confidence) to `--output`, or prints it to stdout if omitted.

📖 For the full command reference, troubleshooting table, and a day-in-the-life workflow, see [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) and [`SETUP_GUIDE.md`](SETUP_GUIDE.md).

---

## 🏗️ Full Installation

The generation/verification tools (AutoProve, AI Composer, Foundry generation, Sanity Analyzer) need a local Prover build, a set of Postgres databases, and a RAG index built from the CVL documentation. This is heavier than the Quick Start above.

### Requirements

- Python **3.12+** and [`uv`](https://docs.astral.sh/uv/)
- Docker with Compose
- An Anthropic API key
- A local build of the [Certora Prover](https://github.com/Certora/CertoraProver), **or** a `CERTORAKEY` for cloud mode
- The Solidity compiler(s) your target project needs, on `$PATH` (see [Solidity compilers](#solidity-compilers))
- Access to Certora's private [`graphcore`](https://github.com/Certora/graphcore) repository (see below)

### Step 0 — Clone with submodules

`graphcore` is a private Certora dependency vendored in as a **git submodule**, not a plain pip package — a plain `git clone` will leave it empty and `uv sync` will fail. Clone (or fix up an existing clone) with:

```bash
git clone --recurse-submodules https://github.com/0xbrett8571/CertoraAIComposer.git
# or, if you already cloned without --recurse-submodules:
git submodule update --init
```

You'll need SSH access to `Certora/graphcore` for this to succeed.

### Step 1 — Provision the databases

From `scripts/`:

```bash
cd scripts/
docker compose create && docker compose start
```

This starts a `pgvector/pgvector:pg16` container pre-initialized (via `init-db.sql`) with the databases AI Composer uses: `rag_db` (CVL manual embeddings), `langgraph_store_db`, `langgraph_checkpoint_db` (workflow checkpointing/resume), `memory_tool_db`, and `audit_db` (run history). No attempt has been made to secure this database — treat it as local-dev-only. You'll need to restart the container whenever your host restarts (or adjust its restart policy).

### Step 2 — Build the RAG index

The LLM grounds its CVL usage in the actual Prover documentation via a local RAG index built from the docs themselves:

```bash
./gen_docs.sh               # builds the CVL manual HTML into prover-docs/
./populate_rag.sh           # populates the rag schema (used by AutoProve, AI Composer, cex-analyzer)
./populate_extended_rag.sh  # populates the extended_rag schema (CVL + Prover docs, used by sanity-analyzer)
```

There is only one physical database, `rag_db` — `populate_rag.sh` and `populate_extended_rag.sh` both write into it, as different PostgreSQL users (`rag_user` and `extended_rag_user`) each scoped via `search_path` to their own schema (`rag` and `extended_rag` respectively — see `composer/scripts/init-db.sql`).

The RAG is fully derived from the docs and is read-only at query time, so when the docs change you just rebuild:

```bash
./refresh_rag.sh                  # regenerate docs, then wipe + rebuild the rag schema
./refresh_rag.sh --all            # also rebuild the extended_rag schema
./refresh_rag.sh --skip-gen-docs  # rebuild from HTML already in prover-docs/
```

> [!NOTE]
> Run this offline — it empties the target schema before re-embedding, during which CVL manual search returns nothing.

### Step 3 — Build the Prover

From the root of the Certora Prover repo:

```bash
./gradlew copy-assets
```

Then point `CERTORA` at the build output (`CertoraProver/target`). If you'd rather skip a local build entirely, AutoProve supports `--cloud` mode with a `CERTORAKEY` instead.

### Step 4 — Install AI Composer's Python dependencies

```bash
uv sync --extra ml
```

If you're working in your own virtualenv rather than `uv`'s, also install the `certora-cli` requirements (`uv pip install -r certora_cli_requirements.txt` from the `CertoraProver/scripts` folder) and remember to activate that environment each time you run AI Composer. `certora-cli`, `certora-cli-beta`, and `certora-cli-beta-mirror` are mutually exclusive extras — pick exactly one at install time (`ai-composer[certora-cli]`, or the `prover` alias, for the stable channel).

### Solidity compilers

AI Composer expects `solc` on `$PATH` following the `solcX.Y` naming convention (`X.Y` from the version, e.g. `0.8.29` → `solc8.29`). The prompts currently default to targeting Solidity 0.8.29; adjust them if you need a different version.

---

## 📖 Usage

Each generation/verification tool has its own detailed doc; this section is a map, not the full reference.

### AutoProve — generate & verify CVL specs

Given a Solidity project, a contract, and a design document, AutoProve analyzes the system, formulates properties, generates CVL, and runs the Prover in a loop until the rules verify (or it needs your input).

```bash
tui-autoprove <project_root> <path/to/Contract.sol:ContractName> <design_doc>
# or, for CI/headless logging:
console-autoprove <project_root> <path/to/Contract.sol:ContractName> <design_doc>
```

Useful options include `--cloud` (run the Prover in Certora's cloud instead of locally), `--max-concurrent` (parallel agents for property extraction/CVL generation, default 4), `--cache-ns` (enable cross-run caching so repeated runs skip completed phases), and `--heavy-model`/`--lite-model`/`--tokens`/`--thinking-tokens` to tune the LLM calls (AutoProve has no single `--model` flag). Full details, including the five pipeline phases and cache/memory exploration, are in [`AUTOPROVE.md`](AUTOPROVE.md).

### AI Composer — generate code from a spec

The original workflow: given a CVL spec, an interface the implementation must satisfy, and a system description, AI Composer iterates with the LLM (proposing code, running the Prover, revising) until the spec verifies or it needs help via its human-in-the-loop tooling. It supports resuming a prior session from a checkpoint (each run has a thread ID and a sequence of checkpoint IDs it prints as it goes) so you can "time travel" back to an earlier decision, and a "meta-iteration" mode for refining a spec and re-running against a previous session's output.

This workflow is invoked through `composer.console.app`; see the module and [`TOOL_STATUS_AND_USAGE.md`](TOOL_STATUS_AND_USAGE.md) for current invocation details, and expect some rough edges — this part of the toolkit predates the AutoProve pipeline above and is being consolidated with it.

### Foundry test generation

```bash
tui-foundry <project_root> <path/to/Contract.sol:ContractName>
# or:
console-foundry <project_root> <path/to/Contract.sol:ContractName>
```

Generates a Foundry test suite for the given contract.

### CEX extraction & analysis

Covered in the [Quick Start](#-quick-start) above. Once the full RAG/DB setup is in place, the packaged `cex-analyzer` command (backed by `analyzer/analysis.py`) gives the same functionality with RAG-grounded analysis; the standalone `extract_and_analyze_cex.py` script works without any of that setup.

### Sanity Analyzer — diagnosing unsat rules

When a rule comes back **unsat** rather than verified/violated, the Prover emits an unsat core showing which constraints conflict. Sanity Analyzer asks Claude to explain the conflict and suggest a fix:

```bash
sanity-analyzer /path/to/report/Reports/UnsatCoreTAC-myRule-myMethod-description-0.txt
```

It needs the extended RAG schema (populated by `populate_extended_rag.sh`, into the same `rag_db` — see the RAG index section under [installation](#-full-installation) above) rather than a separate database. See [`sanity_analyzer/README.md`](sanity_analyzer/README.md) for the full CLI reference.

### Inspecting past runs

`ap-trail` lists and inspects prior AutoProve sessions recorded in the audit database; `cache-natspec` browses the cache/memory namespaces produced by the NatSpec pipeline; `scripts/traceDump.py` renders a full HTML visualization of a session's message history given its thread ID and the audit DB connection string.

---

## 🗂️ Repository layout

```text
composer/          Core library
├── prover/         Prover invocation & result parsing (local + cloud)
├── cvl/            CVL parsing/generation utilities
├── spec/           NatSpec & AutoProve pipelines, report rendering
├── foundry/        Foundry test-generation pipeline
├── rag/            RAG index build/query for the CVL manual
├── kb/             Knowledge-base utilities backing the RAG
├── input/, io/     Input parsing and VFS/materialization
├── audit/          Run history / audit database
├── diagnostics/    ap-trail and related inspection tools
├── human/          Human-in-the-loop tooling
├── workflow/       LangGraph workflow orchestration
├── ui/, console/   Textual TUI and console front ends
├── cli/            Console-script entry points
├── scripts/        RAG build/population scripts
└── templates/      Prompt templates (Jinja2)

analyzer/           Packaged CEX analyzer (`cex-analyzer`)
sanity_analyzer/    Unsat-core analyzer (`sanity-analyzer`)
extract_and_analyze_cex.py   Standalone, dependency-light CEX tool
graphcore/          Private Certora dependency (git submodule)
examples/           Sample inputs for the AI Composer / AutoProve workflows
scripts/            DB/RAG setup, cache tooling, trace visualization
tests/              pytest suite
```

---

## 🧪 Examples

`examples/` contains sample inputs you can use to sanity-check your setup:

| Example | Purpose |
|---|---|
| [`examples/trivial`](examples/trivial) | A minimal interface/spec/system-doc trio. The generated code isn't interesting, but it's the fastest way to confirm your AI Composer setup works end-to-end. |
| [`examples/cccp_fixed`](examples/cccp_fixed) | A realistic pool contract with a clean spec — good for a "does it verify" smoke test. |
| [`examples/cccp_buggy`](examples/cccp_buggy) | The same pool contract, but with malformed rules and syntax errors on purpose, to demonstrate the tooling's behavior on imperfect inputs. |

---

## 📚 Documentation map

| Document | Covers |
|---|---|
| [`README.md`](README.md) | The original overview doc (kept as-is in the repo) |
| [`README2.0.md`](README2.0.md) | This file — an expanded, reorganized overview, installation guide, and usage map |
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | Single ordered checklist from clean machine to working install (both tracks) |
| [`AUTOPROVE.md`](AUTOPROVE.md) | AutoProve pipeline: setup, arguments, options, pipeline phases, caching |
| [`SETUP_GUIDE.md`](SETUP_GUIDE.md) | Full, step-by-step setup instructions for the CEX tool |
| [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) | One-page CEX tool command cheat sheet |
| [`TOOL_STATUS_AND_USAGE.md`](TOOL_STATUS_AND_USAGE.md) | Tool-by-tool status and structure notes |
| [`VISUAL_SUMMARY.md`](VISUAL_SUMMARY.md) | Architecture and data-flow diagrams |
| [`CEX_AND_REQUIREMENTS_GUIDE.md`](CEX_AND_REQUIREMENTS_GUIDE.md) | Deep dive on `req_extraction_prompt.j2` and `cex_instructions.j2`, how CEX extraction works internally, and how to extract a CEX with or without AI Composer |
| [`sanity_analyzer/README.md`](sanity_analyzer/README.md) | Sanity Analyzer setup and CLI |
| [`MASTER_AUDIT_GUIDE_V2.md`](MASTER_AUDIT_GUIDE_V2.md), [`AUDIT_WORKFLOW.md`](AUDIT_WORKFLOW.md), [`BeforeStartingAnyAudit.md`](BeforeStartingAnyAudit.md) | Reference material for using AI Composer as part of a broader manual audit workflow |
| [`REFERENCE.md`](REFERENCE.md), [`cvl_guidelines.md`](cvl_guidelines.md), [`REAL_VULNERABILITY_EXAMPLES.md`](REAL_VULNERABILITY_EXAMPLES.md) | CVL/Prover reference material and real-world vulnerability case studies |

---

## 🛠️ Development

```bash
uv sync --group test --group ci
uv run pytest
uv run pyright
```

`uv sync --group test` pulls in `pytest`, `pytest-asyncio`, and `testcontainers`; `--group ci` adds `pyright`. See `tests/` for the existing suite, and `.github/workflows/` for the CI checks (lockfile consistency, submodule pin sync, `pyright`, `pytest`) that run on every PR.

---

## ⚠️ Known issues

- The original AI Composer workflow (spec → implementation) predates the AutoProve pipeline and its exact current invocation is in flux; check `composer/console/app.py` and [`TOOL_STATUS_AND_USAGE.md`](TOOL_STATUS_AND_USAGE.md) rather than relying solely on this README if something doesn't match.
- Two prompt-template gaps identified in [`CEX_AND_REQUIREMENTS_GUIDE.md`](CEX_AND_REQUIREMENTS_GUIDE.md) — a typo in `req_extraction_prompt.j2` and missing non-code-root-cause guidance in `cex_instructions.j2` — have since been fixed; see that guide's "Known gaps" section for anything still open.

---

## 🚦 Project status

AI Composer is a **research prototype** released by Certora Labs.

> [!WARNING]
> Code generated by AI Composer, and fixes suggested by the CEX/Sanity analyzers, should **not** be used in production without thorough independent review, testing, and auditing. The CEX Analysis tool is a standalone utility for extracting and explaining counterexamples from Certora Prover output; its suggested fixes are a starting point, not a substitute for verifying the fix actually holds.

---

## License

[MIT](LICENSE) © Certora, Inc.
