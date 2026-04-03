# Auto-Prove Pipeline

Auto-prove is a multi-agent pipeline that automatically generates and verifies CVL specifications for Solidity smart contracts. Given a project root, a main contract, and a design document, it analyzes the system's components, formulates properties, and generates CVL specs — running the Certora Prover in a loop to verify them.

## Prerequisites

You need everything from the main [README](README.md) setup:

- Python 3.12+, `uv`, Docker with compose
- `ANTHROPIC_API_KEY` in your environment
- PostgreSQL databases running (see below)
- RAG database populated
- Solidity compiler(s) on `$PATH` (naming convention `solcX.Y`, e.g. `solc8.29`)

### PreAudit

Auto-prove depends on PreAudit for compilation analysis and harness generation. PreAudit is a separate internal repository. Clone it and set the `PREAUDIT_PATH` environment variable to its root directory:

```bash
export PREAUDIT_PATH=/path/to/preaudit
```

The pipeline will fail at import time if this is not set.

### Certora Prover

**Cloud mode (`--cloud`):** You simply need a CERTORAKEY set in your environment.

**Local mode (default):** You need a full local prover build. From the prover repo root, run `./gradlew copy-assets` and set `CERTORA` to point to `CertoraProver/target`.

NB: you still should specify the `--group prover` flag to `uv` as it forces the installation of the dependences of the certora prover python.

## (One Time) Database Setup

### 1. Start PostgreSQL

From the `scripts/` directory:

```bash
cd scripts/
docker compose create && docker compose start
```

This launches a `pgvector/pgvector:pg16` container with all databases pre-initialized via `init-db.sql`. The databases created are:

| Database | Purpose |
|---|---|
| `rag_db` | CVL manual search (pgvector embeddings) |
| `langgraph_store_db` | LangGraph document/index store |
| `langgraph_checkpoint_db` | Workflow checkpoints for resumption |
| `memory_tool_db` | LLM context memory (hierarchical filesystem) |
| `audit_db` | Execution history and prover results |

By default the databases are on `localhost:5432`. Override with environment variables if needed:

```bash
export CERTORA_AI_COMPOSER_PGHOST=myhost
export CERTORA_AI_COMPOSER_PGPORT=5433
```

Note: the RAG database has its own connection string (default `postgresql://rag_user:rag_password@localhost:5432/rag_db`), overridable via `--rag-db`.

### 2. Populate the RAG Knowledge Base

The RAG database must be populated before running auto-prove. This is a two-step process:

```bash
# Build the Certora documentation HTML
./scripts/gen_docs.sh

# Populate the RAG DB from the CVL documentation
./scripts/populate_rag.sh
```

`gen_docs.sh` clones the [Certora Documentation](https://github.com/Certora/Documentation) repo and builds single-page HTML files via Sphinx. `populate_rag.sh` then chunks and indexes the CVL documentation into `rag_db`.

### 3. Populate the CVL Knowledge Base

Run the knowledge base population script to load common CVL pitfall articles into the LangGraph store:

```bash
python scripts/kb_populate.py
```

This inserts ~30 curated articles (summary misapplication, vacuity traps, ghost semantics, etc.) that agents consult during spec generation.

## Installation

```bash
uv sync --group ml --group prover
```

The `ml` group includes `sentence-transformers` and `einops`, required for the embedding model (`nomic-embed-text-v1`) used by RAG and the indexed store. `prover` includes all of the dependencies
for running the prover scripts (in local mode), and the certoraRun scripts themselves (cloud mode).

## Usage

Auto-prove has two entry points: a Textual-based TUI and a headless console mode.

### TUI Mode

```bash
python tui_autoprove.py <project_root> <path/to/Contract.sol:ContractName> <design_doc>
```

### Console Mode

```bash
python console_autoprove.py <project_root> <path/to/Contract.sol:ContractName> <design_doc>
```

Console mode prints the same pipeline output to stdout without the interactive TUI. Useful for CI or logging. NB if you need to do `print` debugging, the `console_autoprove.py` is your best
bet. Debugging in the tui_autoprove.py workflow *mandates* the use of a python logger.

### Arguments

| Argument | Description |
|---|---|
| `project_root` | Root directory of the Solidity project |
| `main_contract` | Path to the contract file and contract name, separated by `:`. The path must be relative to or within `project_root`. Example: `src/Token.sol:Token` |
| `system_doc` | Path to a design document (plain text or PDF) describing the system |

### Options

| Option | Default | Description |
|---|---|---|
| `--max-concurrent` | 4 | Maximum number of parallel agents for property extraction and CVL generation |
| `--cache-ns` | None | Cache namespace string. Enables cross-run caching so repeated runs skip completed phases |
| `--memory-ns` | None | Memory namespace. Defaults to the auto-generated thread ID |
| `--cloud` | off | Run prover jobs in the Certora cloud instead of locally |
| `--model` | `claude-opus-4-6` | Anthropic model to use |
| `--tokens` | 10000 | Token budget for LLM responses |
| `--thinking-tokens` | 2048 | Thinking token budget |
| `--rag-db` | `postgresql://rag_user:rag_password@localhost:5432/rag_db` | RAG database connection string |

### Example

```bash
python tui_autoprove.py \
    ~/projects/my-defi-protocol \
    src/Vault.sol:Vault \
    docs/vault-design.pdf \
    --cloud \
    --max-concurrent 2 \
    --cache-ns my-vault-run
```

## Pipeline Phases

The pipeline executes the following phases in order:

### Phase 0: System Analysis

Analyzes the source code to identify the system's components, contracts, and external actors. Uses filesystem exploration tools to read and understand the codebase. The result is cached and feeds into all subsequent phases.

### Phase 1: Harness Setup

Runs PreAudit to analyze compilation and classify external contracts (ERC20s, interfaces, etc.). Generates harness contracts and a prover configuration (`compilation_config.conf`). Also produces summaries for known external contracts.

### Phase 2: Custom Summaries

Generates CVL summaries for ERC20 contracts and external interfaces discovered in Phase 1. Only runs if the system has external contracts that need summarizing.

### Phase 3: Structural Invariants

Formulates and generates CVL for system-wide structural invariants (e.g. total supply consistency, balance accounting). The resulting `certora/invariants.spec` is made available as a resource that later phases can import and use as preconditions.

### Phase 4: Per-Component Property Extraction (parallel)

For each component identified in Phase 0, an agent analyzes the code and formulates properties to verify. Runs in parallel, bounded by `--max-concurrent`. Produces a list of property formulations per component.

### Phase 5: Per-Component CVL Generation (parallel)

For each component's properties, an agent generates CVL specs and runs the prover to verify them. Failed specs are revised in a feedback loop. Results are written to `certora/autospec_{n}.spec` with accompanying commentary files. Also bounded by `--max-concurrent`.

### Output

Auto-prove writes its output into the `certora/` directory within the project root:

- `certora/invariants.spec` — structural invariants (if any were formulated)
- `certora/autospec_0.spec`, `certora/autospec_1.spec`, ... — per-component specs
- `certora/*.commentary.md` — LLM commentary explaining each spec

The pipeline returns an `AutoProveResult` with counts of components analyzed, properties generated, and any failures.

## Caching

When `--cache-ns` is provided, auto-prove caches the results of expensive phases (system analysis, property extraction, invariant CVL generation) in the LangGraph store. On subsequent runs with the same `--cache-ns`, cached results are reused if the inputs (project root, contract path, design doc content) haven't changed.

The cache key is derived from a SHA-256 hash of the project root, design document content, contract path, and contract name. Changing any of these invalidates the cache.

### Exploring the cache/memory

You can view the cache/memory of the various phases by running `scripts/autoprove_cache_explorer.py`. This script takes the same positional arguments as the `*_autoprove.py` entrypoints, as well as the `--memory-ns` and `--cache-ns`.
This should load up a TUI frame that lets you inspect the currently cached values. You can toggle to memory mode to view (and potentially edit) the memories of the various sub agents.

### Debugging Agents

In headless mode, the CVL Generation agents will produce a mnemonic name. You can pass this mnemonic into the `snapshot_viewer.py` to view the conversation history of that agent.