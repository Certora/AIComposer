# Vacuity Analyzer

An AI-powered tool for diagnosing and explaining vacuity issues in unsatisfiable Certora Prover rules.

## What It Does

When the Certora Prover encounters an unsatisfiable (unsat) rule, it generates an unsat core that shows which constraints are conflicting. The Vacuity Analyzer uses Claude to:

- Analyze the unsat core and identify the root cause of unsatisfiability
- Explain why the constraints cannot be satisfied together
- Suggest concrete solutions including configuration changes, CVL modifications, or implementation fixes
- Provide structured explanations with detailed analysis

## Installation

From the repository root:

```bash
pip install -e .
```

This will install the `vacuity-analyzer` command.

## Prerequisites

Before using the vacuity analyzer, you need to set up the extended RAG database. See the [main README](../README.md) for instructions on:

1. Setting up the PostgreSQL database
2. Building the extended documentation (`gen_docs_extended.sh`)
3. Populating the extended RAG database

## Usage

### Basic Usage

The simplest usage automatically extracts rule and method names from the filename:

```bash
vacuity-analyzer /path/to/report/Reports/UnsatCoreTAC-myRule-myMethod-description-0.txt
```

### Explicit Rule and Method

You can also specify the rule and method explicitly:

```bash
vacuity-analyzer /path/to/report/Reports/unsat_core.txt --rule myRule --method myMethod
```

### Filename Format

The analyzer expects unsat core files to follow the naming pattern:
```
UnsatCoreTAC-{rule}-{method}-{description}-{counter}.txt
```

For rules without a method:
```
UnsatCoreTAC-{rule}-{description}-{counter}.txt
```

## Options

- `--rule NAME` - Rule name (extracted from filename if not provided)
- `--method NAME` - Method name (extracted from filename if not provided)
- `--quiet` - Suppress intermediate output, only show final result
- `--recursion-limit N` - Maximum workflow iterations (default: 30)
- `--thinking-tokens N` - Claude thinking token budget (default: 2048)
- `--tokens N` - Claude output token budget (default: 4096)
- `--rag-db CONNECTION` - Override RAG database connection string
- `--thread-id ID` - Resume from a specific thread
- `--checkpoint-id ID` - Resume from a specific checkpoint

## Output Format

The analyzer produces a structured analysis with:

### Root Cause
Brief summary of what's causing the unsatisfiability

### Detailed Analysis
Step-by-step explanation of:
- Which constraints in the unsat core are conflicting
- Why they create a logical contradiction
- How the conflict manifests

### Solution
Concrete recommendations including:
- Configuration flag changes with explanations
- CVL rule modifications if needed
- Implementation changes if needed

### Summary
Quick overview with:
- Issue type (e.g., "Prover Configuration", "Specification Issue")
- One-line root cause
- Impact on verification
- Actionable fix steps
- Expected outcome

## Examples

### Example 1: Basic Usage
```bash
vacuity-analyzer ./my_report/Reports/UnsatCoreTAC-sanity-checkBalance-Satisfy_end_of_methods-0.txt
```

### Example 2: With Explicit Parameters
```bash
vacuity-analyzer ./Reports/unsat_core.txt \
  --rule sanity \
  --method MyContract.checkBalance \
  --quiet
```

### Example 3: Resuming a Session
```bash
vacuity-analyzer ./Reports/unsat_core.txt \
  --thread-id vacuity-analysis-abc123 \
  --checkpoint-id checkpoint-def456
```

## Troubleshooting

### "Could not determine rule name from filename"
The filename doesn't match the expected pattern. Use `--rule` to specify explicitly.

### "Expected txt file to be in 'Reports' directory"
The unsat core file must be in a directory named "Reports" within the report structure.

### Database Connection Errors
Ensure the extended RAG database is set up correctly. See the main README for setup instructions.

### "Unsat txt file not found"
Check that the path to the unsat core file is correct and the file exists.

## Environment Variables

- `ANTHROPIC_API_KEY` - Required. Your Claude API key.

## Technical Details

The vacuity analyzer uses:
- **Claude Sonnet 4.5** for analysis with extended thinking
- **Extended RAG database** with CVL manual + prover documentation
- **LangGraph** for workflow orchestration
- **PostgreSQL with pgvector** for documentation search

The analyzer has access to:
- CVL specification language documentation
- Prover CLI options and configuration
- Sanity checking documentation
- Loop handling and approximation techniques
- Hashing and coverage information
- Source files from the verification report

## Related Tools

- **cex-analyzer** - Analyzes counterexamples for violated rules
- **AI Composer** - Generates verified implementations from specifications
