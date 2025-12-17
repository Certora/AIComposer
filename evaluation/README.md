# AI Composer – Governance Evaluation Module

This directory defines the **evaluation framework** for validating and improving **AI Composer** on real-world **Aave Governance AIPs**.

The evaluation module is designed to answer two core questions:

1. **Can AI Composer demonstrably reduce audit cost and risk for sophisticated governance customers (e.g., Aave)?**
2. **What are the recurring, fundamental failure modes of AI Composer when applied at scale to real proposals?**

---

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [Goals](#goals)
  - [Primary Goals](#primary-goals)
  - [Secondary Goals](#secondary-goals)
- [Evaluation Dataset](#evaluation-dataset)
  - [1. Governance SPEC (Input)](#1-governance-spec-input)
  - [2. Reference Implementation (Ground Truth)](#2-reference-implementation-ground-truth)
    - [Interface Resolution](#interface-resolution)
  - [3. Behavioral Oracle: Seatbelt (Mandatory)](#3-behavioral-oracle-seatbelt-mandatory)
  - [4. Certora Governance Review Metadata (Required Annotations)](#4-certora-governance-review-metadata-required-annotations)
  - [5. Proposal Metadata (Stratification Signals)](#5-proposal-metadata-stratification-signals)
- [Evaluation Pipeline](#evaluation-pipeline)
  - [Stage 0 — Data Preparation (High Risk)](#stage-0--data-preparation-high-risk)
  - [Stage 1 — AI Composer Generation](#stage-1--ai-composer-generation)
  - [Stage 2 — Formal Equivalence Checking (Phase 1 Ranker)](#stage-2--formal-equivalence-checking-phase-1-ranker)
  - [Stage 3 — Behavioral Comparison via Seatbelt (Phase 2)](#stage-3--behavioral-comparison-via-seatbelt-phase-2)
  - [Stage 4 — Concordance (Phase 3, Optional)](#stage-4--concordance-phase-3-optional)
- [Development Phases](#development-phases)
  - [Phase 1 (MVP)](#phase-1-mvp)
  - [Phase 2](#phase-2)
  - [Phase 3](#phase-3)
- [Metrics](#metrics)
  - [Per-AIP](#per-aip)
  - [Aggregate](#aggregate)
- [Known Gaps & Open Questions (Tracked)](#known-gaps--open-questions-tracked)
  - [Spec & Prompting](#spec--prompting)
  - [Human-in-the-Loop Dependency](#human-in-the-loop-dependency)
  - [Technical Gaps](#technical-gaps)
    - [Composer Gaps](#composer-gaps)
    - [Evaluation Flow Gaps](#evaluation-flow-gaps)
    - [Scalability](#scalability)
  - [Score Normalization and Aggregation](#score-normalization-and-aggregation)
    - [Problem Statement](#problem-statement)
    - [Immediate Scoring Solution (v1, Subject to Revision)](#immediate-scoring-solution-v1-subject-to-revision)
      - [Track A — Correctness Scoring](#track-a--correctness-scoring)
        - [Primary Score: Equivalence (0–1)](#primary-score-equivalence-01)
        - [Secondary Score: Seatbelt Similarity (0–1) *(Phase 2)*](#secondary-score-seatbelt-similarity-01-phase-2)
        - [Total Score](#total-score)
        - [Non-Determinism Handling](#non-determinism-handling)
      - [Track B — Bug Sensitivity Scoring](#track-b--bug-sensitivity-scoring)
        - [Bug Sensitivity Score (0–1)](#bug-sensitivity-score-01)
      - [Scoring Summary (At a Glance)](#scoring-summary-at-a-glance)
- [Success Criteria (Phase 1)](#success-criteria-phase-1)
- [High-Risk Components (Explicit)](#high-risk-components-explicit)
- [Status](#status)

---

## High-Level Overview

Each evaluation datapoint corresponds to **one Aave Improvement Proposal (AIP)** and combines:

* A natural-language governance SPEC
* A ground-truth on-chain implementation
* Formal equivalence checking
* Behavioral comparison via Seatbelt
* Expert audit annotations from Certora governance reviews

The evaluation pipeline is **incremental by design**:

* v1 focuses on **AI Composer + Equivalence**
* v2 adds **Seatbelt diffing**
* v3 integrates **Concordance-based normalization**

---

## Goals

### Primary Goals

1. **Demonstrate value quickly**
   Show that for AIPs where Certora found *no bugs*, AI Composer can:

   * Generate correct implementations
   * Reduce audit time
   * Improve early detection of governance risks

2. **Detect systemic weaknesses at scale**
   Identify repeated Composer failure modes across many real AIPs:

   * Spec misinterpretation
   * Missing assumptions
   * Environment modeling gaps
   * Pattern-generalization failures

### Secondary Goals

* Provide a **reproducible benchmark** for Composer regression testing
* Surface **product gaps** before scaling to customers
* Enable future **model-to-model benchmarking**

---

## Evaluation Dataset

Each datapoint (AIP) consists of the following **mandatory artifacts**.

### 1. Governance SPEC (Input)

* Source: Aave governance portal (e.g., [https://vote.onaave.com](https://vote.onaave.com))
* Format: Raw governance description (natural language)
* Used *as-is*, reflecting real customer input

---

### 2. Reference Implementation (Ground Truth)

* Source: [https://github.com/bgd-labs/aave-proposals-v3](https://github.com/bgd-labs/aave-proposals-v3)
* Includes:

  * Solidity payload contract(s)
  * Execution entrypoints

#### Interface Resolution

AI Composer requires explicit interfaces.
Interfaces are deterministically derived from:

* Payload inheritance tree
* Aave governance executor interfaces
* Interfaces used in the reference repository

**Interface discovery is logged and versioned**, as it is a critical and error-prone step.

---

### 3. Behavioral Oracle: Seatbelt (Mandatory)

Seatbelt computes **storage diffs between pre- and post-execution states**.

For evaluation:

1. Run Seatbelt on the **reference implementation** → `seatbelt_ref`
2. Run Seatbelt on the **AI Composer output** → `seatbelt_gen`
3. Compare the **two Seatbelt outputs**

Seatbelt is therefore used as a **diff-of-diffs behavioral oracle**, not a single validator.

References:

* Seatbelt repo: [https://github.com/bgd-labs/seatbelt-gov-v3](https://github.com/bgd-labs/seatbelt-gov-v3)
* Seatbelt README: explains execution model and diff semantics

---

### 4. Certora Governance Review Metadata (Required Annotations)

Source: Certora internal governance review board (Monday).

This is the **annotation layer** for evaluation and is **mandatory**.

Includes:

* Whether bugs were found
* Bug categories (logic, config, edge cases, assumptions)
* Expert audit notes
* Links to proposals
* (Planned) **Audit time / effort tracking**

This allows distinguishing:

* Composer failures on *easy* proposals
* Composer failures where humans also struggled

---

### 5. Proposal Metadata (Stratification Signals)

Each datapoint must include:

* Proposal author / proposer
* Execution chain (mainnet, L2, etc.)
* Proposal type (per Monday taxonomy; more granular than logic/config)
* Affected protocol modules
* Execution pattern (single payload, multi-call, cross-module)

This supports the hypothesis that **some proposal classes are inherently easier than others**.

---

## Evaluation Pipeline

### Stage 0 — Data Preparation (High Risk)

* Collect all artifacts
* Resolve interfaces
* Normalize execution environment
* Produce a **single canonical YAML datapoint**

> Data preparation is explicitly treated as a fragile, observable component.

---

### Stage 1 — AI Composer Generation

**Inputs**

* Governance SPEC
* Interface file(s)
* System documentation (Aave governance context)
* Composer configuration (model, tokens, etc.)

**Outputs**

* Generated Solidity payload(s)
* Composer trace metadata:

  * Iterations
  * Human interventions
  * Prompt variants
  * Failure signals

All outputs must be **materialized and frozen**.

---

### Stage 2 — Formal Equivalence Checking (Phase 1 Ranker)

* Run Certora Equivalence Checker:

  * Generated vs reference implementation
* Outputs:

  * PASS / FAIL / INCONCLUSIVE
  * Counterexamples
  * Diff localization

This is the **first and mandatory ranking signal**.

---

### Stage 3 — Behavioral Comparison via Seatbelt (Phase 2)

* Compare `seatbelt_gen` vs `seatbelt_ref`
* Output:

  * Storage diff mismatches
  * Behavioral severity classification

---

### Stage 4 — Concordance (Phase 3, Optional)

* If equivalence fails:

  * Apply Concordance to simplify one or both implementations
  * Re-run equivalence checking

Concordance is **not required in v1**, but the pipeline is designed to support it.

Reference:

* Concordance docs: [https://github.com/Certora/CertoraProver/tree/master/scripts/concordance](https://github.com/Certora/CertoraProver/tree/master/scripts/concordance)

---

## Development Phases

### Phase 1 (MVP)

* AI Composer
* Equivalence checker
* Governance annotations
* YAML dataset
* Minimal automation

### Phase 2

* Seatbelt diff comparison
* Proposal stratification metrics
* Automated scoring

### Phase 3

* Concordance-assisted equivalence
* Advanced ranking / reranking
* Partial end-to-end automation

---

## Metrics

### Per-AIP

* Equivalence result
* Seatbelt diff result
* Composer iterations / interventions
* Alignment with audit annotations
* (Planned) Audit time vs baseline

### Aggregate

* Success rate by proposal type
* Failure clustering
* Regression across Composer versions / models

---

## Known Gaps & Open Questions (Tracked)

### Spec & Prompting

* Is governance text alone sufficient input for Composer? according to the current dev-phase of the Composer: No.
* Are intermediate spec transformations required?
* Do we need few-shot examples from prior AIPs?
* Risk of overfitting vs generalization


### Human-in-the-Loop Dependency

AI Composer currently expects human guidance.

Challenges:

1. Reproducibility for benchmarking
2. Parallelism at scale

Potential mitigations:

* Scripted “virtual operator”
* Bounded intervention modes
* Replayable guidance traces

### Technical Gaps

#### Composer Gaps
* Missing assumptions (permissions, caller context, execution environment)?
* Inability to infer Aave-specific governance patterns?

#### Evaluation Flow Gaps
* What does equivalence not catch that Seatbelt does (or vice versa)?
* Are there proposal classes that systematically fail?

#### Scalability
* Other than the Human in the loop dependency, can this pipeline run fully unattended?

### Score Normalization and Aggregation
#### Problem Statement

The evaluation pipeline produces multiple heterogeneous signals:
* **Equivalence checking** (hard, formal correctness)
* **Seatbelt behavioral comparison** (state-level similarity)
* (Future) **Concordance-assisted equivalence**

These signals differ in:
* **Semantics** (formal, machine-checked equivalence results vs. behavioral similarity inferred from storage diffs)
* **Structure and resolution** (categorical outcomes vs. rich, structured diff data from which graded similarity metrics can be deterministically derived)
* **Availability** across development phases


The core challenge is to convert them into numeric scores that are:
* **Normalized** across AIPs, proposal types, and execution chains
* **Reproducible** despite LLM **non-determinism**
* **Verifiable** from stored artifacts
* **Stable** enough for regression testing and model comparison (for example, seed-wise)

In addition, we need to ask ourself: how do we score AIPs with known bugs without corrupting correctness benchmarks?

### Immediate Scoring Solution (v1, Subject to Revision)

We adopt a **two-track scoring model** that separates:

* **Correctness measurement**
* **Bug sensitivity measurement**

This avoids mixing incompatible semantics.


#### Track A — Correctness Scoring

*(AIPs where Certora found no bugs)*

This track is used for:

* Customer-facing claims
* Regression testing
* Progress tracking

##### Primary Score: Equivalence (0–1)

* `1.00` — Equivalence **PASS**
* `0.25` — Equivalence **INCONCLUSIVE**
* `0.00` — Equivalence **FAIL**

This reflects that inconclusive proofs are weaker than passes but not equivalent to failure.

##### Secondary Score: Seatbelt Similarity (0–1) *(Phase 2)*

Computed by comparing Seatbelt outputs of:

* Reference implementation
* AI Composer–generated implementation
* `1.00` — Identical storage diffs
* Otherwise:
  `1 / (1 + D)`
  where `D` is a deterministic diff magnitude (e.g., number of mismatching storage slots)

##### Total Score

* **Phase 1 (MVP):**
  `Total = Primary`
* **Phase 2+:**
  `Total = Primary × Secondary`

Equivalence acts as a **hard gate**; behavioral similarity only refines scores when implementations are close to correct.

##### Non-Determinism Handling

For each AIP, run AI Composer `k` times (initially `k = 3`) and report:

* **Best-of-k score** → measures *potential*
* **Median-of-k score** → measures *reliability*

Only stored artifacts are used in scoring.

#### Track B — Bug Sensitivity Scoring

*(AIPs where Certora found bugs)*

This track is **diagnostic only** and never combined with correctness scores.

The question here is:

> Does the evaluation pipeline surface a divergence signal consistent with known bugs?

##### Bug Sensitivity Score (0–1)

| Condition                                    | Score |
| -------------------------------------------- | ----- |
| Equivalence **FAILS**                        | `1.0` |
| Equivalence **INCONCLUSIVE**                 | `0.7` |
| Equivalence **PASSES**, Seatbelt **DIFFS**   | `0.5` |
| Equivalence **PASSES**, Seatbelt **MATCHES** | `0.0` |

Higher scores indicate **better sensitivity to faulty proposals**.

Optionally (v2), a small bonus may be added if the detected divergence aligns with the bug location noted in Certora’s review.

#### Scoring Summary (At a Glance)

* **Clean AIPs** → scored for *correctness*
* **Buggy AIPs** → scored for *sensitivity*
* Scores are:

  * Numeric
  * Artifact-derived
  * Reproducible
  * Phase-aware
* Equivalence is always the **primary signal**
* Seatbelt and Concordance refine, but never replace, formal correctness

This scoring framework provides a **defensible baseline** that can evolve as empirical data accumulates, without invalidating early results.



---

## Success Criteria (Phase 1)

The evaluation module is successful when:

1. For AIPs where **Certora found no bugs**, AI Composer achieves a **high equivalence pass rate**
2. We produce a **ranked list of recurring Composer failure modes**
3. The pipeline runs end-to-end with **minimal manual glue**
4. We can evaluate whether higher AI Composer score aligns with better governance review outcomes (fewer issues and lower recorded effort), **and use that correlation to support triage and early risk flagging**. 
    
    In practice, we treat the Monday governance review metadata as an audit-outcome proxy (bugs found, severity/notes, and - when available - time spent or review duration). We treat the evaluation outputs as an objective quality proxy (equivalence PASS/FAIL/INCONCLUSIVE, Seatbelt diff magnitude, and an aggregated correctness score for clean AIPs). We then test whether these signals move together across many AIPs—for example, whether high Composer scores tend to coincide with proposals where auditors reported fewer issues or lower effort, and whether low scores cluster around complex or bug-prone proposals. This statistical correlation makes the score defensible and operationally useful: it can help prioritize human review and surface risky proposals earlier without over-claiming that Composer directly reduces audit time.


---

## High-Risk Components (Explicit)

* **Data preparation**
* **Interface discovery**
* **Evaluation orchestration**
* **I/O schema stability (YAML)**

These components must be observable, testable, and versioned.

---

## Status

This evaluation framework is **actively evolving**.
Not all gaps are solved; they are intentionally documented to guide iteration.
