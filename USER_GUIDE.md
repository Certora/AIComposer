# Certora AIComposer — User Guide

> Complete theory, architecture, and advanced usage. For a quick start, see [GETTING_STARTED.md](./GETTING_STARTED.md).

---

## Table of Contents

1. [Architecture & Core Concepts](#architecture--core-concepts)
2. [The Two Workflows](#the-two-workflows)
3. [Auto-Prove Pipeline: Deep Dive](#auto-prove-pipeline-deep-dive)
4. [How Property Extraction Works](#how-property-extraction-works)
5. [The CVL Generation Feedback Loop](#the-cvl-generation-feedback-loop)
6. [The Auditor Mindset: Hunt Impact, Not Bugs](#the-auditor-mindset-hunt-impact-not-bugs)
7. [The 12 Business Logic & Infrastructure Bug Categories](#the-12-business-logic--infrastructure-bug-categories)
8. [Catastrophic Failure Scenarios](#catastrophic-failure-scenarios)
9. [Working Without AutoSetup](#working-without-autosetup)
10. [Caching & Resumption](#caching--resumption)
11. [Codegen: Generating Implementations from Specs](#codegen-generating-implementations-from-specs)

---

## Architecture & Core Concepts

### System Architecture

```
┌──────────────────────────────────────────────────────────┐
│  ENTRY POINTS                                             │
│  main.py (codegen)  │  console_autoprove.py (auto-prove) │
├──────────────────────────────────────────────────────────┤
│  IO LAYER (composer/io/)                                  │
│  Event-driven handler protocol, nested graph execution,   │
│  parallel task orchestration                               │
├──────────────────────────────────────────────────────────┤
│  WORKFLOW LAYER (composer/spec/, composer/workflow/)      │
│  Pipeline phases, component analysis, property extraction │
│  CVL generation, feedback loops, caching hierarchy        │
├──────────────────────────────────────────────────────────┤
│  TOOLS LAYER (composer/tools/)                            │
│  Prover, CVL RAG search, HITL, spec changes, VFS          │
├──────────────────────────────────────────────────────────┤
│  GRAPHCORE (graphcore/)                                   │
│  Generic LLM agent framework: Builder pattern,            │
│  state graphs, summarization, tool loops                  │
├──────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE                                           │
│  PostgreSQL (checkpoints, store, RAG, audit, memory)      │
│  LangGraph, LangChain, Anthropic API                       │
└──────────────────────────────────────────────────────────┘
```

### The graphcore Foundation

graphcore is a **reusable agent framework** with zero domain knowledge about Solidity or CVL. It provides:

- **Builder pattern** — Fluent, type-safe construction of LLM-agent graphs
- **State graph** — `initial → LLM → tools → tool_result → LLM → ...` with automatic completion detection
- **Summarization** — Automatic context window compaction when token limits are exceeded
- **VFS (Virtual File System)** — Sandboxed file operations for the LLM

The state graph pattern:

```
initial ──→ LLM call ──→ tools ──→ tool_result ──→ LLM call ──→ ...
                         ↑                                    │
                         └────────────────────────────────────┘
                                        (loop)

Completion: when output_key field becomes non-None
Summarization: when token count exceeds threshold → compact history
```

---

## The Two Workflows

### Auto-Prove: Code → Specifications

**Goal**: Given source code and a design document, extract security properties, generate CVL specifications, and find bugs.

```bash
console-autoprove <project_root> <path/to/Contract.sol:ContractName> <design_doc>
```

The pipeline runs **5 phases** (detailed below), with parallel property extraction and CVL generation. Output goes to `certora/` in the project root.

### AIComposer (Codegen): Specifications → Code

**Goal**: Given a CVL specification, a Solidity interface, and a system description, generate an implementation that satisfies the spec.

```bash
python3 ./main.py cvl_input.spec interface_file.sol system_doc.txt
```

The LLM iterates: write code → run prover → analyze counterexamples → fix → repeat until all rules pass.

---

## Auto-Prove Pipeline: Deep Dive

### Phase 0: System Analysis

The LLM reads source code and the design document to identify:
- Explicit contracts (their components, state, functions)
- External actors (ERC20s, oracles, governance tokens)
- Dependencies between components

Output: An `Application` model that feeds all subsequent phases.

### Phase 1-2: Setup & Summaries

**Harness Creation** generates harness contracts for compilation. **AutoSetup** analyzes compilation output and classifies external contracts. **Custom Summaries** generates CVL method summaries for ERC20s and external interfaces.

### Phase 3: Structural Invariants

Formulates system-wide invariants (e.g., total supply consistency, balance accounting). These become `invariants.spec` — importable by later per-component specs as preconditions.

### Phase 4: Per-Component Property Extraction (parallel)

For each component, an LLM agent extracts properties in multiple **rounds**:

```
Round 1: Broad sweep — analyze component, extract invariants + safety + attack vectors
Round 2: New angles — review Round 1 results, look for missed edge cases
Round 3+: Convergence — continue until output is empty (no new properties found)
```

Output per component: `{component}.properties.json` with `title`, `sort`, `methods`, `description` for each property.

Runs in parallel across components, bounded by `--max-concurrent`.

### Phase 5: Per-Component CVL Generation (parallel)

For each component's properties, an agent:
1. Gathers component info from the VFS and phase results
2. Writes a CVL specification (rules + invariants)
3. Gets feedback from the **CVL Judge** (see below)
4. Refines based on feedback
5. Runs the Certora Prover
6. Iterates until all properties are verified or skipped

Output per component: `autospec_{component}.spec`, `.conf`, `.commentary.md`.

---

## How Property Extraction Works

### The Multi-Round Convergence Pattern

The key insight: **"nothing new" is the right answer when it's true.** The system prompt guides the LLM to:
- Prioritize quality over quantity
- Never pad the list
- Return empty when converged
- Every property needs defensible reasoning

### Property Categories

1. **Invariants** — Representational invariants that should always hold. Good: "Sum of all pool reserves ≥ sum of LP token backing." Bad: "The state fields should be correct."

2. **Safety Properties** — Concrete statements about what should not be possible. Good: "A user withdrawal decreases their LP token balance by proportional amount." Bad: "A user should not be able to hack the protocol."

3. **Attack Vectors** — Potential issues/edge cases that could be exploited. Good: "Malicious actor could use stale oracle price to unbalance pool." Bad: "The protocol could be hacked."

### Avoid in Property Extraction

- Restating prior properties with different words
- Non-security-relevant properties
- Trivial properties implied by the type system (e.g., overflow assertions)
- Off-chain events, hash collisions, event emission (not formally verifiable)

---

## The CVL Generation Feedback Loop

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Agent writes │────▶│ CVL Judge    │────▶│ Agent revises│
│ CVL spec     │     │ reviews      │     │ based on     │
│              │     │              │     │ feedback     │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                         │
       │         ┌──────────────┐                │
       └────────▶│ Certora      │◀───────────────┘
                 │ Type Check   │
                 └──────┬───────┘
                        │ (passes)
                 ┌──────▼───────┐
                 │ Certora      │
                 │ Prover       │
                 └──────────────┘
```

### What the CVL Judge Evaluates

| Criterion | Description |
|-----------|-------------|
| **Code Smells** | Tautologies, pointless bounds (e.g., `uint256 >= 0`) |
| **CVL Guidelines** | Compliance with best practices |
| **Precondition Validity** | Sufficient input constraints |
| **Overconstrained Inputs** | May hide real bugs |
| **Trivializing Summaries** | Don't abstract away crucial behavior |
| **Manifest Errors** | Obvious contradictions with design |
| **Property Coverage** | All properties addressed or skipped with justification |

### The Rebuttal Mechanism

The generating agent can push back with evidence:

| Evidence Type | Weight | Example |
|---------------|--------|---------|
| `typecheck_failure` | Strong | "Your suggestion doesn't compile" |
| `counterexample` | Strong | "Your fix doesn't resolve the violation" |
| `manual_citation` | Strong | "The CVL manual says otherwise" |
| `reasoned` | Weak | Logical argument |

---

## The Auditor Mindset: Hunt Impact, Not Bugs

### The Paradigm Shift

**Most high-value findings in modern audits are business logic bugs, not reentrancy, simple oracle, or overflow bugs.** Analysis of over $1B in real-world DeFi losses shows the majority of catastrophic exploits trace back to **protocol design flaws and broken invariants** — not classic coding mistakes that static analyzers catch.

When building a design document, don't just ask:

> *"What does the contract do?"*

Also ask:

> *"If this invariant breaks, can someone **steal money, print money, freeze money, or brick the protocol**?"*

Those answers become your highest-value properties for AIComposer and the Certora Prover.

### Old Thinking vs. New Thinking

| Old (Bug Hunting) | New (Impact Hunting) |
|---|---|
| "Find reentrancy" | "Can someone steal funds?" |
| "Find overflow" | "Can someone create value from nothing?" |
| "Find oracle manipulation" | "Can someone break the protocol's solvency?" |
| "Find access control gaps" | "Can someone take over governance?" |
| Mindset: "What's wrong with this code?" | Mindset: "What can break this protocol?" |
| Value: $5k-$20k per bug | Value: $10k-$50k+ per impact |

A protocol can be **completely bug-free** (no reentrancy, no overflow, no access control gaps) yet still have **devastating business logic vulnerabilities** — asset conservation violations, solvency failures, share price manipulation, state machine bypasses.

### The Properties That Find the Most Severe Bugs

In practice, the properties that uncover the highest-value findings are:

1. **Asset conservation** — "Can value be created from nothing?"
2. **Solvency** — "Can liabilities exceed assets?"
3. **Share/accounting integrity** — "Can shares be inflated or accounting broken?"
4. **State machine correctness** — "Can invalid state transitions occur?"
5. **Cross-contract invariant preservation** — "Do multiple contracts stay in sync?"
6. **Access control** — "Can unauthorized actors perform privileged actions?"
7. **Fund lockup prevention** — "Can funds become permanently inaccessible?"

These seven categories cover the vast majority of devastating business-logic exploits seen in modern DeFi.

---

## The 12 Business Logic & Infrastructure Bug Categories

> Informed by an analysis of 50 real-world DeFi attacks totaling over $1B in losses (arXiv:2507.20175,
> "SoK: Root Cause of $1 Billion Loss in Smart Contract Real-World Attacks"). These bugs are more valuable
> to find than classic security bugs (which static analyzers already catch), because they require domain
> reasoning rather than pattern-matching. This 12-category list is this document's own extraction, organized
> for audit-time actionability, rather than a verbatim restatement of the cited paper's four-tier root-cause
> framework — see `MASTER_AUDIT_GUIDE_V2.md` Part 4 for the full treatment of each category.

### 1. Asset Conservation

**Claim**: Assets cannot be created from nothing.

**Examples**: Inflation bugs, double-mint, share inflation, accounting mismatches.

**Design Doc Property**:
```
PROPERTY:
Total assets accounted for by the protocol must never
exceed actual assets held by the protocol.

PROPERTY:
A user cannot increase their claim on protocol assets
without providing equivalent value.
```

**CVL**:
```cvl
invariant asset_conservation() {
    uint256 total_claimed = getTotalClaimedAssets();
    uint256 total_actual = getActualBalance();
    assert total_claimed <= total_actual;
}

rule no_value_creation() {
    uint256 claimBefore = getUserClaim(user);
    uint256 valueBefore = getProvidedValue(user);
    // ... perform action ...
    uint256 claimAfter = getUserClaim(user);
    uint256 valueAfter = getProvidedValue(user);
    assert (claimAfter - claimBefore) <= (valueAfter - valueBefore);
}
```

**Severity**: CRITICAL ($10k-$50k). Inflation is irreversible and drains all LPs/stakers.

---

### 2. Funds Cannot Be Withdrawn Twice

**Claim**: The same value cannot be redeemed twice.

**Examples**: Withdrawal accounting bugs, claim tracking bugs, reward claim bugs.

**Design Doc Property**:
```
PROPERTY:
After assets are withdrawn,
the user's claim must decrease accordingly.
```

**CVL**:
```cvl
rule withdraw_decreases_claim() {
    uint256 claimBefore = getUserClaim(user);
    withdraw(amount);
    uint256 claimAfter = getUserClaim(user);
    assert claimAfter == claimBefore - amount;
}

rule cannot_double_withdraw() {
    uint256 claim = getUserClaim(user);
    withdraw(claim);
    withdraw(1)@withrevert();
    assert lastReverted;
}
```

**Severity**: CRITICAL ($10k-$50k). Double withdrawal empties entire protocol.

---

### 3. Reserved Funds Are Untouchable

**Claim**: Specifically reserved funds (staking deposits, beacon funds, escrow) cannot be withdrawn by any actor.

**Examples**: StakingVault beacon ETH, escrow locked amounts, recovery reserves.

**Design Doc Property**:
```
PROPERTY:
Staged ETH cannot be withdrawn by any actor.

PROPERTY:
Owner withdrawals cannot reduce stagedBalance.
```

**CVL**:
```cvl
rule staged_eth_protected() {
    uint256 stagedBefore = stagedBalance();
    withdraw(amount)@withrevert();
    if (stagedBefore > 0) {
        assert lastReverted || stagedBalance() == stagedBefore;
    }
}

rule owner_cannot_drain_reserved() {
    uint256 reserved = getReservedBalance();
    ownerWithdraw@withrevert();
    assert getReservedBalance() == reserved;
}
```

**Severity**: CRITICAL ($10k-$50k). Losing reserved funds breaks core functionality.

---

### 4. Solvency

**Claim**: Protocol remains fully collateralized. Total liabilities never exceed total assets.

**Examples**: Euler ($196M loss), lending protocols, vault systems.

**Design Doc Property**:
```
PROPERTY:
Total liabilities must never exceed total assets.

PROPERTY:
A user cannot borrow more than allowed collateral.
```

**CVL**:
```cvl
invariant solvency() {
    uint256 totalAssets = getTotalAssets();
    uint256 totalLiabilities = getTotalLiabilities();
    assert totalLiabilities <= totalAssets;
}

rule liquidation_before_insolvency() {
    uint256 health = getHealthFactor(user);
    if (health < 1e18) {
        liquidate@withrevert(user);
        assert !lastReverted; // Liquidation must succeed
    }
}
```

**Severity**: CRITICAL ($10k-$50k). Insolvency causes total protocol collapse.

---

### 5. Exchange Rate / Share Price Integrity

**Claim**: Share value cannot be artificially inflated. Share price must stay proportional to assets.

Extremely important for: ERC4626 vaults, LSTs, LRTs, yield vaults. Donation-style attacks manipulate exchange rates and cascade into catastrophic borrowing exploits.

**Design Doc Property**:
```
PROPERTY:
External transfers must not artificially inflate share value.

PROPERTY:
Share issuance must be proportional to assets deposited.
```

**CVL**:
```cvl
invariant share_price_consistency() {
    uint256 assets = totalAssets();
    uint256 priceA = convertToAssets(1e18);
    uint256 priceB = (assets * 1e18) / totalSupply();
    assert abs(priceA - priceB) <= 1; // Minimal rounding
}

rule deposit_doesnt_inflate_shares() {
    uint256 priceBefore = getSharePrice();
    deposit(1000 ether);
    uint256 priceAfter = getSharePrice();
    assert priceAfter >= priceBefore - 1; // Allow 1 wei rounding
}
```

**Severity**: CRITICAL ($10k-$50k). Donation attacks cascade into catastrophic exploits.

---

### 6. Access Control Guarantees

**Claim**: Only authorized actors can perform privileged actions.

**Design Doc Property**:
```
PROPERTY:
Only owner may pause.

PROPERTY:
Only depositor may beacon deposit.

PROPERTY:
No unauthorized actor can move protocol funds.
```

**CVL**:
```cvl
rule only_owner_pause() {
    env e;
    require e.msg.sender != owner;
    pause@withrevert(e);
    assert lastReverted; // Non-owner blocked
}

rule only_valid_depositor_beacon() {
    env e;
    require !isAuthorizedDepositor(e.msg.sender);
    depositToBeaconChain@withrevert(e);
    assert lastReverted; // Unauthorized fails
}
```

**Severity**: CRITICAL ($10k-$50k). Unauthorized control = full protocol theft.

---

### 7. State Machine Correctness

**Claim**: Actions happen in valid sequence. Invalid transitions are blocked.

**Example**: Stage → Deposit is valid; Deposit → Stage is not.

**Design Doc Property**:
```
PROPERTY:
Beacon deposits may only occur from staged funds.
```

**CVL**:
```cvl
rule valid_state_transitions() {
    State before = currentState();
    doSomething();
    State after = currentState();
    assert isValidTransition(before, after);
}

rule beacon_only_from_staged() {
    uint256 stagedBefore = stagedBalance();
    depositToBeaconChain();
    assert stagedBalance() == stagedBefore - 1 ether;
}
```

**Severity**: HIGH ($5k-$20k). Flow bypass can unlock funds early or drain reserves.

---

### 8. Accounting Synchronization

**Claim**: Multi-contract accounting stays synchronized.

**Examples**: Vault balance ≠ share accounting, multi-layer discrepancies.

**Design Doc Property**:
```
PROPERTY:
Whenever assets increase,
corresponding accounting must increase.

PROPERTY:
Whenever assets decrease,
corresponding accounting must decrease.
```

**CVL**:
```cvl
invariant accounting_sync() {
    uint256 vaultBalance = token.balanceOf(vault);
    uint256 accountedAssets = getTotalAccountedAssets();
    assert vaultBalance >= accountedAssets;
}

rule assets_increase_syncs_accounting() {
    uint256 assetsBefore = token.balanceOf(vault);
    uint256 accountedBefore = getTotalAccountedAssets();
    deposit(amount);
    assert (token.balanceOf(vault) - assetsBefore) ==
           (getTotalAccountedAssets() - accountedBefore);
}
```

**Severity**: CRITICAL ($10k-$50k). Accounting drift enables phantom withdrawals.

---

### 9. Reward Integrity

**Claim**: Rewards are distributed fairly. Cannot claim same reward twice.

**Design Doc Property**:
```
PROPERTY:
A user cannot claim rewards twice.

PROPERTY:
Rewards claimed cannot exceed rewards accrued.
```

**CVL**:
```cvl
rule cannot_double_claim() {
    uint256 earned = getClaimableRewards(user);
    claimRewards();
    uint256 earned2 = getClaimableRewards(user);
    assert earned2 == 0; // Nothing left to claim
}

rule rewards_bounded() {
    uint256 claimed = getClaimedRewards(user);
    uint256 accrued = getAccruedRewards(user);
    claimRewards();
    assert getClaimedRewards(user) <= accrued + claimed;
}
```

**Severity**: HIGH ($2k-$10k). Phantom reward claims drain reward pool.

---

### 10. Funds Cannot Become Permanently Locked

**Claim**: Protocol funds remain recoverable. Valid withdrawal paths always exist.

**Design Doc Property**:
```
PROPERTY:
Valid withdrawal paths always exist.

PROPERTY:
Pause cannot permanently lock user funds.
```

**CVL**:
```cvl
rule withdrawal_always_possible() {
    uint256 balance = balanceOf(user);
    require balance > 0;
    if (isPaused()) {
        emergencyWithdraw@withrevert();
        assert !lastReverted; // Emergency must be available
    } else {
        withdraw@withrevert();
        assert !lastReverted; // Normal must be available
    }
}

rule pause_not_permanent_lock() {
    pause();
    emergencyWithdraw@withrevert();
    assert !lastReverted; // Emergency exits work even when paused
}
```

**Severity**: HIGH ($2k-$10k). Permanent lockup = total loss for users.

---

### 11. Upgrade & Proxy Safety

**Claim**: Only the authorized upgrade path can change contract logic or reinterpret storage. The
implementation contract cannot be initialized or hijacked independently of the proxy.

**Design Doc Property**:
```
PROPERTY:
The upgrade function is only callable by the designated,
time-locked upgrade authority.

PROPERTY:
The implementation contract's initializers are disabled at
deployment, so it cannot be initialized directly.

PROPERTY:
Storage layout (slot, order, type) is identical across every
deployed implementation version.
```

**CVL**:
```cvl
rule only_authority_can_upgrade(address caller) {
    env e;
    require e.msg.sender == caller;
    require caller != upgradeAuthority();
    upgradeToAndCall@withrevert(e, _, _);
    assert lastReverted; // Anyone other than the authority must revert
}

rule implementation_cannot_be_reinitialized() {
    env e;
    // Called directly against the implementation address, not via the proxy
    initialize@withrevert(e, _);
    assert lastReverted; // Must already be initialized/disabled
}
```

**Severity**: CRITICAL. Historical precedent: Parity multisig library
self-destruct froze $280M+ permanently; Nomad bridge lost $190M to a
misconfigured initializer during an upgrade.

---

### 12. Cross-Chain / Bridge Trust

**Claim**: A cross-chain message is only accepted on the destination chain if it was genuinely
authorized on the source chain, and can never be replayed — on the same chain or across chains.
*(Skip this category if the contract has no cross-chain component.)*

**Design Doc Property**:
```
PROPERTY:
Each cross-chain message carries a unique identifier (nonce +
source chain ID) and can never be processed twice.

PROPERTY:
Message authenticity is verified against real source-chain state
(light client, merkle proof, or supermajority validator
signatures) — never trusted from a single relayer.

PROPERTY:
Changing the trusted validator set or verification root requires
passing through the same time-locked authority as any other
upgrade (see Category 11).
```

**CVL**:
```cvl
rule message_cannot_be_replayed(bytes32 messageId) {
    env e;
    require isProcessed(messageId);
    processMessage@withrevert(e, messageId, _);
    assert lastReverted; // Already-processed messages must be rejected
}

rule below_threshold_signatures_rejected(uint256 numSigners) {
    env e;
    require numSigners < requiredThreshold();
    verifyAndExecute@withrevert(e, _, numSigners);
    assert lastReverted; // Must not execute below the trust threshold
}
```

**Severity**: CRITICAL. The single largest cumulative DeFi loss category
historically: Ronin ($625M), Poly Network ($611M), Wormhole ($325M),
Nomad ($190M).

---

## Catastrophic Failure Scenarios

Every design document should include this section near the end. It forces you to think: **"If this invariant breaks, can someone steal, print, freeze, or brick the protocol?"**

### Template

```markdown
# Catastrophic Failure Scenarios

## CF-1: Theft of Reserved Funds

Impact:
Loss of beacon-chain deposit funds.

Must Never Happen:
Owner withdrawal uses staged ETH.

Affected Functions:
withdraw(), stage(), depositToBeaconChain()

---

## CF-2: Insolvency

Impact:
User claims exceed protocol assets.

Must Never Happen:
Total liabilities > total assets.

Affected Functions:
borrow(), deposit(), withdraw(), liquidate()

---

## CF-3: Share Inflation

Impact:
Attacker obtains disproportionate ownership.

Must Never Happen:
Shares minted exceed value deposited.

Affected Functions:
deposit(), mint(), convertToShares()

---

## CF-4: Permanent Lockup

Impact:
Funds become unrecoverable.

Must Never Happen:
All withdrawal paths become inaccessible.

Affected Functions:
withdraw(), emergencyWithdraw(), pause()

---

## CF-5: Unauthorized Control

Impact:
Attacker gains privileged authority.

Must Never Happen:
Non-owner performs owner-only action.

Affected Functions:
pause(), setFees(), changeVotingPower(), upgradeTo()
```

### Key Questions for Each CF

1. **How bad is it?** — $$ loss estimate
2. **How could it happen?** — attack path
3. **What stops it?** — safeguards currently in place
4. **Is that safeguard guaranteed?** — can it be formally verified with CVL?
5. **What if that safeguard fails?** — is there a fallback?

### Real Example: StakingVault CF Scenarios

```markdown
## CF-1: Beacon Deposit Funds Stolen

Impact:
$50M beacon deposit never happens, staking rewards lost forever. CRITICAL.

Must Never Happen:
Owner calls withdraw() and it touches stagedBalance.

Prevented By:
- stagedBalance stored in separate storage slot
- withdraw() checks: require(amount <= unstagedBalance)

CVL:
rule staged_balance_protected() {
    uint256 stageBefore = stagedBalance();
    withdraw(amount)@withrevert();
    if (stageBefore > 0) {
        assert lastReverted || stagedBalance() == stageBefore;
    }
}

## CF-2: Insolvency from Share Inflation

Impact:
First depositor inflates share price, subsequent depositors get near-zero shares.
All deposits effectively stolen. CRITICAL.

Must Never Happen:
Share price deviates from 1:1 by more than minimal rounding.

Prevented By:
- Minimum deposit amount
- Dead shares minted on initialization

CVL:
rule no_share_price_inflation() {
    uint256 priceBefore = sharePrice();
    deposit(1 wei);
    uint256 priceAfter = sharePrice();
    assert abs(priceAfter - priceBefore) <= 1;
}

## CF-3: Permanent Fund Lockup

Impact:
Protocol pauses, users cannot withdraw funds indefinitely. HIGH.

Must Never Happen:
withdraw() fails AND emergencyWithdraw() fails.

Prevented By:
- emergencyWithdraw() bypasses pause check
- emergencyWithdraw() is always callable

CVL:
rule emergency_withdrawal_available() {
    pause();
    emergencyWithdraw@withrevert();
    assert !lastReverted;
}
```

---

## Working Without AutoSetup

AutoSetup (private Certora infrastructure) handles:
- Solidity compiler analysis (storage layout, function visibility)
- External contract classification (ERC20s, interfaces, dependencies)
- Harness contract generation
- Prover configuration setup

### Option A: Cloud Prover (Recommended)

```bash
console-autoprove . src/C.sol:C design_doc.md --cloud
```

Cloud mode eliminates the need for local AutoSetup entirely. No setup, 10x faster.

### Option B: Manual Replacement with Claude

#### Step 1: Analyze Contract

```
Analyze this Solidity contract for:
1. Storage layout (all state variables with byte offsets)
2. Function visibility, mutability, external calls, state mutations
3. External contract classification (ERC20s, interfaces)
4. Potential reentrancy points

[Paste contract code]
```

#### Step 2: Claude Analyzes Business Logic

```
Analyze this contract for ALL 10 business logic bug categories:

"Don't just look for bugs — look for IMPACT. For each category,
ask: can someone steal, print, freeze, or brick the protocol?"

1. Asset Conservation — Can assets be created from nothing?
2. Double Withdrawal — Can same value be withdrawn twice?
3. Reserved Funds — Are reserved funds protected from all access?
4. Solvency — Total liabilities ≤ total assets always?
5. Share Price Integrity — Can share value be artificially inflated?
6. Access Control — Only authorized actors for privileged functions?
7. State Machine — Are all state transitions valid?
8. Accounting Sync — Do all accounting systems stay synchronized?
9. Reward Integrity — Cannot claim rewards twice?
10. Fund Lockup Prevention — Can funds always be recovered?

For each: specific vulnerability (if any), exploit steps, financial impact, fix.

[Paste contract code]
```

#### Step 3: Generate Harness & Config

```
Generate a Certora prover configuration (.conf file) for [ContractName].
Contract: [paste]
External Contracts: [list]
Compilation Target: Solidity 0.8.29

Include: compilation rules, method summaries for externals,
verification rules, resource limits.
```

#### Step 4: Combine & Run

Merge Claude's findings with auto-extracted properties into one CVL file, then run:

```bash
certoraRun certora/MyContract.conf
```

---

## Caching & Resumption

### Cache Namespace (`--cache-ns`)

Phase results (system analysis, property extraction, invariant CVL) are cached in the LangGraph store. On subsequent runs with the same `--cache-ns`, cached results are reused if inputs haven't changed.

```bash
# First run — caches results
console-autoprove . src/C.sol:C doc.md --cache-ns my-run --cloud

# Second run — reuses cache (fast!)
console-autoprove . src/C.sol:C doc.md --cache-ns my-run --cloud

# If design doc changes — cache auto-invalidates, re-extracts
echo "new content" >> doc.md
console-autoprove . src/C.sol:C doc.md --cache-ns my-run --cloud
```

### Checkpoint Resumption

LangGraph checkpoints allow resuming from any point in a crashed or interrupted run:

```bash
# Note the thread-id and checkpoint-id from the crashed run
python3 ./main.py \
  --thread-id crypto_session_abc123 \
  --checkpoint-id 1f0cfbf9-bbd9-6365-8001-90d0fca3dbdf \
  cvl_input.spec interface_file.sol system_doc.txt
```

### Meta-Iteration

After a codegen run completes, refine the spec and resume:

```bash
# Materialize output
python3 ./resume.py materialize thread-id path/

# Edit files in path/ arbitrarily, then resume
python3 resume.py resume-dir thread-id path/
```

---

## Codegen: Generating Implementations from Specs

The codegen workflow (`main.py`) uses the same graphcore infrastructure but with different tools:

### Tools Available in Codegen

| Tool | Purpose |
|------|---------|
| `certora_prover` | Run the Certora Prover on generated code |
| `propose_spec_change` | Propose weakening a spec rule if impossible |
| `human_in_the_loop` | Ask the user for guidance |
| `code_result` | Deliver the final implementation |
| `cvl_manual_search` | Query CVL documentation via RAG |
| `requirements_relaxation` | Relax natural language requirements |
| `requirements_judge` | LLM-as-judge: does implementation satisfy requirements? |
| `read/write/commit_working_spec` | Iterate on spec without touching master copy |
| `CVL research sub-agent` | Deep-dive into CVL manual |

### The Iteration Loop

```
1. LLM writes Solidity code to VFS
2. LLM calls certora_prover on specific rules
3. Prover returns VERIFIED / VIOLATED / TIMEOUT
4. If VIOLATED:
   a. CEX Analyzer analyzes the counterexample
   b. LLM reads analysis and fixes code
   c. Re-run prover
5. If VERIFIED: move to next rule
6. If truly stuck: propose_spec_change or human_in_the_loop
7. When all rules VERIFIED: deliver result via code_result
```

### The Virtual File System (VFS)

All code generation happens in a sandboxed VFS, not the real disk:
- Prevents overwriting the spec file
- Enables sandboxed iteration
- Allows state snapshots for auditing
- Materialized to real files only for prover invocation

---

## Red Flags in Generated CVL

When reviewing generated CVL, watch for:

| Red Flag | Example | Fix |
|----------|---------|-----|
| No precondition check | `assert after > before` without `require amount > 0` | Add preconditions |
| NONDET on important behavior | `transfer(...) NONDET` — abstracts away actual transfer | Use DISPATCHER(true) or real implementation |
| Overconstrained inputs | `require msg.sender == owner` on non-admin function | Relax constraints |
| Missing coverage | Property SKIPPED with weak justification | Challenge the skip |
| Pointless bounds | `require uint256 >= 0` | Remove |
| Tautologies | `assert x == x` | Remove |

---

## Performance Tuning

| Issue | Solution |
|-------|----------|
| Slow extraction | Reduce thinking tokens: `--thinking-tokens 1000` |
| Prover timeouts | Use `--cloud` (10x faster), reduce `--max-concurrent` |
| Too many rounds | Reduce `--max-bug-rounds 2` |
| High API cost | Reduce `--tokens 5000`, `--thinking-tokens 1000` |

---

## Reviewing Results

```bash
# View extracted properties
cat certora/properties/autospec_*.properties.json | jq .

# View generated CVL
cat certora/specs/autospec_*.spec

# Find which specs had failures
for spec in certora/specs/autospec_*.spec; do
  echo "=== $(basename $spec) ==="
  grep -c "FAILED" "$spec" || echo "0 failures"
done

# View cache state
python scripts/autoprove_cache_explorer.py . src/C.sol:C doc.md --cache-ns my-run

# View agent conversation history
python scripts/snapshot_viewer.py <agent_name>
```
