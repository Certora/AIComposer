# Master Audit Guide V2: GitHub Copilot Pro + AIComposer
## Smart Contract Bug Bounty — Everything in One Place

> A consolidated, single-source workflow for smart contract bug-bounty auditing that combines manual analysis (via an LLM coding assistant such as GitHub Copilot) with AIComposer's automated CVL generation and Certora Prover verification.

---

## TABLE OF CONTENTS

1.  The Correct Order — How Everything Connects
2.  Command Reference — 4 Tools, Not Interchangeable
3.  Impact-Driven Paradigm + Severity Framework
4.  The 12 High-Value Business Logic & Infrastructure Bug Categories
5.  Phase A: 7-Step Property Extraction (GitHub Copilot Pro Prompts)
6.  Phase A: Business Logic Manual Analysis (Copilot Prompts)
7.  Phase B: AIComposer Pipeline (After AutoSetup)
8.  Counter-Example Analysis (cex-analyzer)
9.  Report Writing — Complete Template
10. Finding Classification — 4 Types
11. PoC Test Format — Foundry
12. Contest Submission Strategy
13. Catastrophic Failure Scenarios Template
14. Complete Worked Example
15. WSL Ubuntu Setup Reference
16. Quick Reference Checklist
17. Post-Deployment — Monitoring, Incident Response & Supply Chain

---

## PART 1: THE CORRECT ORDER — HOW EVERYTHING CONNECTS

```
THE WINNING FORMULA
════════════════════════════════════════════

Design Doc (What protocol claims it does)
           ↓
GitHub Copilot Pro (Helps you extract + write)
           ↓
AIComposer (Extracts properties → generates CVL automatically)
           ↓
Certora Prover (Finds violations mathematically)
           ↓
cex-analyzer (Explains violations in plain English)
           ↓
Contest Report (Documented → Formal → Practical)
           ↓
VALIDATED FINDINGS ✓ (Judges cannot invalidate what the
                       protocol itself promised)

════════════════════════════════════════════
PHASE A — RIGHT NOW (No AutoSetup needed)
────────────────────────────────────────────
You + GitHub Copilot Pro
  → 7-step property extraction
  → Business logic manual analysis
  → design_doc.md (2000+ words, plain English only)
  → PoC test drafts

PHASE B — AFTER AUTOSETUP (Certora access granted)
────────────────────────────────────────────────────
Your design_doc.md → AIComposer phases 0-5
  → CVL specs generated automatically
  → Prover finds violations
  → cex_analyzer explains them
  → You write contest report
```

---

## PART 2: COMMAND REFERENCE — 4 TOOLS, NOT INTERCHANGEABLE

| Command | What It Does | When to Use |
|---------|-------------|-------------|
| `tui-autoprove` (installed) or `python -m composer.cli.tui_autoprove` | Full audit pipeline with live TUI dashboard | PRIMARY — use this for all audits |
| `console-autoprove` (installed) or `python -m composer.cli.console_autoprove` | Same pipeline, terminal output | When TUI has rendering issues, or for CI/logging |
| `certoraRun contract.conf` | Runs Certora Prover on a manually-written .conf | After you write CVL specs manually |

**Standard audit command:**
```bash
tui-autoprove \
  ~/projects/protocol \
  src/Contract.sol:ContractName \
  design_doc.md \
  --cloud \
  --cache-ns my-audit-v1 \
  --max-concurrent 4
```

**Critical clarification:** `--cloud` only means the Certora Prover runs on
Certora's cloud servers instead of your local machine. It does NOT bypass
the AutoSetup requirement. Phase 1 still needs AutoSetup regardless.

---

## PART 3: THE REAL THREAT LANDSCAPE

### Why Classical Bugs Are No Longer Enough

Modern automated tools (Slither, MythX, Echidna) catch most classical bugs:
- Reentrancy (SWC-107)
- Integer overflow/underflow (SWC-101 — largely mitigated by Solidity 0.8's built-in checked arithmetic, but still relevant in `unchecked {}` blocks and pre-0.8 code)
- Unchecked return values (SWC-104)
- Basic access control gaps (SWC-105, SWC-106)

These are valuable but increasingly expected. They are found by automated
scans before the audit even begins. Bug bounty programs reward them at
Medium to Low severity in most cases.

Two classical categories are worth calling out explicitly, because they're *not* reliably
caught by static analysis or naive fuzzing the way the four above are — they require
understanding external-call gas semantics and call-site context, which puts them closer to
the business-logic reasoning this document focuses on than to a simple pattern match:

- **Insufficient gas griefing (SWC-126)**: a relayer or meta-transaction forwarder is given
  enough gas to *appear* to succeed but not enough for the inner call to complete, causing a
  silent partial failure the caller can't easily detect. Relevant to any contract accepting
  a caller-supplied gas amount for a sub-call (e.g., ERC-2771-style meta-transactions,
  multisend/batch-execution contracts). Related: SWC-113 (DoS with Failed Call — an external
  call reverting, deliberately or not, blocks a loop or shared code path) and SWC-128 (DoS
  With Block Gas Limit — an unbounded loop that eventually can't fit in a block).
- **Missing signature-replay protection (SWC-121)**: an off-chain-signed message (a permit,
  a meta-transaction, a bridge attestation) is accepted more than once, or is valid across a
  contract/chain it wasn't intended for, because the signed payload doesn't include a nonce
  and a domain separator (chain ID + contract address, per EIP-712). This overlaps directly
  with Category 12 (Cross-Chain / Bridge Trust) below, where signature replay across chains
  is the dominant failure mode.

*(A note on SWC as a reference: the SWC Registry's own maintainers flag that it hasn't been
substantively updated since 2020 and may be incomplete — for currently-maintained guidance,
they point to the EEA EthTrust Security Levels specification and the Smart Contract Security
Verification Standard (SCSVS). SWC IDs above are cited as a widely-recognized common
vocabulary for cross-referencing findings, not as a claim that the registry itself is current.)*

### Where the Real Money Is

Research analyzing 50 severe real-world attacks totaling over $1.09B in losses (arXiv:2507.20175)
found that a large share of catastrophic exploits traced back to:

- Protocol logic design flaws
- Broken invariants that "bug-free" code violated
- Business logic that worked in isolation but failed under composition

The Euler Finance exploit ($197M, per the cited paper) is the canonical example. The code had
no reentrancy, no overflow, no oracle manipulation. The access control was
correct. But the solvency invariant could be broken through a specific
sequence of valid operations. Business logic. Not a bug. A broken guarantee.

### The Two Questions Every Auditor Must Ask

When reading ANY function in ANY contract, ask BOTH of these:

```
QUESTION 1: "What does this contract do?"
  → Describes features
  → Finds implementation bugs
  → Table stakes for any auditor

QUESTION 2: "If this invariant breaks, can someone:
             STEAL money?
             PRINT money from nothing?
             FREEZE money permanently?
             BRICK the protocol?"
  → Describes impact
  → Finds business logic flaws
  → This is where $10k–$50k+ findings live
```

If Question 2 has a "yes" answer for any function, that function is your
highest-priority audit target. That is what goes into your design_doc.md
as a formal property.

### Priority Order for Audit Time Allocation

Spend your time in this order. These categories cover the majority of
catastrophic losses in modern DeFi:

```
Priority 1: Asset Conservation
            Can value be created from nothing?
            → Highest frequency of critical findings

Priority 2: Solvency
            Can liabilities exceed assets?
            → Highest dollar impact per exploit (Euler: $196M)

Priority 3: Share / Accounting Integrity
            Can share price be manipulated?
            → Common in ERC4626, vaults, LSTs, LRTs

Priority 4: State Machine Correctness
            Can operations happen in wrong order?
            → Often invisible to classical auditors

Priority 5: Cross-Contract Invariant Preservation
            Do multi-contract systems stay synchronized?
            → Missed by single-contract analysis tools

Priority 6: Access Control
            Can unauthorized actors act?
            → Still important but more often caught automatically

Priority 7: Fund Lockup Prevention
            Can funds be permanently frozen?
            → Often overlooked, high impact when found
```

### Severity Framework (Immunefi / Sherlock / HackenProof)

| Severity | Value | Trigger |
|----------|-------|---------|
| CRITICAL | $10k–$50k+ | Theft, permanent freezing, insolvency, unauthorized minting |
| HIGH | $5k–$20k | Temp freeze >1 month, yield theft, broken accounting |
| MEDIUM | $1k–$5k | DoS >1 week, state inconsistency |
| LOW | <$1k | Gas, informational |

### Realistic Expectations

Starting out (months 1–3): $500–$3k per valid finding
Growing (months 3–12): $2k–$10k per finding
Established: $5k–$50k+ per finding on complex protocols

---

---

## PART 4: THE 12 BUSINESS LOGIC & INFRASTRUCTURE BUG CATEGORIES
### CLAIM → PROPERTY: a useful convention for design_doc.md, not a required format

These are not theoretical. Each category maps to real exploits that cost protocols millions.

**Note on format:** AI Composer doesn't parse or require any particular syntax in your design doc — the whole
document is read as prose by an LLM classification step that derives structured requirements from it, so what
matters is writing a clear, unambiguous, standalone claim, not matching a specific template. The `PROPERTY:`
labeling below is a readability convention worth adopting anyway (it keeps a long design doc organized and makes
each claim easy to spot on review), but there's nothing special about the literal string — a plain paragraph
saying the same thing works identically well.

---

### Category 1: Asset Conservation

**The Core Claim:**
```
Assets cannot be created from nothing.
```

**Examples of violations:**
- Inflation bugs (totalSupply increases without mint)
- Double-mint bugs (same value minted twice)
- Share inflation (share count inflates without deposit)
- Accounting mismatch (tracked value diverges from real value)

**Properties to include in design_doc.md:**
```
PROPERTY:
Total assets accounted for by the protocol must never exceed
actual assets held by the protocol.

PROPERTY:
A user cannot increase their claim on protocol assets
without providing equivalent value.
```

**The Question to Ask:**
"Is there any sequence of valid function calls that increases
total claimable value without real assets entering the protocol?"

---

### Category 2: Funds Cannot Be Withdrawn Twice

**The Core Claim:**
```
The same value cannot be redeemed twice.
```

**Examples of violations:**
- Withdrawal accounting bugs (claim not decremented)
- Claim tracking bugs (per-user record not updated)
- Reward claim bugs (double-claim in same block)

**Properties to include in design_doc.md:**
```
PROPERTY:
After assets are withdrawn, the user's claim must
decrease by exactly the withdrawn amount.

PROPERTY:
A second withdrawal by the same user for the same
assets must revert.
```

**The Question to Ask:**
"Is the user's claim decremented BEFORE or AFTER the transfer executes?
If AFTER, reentrancy or state manipulation can double-spend."

---

### Category 3: Reserved Funds Are Untouchable

**The Core Claim:**
```
Reserved funds cannot be accessed through any path
not intended for them.
```

**Examples of violations:**
- Admin withdrawal that counts reserved funds as available
- Emergency functions that bypass reservation checks
- Rounding that allows small amounts to leak from reserved pool

**Properties to include in design_doc.md:**
```
PROPERTY:
Reserved funds cannot be withdrawn by any actor,
including the owner or admin.

PROPERTY:
Withdrawals cannot reduce the reserved balance.
The invariant availableBalance = contract.balance - reservedBalance
must hold after every call to withdraw().
```

**The Question to Ask:**
"Is there any function — including owner-only, emergency, or recovery
functions — that can reduce the reserved balance without going
through the intended reserve mechanism?"

---

### Category 4: Solvency

**The Core Claim:**
```
Protocol remains fully collateralized.
Total liabilities must never exceed total assets.
```

**Real-world impact:**
Euler Finance ($196M) — broken solvency through valid operations.
Lending protocols across DeFi — broken LTV after price movement.

**Properties to include in design_doc.md:**
```
PROPERTY:
Total liabilities must never exceed total assets.
At all times: sum(what protocol owes) ≤ sum(what protocol holds).

PROPERTY:
A user cannot borrow beyond their allowed collateral.
The protocol's LTV ratio must be enforced on every borrow call.
```

**The Question to Ask:**
"What does this protocol OWE to its users?
What does it HOLD as real assets?
Under what sequence of operations can OWED > HELD?"

---

### Category 5: Exchange Rate / Share Price Integrity

**The Core Claim:**
*(Critical for ERC4626, vaults, LSTs, LRTs, yield vaults)*
```
Share value cannot be artificially inflated.
Share issuance must be proportional to assets deposited.
```

**Real-world impact:**
Donation-style attacks (Euler, multiple ERC4626 vaults) manipulate
exchange rates and cascade into catastrophic borrowing exploits.

**Properties to include in design_doc.md:**
```
PROPERTY:
External ETH or token transfers to the contract must not
artificially inflate the share price.

PROPERTY:
Share issuance must be proportional to assets deposited.
A deposit of X assets must yield Y shares where Y is computed
consistently before and after the deposit.

PROPERTY:
The first depositor cannot manipulate the initial share price
to disadvantage subsequent depositors.
```

**The Question to Ask:**
"What happens if someone sends ETH or tokens directly to the contract
(not through the deposit function) before anyone else deposits?
Does this inflate the share price for the first depositor?"

---

### Category 6: Access Control Guarantees

**The Core Claim:**
```
Only authorized actors can perform privileged actions.
```

**Properties to include in design_doc.md:**
```
PROPERTY:
Only the owner address may call privileged administrative
functions. Any other caller must revert.

PROPERTY:
Only authorized actors may call restricted operations.
Any other caller must revert.

PROPERTY:
No unauthorized actor can move protocol funds, regardless of
the sequence of calls made.
```

**The Question to Ask:**
"Is the access check at the TOP of the function, before any state changes?
Or is there a path through the function that bypasses the check?"

---

### Category 7: State Machine Correctness

**The Core Claim:**
```
Actions must happen in a valid sequence.
The protocol must enforce operation ordering.
```

**Example of a state machine violation:**
```
Correct sequence:
  deposit() → lock() → finalizeOperation()

Must never be possible:
  finalizeOperation() without prior lock()
  (would use unlocked funds, bypassing the lock requirement)
```

**Properties to include in design_doc.md:**
```
PROPERTY:
Critical operations may only occur from the correct prior state.
A call to finalizeOperation() must require that funds are locked,
not just that they exist in the contract.

PROPERTY:
The locked state and the finalized state must transition
atomically. There must be no intermediate state where funds
are neither locked nor finalized.
```

**The Question to Ask:**
"Draw the valid state transitions as a flowchart.
Can a caller reach a final state by skipping one of the required
intermediate steps? What is the impact if they can?"

---

### Category 8: Accounting Synchronization

**The Core Claim:**
*(Very common in multi-contract systems)*
```
All accounting systems tracking the same value
must stay synchronized at all times.
```

**Example of the problem:**
```
Vault balance shows: 1000 ETH
Share accounting shows: 900 ETH equivalent

Where is the 100 ETH? Who can claim it?
Can someone withdraw the discrepancy?
```

**Properties to include in design_doc.md:**
```
PROPERTY:
Whenever assets enter the protocol, total accounted assets
must increase by exactly the same amount.

PROPERTY:
Whenever assets leave the protocol, total accounted assets
must decrease by exactly the same amount.

PROPERTY:
At all times: actual_balance == sum(all_user_claims) + protocol_reserves.
No assets should be unaccounted for.
```

**The Question to Ask:**
"Are there two or more variables that both track the same underlying value?
Is there any function that updates one but not the other?
What can an attacker do with the gap?"

---

### Category 9: Reward Integrity

**The Core Claim:**
```
Rewards are distributed fairly.
Each reward can only be claimed once.
```

**Properties to include in design_doc.md:**
```
PROPERTY:
A user cannot claim rewards twice for the same accrual period.
After claimRewards() succeeds, the user's claimable amount
must be zero.

PROPERTY:
Total rewards claimed across all users cannot exceed
total rewards accrued by the protocol.
```

**The Question to Ask:**
"Is the user's claimable reward amount set to zero BEFORE
the token transfer executes?
If the token has a callback (ERC777, ERC1155), can the user
re-enter during the transfer and claim again?"

---

### Category 10: Funds Cannot Become Permanently Locked

**The Core Claim:**
*(Often overlooked, high impact when found)*
```
Protocol funds remain recoverable.
Valid withdrawal paths always exist.
Pause cannot permanently lock user funds.
```

**Properties to include in design_doc.md:**
```
PROPERTY:
Valid withdrawal paths must always exist for users
with non-zero balances.

PROPERTY:
The pause mechanism is a temporary emergency measure.
It must not permanently prevent fund recovery.
At minimum, an emergency withdrawal path must remain
accessible even while the contract is paused.
```

**The Question to Ask:**
"If the contract enters its paused/emergency state permanently,
can users still get their funds out?
Is there ANY scenario where both the normal AND emergency
withdrawal paths fail simultaneously?"

---

### Category 11: Upgrade & Proxy Safety

**The Core Claim:**
*(SWC-106 / SWC-118-adjacent; caused the Parity multisig freeze — $280M+ permanently
locked when a library contract was accidentally left uninitialized and self-destructed;
caused the Nomad bridge's $190M loss via a misconfigured initializer during an upgrade)*
```
Only authorized parties can trigger an upgrade.
The implementation contract cannot be independently initialized or hijacked.
Storage layout is preserved across upgrades — no field ever changes type or slot.
An upgrade cannot silently change the meaning of already-stored user balances.
```

**Properties to include in design_doc.md:**
```
PROPERTY:
The upgrade function must only be callable by the designated
upgrade authority (timelocked multisig, DAO governance, etc.),
never by an arbitrary caller.

PROPERTY:
The logic/implementation contract must have its initializers
disabled at deployment (e.g., via _disableInitializers() or
equivalent), so it cannot be initialized and taken over directly,
bypassing the proxy.

PROPERTY:
Storage variables must occupy the same slot, in the same order,
with the same type, across every implementation version ever
deployed. A new version must only ever append new variables,
never reorder, retype, or remove existing ones (respect any
storage-gap convention already in use).

PROPERTY:
An upgrade must not change the semantic meaning of existing
stored values (e.g., reinterpreting a balance field as a
share-count field) without an explicit, audited migration step.
```

**The Question to Ask:**
"Who can call the upgrade function, and is that authority itself
time-locked or otherwise resistant to a single compromised key?
Can the *implementation* contract be initialized directly, without
going through the proxy? If I diff this version's storage layout
against the last deployed version, does anything change type,
order, or meaning?"

**Recommended primitives:** OpenZeppelin's `UUPSUpgradeable` /
`TransparentUpgradeableProxy` (with `_disableInitializers()` called
in the implementation's constructor) rather than a hand-rolled proxy;
OpenZeppelin Upgrades plugin or `storage-layout`-style tooling to
diff storage layout between versions before every upgrade.

---

### Category 12: Cross-Chain / Bridge Trust

**The Core Claim:**
*(The single largest loss category in DeFi history by cumulative $ — Ronin $625M,
Poly Network $611M, Wormhole $325M, Nomad $190M)*
```
A message can only be considered valid on the destination chain if it was
genuinely authorized on the source chain.
A message cannot be replayed — neither on the same chain twice, nor
replayed identically across two different destination chains.
The set of parties trusted to attest to source-chain state is itself
protected against compromise of a subset of its members.
```

**Properties to include in design_doc.md:**
```
PROPERTY:
Each cross-chain message must be uniquely identified (e.g., by a
monotonically increasing nonce plus source-chain ID) such that it
can never be accepted twice on any destination chain.

PROPERTY:
Message authenticity must be verified against the actual
source-chain state (light client, merkle proof, or a validator
set requiring a supermajority signature threshold) — never
trusted purely on the say-so of a single relayer or a minority
of validators.

PROPERTY:
Compromising fewer than [threshold] of the trusted validator/relayer
set must be insufficient to forge a valid message.

PROPERTY:
Any admin function that can change the trusted validator set,
the trusted root, or the message-verification logic must be
time-locked and must not be reachable via a single uninitialized
or misconfigured call (see Category 11 — this is exactly the
class of bug that caused the Nomad bridge loss).
```

**The Question to Ask:**
"If I control a bare majority (or, depending on the design, even
just one) of the parties who attest to what happened on the source
chain, can I mint/release funds on the destination chain for an
event that never actually happened? Is the message-verification
root itself protected by the same upgrade-safety properties as
Category 11?"

**Recommended primitives:** Prefer canonical, audited bridge
infrastructure (LayerZero, Axelar, Wormhole's own guardian-set
design, or a rollup's native canonical bridge) over a hand-rolled
trusted-relayer bridge wherever the protocol's design allows it;
where a custom bridge is unavoidable, treat its validator-set and
message-verification logic as the single highest-priority target
for both formal verification and external audit.

---


## THE AUDITOR MINDSET (Add to Every Design Doc Session)

Before writing any property, run this mental filter:

```
FOR EVERY INVARIANT YOU FIND, ASK:

"If this invariant breaks..."

→ Can someone STEAL money?
  YES → CRITICAL finding. Write this property FIRST.

→ Can someone PRINT money from nothing?
  YES → CRITICAL finding. Asset conservation violation.

→ Can someone FREEZE money permanently?
  YES → CRITICAL/HIGH finding. Lockup violation.

→ Can someone BRICK the protocol entirely?
  YES → HIGH/CRITICAL finding. State machine or governance violation.

→ Can someone gain UNFAIR ADVANTAGE?
  YES → HIGH finding. Accounting or reward integrity violation.

→ None of the above?
  → LOW/MEDIUM or informational. Not your priority.
```

This filter determines which properties go into your design_doc.md.
Only properties that produce a "yes" to one of these questions are
worth formalizing and verifying with Certora.

---

---

## PART 5: PHASE A — 7-STEP PROPERTY EXTRACTION
### GitHub Copilot Pro Prompts (All Exact, Ready to Use)

**Correct order — this is important:**
```
Step 1: Understand → Step 2: Extract Claims → Step 3: Map Components
Step 4: Convert to Properties → Step 5: Pre/Postconditions
Step 6: Draft design_doc.md → Step 7: Review & Refine
```

---

### STEP 1: Understand the Contract
**Tool:** GitHub Copilot Pro Chat
**Who does it:** Copilot (you read the output)

**Exact Prompt:**
```
@workspace I need to audit this smart contract for a bug bounty submission.
Please analyze it and provide:

1. SUMMARY: What does this contract do in 3–4 sentences?
2. STATE VARIABLES: List every state variable with its type and purpose
3. FUNCTIONS: List every function with visibility, mutability, and a
   one-line description of what it does
4. ACTORS: Who are the key actors? (owner, admin, user, etc.)
   What can each actor do?
5. MODIFIERS: List all modifiers and what they enforce
6. EXTERNAL DEPENDENCIES: What external contracts does this call?
7. EVENTS: List all events and when they are emitted

Format clearly with a header for each section.
```

**What you do with the output:** Read carefully. Ask follow-up questions
about anything you do not fully understand before moving to Step 2.

---

### STEP 2: Extract Documented Claims
**Tool:** You read the code, Copilot validates
**Who does it:** YOU first (critical thinking), then Copilot

**What you do MANUALLY:**
Read the contract and extract every explicit promise. Look for:
- NatSpec @notice and @dev comments
- require() error messages (reveal what SHOULD be true)
- Function names (onlyOwner, whenNotPaused = claims in themselves)
- Variable names (reservedBalance, availableBalance = claims about fund separation)
- Modifiers (each one is a claim about who can call what)

**Exact Prompt (after your own reading):**
```
Based on this smart contract's code comments, NatSpec documentation,
require() statements, error messages, function names, and modifier names,
extract ALL explicit and implicit claims the protocol makes about its
own behavior.

For each claim:
CLAIM: [what the protocol promises]
SOURCE: [exact location — function name, line, or NatSpec comment]
TYPE: [access control / accounting / fund protection / state transition / other]

Be exhaustive. Even function names like "onlyOperator" are claims.
Even require(amount > 0, "Zero amount") is a claim.
```

---

### STEP 3: Identify Core System Components
**Tool:** GitHub Copilot Pro Chat
**Who does it:** Copilot (you verify accuracy)

**Exact Prompt:**
```
Analyze this smart contract and identify 3–6 major functional components
or areas of responsibility.

For each component provide:
- COMPONENT NAME: (descriptive, not just the contract name)
- PURPOSE: What does this component do?
- STATE IT MANAGES: Which state variables belong to this component?
- FUNCTIONS: Which functions implement this component?
- EXTERNAL DEPENDENCIES: What external contracts does it interact with?
- RISK AREA: Where could this component have security issues?

Format as a separate section for each component.
```

---

### STEP 4: Convert Claims to Formal Properties
**Tool:** YOUR thinking, Copilot validates
**Who does it:** YOU (most critical step)

**Your job:** Take each claim from Step 2 and write it as a testable
mathematical statement. Use this format:

```
CLAIM:    "Only designated operator can perform critical operations"
PROPERTY: finalizeOperation() → msg.sender must equal operator address

CLAIM:    "Reserved balance cannot be withdrawn"
PROPERTY: withdraw(amount) requires amount ≤ (address(this).balance − reservedBalance)
          AND reservedBalance does not decrease as result of withdraw()

CLAIM:    "Pausing prevents all deposits"
PROPERTY: depositsPaused == true → deposit() reverts
          depositsPaused == true → addLiquidity() reverts
```

**Exact Prompt (to validate your translations):**
```
I have translated these documented claims from the contract into
formal properties. Please review each one:

[PASTE YOUR CLAIM → PROPERTY TRANSLATIONS]

For each property tell me:
1. Is my translation accurate and complete?
2. Are there edge cases or conditions I missed?
3. Could this property be violated in any way I haven't considered?
4. Is the property too narrow? (Missing related cases?)
5. Is the property too broad? (Would catch false positives?)
```

---

### STEP 5: Define Pre/Postconditions for Critical Functions
**Tool:** Copilot Pro (you specify which functions)
**Who does it:** You + Copilot

**Exact Prompt:**
```
For each of these critical functions in the contract, define complete
pre/postconditions. Focus on security-relevant conditions only.

Functions to analyze:
- [function 1]
- [function 2]
- [function 3]

For EACH function provide:

PRECONDITIONS (must be true BEFORE the function can execute):
- permission check: [who is allowed to call this?]
- state check: [what contract state must be true?]
- input check: [what must be true about the parameters?]
- balance check: [what balance conditions must hold?]

POSTCONDITIONS (must be true AFTER successful execution):
- state changes: [exactly what state variables changed and how?]
- balance changes: [what balances changed and by how much?]
- events: [which events are emitted?]
- what did NOT change: [what state must remain unchanged?]

REVERTS IF:
- [list every condition that should cause a revert]

INVARIANTS THAT MUST HOLD THROUGHOUT:
- [what must remain true even during execution?]
```

---

### STEP 6: Draft design_doc.md
**Tool:** Copilot Pro (structures and writes)
**Who does it:** Copilot + You (you provide all content from Steps 1–5)

**CRITICAL RULE: design_doc.md = PLAIN ENGLISH ONLY**
- NO Solidity code
- NO CVL code (invariant, rule, etc.)
- NO pseudocode
- Only clear prose and structured lists

AIComposer converts your English into CVL automatically.
Putting CVL in the doc confuses the LLM and reduces quality.

**Exact Prompt:**
```
Help me write a design document for AIComposer formal verification
using the content I have gathered.

STRICT RULE: The document must contain ONLY plain English.
No Solidity code, no CVL syntax, no pseudocode.
If I have included any code, convert it to plain English descriptions.

Use this exact 9-section structure:

1. Executive Summary (2–3 sentences only)
2. Core System Components (one subsection per component from Step 3)
3. Documented Claims (all claims from Step 2, organized by type)
4. Formal Properties (translations from Step 4, plain English only)
5. Critical Function Specifications (pre/postconditions from Step 5)
6. High-Level Properties for Verification (6–10 most important)
7. Edge Cases and State Transitions
8. Business Logic Concerns (all 12 categories addressed for this protocol — skip category 12 if no cross-chain component)
9. Assumptions

Here is my content to organize:
[PASTE ALL YOUR FINDINGS FROM STEPS 1–5]

Make each property specific to THIS protocol, not generic.
Target length: 2000–5000 words.

Example of GENERIC (bad — could apply to ANY protocol):
  "Users cannot withdraw more than they deposited."

Example of SPECIFIC (good — tied to THIS protocol's actual logic):
  "When withdraw() is called, the amount available to withdraw is
   calculated as (userShares * (address(this).balance - reservedBalance))
   / totalSupply, not as (userShares * address(this).balance) / totalSupply.
   This ensures reserved funds are excluded from the withdrawable pool."
```

---

### STEP 7: Review and Refine
**Tool:** Copilot Pro (quality check)
**Who does it:** You first (self-review), then Copilot

**Self-Review Questions:**
- Does every property tie to a documented claim? (no invented properties)
- Is every property specific? (not "fees work correctly")
- Are all 12 business logic & infrastructure categories addressed?
- Are edge cases covered? (zero amounts, empty states, pause states)
- Does the doc contain ANY code? (remove it if so)
- Would a judge accept each finding based on this documentation?

**Exact Prompt:**
```
Review this design_doc.md I have written for AIComposer formal verification.

[PASTE YOUR COMPLETE design_doc.md]

Check for these specific issues:

1. SPECIFICITY: Are properties specific to this protocol or are they generic?
   Flag any property that could apply to ANY protocol unchanged.

2. COMPLETENESS: Are all 12 business logic & infrastructure categories addressed?
   (asset conservation, double withdrawal, reserved funds, solvency,
   share price integrity, access control, state machine correctness,
   accounting synchronization, reward integrity, fund lockup prevention)

3. TRACEABILITY: Does each property trace back to a documented claim?
   Flag any property I invented without a documented source.

4. CODE PRESENCE: Does the document contain any CVL, Solidity, or pseudocode?
   Convert anything found to plain English.

5. EDGE CASES: Are these edge cases covered?
   - Zero amount inputs
   - Maximum amount inputs
   - Empty/zero state (empty pool, zero balance)
   - Paused state operations
   - Transition states (mid-operation)

6. IMPACT CLARITY: For each property, is it clear what breaks if it fails?
   (theft / freezing / insolvency / minting / none)

Give specific line-by-line feedback on what to improve.
```

---

## PART 6: PHASE A — BUSINESS LOGIC MANUAL ANALYSIS
### Replaces AIComposer Phase 4 While Waiting for AutoSetup

---

### Prompt A: 12-Category Business Logic & Infrastructure Analysis

*Use this exact prompt with GitHub Copilot Pro or Claude on any contract.*
*This replaces the previous generic version.*

```
Analyze this smart contract for business logic vulnerabilities.
This is for a bug bounty submission targeting $10k–$50k+ findings.

Ignore classical bugs like reentrancy and overflow — assume those are
handled by automated tools. Focus entirely on business logic:
invariants whose violation leads to theft, inflation, freezing, or
protocol shutdown.

CONTRACT CODE:
[PASTE FULL CONTRACT]

ANALYZE EACH OF THESE 12 CATEGORIES (skip category 12 if no cross-chain component exists):

────────────────────────────────────────────────────────
1. ASSET CONSERVATION
   Claim: Assets cannot be created from nothing.

   Check:
   - Can total claimable value exceed real assets held?
   - Can a user increase their claim without depositing value?
   - Does totalSupply ever increase without an explicit mint call?
   - Are there rounding paths that create dust value over time?

   If violated: state the EXACT function, the EXACT sequence
   of calls, and the dollar impact.
   Also state the plain-English property for design_doc.md.

────────────────────────────────────────────────────────
2. FUNDS CANNOT BE WITHDRAWN TWICE
   Claim: The same value cannot be redeemed twice.

   Check:
   - Is the user's claim decremented BEFORE or AFTER the transfer?
   - Can withdraw() complete and then succeed again immediately?
   - Is there any path where claim is not cleared but funds are sent?

────────────────────────────────────────────────────────
3. RESERVED FUNDS ARE UNTOUCHABLE
   Claim: Designated reserved funds cannot be accessed through
   any path not intended for them.

   Check:
   - What funds are staged, reserved, or locked for specific use?
   - Can the owner, admin, or any actor access them through
     withdraw(), emergencyWithdraw(), or any other function?
   - Does availableBalance correctly exclude reserved amounts?

────────────────────────────────────────────────────────
4. SOLVENCY
   Claim: Protocol remains fully collateralized.
   Total liabilities must never exceed total assets.

   Check:
   - What does this protocol OWE to users? (list it explicitly)
   - What does this protocol HOLD in real assets? (list it)
   - Under what sequence of valid operations can OWED > HELD?

────────────────────────────────────────────────────────
5. EXCHANGE RATE / SHARE PRICE INTEGRITY
   Claim: Share value cannot be artificially inflated.
   Share issuance is proportional to assets deposited.

   Check:
   - What happens if ETH/tokens are sent directly to the contract?
   - Can a first depositor attack manipulate the initial price?
   - Is the share price formula stable at extreme reserve ratios?

────────────────────────────────────────────────────────
6. ACCESS CONTROL GUARANTEES
   Claim: Only authorized actors can perform privileged actions.

   Check:
   - List every function that changes critical state
   - For each: where exactly is the access check? Is it first?
   - Can any check be bypassed through any code path?

────────────────────────────────────────────────────────
7. STATE MACHINE CORRECTNESS
   Claim: Actions must happen in a valid sequence.

   Check:
   - Draw the valid state transitions
   - Can any required step be skipped?
   - Can the contract reach a state through invalid ordering?

────────────────────────────────────────────────────────
8. ACCOUNTING SYNCHRONIZATION
   Claim: All accounting systems tracking the same value
   stay synchronized.

   Check:
   - List all variables that track the same underlying value
   - Is there any function that updates one but not the other?
   - What can an attacker do with any gap between them?

────────────────────────────────────────────────────────
9. REWARD INTEGRITY
   Claim: Rewards are distributed fairly.
   Each reward can only be claimed once.

   Check:
   - Is claimable amount zeroed BEFORE the transfer?
   - What if claimRewards() is called twice in the same block?
   - Can a token callback enable double-claim?

────────────────────────────────────────────────────────
10. FUNDS CANNOT BECOME PERMANENTLY LOCKED
    Claim: Valid withdrawal paths always exist.
    Pause cannot permanently lock user funds.

    Check:
    - If the contract is paused indefinitely, can users withdraw?
    - Is there an emergency path? Is it always accessible?
    - Can both normal AND emergency paths fail simultaneously?

────────────────────────────────────────────────────────
11. UPGRADE & PROXY SAFETY
    Claim: Only the authorized upgrade path can change logic or
    storage semantics; nothing else can.

    Check:
    - Who can call the upgrade function? Is that authority
      time-locked or a single key?
    - Can the implementation contract be initialized directly,
      bypassing the proxy?
    - Does the storage layout match the previous version exactly
      (same slots, same types, same order)?
    - Could an upgrade silently reinterpret existing stored values?

────────────────────────────────────────────────────────
12. CROSS-CHAIN / BRIDGE TRUST
    Claim: A message is only valid on the destination chain if it
    was genuinely authorized on the source chain, and can never be
    replayed.

    Check: (skip this category if the contract has no cross-chain
    component)
    - Is each message uniquely identified (nonce + source chain ID)
      so it cannot be replayed?
    - Is authenticity checked against real source-chain state
      (light client / merkle proof / supermajority signatures),
      or trusted on a single relayer's say-so?
    - How many compromised validators/relayers would it take to
      forge a valid message?
    - Is the function that changes the trusted validator set or
      verification root itself time-locked (see Category 11)?

────────────────────────────────────────────────────────

FOR EACH CATEGORY PROVIDE:

VULNERABILITY FOUND: YES / NO

If YES:
- EXACT FUNCTION: [function name]
- EXPLOIT SEQUENCE: [step-by-step — what the attacker does]
- FINANCIAL IMPACT: [$ estimate]
- AUDITOR MINDSET CHECK: does this let someone STEAL / PRINT /
  FREEZE / BRICK? → determines severity
- DESIGN DOC PROPERTY: [exact plain-English property statement
  to add to design_doc.md]
- SUGGESTED FIX: [brief code change]

If NO:
- ONE SENTENCE explaining why the category is safe here
```

---

---

### Prompt B: Storage Layout and State Analysis
*Replaces AutoSetup Function #1: Solidity Compiler Analysis*

```
Analyze this Solidity contract for:

1. STORAGE LAYOUT
   List every state variable with:
   - Name and type
   - Storage slot (approximate)
   - Purpose in the protocol
   - Which functions read it / write it

2. FUNCTION ANALYSIS
   For every function list:
   - Visibility (public/external/internal/private)
   - Mutability (view/pure/payable/nonpayable)
   - External calls made (if any)
   - State variables modified
   - Reentrancy risk (low/medium/high and why)

3. REENTRANCY MAP
   For every external call:
   - Which function contains it?
   - Is state updated BEFORE or AFTER the external call?
   - Could an attacker exploit the callback window?

4. DEPENDENCY MAP
   For every external contract used:
   - Interface name and function called
   - What happens if this external call fails?
   - What happens if this external contract is malicious?

CONTRACT:
[PASTE CONTRACT]
```

---

### Prompt C: External Contract Classification
*Replaces AutoSetup Function #2: External Contract Classification*

```
For this Solidity project, classify all external contracts and imports:

[PASTE ALL CONTRACT FILES]

For each external or imported contract:
1. TYPE: ERC20 / ERC721 / Interface / Abstract / Library / Oracle / Other
2. IMPORTED BY: which contracts in the project use it
3. KEY FUNCTIONS CALLED: which functions from this external contract are called
4. POTENTIAL ISSUES:
   - Could this token have transfer fees?
   - Could this token be pausable?
   - Could this token have callbacks (ERC777, ERC1155)?
   - Is this a rebasing token?
5. CVL SUMMARY SUGGESTION: How should this be summarized in Certora?
   (DISPATCHER / CONSTANT / NONDET / custom summary)
```

---

### Prompt D: Harness Contract Generation
*Replaces AutoSetup Function #3: Harness Contract Generation*

```
Generate a CVL harness setup for testing [CONTRACT NAME].

Contract to test:
[PASTE CONTRACT]

External contracts it uses:
[LIST THEM]

Properties I want to test:
[LIST YOUR PROPERTIES FROM Step 4]

Please generate:
1. A harness contract (Solidity) that:
   - Inherits from the main contract
   - Adds ghost variables to track state across calls
   - Exposes helper functions for property testing

2. Mock implementations for each external dependency

3. A Certora .conf file with:
   - File paths and solc version
   - Method summaries for all external calls
   - verify directive pointing to main contract
   - Prover timeout set to 600 seconds

Use Solidity version 0.8.20.
```

---

## PART 7: PHASE B — AICOMPOSER PIPELINE (After AutoSetup)

### Setup (One Time After AutoSetup Access Granted)

```bash
# 1. Clone AutoSetup
git clone https://github.com/Certora/AutoSetup.git ~/AutoSetup

# 2. Set environment variable permanently
echo 'export AUTOSETUP_PATH=~/AutoSetup' >> ~/.bashrc
source ~/.bashrc

# 3. Verify
echo $AUTOSETUP_PATH
```

### Run the Pipeline

**PREREQUISITE:** `AUTOSETUP_PATH` must be set and AutoSetup cloned (see Setup section above).
The commands below **WILL FAIL** without AutoSetup configured.

```bash
# Standard audit run
tui-autoprove \
  ~/projects/protocol \
  src/Contract.sol:ContractName \
  design_doc.md \
  --cloud \
  --max-concurrent 4 \
  --cache-ns protocol-audit-v1

# Second run after updating design_doc (uses cache for unchanged parts)
tui-autoprove \
  ~/projects/protocol \
  src/Contract.sol:ContractName \
  design_doc.md \
  --cloud \
  --cache-ns protocol-audit-v1  # same namespace = reuses cache

# With more extraction rounds (default is 3, try 5 for complex protocols)
tui-autoprove \
  ~/projects/protocol \
  src/Contract.sol:ContractName \
  design_doc.md \
  --cloud \
  --max-bug-rounds 5 \
  --cache-ns protocol-audit-v1
```

### Phase Summary

| Phase | What Happens | Needs AutoSetup? |
|-------|-------------|-----------------|
| 0 | Reads code + design_doc, identifies components | NO |
| 1 | Harness generation, compiler analysis | YES — REQUIRED |
| 2 | CVL summaries for external contracts | YES (depends on Phase 1) |
| 3 | Structural invariants (total supply, etc.) | NO |
| 4 | Per-component property extraction (multi-round) | NO |
| 5 | CVL generation + Certora Prover verification | YES for harness |

### Output Files

```
certora/
├── specs/
│   ├── invariants.spec              # Structural invariants
│   └── autospec_{Component}.spec   # Per-component CVL specs
├── confs/
│   └── autospec_{Component}.conf   # Prover configuration
└── properties/
    ├── {Component}.properties.json      # Extracted properties
    ├── {Component}.property_rules.json  # Property → rule mapping
    └── {Component}.commentary.md        # LLM explanation of specs
```

---

## PART 8: COUNTER-EXAMPLE ANALYSIS

### Running the CEX Analyzer

```bash
cex-analyzer \
  certora/output/report_folder_name \
  ruleName \
  --method ContractName.functionName
```

### What cex_analyzer Shows

```
1. SUMMARY OF PROVING PROCESS
   What was checked and the exact path that led to the violation

2. DESCRIPTION OF SURFACED BUG
   Which property was violated and exactly why, with variable values

3. CONVERSATION ON HOW TO PROCEED
   Suggested fixes and alternative interpretations of the violation
```

### Prover Result Types

| Result | Meaning | Your Action |
|--------|---------|------------|
| VERIFIED | Property holds for all inputs | Record as confirmed safe |
| VIOLATED | Counter-example found — potential bug! (the Prover's actual status string is `VIOLATED`, not `FAILED`) | Investigate immediately with cex_analyzer |
| TIMEOUT | Too complex to finish | Simplify the rule, add more preconditions |
| SANITY_FAILED | The rule/invariant's assertion is unreachable — a precondition conflict, not a code bug | Check for vacuity; see `cex_instructions.j2` guidance |
| ERROR | The rule failed to run (e.g. a typecheck or setup error), not a property violation | Fix the underlying error and re-run |
| SKIPPED | The rule was not run this pass (e.g. filtered out, or blocked by an earlier error) | Check why it didn't run before treating it as passing |

### Translating Counter-Examples for Judges

Judges are often not formal verification experts. Use this format:

**Step 1: Plain-English Summary (No Jargon)**
```
The contract allows funds reserved for a specific purpose to be withdrawn
through the general withdrawal function, which means the reserved operation
can fail silently while funds have already been removed.
```

**Step 2: Concrete Scenario**
```
"Alice locks 100 tokens into the protocol's reserve. Before the reserved
operation executes, Alice calls withdraw(100 tokens). The contract allows
this because withdraw() does not check the reserved balance. The reserved
operation later fails — but the 100 tokens are already gone."
```

**Step 3: State Visualization**
```
Initial state:
  contract.balance   = 100 tokens
  reservedBalance    = 100 tokens
  availableBalance   = 0 tokens

Step 1: Alice calls withdraw(100 tokens)
  → withdraw() checks: 100 ≤ contract.balance ✓  (BUG: wrong check!)
  → Should check: 100 ≤ availableBalance          (which is 0)
  → Transfer executes: Alice receives 100 tokens

Final state:
  contract.balance   = 0 tokens    (WRONG — should still be 100)
  reservedBalance    = 100 tokens  (still says 100 is reserved — IMPOSSIBLE)
  availableBalance   = -100 tokens (INVALID STATE — cannot be negative)
```

**Step 4: Connect to Formal Verification**
```
This scenario was generated by Certora Prover as a counter-example
to the rule reservedFundsProtected(). The prover exhaustively verified
all possible inputs and found this as the minimal violation case.
This is not a theoretical edge case — it is the simplest possible
exploitation path.
```

---

## PART 9: REPORT WRITING — COMPLETE TEMPLATE

### The Multi-Layer Evidence Approach

Every finding you submit should have THREE layers of evidence:
```
Layer 1: MATHEMATICAL PROOF (Certora counter-example)
         → Proves it's POSSIBLE (unrefutable)

Layer 2: DOCUMENTED CLAIM (from protocol's own documentation)
         → Proves it was PROMISED (makes invalidation impossible)

Layer 3: RUNNABLE PoC (Foundry test)
         → Proves it's EXPLOITABLE (shows real impact)
```

### Complete Finding Report Template

```markdown
# Finding: [Specific Title — What Can Happen, Not Just What's Wrong]

## Severity: [Critical / High / Medium / Low]

## Summary
[2–3 sentences in plain English. What is the vulnerability?
What can an attacker do because of it? Non-technical audience.]

## Protocol's Own Documentation
"[EXACT QUOTE from contract comment, NatSpec, or whitepaper]"
Source: [exact location]

This is the documented guarantee that is violated. The protocol
itself promised this behavior. The violation is not a matter of
interpretation.

## Vulnerability Details
[Technical explanation of WHY the guarantee is violated.
Which function, which check is missing, what path leads to violation.]

## Certora Formal Verification
The following CVL rule was generated by AIComposer and verified
by Certora Prover:

[PASTE THE CVL RULE from certora/specs/autospec_*.spec]

Prover result: FAILED
Counter-example: [paste or summarize the cex_analyzer output]

## How the Counter-Example Works
Step-by-step trace of the violation:

Initial state:
  [key variable] = [value]
  [key variable] = [value]

Step 1: [Actor] calls [function]([params])
  → [what happens]
  → [state change]

Step 2: [what changes next]

Final state:
  [key variable] = [value] ← INVALID
  [key variable] = [value]

[Explain what makes the final state invalid and dangerous]

## Impact
If exploited:
- Attacker can: [concrete action in plain English]
- Potential loss: [$ estimate based on current TVL or pool size]
- Affected users: [who is affected]
- Severity justification: [why this is Critical/High/Medium/Low
  per Immunefi/Sherlock standards]

## Proof of Concept
The following Foundry test demonstrates the vulnerability:

[PASTE YOUR FOUNDRY TEST — see Part 11 for format]

Run with:
forge test --match-test testVulnerabilityName -vvv

Expected output:
[Describe what the test shows when it passes]

## Addressing Potential Counter-Arguments

"Is this a false positive?"
The counter-example is realizable because:
- [Reason 1: e.g., no special permissions required]
- [Reason 2: e.g., standard ERC20 tokens involved]
- [Reason 3: e.g., publicly callable function]

"Is this intended behavior?"
The protocol's own documentation (quoted above) states the opposite.
This is a violation of the protocol's stated guarantees.

## Recommendation
[Specific code change to fix the issue. Reference the exact function.]

The fix ensures the property holds because:
[Explain why the fix resolves the violation]

## Additional Notes
Verified on: Certora Prover [version]
AIComposer version: [version]
Environment: All findings are verified counter-examples, not
theoretical violations.
```

---

## PART 10: FINDING CLASSIFICATION — 4 TYPES

### Critical for Report Framing

Before writing a report, classify your finding as one of these:

**Type 1: VULNERABILITY**
Direct exploitable issue with concrete financial impact.
```
Use when: You can show step-by-step how an attacker steals/freezes funds.
Label as: "This finding represents an exploitable vulnerability where..."
Severity: Critical or High
```

**Type 2: DESIGN FLAW**
Contract meets its formal specification but the design itself creates risk.
```
Use when: The code does what it says, but what it says is dangerous.
Label as: "While the contract meets its specification, this design pattern
           introduces risk because..."
Severity: Medium to High
```

**Type 3: SPECIFICATION VIOLATION**
Contract does not match its own documentation.
```
Use when: Code comment says X, but code actually does Y.
Label as: "The protocol documentation states [quote], but formal
           verification shows this constraint is not enforced."
Severity: High (documentation lies = trust broken)
```

**Type 4: CORRECTNESS ISSUE**
Logical inconsistency without a clear exploit path.
```
Use when: You found an invariant violation but can't make it pay.
Label as: "This invariant violation represents a logical inconsistency
           in the contract state. No direct exploit path was identified,
           however..."
Severity: Low to Medium
```

### Why Judges Invalidate Findings (and How to Prevent It)

| Reason for Invalidation | Prevention |
|------------------------|-----------|
| "No exploit path shown" | Always include a step-by-step scenario AND PoC test |
| "This is intended behavior" | Quote the protocol's OWN documentation that says otherwise |
| "False positive from prover" | Explain why counter-example is realizable in practice |
| "Insufficient impact" | Quantify dollar amount; cite Immunefi severity criteria |
| "No proof of concept" | Always include a running Foundry test |
| "Already known / out of scope" | Check existing audit reports before submitting |

---

## PART 11: POC TEST FORMAT — FOUNDRY

### Standard Foundry PoC Template

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/ContractUnderTest.sol";

contract VulnerabilityNamePoC is Test {
    ContractUnderTest internal protocol;
    address internal attacker = makeAddr("attacker");
    address internal admin = makeAddr("admin");

    function setUp() public {
        // Deploy contract
        protocol = new ContractUnderTest();

        // Initialize state
        vm.deal(admin, 100 ether);
        vm.startPrank(admin);
        protocol.initialize();
        vm.stopPrank();
    }

    function testVulnerabilityName() public {
        // ─── SETUP ───────────────────────────────────────────────
        vm.deal(attacker, 10 ether);
        vm.startPrank(attacker);

        // Record state BEFORE the exploit
        uint256 balanceBefore = address(protocol).balance;
        uint256 reservedBefore  = protocol.reservedBalance();

        console.log("=== BEFORE EXPLOIT ===");
        console.log("Contract balance:", balanceBefore);
        console.log("Reserved balance:", reservedBefore);
        console.log("Available:       ", protocol.availableBalance());

        // ─── EXPLOIT ──────────────────────────────────────────────
        // Step 1: [describe what attacker does first]
        protocol.deposit{value: 10 ether}();

        // Step 2: [describe the vulnerability trigger]
        protocol.withdraw(attacker, 10 ether);

        vm.stopPrank();

        // ─── VERIFY VIOLATION ─────────────────────────────────────
        uint256 balanceAfter = address(protocol).balance;
        uint256 reservedAfter  = protocol.reservedBalance();

        console.log("=== AFTER EXPLOIT ===");
        console.log("Contract balance:", balanceAfter);
        console.log("Reserved balance:", reservedAfter);
        console.log("Attacker gained: ", attacker.balance - 10 ether);

        // This assertion PASSES, showing the invariant IS violated
        assertLt(
            balanceAfter,
            reservedAfter,
            "VULNERABILITY: contract balance < reserved balance — invalid state"
        );
    }
}
```

**Run command:**
```bash
forge test --match-test testVulnerabilityName -vvv
```

### Copilot Pro PoC Writing Prompt

```
I found a potential vulnerability in this smart contract:

DOCUMENTED CLAIM: [exact quote from contract or whitepaper]
PROPERTY VIOLATED: [plain English statement of what should be true]
HOW IT'S VIOLATED: [which function, what check is missing]
COUNTER-EXAMPLE: [paste cex_analyzer output or your trace]

Write a complete Foundry test (PoC) that demonstrates this vulnerability.

Requirements:
- Use vm.startPrank() for actor impersonation
- Record state with console.log() BEFORE and AFTER the exploit
- Use descriptive variable names (balanceBefore not b1)
- The test should PASS to prove the bug exists (not revert)
- The final assertion should show the invariant IS violated
- Include a comment on each step explaining what the attacker does
- Include a comment explaining what SHOULD have happened instead

Also provide the one-line fix that would make this test FAIL
(i.e., the fix that closes the vulnerability).
```

---

## PART 12: CONTEST SUBMISSION STRATEGY

### The Key Insight

```
When you say:
  "This is a bug"
Judges can say:
  "Intended behavior" → INVALIDATED

When you say:
  "The protocol's own documentation at [source] states [claim].
   Certora Prover formally proves this claim is violated.
   Here is the mathematical counter-example.
   Here is the runnable test."
Judges cannot say:
  "Intended behavior" → MUST BE VALIDATED
```

### Packaging Your Submission

Every submission should include:

```
submission/
├── AUDIT_REPORT.md          # Your full report (Part 9 template)
├── design_doc.md            # Shows your research methodology
├── certora/
│   └── specs/
│       └── autospec_*.spec  # Generated CVL specs (proof)
├── test/
│   └── Exploit.t.sol        # Runnable PoC test
└── README.md                # How to run the PoC
```

### README.md for Your Submission

```markdown
# Bug Bounty Submission — [Protocol Name]

## How to Reproduce

### Prerequisites
- Foundry installed
- Protocol repository cloned

### Run the PoC
git clone [protocol repo]
cd [protocol]
forge install
forge test --match-test testVulnerabilityName -vvv

### Expected Output
[Describe what the test outputs that proves the bug]

### How Formal Verification Found This
This vulnerability was identified through formal verification using
Certora Prover via AIComposer. The CVL rule [rule_name] in
certora/specs/autospec_[component].spec produces a counter-example
that corresponds to the PoC test above.
```

---

## PART 13: CATASTROPHIC FAILURE SCENARIOS TEMPLATE
### A recommended section for every design_doc.md — not a format AI Composer recognizes structurally

Add this section to EVERY design_doc.md. This forces you to name the five worst outcomes and what prevents each
one. **Note on mechanism:** there's no code in AI Composer that looks for a section literally titled "Catastrophic
Failure Scenarios," a `CF-N` numbering scheme, or the `Must Never Happen:`/`Affected Functions:` labels below —
the design doc is read as prose by an upstream classification step (not string-matched by heading), so nothing
about this specific structure gets special treatment over any other clearly-written part of the document. The
value of doing this exercise is in the thinking it forces (naming the worst outcomes explicitly tends to surface
requirements you'd otherwise miss), and clear, explicit language here is more likely to be picked up accurately
regardless of the heading style you use.

```markdown
# Catastrophic Failure Scenarios

## CF-1: Theft of Reserved Funds

Impact:
Loss of reserved protocol funds. Core operations permanently broken.
Estimated loss: [$ amount based on TVL]

Must Never Happen:
Any withdrawal or transfer reduces the reserved balance.
Any actor accesses reserved funds through non-intended paths.
availableBalance() returns a value that includes reserved funds.

Affected Functions:
withdraw()
emergencyWithdraw()
transfer()
[any function that moves funds]

What AIComposer should verify:
After any call to withdraw(), reservedBalance is unchanged.
availableBalance() always equals contract.balance minus reservedBalance.
Reserved funds can only be reduced through their intended mechanism.

---

## CF-2: Protocol Insolvency

Impact:
User claims exceed protocol assets. Protocol cannot honor withdrawals.
Estimated loss: [$ amount — potentially entire TVL]

Must Never Happen:
Total liabilities (what the protocol owes) exceeds
total assets (what the protocol holds).

Affected Functions:
deposit(), withdraw(), borrow(), repay(), liquidate(), getHealthFactor()

What AIComposer should verify:
At all times: sum(user_claims) ≤ address(this).balance + external_assets.
No sequence of valid operations makes the protocol insolvent.

---

## CF-3: Share / Accounting Inflation

Impact:
Attacker obtains disproportionate ownership of the protocol.
Subsequent depositors receive far fewer shares than they should.
Estimated loss: [$ amount per attack]

Must Never Happen:
Shares minted to a user exceed the value of assets they deposited.
Share price changes as a result of external transfers, not deposits.
First depositor can set an initial share price that harms others.

Affected Functions:
deposit(), mint(), redeem(), withdraw(), convertToShares(), convertToAssets()

What AIComposer should verify:
Share price remains stable before and after any deposit.
Shares minted always proportional to assets received.
External ETH transfers do not change the share price calculation.

---

## CF-4: Permanent Lockup

Impact:
User funds become permanently unrecoverable.
Protocol effectively shut down.
Estimated loss: [$ amount — all user deposits]

Must Never Happen:
A state exists where withdraw() fails AND emergencyWithdraw() fails.
The pause mechanism eliminates all withdrawal paths.
Owner can freeze the contract with no recovery mechanism.

Affected Functions:
withdraw(), emergencyWithdraw(), pause(), unpause()

What AIComposer should verify:
For any user with balance > 0, at least one withdrawal path succeeds.
Pause does not block ALL fund recovery paths simultaneously.

---

## CF-5: Unauthorized Control

Impact:
Attacker gains privileged authority over the protocol.
Can drain treasury, change parameters, or halt operations.
Estimated loss: [$ amount — entire protocol TVL at risk]

Must Never Happen:
Non-owner successfully calls any owner-only function.
Ownership transfers to an unauthorized address.
Admin privileges are escalated without explicit owner action.

Affected Functions:
pause(), unpause(), setFees(), addAuthorizedDepositor(),
transferOwnership(), upgradeTo()

What AIComposer should verify:
Every privileged function reverts for non-authorized callers.
Ownership cannot change without explicit action from current owner.
```

---

## HOW THIS CHANGES YOUR DESIGN_DOC.MD

The properties you write in design_doc.md should now follow this pattern
for every single entry:

```
PROPERTY NAME: [descriptive name]

Claim: [one sentence — what the protocol promises]

If violated: [STEAL / PRINT / FREEZE / BRICK] → severity [CRITICAL/HIGH]

Statement:
[exact plain-English description of what must always be true]

What breaks if this fails:
[specific attack, dollar impact, who is affected]

Priority: [1-7 based on the priority order in Part 3]
```

**Example using a vault contract:**

```
PROPERTY NAME: Reserved Fund Protection

Claim: Reserved funds are exclusively for their intended purpose.

If violated: STEAL → CRITICAL ($50k+ bounty)

Statement:
The reserved balance can only decrease as a result of a successful
call to the intended reserve mechanism. No other function — including
withdraw(), emergencyWithdraw(), or any admin-only function —
may reduce the reserved balance.

What breaks if this fails:
An admin or attacker calls withdraw() to drain funds that were
reserved for protocol operations. The critical operation fails
or succeeds with wrong parameters. Up to the full reserved
amount is permanently lost from the protocol.

Priority: 1 (Asset Conservation — reserved funds variant)
```

---

## PART 14: COMPLETE WORKED EXAMPLE

This is a complete worked example showing how to apply all steps to a
real smart contract. The pattern applies to any protocol — DeFi, staking,
lending, governance, or NFT.

### Step 2 Output: Claims Extracted from the Contract

```
CLAIM 1: "Designed to handle deposits with a designated operator"
SOURCE:  Contract description comment
TYPE:    Access control
→ MEANING: Only designated entities can trigger critical operations

CLAIM 2: "A portion of funds is locked and cannot be withdrawn"
SOURCE:  Comment in the contract
TYPE:    Fund protection
→ MEANING: Locked/reserved funds are unavailable for general withdrawal

CLAIM 3: "Available balance excludes reserved amounts"
SOURCE:  @notice comment on availableBalance()
TYPE:    Accounting
→ MEANING: availableBalance = contract.balance − reservedBalance

CLAIM 4: Implicit from onlyOwner modifier on withdraw()
TYPE:    Access control
→ MEANING: Only owner can withdraw available funds

CLAIM 5: "Pauses critical operations"
SOURCE:  Function description for pause()
TYPE:    State transition
→ MEANING: When paused, critical operations must revert

CLAIM 6: "Locked funds are set aside for specific use later" +
         "Unlock returns funds back to available balance"
SOURCE:  Function descriptions for lock() and unlock()
TYPE:    Accounting
→ MEANING: lock() increases reservedBalance, unlock() decreases it

CLAIM 7: "The contract becomes immutable after finalization"
SOURCE:  Context from finalize() function
TYPE:    State machine
→ MEANING: finalize() is a permanent, irreversible state change
```

### Step 4 Output: Formal Properties

```
PROPERTY 1: Balance Invariant
The contract's total balance must always equal the sum of the
available balance plus the reserved balance. After every function call,
this equality must hold. If it does not, the contract's accounting
has become desynchronized — there are funds that are neither
available nor properly reserved.

PROPERTY 2: Withdrawal Access Control
The withdraw() function can only be called by the owner address.
Any caller other than the owner must cause the transaction to revert.
There is no path through any other function that allows a non-owner
to withdraw funds from the contract.

PROPERTY 3: Reserved Balance Cannot Be Withdrawn
When the owner calls withdraw(), the reserved balance must not decrease.
The amount withdrawn must not exceed the available balance, where
available balance is defined as the contract's total balance
minus the reserved balance. After withdraw() completes, the reserved
balance must be exactly the same as it was before the call.

PROPERTY 4: Pause Enforcement
When the paused flag is true, calls to critical operations must
always revert. No state-changing operations should succeed while
the contract is paused.

PROPERTY 5: Lock Increases Reserved Balance
A successful call to lock(amount) increases reservedBalance by exactly
amount and decreases availableBalance by exactly amount.
Total contract balance does not change.

PROPERTY 6: Unlock Decreases Reserved Balance
A successful call to unlock(amount) decreases reservedBalance by exactly
amount and increases availableBalance by exactly amount. The total
contract balance does not change, since unlock() only reclassifies
funds from reserved to available — no funds enter or leave the contract.

PROPERTY 7: Operator-Only Access to Critical Functions
Only the authorized operator address can call lock(), unlock(),
and executeOperation(). All other callers must revert.
There is no path through any other function that allows a non-operator
to reserve funds or trigger a critical operation.

PROPERTY 8: Available Balance Cannot Be Negative
The available balance calculated as the contract's total balance minus
the reserved balance must never be negative. This would indicate that
reservedBalance was incremented beyond the actual funds held, which would
mean the contract's accounting has lost synchronization. A negative
available balance would make it impossible to determine how much
can safely be withdrawn without touching reserved funds.
```

### Step 5 Output: Pre/Postconditions

#### withdraw(address recipient, uint256 amount)

```
PRECONDITIONS:
- msg.sender == owner
- recipient != address(0)
- amount > 0
- amount ≤ availableBalance()
- availableBalance() == address(this).balance − reservedBalance

POSTCONDITIONS:
- recipient.balance increased by exactly amount
- address(this).balance decreased by exactly amount
- reservedBalance is UNCHANGED (critical — reserved funds are protected)
- Withdrawal event is emitted

REVERTS IF:
- msg.sender != owner (onlyOwner modifier)
- recipient == address(0)
- amount == 0
- amount > availableBalance()
- Transfer to recipient fails
```

#### lock(uint256 amount)

```
PRECONDITIONS:
- msg.sender == operator
- paused == false
- amount > 0
- amount ≤ availableBalance()

POSTCONDITIONS:
- reservedBalance increased by exactly amount
- availableBalance decreased by exactly amount
- address(this).balance UNCHANGED (funds stay in contract)
- FundsLocked event is emitted

REVERTS IF:
- msg.sender != operator
- paused == true
- amount == 0
- amount > availableBalance()
```

#### executeOperation(Operation calldata op)

```
PRECONDITIONS:
- msg.sender == operator
- paused == false
- op.amount > 0
- op.amount ≤ availableBalance()
- TARGET_CONTRACT address is set and valid

POSTCONDITIONS:
- Operation submitted to target contract
- address(this).balance decreased by op.amount
- OperationExecuted event emitted

REVERTS IF:
- msg.sender != operator
- paused == true
- op.amount > availableBalance()
- TARGET_CONTRACT call fails
```

---

## PART 15: WSL UBUNTU SETUP REFERENCE

### Environment Variables (Add to ~/.bashrc)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Required: AIComposer won't start without this
export CERTORAKEY="..."                    # Required for cloud prover mode
export AUTOSETUP_PATH="~/AutoSetup"       # Required: set after Certora grants access
export CERTORA_AI_COMPOSER_PGHOST="localhost"
export CERTORA_AI_COMPOSER_PGPORT="5432"
```

### One-Time Setup Commands

```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip git

# 2. Install uv
pip install uv

# 3. Clone AIComposer
cd ~
git clone https://github.com/Certora/AIComposer.git
cd AIComposer

# 4. Start PostgreSQL (Docker Desktop must be running on Windows
#    with WSL integration enabled in Docker Desktop settings)
cd scripts/
docker compose create && docker compose start

# 5. Build RAG database (takes ~15 minutes, run once)
./gen_docs.sh
./populate_rag.sh

# 6. Install Solidity compilers
pip install solc-select
solc-select install 0.8.29
ln -s ~/.solc-select/artifacts/solc-0.8.29/solc /usr/local/bin/solc8.29

# 7. Install AIComposer
cd ~/AIComposer
uv sync --group ml

# 8. Verify
python -c "import composer; print('AIComposer ready')"
```

### Docker Desktop WSL Integration (Critical)

```
Open Docker Desktop on Windows
→ Settings → Resources → WSL Integration
→ Enable integration with your Ubuntu distribution
→ Apply & Restart

Then in WSL terminal:
docker ps    # Should work without sudo
```

---

## PART 16: QUICK REFERENCE CHECKLIST

### Before Starting Any Audit

- [ ] Target protocol selected (has public docs/whitepaper)
- [ ] VS Code open with contract file and Copilot Pro active
- [ ] Notepad/file ready for extracted claims (Step 2)

### During Property Extraction (7 Steps)

- [ ] Step 1: Copilot summarized contract (understand before extracting)
- [ ] Step 2: YOU manually extracted claims from comments/require/modifiers
- [ ] Step 3: Copilot mapped 3–6 components
- [ ] Step 4: YOU converted each claim to a formal property (plain English)
- [ ] Step 5: Pre/postconditions defined for 3+ critical functions
- [ ] Step 6: design_doc.md drafted — ZERO code in it
- [ ] Step 7: Copilot reviewed doc — all 12 categories addressed (categories 11-12: upgrade/proxy safety and cross-chain/bridge trust — see [Part 4](#part-4-the-12-business-logic--infrastructure-bug-categories))

### Before Running AIComposer

- [ ] design_doc.md is 2000+ words
- [ ] design_doc.md has NO CVL or Solidity code
- [ ] All 12 business logic & infrastructure categories have entries (skip category 12 if no cross-chain component)
- [ ] Edge cases listed (zero, max, empty, paused states)
- [ ] ANTHROPIC_API_KEY set in WSL
- [ ] CERTORAKEY set in WSL
- [ ] AUTOSETUP_PATH set in WSL (after access granted)
- [ ] Docker PostgreSQL running: `docker ps | grep pgvector`
- [ ] RAG database populated (one-time setup done)

### After Getting Prover Results

- [ ] All VIOLATED rules investigated with cex_analyzer (the Prover's actual status string is `VIOLATED`, not `FAILED` — see Prover Results Reference)
- [ ] Each failure verified as real (not a proof artifact)
- [ ] Finding classified as Vulnerability / Design Flaw / Spec Violation / Correctness
- [ ] PoC test written and passing (using Part 11 template)
- [ ] Impact quantified in dollars
- [ ] Severity justified using Immunefi/Sherlock criteria
- [ ] Report written using Part 9 template
- [ ] Finding ties back to documented protocol claim
- [ ] Submission package prepared (report + specs + PoC + README)

---

## PART 17: POST-DEPLOYMENT — MONITORING, INCIDENT RESPONSE & SUPPLY CHAIN

A verified spec and a clean audit report are necessary, not sufficient, for production safety.
Everything in this Part happens *after* deployment, and is commonly under-specified relative to
the pre-deployment audit workflow above — this section is intentionally short and practical
rather than exhaustive, since each of these is a discipline in its own right.

### Monitoring

- **Alert on the properties you just verified.** Every CVL invariant you proved pre-deployment
  is also a candidate for a post-deployment monitor — if `totalLiabilities <= totalAssets` was
  worth proving, it's worth continuously checking on-chain too (a proof covers the code as
  written; it does not cover a future upgrade, a misconfiguration, or an assumption about an
  external contract that later stops holding).
- Tools in this space (evaluate independently; not an endorsement): Forta (decentralized
  detection bots), OpenZeppelin Defender (monitoring + automated response actions), Tenderly
  (alerting + simulation). At minimum, monitor: large/unusual withdrawals, admin-function calls,
  upgrade events, and any oracle price deviating sharply from a reference source.

### Incident Response

- **Pause authority should be pre-delegated and rehearsed, not improvised.** Category 6
  (Access Control) properties should already guarantee *who* can pause; make sure that party
  has a tested, fast path to actually doing so (a multisig with an unfamiliar UI at 3am is not
  a fast path).
- **Have a drafted (not written-from-scratch-during-the-incident) communication template** —
  what gets disclosed, when, and through which channel, decided calmly in advance rather than
  under pressure.
- **Postmortem structure**: what was verified vs. not, why the verified properties didn't
  prevent this incident (new code path? compromised key? external dependency change? a gap in
  the design doc that never became a CVL rule?), and — concretely — what new property or
  category should be added to `design_doc.md` as a result. Feed real incidents back into the
  audit process that produced this document.

### Supply Chain

- **Pin dependencies to exact versions/commits, not ranges.** This repository's own
  `pyproject.toml` pins the `graphcore` dependency to an exact git commit SHA rather than a
  branch or tag — apply the same discipline to your own Solidity dependencies (OpenZeppelin,
  Solmate, etc.) and npm/Foundry lockfiles.
- **Prefer reproducible builds.** Verify that the deployed bytecode matches the audited source
  (Etherscan/Sourcify verification, or a documented deterministic-build process) — an audit of
  source code that doesn't match what's actually deployed provides no real guarantee.
- **Treat your build toolchain as part of the attack surface.** A compromised compiler,
  dependency, or CI runner can introduce a vulnerability that source-level audit and formal
  verification will never see, because they never see the actual build step.

---

*Master Audit Guide V2 — Corrected and Comprehensive*
*Combines all 18 items from conversation + 5 valuable items from uploaded docs*
*17 Parts covering the complete end-to-end workflow, including post-deployment*