# AIComposer Audit Workflow — Complete Manual Guide

> **Status**: AutoSetup is currently private to the Certora team. This guide covers
> everything you CAN do right now with GitHub Copilot Pro (Claude) to perform a
> professional whitehat audit, plus what changes when AutoSetup becomes available.

---

## The Current Reality

### What AIComposer Actually Does in One Command

AIComposer's pipeline is a single command that runs all phases sequentially.
`--cloud` is not a separate step — it's a flag on Phase 5 that tells the
prover where to run (Certora's servers vs your machine). The pipeline generates
CVL AND runs the prover in one go.

```
YOU RUN ONE COMMAND:
tui-autoprove \
  ~/projects/protocol \
  src/Contract.sol:ContractName \
  design_doc.md \
  --cloud

WHAT HAPPENS INSIDE (all automatic, one invocation):

Phase 0 ─── Reads code + design_doc.md          ✓ Works without AutoSetup
            Identifies components
            ↓
Phase 1 ─── Harness Setup                        ✗ BLOCKED without AutoSetup
            Compiler analysis                         Pipeline stops here
            Generates harness contracts
            ↓
Phase 2 ─── CVL summaries for external           ✗ Never reached
            contracts
            ↓
Phase 3 ─── Structural invariants                ✗ Never reached
            ↓
Phase 4 ─── Property extraction per component    ✗ Never reached
            ↓
Phase 5 ─── CVL generation                       ✗ Never reached
            Runs Certora Prover ←── --cloud means
            (cloud or local)        THIS step runs on
                                    Certora's servers,
                                    not your machine

RESULT WITHOUT AUTOSETUP: Pipeline stops at Phase 1.
You cannot generate CVL via AIComposer without AutoSetup.
```

### The Practical Workaround (Works TODAY)

When you have manually prepared the harness, `.conf` files, and `design_doc.md`,
you bypass AIComposer's pipeline entirely and use the Certora Prover directly:

```bash
# Step 1: You write CVL manually (Copilot helps — one .spec file, all rules inside)
# Saved to: certora/specs/MyContract.spec

# Step 2: You run the prover directly on your manually written spec
certoraRun certora/confs/MyContract.conf \
  --cloud \
  --msg "Audit — reserved balance protection"

# That's the complete command.
# certoraRun reads your .conf file which points to your .spec file.
# --cloud sends the job to Certora's servers.
# Results come back with VERIFIED / FAILED / TIMEOUT per rule.
```

| Situation | What you use | What is automatic |
|-----------|-------------|-------------------|
| After AutoSetup | `tui-autoprove --cloud` | Everything — Phases 0 through 5 |
| Before AutoSetup | `certoraRun --cloud` | Nothing — you write CVL manually |
| Before AutoSetup + Copilot | Copilot writes CVL → `certoraRun --cloud` | CVL writing is assisted but not automated |

### What you have and what you can do

```
WHAT YOU HAVE TODAY:
  ✓ GitHub Copilot Pro (Claude Opus) in VS Code
  ✓ AIComposer codebase (installed, ready)
  ✓ All documentation (MASTER_AUDIT_GUIDE_V2, REFERENCE)

WHAT YOU CAN DO TODAY:
  ✓ Full contract analysis (storage, functions, dependencies)
  ✓ Extract documented claims from source code
  ✓ 10-category business logic vulnerability analysis
  ✓ Write professional design_doc.md
  ✓ Write CVL specifications manually (Copilot helps)
  ✓ Generate prover .conf files + harness contracts manually
  ✓ Run certoraRun directly with --cloud (cloud prover)
  ✓ Write Foundry PoC tests
  ✓ Find real $10k-$50k+ business logic bugs

WHAT NEEDS AUTOSETUP (coming later):
  ✗ Run console-autoprove / tui-autoprove (the AIComposer pipeline)
  ✗ Automatic CVL generation from design_doc.md
  ✗ CVL Judge automated quality review
  ✗ Auto-generated harness contracts and prover configs
```

### How CVL files are structured: one .spec per component, all rules inside

**Each property becomes one `rule` or `invariant` block. All rules for one contract live together in one `.spec` file.**

```
ONE .spec FILE PER CONTRACT/COMPONENT:
├── Contains ALL properties for that contract
├── Each property = one rule block or one invariant block
└── Properties are NOT in separate files

STRUCTURE OF A .spec FILE:

certora/specs/Vault.spec
─────────────────────────────────────────────────────
// PROPERTY 1: Balance Invariant
invariant balanceInvariant()
    address(this).balance == availableBalance() + reservedBalance()

// PROPERTY 2: Withdrawal Access Control
rule onlyOwnerCanWithdraw(env e, address recipient, uint256 amount) {
    require e.msg.sender != owner();
    withdraw@withrevert(e, recipient, amount);
    assert lastReverted;
}

// PROPERTY 3: Reserved Balance Protected
rule withdrawDoesNotReduceReservedBalance(env e, address r, uint256 a) {
    uint256 reservedBefore = reservedBalance();
    withdraw(e, r, a);
    assert reservedBalance() == reservedBefore;
}

// PROPERTY 4: Pause Enforcement
rule pauseBlocksOperations(env e, uint256 amount) {
    require paused() == true;
    deposit@withrevert(e, amount);
    assert lastReverted;
}
─────────────────────────────────────────────────────
All 8 properties → all 8 rules/invariants → one .spec file
```

**How AIComposer does it:** One `.spec` file per component (`autospec_{ComponentName}.spec`).
Inside that file, every extracted property gets its own `rule` or `invariant` block.
If StakingVault has 8 properties, AIComposer writes 8 separate blocks inside one
`autospec_StakingVault.spec` file.

**How you do it manually:** Same structure. One `.spec` file. Each property from
your design_doc.md becomes one `rule` or `invariant` block inside that file.
Separate them with comments so each rule is clearly identified.

### The two CVL keywords and when to use each

```
invariant propertyName()
    [mathematical expression that must always be true]

→ Use for: things that must hold in ALL states at ALL times
→ Example: total supply always equals sum of all balances
→ Certora checks: constructor + after every function call in the contract

rule propertyName(env e, params...) {
    [setup — preconditions]
    [call the function]
    assert [what must be true after]
}

→ Use for: things that must hold after specific function calls
→ Example: onlyOwnerCanWithdraw — specific function must revert for non-owner
→ Certora checks: the specific execution path you describe in the rule body
```

### Your Copilot prompt to write CVL from design_doc.md

```
Convert these formal properties from my design_doc.md into a
single Certora CVL specification file.

PROPERTIES TO FORMALIZE:
[PASTE ALL PROPERTIES FROM SECTION 4 OF DESIGN DOC]

CONTRACT:
[PASTE CONTRACT CODE]

Rules:
- Use invariant for properties that must hold in all states
- Use rule for properties about specific function behavior
- Put ALL rules in ONE .spec file
- Give every rule a clear descriptive name
- Add a comment above each rule quoting the design_doc.md
  property it formalizes
- Use @withrevert when testing that functions should revert
- Use env e as first parameter on all rules

Output: one complete .spec file ready to run with certoraRun
```

### What "business logic flaw" actually means — the Type A / Type B distinction

This is the most important distinction in the entire guide. It changes
how you think about every audit.

**Type A: Business Logic BUG (Code is Wrong)**

The code does something different from what it was designed to do.
A check is missing. A variable is wrong. Math is incorrect. This is a
traditional bug that happens to be in the business logic area.

```
Type A Example — Double Withdrawal Bug:

Intent:   After withdraw(), user's claim should be zeroed
Code:     user.claim -= amount;  ← subtraction happens AFTER transfer
          token.transfer(user, amount);

THE BUG:  Order is wrong. Claim is zeroed after transfer.
          Reentrancy during token.transfer() can call withdraw() again
          before claim is zeroed.

This IS a bug. The code does NOT correctly implement what it
was designed to do. The design said "zero the claim before
transferring." The code does the opposite. Fix: reorder the code.
```

**Type B: Business Logic FLAW (Code is Perfect — Design is Broken)**

The code does EXACTLY what it was designed to do. Every function
is correct. Every check passes. Every calculation is correct. But a
sequence of VALID function calls produces an outcome that violates
the protocol's fundamental guarantee.

```
Type B Example — Share Price Inflation Flaw:

ALL OF THIS CODE IS CORRECT:
  deposit()  correctly calculates shares = amount * totalShares / totalAssets
  withdraw() correctly calculates amount = shares * totalAssets / totalShares
  receive()  correctly accepts ETH donations (by design)

THE FLAW:
  Step 1: Attacker deposits 1 wei → receives 1 share (1:1 ratio, pool is empty)
  Step 2: Attacker sends 100 ETH directly to contract (valid operation!)
  Step 3: Now totalAssets = 100 ETH + 1 wei, totalShares = 1
  Step 4: Next depositor deposits 1 ETH
          shares = 1 ETH * 1 / (100 ETH + 1 wei) = 0 (rounds to 0)
  Step 5: Depositor receives 0 shares, loses 1 ETH
  Step 6: Attacker withdraws: 1 share * (101 ETH + 1 wei) / 1 = ~101 ETH

NOT A BUG. Every single line of code is correct.
The design itself allows this sequence of valid operations.
This is a FLAW in the protocol's guarantee that depositors receive
fair share value.

Fix: redesign the share minting logic — not just fix a line of code.
```

**The critical difference:**

```
CLASSICAL BUG:
  Code should do X → code actually does Y → fix the code

BUSINESS LOGIC BUG (Type A):
  Code should prevent double withdrawal → order of operations is wrong → reorder

BUSINESS LOGIC FLAW (Type B):
  Code correctly implements the design → the design allows harmful
  sequences of valid calls → redesign the protocol
```

### How Each Type Is Found

| Finding Type | Manual Review | Fuzzing | Certora Prover |
|-------------|--------------|---------|---------------|
| Type A — Business Logic Bug | ✓ Yes | ✓ Sometimes | ✓ Yes |
| Type B — Design Flaw in Perfect Code | Rarely | Rarely | ✓ **Best tool** |

**Why Certora is the best tool for Type B:**

Manual review asks: "Is this code correct?" → finds Type A bugs.
Certora asks: "Does any sequence of valid calls violate this invariant?"
→ finds Type B flaws.

Certora does not care whether individual functions are correct. It checks
whether the INVARIANT holds across ALL possible sequences of ALL possible
function calls. This is mathematically exhaustive in a way that manual
review and fuzzing cannot match.

```
CERTORA CHECKS:
  Can ANY sequence of:
    deposit() → lock() → withdraw() → unlock() → emergencyWithdraw() → ...
  (in any order, any number of times, with any parameters)
  violate the invariant: reservedBalance ≤ address(this).balance?

MANUAL REVIEW CHECKS:
  Does withdraw() look correct?  ← only checks one function
  Does lock() look correct?      ← only checks one function
  Misses the COMBINATION of calls

FUZZING CHECKS:
  Random sequences of calls
  May never find the specific harmful combination
  Has no mathematical proof of exhaustiveness
```

### Which categories are typically which type

```
MOSTLY TYPE B — Design Flaws in Correct Code:
  1. Asset Conservation        ← donation attacks, share inflation
  4. Solvency                  ← valid borrowing sequences break LTV
  5. Share Price Integrity      ← first depositor attack, donations
  7. State Machine Correctness ← valid calls in wrong order
  8. Accounting Synchronization ← valid updates that desync systems

CAN BE EITHER TYPE A OR TYPE B:
  2. Double Withdrawal         ← Type A if reentrancy, Type B if claim tracking
  3. Reserved Funds            ← Type A if missing check, Type B if design gap
  6. Access Control            ← Type A usually (missing modifier)
  9. Reward Integrity          ← Type A if reentrancy, Type B if calculation flaw
  10. Fund Lockup              ← Type A if emergency path missing, Type B if
                                  sequence blocks all paths
```

### The practical answer

The 10 categories find BOTH bugs and flaws. The reason they are worth
$10k-$50k+ is precisely because they frequently contain **Type B flaws**
— where the code is perfectly correct and only formal verification can
prove the design guarantee is broken. No static analyzer, no linter, no
classical audit catches Type B. This is the $50k finding class.

---

## Why Business Logic Matters

### The Paradigm Shift

Classical bugs (reentrancy, overflow, unchecked returns) are table stakes.
Automated tools like Slither and MythX catch them before you even start.
Bug bounty programs reward them at Low to Medium severity.

**The real money ($10k-$50k+) is in business logic.** These are flaws where
code is *perfectly correct* but the protocol's guarantees are broken through
valid sequences of function calls.

**The Euler Finance exploit ($196M)**: No reentrancy. No overflow. No oracle
manipulation. Access control was correct. But the solvency invariant could
be broken through a sequence of valid operations. The code had zero bugs —
but the protocol's guarantee was violated.

### The Two Questions Every Auditor Must Ask

```
QUESTION 1: "What does this contract DO?"
  → Finds implementation bugs (table stakes)

QUESTION 2: "If this invariant breaks, can someone:
             STEAL money?     → CRITICAL
             PRINT money?     → CRITICAL (asset conservation)
             FREEZE money?    → CRITICAL/HIGH (lockup)
             BRICK the protocol? → HIGH/CRITICAL (state machine)"
  → This is where $10k-$50k+ findings live
```

---

## The Complete Manual Workflow (8 Phases)

```
PHASE 1: Contract Analysis          (Claude — 10 min)
PHASE 2: Claim Extraction           (Manual — 30 min)
PHASE 3: 10-Category Business Logic (Claude — 15 min wait)
PHASE 4: Design Document            (Manual + Claude — 60 min)
PHASE 5: CVL Writing               (Claude — 45 min)
PHASE 6: Prover Config + Harness    (Claude — 15 min)
PHASE 7: PoC Test Writing          (Claude + Manual — 30 min)
PHASE 8: When AutoSetup Arrives     (AIComposer — 15 min)
───────────────────────────────────────────────────────
TOTAL: ~3-4 hours per protocol
EXPECTED: 2-5 real business logic vulnerabilities
```

---

## Phase 1: Contract Analysis

**Goal**: Understand what the contract does at every level before hunting for bugs.

**Tool**: GitHub Copilot Pro Chat in VS Code

---

### Step 1A: Storage Layout & Function Map

Open the contract in VS Code. Ask Copilot Chat:

```
Analyze this Solidity contract for:

1. STORAGE LAYOUT
   List every state variable with:
   - Name and type
   - Purpose in the protocol
   - Which functions read it and which write to it

2. FUNCTION ANALYSIS
   For every function, list:
   - Visibility (public/external/internal/private)
   - Mutability (view/pure/payable/nonpayable)
   - External calls made (if any)
   - State variables modified
   - Reentrancy risk (LOW / MEDIUM / HIGH — explain why)

3. REENTRANCY MAP
   For every external call in the contract:
   - Which function contains the external call?
   - Is state updated BEFORE or AFTER the external call?
   - Could an attacker exploit the window between the call and the state update?

4. ACTOR MAP
   List every actor and what they can do:
   - Owner / Admin
   - Authorized operators
   - Regular users
   - External contracts
```

**What you get**: A complete map of every variable, function, and actor in the contract. You cannot audit what you do not understand.

---

### Step 1B: External Contract Classification

Ask Copilot Chat:

```
Classify all external contracts and imports used by this project:

[PASTE ALL CONTRACT FILES]

For each external or imported contract:
1. TYPE: ERC20 / ERC721 / Interface / Abstract / Library / Oracle / Other
2. IMPORTED BY: which contracts in the project use it
3. KEY FUNCTIONS CALLED: which functions does this project call on it
4. POTENTIAL ISSUES:
   - Could this token have transfer fees? (USDT, some ERC20 variants)
   - Could this token be pausable? (USDC, USDT)
   - Could this token have callbacks? (ERC777, ERC1155 — reentrancy risk)
   - Is this a rebasing token? (stETH, aToken — balance changes autonomously)
   - Does this oracle have a staleness check? How recent are its updates?
5. CVL SUMMARY SUGGESTION: how should Certora summarize this?
   - DISPATCHER(true) — if the implementation matters
   - NONDET — if behavior is unpredictable
   - CONSTANT — if return values never change
   - Custom summary — if a specific model is needed
```

**What you get**: You now know which external contracts could introduce hidden risks (transfer fees, callbacks, rebasing) and how to model them in CVL.

---

## Phase 2: Claim Extraction

**Goal**: Extract every promise the protocol makes about itself. These become the
properties you verify. When the prover finds a violation, the protocol cannot
argue — they made the claim.

**Tool**: Your eyes + brain (critical thinking). Copilot validates afterward.

---

### What to Look For

Read the contract source code. Do not skim. Extract every statement that implies
a guarantee:

| Source | Example | What It Reveals |
|--------|---------|-----------------|
| **NatSpec @notice** | `@notice Only the owner can pause` | Access control claim |
| **NatSpec @dev** | `@dev Staged ETH is excluded from availableBalance` | Accounting claim |
| **require() messages** | `require(amount <= available, "Exceeds available")` | Balance constraint claim |
| **Modifier names** | `onlyOwner`, `onlyDepositor`, `whenNotPaused` | Access + state claims |
| **Variable names** | `reservedBalance`, `availableBalance` | Fund separation claim |
| **Function names** | `emergencyWithdraw()` | Recovery path claim |
| **Event names** | `FundsLocked`, `OperationExecuted` | State transition claim |

---

### Manual Extraction (You Do This)

Open a new file. For every claim found, write:

```
CLAIM: [exact wording or close paraphrase from the source]
SOURCE: [function name, line number, or NatSpec tag]
TYPE: [access control / accounting / fund protection / state transition / other]
MEANING: [what this claim actually promises in plain English]
```

**Example output from a vault contract:**

```
CLAIM: "Only the owner can withdraw available funds"
SOURCE: onlyOwner modifier on withdraw()
TYPE: Access control
MEANING: No address except the owner can call withdraw() successfully

CLAIM: "Locked funds are excluded from available balance"
SOURCE: @notice on availableBalance()
TYPE: Accounting
MEANING: availableBalance = contract.balance - lockedBalance AFTER every function

CLAIM: "Pausing prevents all state-changing operations"
SOURCE: whenNotPaused modifier on deposit(), withdraw(), lock()
TYPE: State transition
MEANING: When paused == true, deposit, withdraw, and lock must all revert

CLAIM: "Emergency withdrawal bypasses the pause"
SOURCE: Function name emergencyWithdraw(), no whenNotPaused modifier
TYPE: Fund recovery
MEANING: emergencyWithdraw() must succeed even when the contract is paused
```

---

### Copilot Validation (After Your Extraction)

```
I have extracted these documented claims from the contract:

[PASTE YOUR CLAIMS]

Please validate:
1. Did I miss any claims? Check require() statements, modifiers,
   variable names, and NatSpec tags I might have overlooked.
2. Are my interpretations accurate? For each claim, does the
   code actually support what I think it promises?
3. Are there IMPLICIT claims? (Things the code assumes without
   stating them — e.g., "the initial owner is set correctly")
```

---

## Phase 3: Business Logic Analysis

**Goal**: Find the deep, high-value vulnerabilities that static analyzers miss.

**Tool**: GitHub Copilot Pro Chat.

> The prompt below is a condensed, 10-category quick-start variant meant for pasting straight into Copilot Chat. For the canonical, current 12-category version — which adds Upgrade & Proxy Safety and Cross-Chain/Bridge Trust — use **Prompt A** in [`MASTER_AUDIT_GUIDE_V2.md`](MASTER_AUDIT_GUIDE_V2.md) Part 6.

---

### The Prompt

```
Analyze this smart contract for business logic vulnerabilities.
This is for a bug bounty submission targeting $10k–$50k+ findings.

Ignore classical bugs like reentrancy and overflow — assume those are
handled by automated tools. Focus entirely on business logic:
invariants whose violation leads to theft, inflation, freezing, or
protocol shutdown.

CONTRACT CODE:
[PASTE FULL CONTRACT]

ANALYZE EACH OF THESE 10 CATEGORIES:

────────────────────────────────────────────────────────
1. ASSET CONSERVATION
   Claim: Assets cannot be created from nothing.
   - Can total claimable value exceed real assets held?
   - Can a user increase their claim without depositing value?

2. FUNDS CANNOT BE WITHDRAWN TWICE
   Claim: The same value cannot be redeemed twice.
   - Is the user's claim decremented BEFORE or AFTER the transfer?
   - Can withdraw() complete and then succeed again immediately?

3. RESERVED FUNDS ARE UNTOUCHABLE
   Claim: Designated reserved funds cannot be accessed through
   any path not intended for them.
   - What funds are reserved?
   - Can owner/admin access them through withdraw() or emergencyWithdraw()?

4. SOLVENCY
   Claim: Total liabilities must never exceed total assets.
   - What does this protocol OWE to users?
   - What does this protocol HOLD in real assets?
   - Under what operations can OWED > HELD?

5. EXCHANGE RATE / SHARE PRICE INTEGRITY
   Claim: Share value cannot be artificially inflated.
   - Can ETH/tokens sent directly to the contract inflate share price?
   - Can a first depositor attack work?

6. ACCESS CONTROL GUARANTEES
   Claim: Only authorized actors perform privileged actions.
   - List every privileged function. Where exactly is the access check?
   - Can any check be bypassed?

7. STATE MACHINE CORRECTNESS
   Claim: Actions must happen in a valid sequence.
   - Draw the valid state transitions. Can any step be skipped?
   - Can the contract reach a state through invalid ordering?

8. ACCOUNTING SYNCHRONIZATION
   Claim: All accounting systems tracking the same value stay synced.
   - List all variables tracking the same value. Can they diverge?
   - What happens if an external call fails mid-accounting?

9. REWARD INTEGRITY
   Claim: Rewards are distributed fairly. Cannot double-claim.
   - Is claimable amount zeroed BEFORE the transfer?
   - What if claimRewards() is called twice in the same block?

10. FUNDS CANNOT BECOME PERMANENTLY LOCKED
    Claim: Valid withdrawal paths always exist.
    - If paused indefinitely, can users withdraw through ANY path?
    - Can both normal AND emergency paths fail simultaneously?

────────────────────────────────────────────────────────

FOR EACH CATEGORY PROVIDE:

VULNERABILITY FOUND: YES / NO

If YES:
- EXACT FUNCTION: [function name]
- EXPLOIT SEQUENCE: [step-by-step — what the attacker does]
- FINANCIAL IMPACT: [$ estimate]
- AUDITOR MINDSET CHECK: STEAL / PRINT / FREEZE / BRICK?
- DESIGN DOC PROPERTY: [exact plain-English property for design_doc.md]
- SUGGESTED FIX: [brief code change]

If NO:
- ONE SENTENCE explaining why the category is safe here
```

**What you get**: A complete vulnerability assessment covering all 10 categories. Claude identifies exactly which functions have business logic flaws, how to exploit them, what the dollar impact is, and what property to add to your design document.

**Expected output pattern:**

```
CATEGORY 3: RESERVED FUNDS ARE UNTOUCHABLE

VULNERABILITY FOUND: YES

EXACT FUNCTION: withdraw(uint256 shares)
EXPLOIT SEQUENCE:
  1. Owner locks 100 ETH by calling lock(100 ETH)
     → reservedBalance = 100, availableBalance = 0
  2. Owner calls withdraw(100 shares)
     → withdraw() checks: 100 shares ≤ totalSupply ✓
     → Calculates: (100 shares * 100 ETH) / totalSupply = 100 ETH
     → Does NOT check: 100 ≤ availableBalance (which is 0!)
     → Transfers 100 ETH to owner
  3. reservedBalance still shows 100 ETH — but the contract has 0
     → availableBalance is now -100 (invalid state)
  4. Locked operation fails because funds are gone

FINANCIAL IMPACT: Up to the full reserved amount. For a protocol
  with $10M TVL in reserved funds, this is a $10M loss.

AUDITOR MINDSET CHECK: STEAL — owner can drain reserved funds.
  CRITICAL severity.

DESIGN DOC PROPERTY:
  "When withdraw() is called, the amount available to withdraw
   is calculated from availableBalance (total balance minus
   reserved balance), not from total balance alone."

SUGGESTED FIX:
  In withdraw(), replace:
    uint256 amount = (shares * address(this).balance) / totalSupply;
  With:
    uint256 available = address(this).balance - reservedBalance;
    uint256 amount = (shares * available) / totalSupply;
```

---

## Phase 4: Design Document

**Goal**: Produce a professional `design_doc.md` that serves as both your audit
research record and the input to AIComposer when AutoSetup becomes available.

**Tool**: Manual writing, with Copilot structuring.

**Gold standard reference**: See `examples/cccp_buggy/system_doc.txt` — a
200+ line design document with mathematical formulas, economic models,
explicit guarantees, and risk analysis. This is the quality level AIComposer
expects. Your design_doc.md should aim for this level of depth.

---

### Step 4A: Write the Document

Open `design_doc.md` in VS Code. Follow this exact 9-section structure:

```markdown
# [Protocol Name] Design Document for AIComposer

## 1. Executive Summary
[2-3 sentences: what the protocol does, what it guarantees]

## 2. System Architecture
### Actors
[List every actor: users, owner, operators, external contracts]

### Components
[From your Phase 1 analysis — one subsection per component]

## 3. Core Guarantees
[From Phase 2 — every claim you extracted]
For each guarantee:
GUARANTEE: [what the protocol promises]
IMPACT IF VIOLATED: [STEAL/PRINT/FREEZE/BRICK] → severity
SOURCE: [where in the code/docs this guarantee is stated]

## 4. Formal Properties
[From Phase 2 + Phase 3 — convert guarantees to properties]
Organized by the 10 business logic categories.

### Asset Conservation
PROPERTY: [plain English — what must always be true]

### Reserved Fund Protection
PROPERTY: [plain English]

[Continue for all relevant categories]

## 5. Function Specifications
[From Phase 1 — pre/postconditions for critical functions]

### functionName(params)
- Preconditions: [what must be true before calling]
- Postconditions: [what must be true after]
- Reverts if: [conditions that cause revert]

## 6. High-Level Properties for Verification
[Top 5-10 properties for AIComposer to prioritize]
1. property_name — one-line description
2. property_name — one-line description

## 7. Edge Cases & State Transitions
- Boundary conditions: zero amounts, max values, empty states
- State transitions: pause/resume, lock/unlock, emergency modes
- Reentrancy points: where external calls happen
- Oracle manipulation vectors

## 8. Business Logic Concerns
[From Phase 3 — Claude's analysis]
For each of the 10 categories where a vulnerability was found,
summarize the finding in plain English.

## 9. Assumptions
[From REFERENCE.md Section 7 template]
- Token assumptions (ERC20 standard, no callbacks, no fees)
- External contract assumptions (oracle, deposit contract)
- Attacker model (what they CAN and CANNOT do)
- Admin trust model (choose Option A, B, or C)
- Blockchain / environment assumptions
- Formal verification scope limitations
- Out of scope items
```

### Step 4B: Have Copilot Review

```
Review this design_doc.md I have written for AIComposer.

[PASTE YOUR COMPLETE design_doc.md]

Check:
1. Are all properties SPECIFIC to this protocol? (not generic)
2. Are all 10 business logic categories addressed?
3. Does every property trace back to a documented claim?
4. Is the document completely FREE of CVL, Solidity, or pseudocode?
5. Are edge cases covered? (zero, max, empty, paused states)
6. For each property: is it clear what breaks if it fails?
   (theft / freezing / insolvency / minting / none)

Give line-by-line feedback on what to improve.
```

**What you get**: A polished, professional design document that captures every claim, every property, and every vulnerability finding. When AutoSetup becomes available, this document is fed directly into AIComposer.

---

## Phase 5: CVL Writing (Manual)

**Goal**: Write CVL specifications that formalize the properties from your design document into rules the Certora Prover can verify.

**Tool**: GitHub Copilot Pro Chat — give it a property from Phase 4, get a CVL rule back.

---

### Step 5A: Write CVL for Each Property

For each property in your design document, ask Copilot:

```
Write a Certora CVL rule or invariant that verifies the following
property from my audit of [ContractName]:

PROPERTY:
[Paste the exact property statement from your design_doc.md, Section 4]

CONTRACT CODE:
[Paste the contract]

The rule should:
- Use descriptive rule names matching the property
- Include realistic preconditions (not overly constrained)
- Have clear assertions that directly test the property
- Use @withrevert for negative tests
- Include comments explaining what each section verifies

CVL version: use standard CVL syntax. Do not use experimental features.
```

**Example input → output:**

*Input property:*
```
"When withdraw() is called, the amount available to withdraw is
calculated from availableBalance (total balance minus reserved
balance), not from total balance alone. withdraw() must never
reduce the reserved balance."
```

*Copilot generates:*
```cvl
// ==========================================
// PROPERTY: Reserved Balance Protection
// ==========================================

rule reservedBalanceProtectedOnWithdraw(env e, uint256 shares) {
    // Preconditions: ensure a meaningful test
    require shares > 0;
    require shares <= balanceOf(e.msg.sender);

    // Record state before withdrawal
    uint256 reservedBefore = reservedBalance();

    // Attempt withdrawal
    withdraw(e, shares)@withrevert();

    // If withdrawal succeeded, reserved balance must be unchanged
    if (!lastReverted) {
        assert reservedBalance() == reservedBefore,
            "RESERVED VIOLATION: withdraw() reduced the reserved balance";
    }
}

rule withdrawCannotExceedAvailableBalance(env e, uint256 shares) {
    require shares > 0;
    require shares <= balanceOf(e.msg.sender);

    uint256 totalBalance = address(this).balance;
    uint256 reserved = reservedBalance();
    uint256 available = totalBalance - reserved;

    // If available balance is zero, withdrawal must revert
    require available == 0;

    withdraw(e, shares)@withrevert();
    assert lastReverted,
        "RESERVED VIOLATION: withdraw() succeeded when availableBalance was 0";
}
```

### Step 5C: Follow the CVL Guidelines (From AIComposer's Actual Rules)

AIComposer's CVL Judge evaluates specs against 23 specific guidelines
(from `composer/templates/cvl_guidelines.j2`). Your manual CVL should
follow the most critical ones:

| # | Rule | Why It Matters |
|---|------|---------------|
| 1 | Every rule must end with `assert` or `satisfy` — not a conditional containing them | The prover needs a clear final statement |
| 3 | Use `mathint` for all numeric variables by default | Avoids unnecessary type casting |
| 4 | Narrow `mathint` to `uintK`/`intK` only when passing to contract functions or storing results | Use `require_uintK` or `assert_uintK` |
| 5 | Every contract function gets an implicit `env` parameter unless `envfree` | Missing `env` is the most common CVL error |
| 9 | Values are immutable in CVL — no mutation after declaration | Use `require` to constrain, not assignment |
| 13 | Quantifier bodies must NOT contain contract calls or `require_*`/`assert_*` casts | Separate computation from quantification |
| 18 | `preserved` blocks add preconditions ON TOP of the invariant — don't manually `require` the invariant | Avoids redundant assumptions |
| 23 | Use direct storage access instead of mirroring state in ghosts via hooks | Ghosts are only for state that differs from storage |

**The rough draft protocol**: AIComposer's prompts require the LLM to
write a rough draft, read it back, and review it BEFORE submitting the
final result. Copy this pattern: write your CVL, review it against the
checklist above, fix issues, then finalize.

### Step 5D: Review the CVL

Use the CVL Review Checklist from REFERENCE.md:

```
RULE: reservedBalanceProtectedOnWithdraw
[ ] Rule name matches property title
[ ] Parameters correctly initialized
[ ] Preconditions realistic (not overly constraining)
[ ] Assertions directly test the property
[ ] No pointless bounds (uint256 >= 0)
[ ] No NONDET on critical behavior
[ ] No unsummarized external calls

Status: ___________
```

---

## Phase 6: Prover Configuration & Harness

**Goal**: Generate the `.conf` file and harness contract needed to run the Certora Prover on your CVL.

**Tool**: GitHub Copilot Pro Chat (Prompt D from MASTER_AUDIT_GUIDE_V2 Part 6)

---

### Step 6A: Generate the .conf File

```
Generate a Certora prover configuration (.conf file) for [ContractName].

Contract to verify:
[PASTE YOUR CONTRACT]

External contracts it uses:
[LIST FROM PHASE 1B]

Key properties to verify:
[LIST YOUR PROPERTIES FROM PHASE 4]

The .conf file must:
- Set files to the contract source path
- Set solc to "solc8.29"
- Set verify to "src/Contract.sol:ContractName"
- Include method summaries for every external contract
  (use DISPATCHER(true) for ERC20s, CONSTANT for oracles,
   NONDET for unpredictable externals)
- Set prover_args timeout to 600 seconds
- Reference the CVL spec file

Use JSON format.
```

### Step 6B: Generate the Harness Contract

```
Generate a CVL harness contract for testing [ContractName].

The main contract:
[PASTE YOUR CONTRACT]

The harness must:
1. Inherit from the main contract
2. Add ghost variables to track:
   - Total deposits across all users
   - Total withdrawals across all users
   - Whether reserved balance was ever violated
3. Override critical functions to update ghost state:
   - deposit(): increment totalDeposits ghost
   - withdraw(): increment totalWithdrawals ghost, check reservedBalance
4. Expose helper functions for property testing:
   - getTotalDeposits() → returns ghost tracking variable
   - getTotalWithdrawals() → returns ghost tracking variable
   - isReservedIntact() → returns whether reservedBalance was violated

Use Solidity version 0.8.20.
Ghost variables should use the `ghost` keyword for Certora compatibility.
```

**What you get**: A ready-to-use `.conf` file and harness contract.

### Step 6C: Run the Prover Directly (Works TODAY)

You don't need AutoSetup to run the Certora Prover. With your manually-written
CVL and `.conf` file, run the prover directly:

```bash
# Run Certora Prover with your manual CVL and config
certoraRun certora/confs/MyContract.conf \
  --cloud \
  --msg "Audit — reserved balance protection"
```

This is the complete command. `certoraRun` reads your `.conf` file which
points to your `.spec` file. `--cloud` sends the job to Certora's servers.
Results come back with VERIFIED / FAILED / TIMEOUT per rule.

```bash
# Check results
cat certoraRun.log | grep -E "VERIFIED|FAILED|TIMEOUT"

# Analyze counterexamples for FAILED rules
cex-analyzer \
  certora/output/report_folder \
  ruleName \
  --method ContractName.functionName
```

When AutoSetup becomes available, AIComposer's Phase 5 runs the prover
automatically as part of the pipeline. Until then, `certoraRun --cloud`
gives you the same verification power with your manual CVL.

### Step 6D: Analyze Counter-Examples (PhD-Level)

When a rule FAILS, use this prompt with Claude. Based on AIComposer's actual
`cex_instructions.j2` template — the same prompt the system uses internally.

```
A Certora Prover run produced a FAILED result. Analyze the counterexample.

RULE THAT FAILED:
[Paste the exact CVL rule]

COUNTEREXAMPLE DATA:
[Paste prover output or cex-analyzer output]

CONTRACT CODE (for context):
[Paste the relevant contract]

ANALYSIS INSTRUCTIONS:

1. ROOT CAUSE:
   Summarize the root cause of the failure in the implementation.
   Keep the original specification for this rule in mind.
   If you are uncertain, state this explicitly.
   If you have multiple theories, list each theory.

2. CODE CHANGES NEEDED:
   Identify the specific code changes required to fix the defect.
   Reference exact function names and line-level changes.

3. RESPONSE FORMAT:
   - Respond in natural language ONLY. Do NOT use tools or code blocks.
   - Phrase in SECOND PERSON: "You learned that..." "You must change..."
   - This response will be fed back to the developer agent for action.

CRITICAL RULES:
- If the failure is due to ghosts being HAVOCed by an unresolved external
  call, NEVER suggest making the ghosts persistent. Suggest alternative
  fixes: better method summaries, linking changes, dispatch resolution,
  or adding the unresolved contract to the verification scope.
- Do not suggest weakening the specification to make it pass.
- If the counterexample is a false positive (prover over-approximation),
  explain why and suggest how to constrain the rule's preconditions.
```

## Phase 7: PoC Test Writing

**Goal**: Write a Foundry test that proves the vulnerability is exploitable.
This is essential for bug bounty submissions — a runnable PoC is what gets
your finding validated.

**Tool**: GitHub Copilot Pro Chat

---

### The Prompt

```
I found a business logic vulnerability in [ContractName].

DOCUMENTED CLAIM: [exact quote from Phase 2]
PROPERTY VIOLATED: [property statement from Phase 4]
HOW IT'S VIOLATED: [exploit sequence from Phase 3]
VULNERABILITY: withdraw() calculates the withdrawal amount from
              the total contract balance instead of availableBalance
              (total balance minus reserved balance), allowing the
              owner to drain reserved funds.

Write a complete Foundry test (PoC) that demonstrates this.

Requirements:
- Use vm.startPrank() for actor impersonation
- Record state with console.log() BEFORE and AFTER the exploit
- Use descriptive variable names (reservedBefore, not r1)
- The test should PASS to prove the bug exists (the assertion should
  show the invariant IS violated, meaning the test succeeds)
- Include a comment on each step explaining what the attacker does
- Include a comment explaining what SHOULD have happened instead
- Include the one-line fix in a comment at the end

Use Solidity 0.8.20 and forge-std/Test.sol.
```

**What you get**: A complete, runnable Foundry test that demonstrates the exploit. When a judge runs `forge test --match-test testReservedBalanceDrain -vvv`, they see the vulnerability in action with before/after state logging.

---

## Phase 8: When AutoSetup Becomes Available

When Certora grants AutoSetup access, everything you've built becomes dramatically
more powerful. Here's what changes:

```
BEFORE AUTOSETUP (Phase 1-7 above):
  You → Claude → design_doc.md + manual CVL + manual .conf + PoC
  ✓ Already finds real business logic bugs
  ✓ Already produces submission-ready reports

AFTER AUTOSETUP (adds Phase 8):
  Your design_doc.md  → AIComposer Phase 0: System Analysis
                       → AIComposer Phase 1-2: AutoSetup (compilation, harness)
                       → AIComposer Phase 3: Structural Invariants
                       → AIComposer Phase 4: Auto Property Extraction
                       → AIComposer Phase 5: Auto CVL Gen + Prover
                       → Counterexamples found automatically

WHAT THIS MEANS:
- AIComposer reads your design_doc.md and automatically extracts
  additional properties you may have missed
- The CVL Judge reviews all specs for quality (tautologies, vacuity)
- The Prover runs ALL rules and finds counterexamples automatically
- cex-analyzer explains every violation in plain English
- Your manual CVL (Phase 5) can be merged with AIComposer's auto CVL
  for maximum coverage
```

### Setup (One Time)

```bash
git clone https://github.com/Certora/AutoSetup.git ~/AutoSetup
echo 'export AUTOSETUP_PATH=~/AutoSetup' >> ~/.bashrc
source ~/.bashrc
```

### Run the Pipeline

```bash
tui-autoprove \
  ~/projects/protocol \
  src/Contract.sol:ContractName \
  design_doc.md \
  --cloud \
  --max-concurrent 4 \
  --cache-ns protocol-audit-v1
```

---

## Complete Example: Walkthrough

Here's what a full manual audit produces, condensed to show the flow:

### Input Contract (Simplified)

```solidity
contract Vault {
    uint256 public reservedBalance;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;

    function lock(uint256 amount) external onlyOwner {
        require(amount <= address(this).balance - reservedBalance);
        reservedBalance += amount;
    }

    function withdraw(uint256 shares) external onlyOwner {
        // BUG: uses total balance, not available balance
        uint256 amount = (shares * address(this).balance) / totalSupply;
        balanceOf[msg.sender] -= shares;
        totalSupply -= shares;
        payable(msg.sender).transfer(amount);
    }
}
```

### Phase 2 Output: Claims Extracted

```
CLAIM: "lock() sets aside funds that cannot be withdrawn"
SOURCE: Function name 'lock', variable name 'reservedBalance'
TYPE: Fund protection
MEANING: Funds moved to reservedBalance via lock() should not be
         withdrawable through the normal withdraw() path.

CLAIM: "Only owner can withdraw"
SOURCE: onlyOwner modifier on withdraw()
TYPE: Access control
MEANING: No non-owner address can call withdraw() successfully.
```

### Phase 3 Output: Claude Finds the Bug

```
CATEGORY 3: RESERVED FUNDS ARE UNTOUCHABLE
VULNERABILITY FOUND: YES

EXACT FUNCTION: withdraw(uint256 shares)
EXPLOIT SEQUENCE:
  1. Owner locks 100 ETH via lock(100 ETH)
     → reservedBalance = 100
  2. Owner calls withdraw(100 shares)
     → amount = (100 * 100 ETH) / 100 = 100 ETH
     → Transfers entire balance including reserved funds
  3. reservedBalance = 100 BUT contract.balance = 0
     → Accounting broken, reserved funds stolen

FINANCIAL IMPACT: Full reserved amount. CRITICAL.
AUDITOR MINDSET CHECK: STEAL.
```

### Phase 4 Output: design_doc.md Section 4

```
### Reserved Fund Protection
PROPERTY: When withdraw() is called, the withdrawal amount must be
calculated from availableBalance (contract.balance minus reservedBalance),
not from contract.balance alone. withdraw() must never reduce reservedBalance.

IMPACT IF VIOLATED: Owner drains reserved funds → protocol operations fail.
CRITICAL. $50k+ bounty.
```

### Phase 5 Output: CVL Rule

```cvl
rule reservedBalanceProtectedOnWithdraw(env e, uint256 shares) {
    require shares > 0;
    require shares <= balanceOf(e.msg.sender);
    uint256 reservedBefore = reservedBalance();
    withdraw(e, shares)@withrevert();
    if (!lastReverted) {
        assert reservedBalance() == reservedBefore;
    }
}
```

### Phase 7 Output: Foundry PoC

```solidity
function testReservedBalanceDrain() public {
    vm.startPrank(owner);
    vault.lock(100 ether);
    uint256 reservedBefore = vault.reservedBalance();
    vault.withdraw(100 ether);
    assertLt(address(vault).balance, reservedBefore,
        "BUG: balance drained below reserved amount");
    vm.stopPrank();
}
```

---

## Quick Reference: All Claude Prompts

| Phase | What to Ask Claude |
|-------|-------------------|
| **1A: Storage** | "Analyze this contract: storage layout with read/write mapping, function analysis with reentrancy assessment, actor map" |
| **1B: External** | "Classify all external contracts: type, imported by, key functions, potential issues (transfer fees, callbacks, rebasing), CVL summary suggestion" |
| **2: Validate** | "I extracted these claims. Did I miss any? Are my interpretations accurate? Are there implicit claims?" |
| **3: Business logic** | [Condensed 10-category prompt pasted above in Phase 3; canonical 12-category Prompt A is in `MASTER_AUDIT_GUIDE_V2.md` Part 6] |
| **4: Review** | "Review this design_doc.md for specificity, completeness, traceability, code presence, edge cases, impact clarity" |
| **5: CVL** | "Write a Certora CVL rule/invariant for this property: [paste property]. Contract: [paste code]" |
| **6A: Conf** | "Generate a .conf file for [ContractName] with method summaries, solc version, verify directive, 600s timeout" |
| **6B: Harness** | "Generate a CVL harness with ghost variables for totalDeposits, totalWithdrawals, reservedBalance tracking" |
| **7: PoC** | "Write a Foundry test demonstrating this vulnerability: [claim + property + exploit sequence]" |

---

## Do's and Don'ts

### ✅ Do

- Run Phase 3 (10-category analysis) on EVERY contract you audit
- Tie every property to a documented claim from the source code
- Write design_doc.md in plain English — no CVL, no Solidity, no pseudocode
- Include IMPACT annotations (STEAL/PRINT/FREEZE/BRICK) for every property
- Write a Foundry PoC for every finding before submitting
- Use the three-layer evidence approach: formal proof + documented claim + runnable PoC

### ❌ Don't

- Skip Phase 1 (contract analysis) — you cannot audit what you don't understand
- Write vague properties like "the protocol should be secure"
- Put CVL code in your design_doc.md (AIComposer generates CVL from English)
- Submit findings without a PoC test (judges will invalidate)
- Assume the owner is honest unless you explicitly state your trust model
- Wait for AutoSetup to start auditing — Phases 1-7 find real bugs TODAY

---

## Expected Results

| Metric | Target |
|--------|--------|
| Time per protocol | 3-4 hours |
| Claims extracted | 5-10 per contract |
| Business logic categories with findings | 2-5 per contract |
| CVL rules written | 1-2 per finding |
| PoC tests | 1 per finding |
| Valid findings per protocol | 2-5 |
| Value per finding | $5k-$50k+ |
