# Certora AIComposer — Reference

> All commands, CLI flags, templates, error codes, and prompts in one place.

---

## All CLI Commands

### Installation Check

```bash
python3 --version                        # Need 3.12+
echo $ANTHROPIC_API_KEY                 # Must be set
echo $CERTORAKEY                         # For cloud mode
docker ps | grep postgres               # DB should be running
which solc8.29                          # Solidity compiler (format: solcX.Y)
```

### Auto-Prove (Code → Specifications)

```bash
# Quick bug hunt (fastest path) — installed entry point
console-autoprove . src/Contract.sol:ContractName design_doc.md --cloud

# Or via Python module
python -m composer.cli.console_autoprove . src/Contract.sol:ContractName design_doc.md --cloud

# With threat model (focus attack surface)
console-autoprove . src/Contract.sol:ContractName design_doc.md \
  --threat-model threat.md --cloud

# Interactive refinement (realtime feedback)
console-autoprove . src/Contract.sol:ContractName design_doc.md \
  --interactive --cloud

# Parallel components (extract for all at once)
console-autoprove . src/Contract.sol:ContractName design_doc.md \
  --max-concurrent 8 --cloud

# Cached iteration (update doc, re-run fast)
console-autoprove . src/Contract.sol:ContractName design_doc.md \
  --cache-ns my-run --cloud

# Multi-pass strategy (different threat angles)
console-autoprove . src/Contract.sol:ContractName design_doc.md \
  --threat-model threat_oracle.md --cache-ns oracle --cloud
console-autoprove . src/Contract.sol:ContractName design_doc.md \
  --threat-model threat_access.md --cache-ns access --cloud

# Design-doc-only (no source available)
console-autoprove ~/dummy-project src/Target.sol:Target whitepaper.pdf --cloud

# Local prover (no cloud)
console-autoprove . src/Contract.sol:ContractName design_doc.md

# TUI mode (live dashboard)
tui-autoprove . src/Contract.sol:ContractName design_doc.md --cloud
```

### Counter-Example Analysis

```bash
# Installed entry point (replaces root-level cex_analyzer.py)
cex-analyzer certora/output/report_folder ruleName --method ContractName.functionName

# Or via Python module
python -m analyzer certora/output/report_folder ruleName --method ContractName.functionName
```

### Resumption & Meta-Iteration

```bash
# Materialize output from a prior run (via Python module)
python -m composer.cli.ap_trail materialize thread-id path/

# Visualize a completed session
python scripts/traceDump.py thread-id \
  postgresql://audit_db_user:audit_db_password@localhost:5432/audit_db \
  output.html
```

### Inspection & Debugging

```bash
# View extracted properties
cat certora/properties/autospec_*.properties.json | jq .

# View generated CVL
cat certora/specs/autospec_*.spec

# Find failing specs
for spec in certora/specs/autospec_*.spec; do
  echo "=== $(basename $spec) ==="
  grep -c "FAILED" "$spec" || echo "0 failures"
done

# Inspect cache
python scripts/autoprove_cache_explorer.py . src/C.sol:C doc.md --cache-ns my-run

# View agent conversation history
python scripts/snapshot_viewer.py <agent_name>

# Merge all properties into one file
jq -s 'add' certora/properties/autospec_*.properties.json > all_properties.json

# Analyze property distribution
cat all_properties.json | jq '[.[] | .sort] | group_by(.) | map({sort: .[0], count: length})'
```

---

## All CLI Flags

### Auto-Prove Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--cloud` | off | Use Certora cloud prover (no local setup needed) |
| `--max-concurrent` | 4 | Maximum parallel agents for property extraction and CVL generation |
| `--cache-ns` | None | Cache namespace for cross-run caching |
| `--memory-ns` | None | Memory namespace (defaults to thread ID) |
| `--heavy-model` | `claude-opus-4-6` | Anthropic model for complex tasks. Auto-Prove has no single `--model` flag — it uses this and `--lite-model` instead. |
| `--lite-model` | `claude-sonnet-4-6` | Anthropic model for simpler tasks |
| `--tokens` | 128000 | Token budget for code generation |
| `--thinking-tokens` | 2048 | Thinking token budget |
| `--rag-db` | `postgresql://rag_user:rag_password@localhost:5432/rag_db` | RAG database connection string |
| `--interactive` | off | Enable realtime property refinement |
| `--threat-model` | None | Path to threat model document for focused extraction |
| `--recursion-limit` | 1000 | Max graph iterations |
| `--max-bug-rounds` | 3 | Max rounds of property extraction per component |

### Codegen Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--prover-capture-output` | true | Capture prover output instead of printing to stdout |
| `--prover-keep-folders` | false | Keep temporary prover directories |
| `--debug` | false | Enable verbose debug output |
| `--debug-prompt-override` | None | Append text to the initial prompt |
| `--tokens` | 128000 | Token budget for code generation |
| `--thinking-tokens` | 2048 | Thinking token budget |
| `--model` | `claude-opus-4-6` | Anthropic model name |
| `--thread-id` | auto | Thread ID for resumption |
| `--checkpoint-id` | None | Checkpoint ID for resumption |
| `--summarization-threshold` | auto | Token threshold for history summarization |
| `--debug-fs` | false | Dump virtual filesystem state |

---

## Design Document Template

```markdown
# [Protocol Name] Design Document

## 1. System Overview
[2-3 sentences: What is this protocol? What does it do?]

## 2. Components
### Component: [Name]
- Purpose: [What does it do?]
- Functions: [list key functions]
- State: [list state variables]
- External Dependencies: [oracle, tokens, etc.]
- Risk Areas: [What could go wrong?]

[Repeat for each component]

## 3. Stated Guarantees (with Impact)

For each guarantee, state what happens if it breaks:

"LP tokens are always redeemable for proportional share"
  → IMPACT IF VIOLATED: Share dilution → depositor funds stolen (CRITICAL)

"Only owner can pause the protocol"
  → IMPACT IF VIOLATED: Unauthorized pause → funds frozen (HIGH)

## 4. Properties (Claim → Property Format)

### Asset Conservation
CLAIM: Assets cannot be created from nothing.
PROPERTY: Total assets accounted for must never exceed actual assets held.
PROPERTY: A user cannot increase their claim without providing equivalent value.

### Solvency
CLAIM: Protocol remains fully collateralized.
PROPERTY: Total liabilities must never exceed total assets.

### Share Price Integrity
CLAIM: Share value cannot be artificially inflated.
PROPERTY: External transfers must not artificially inflate share value.

[Continue for all 12 categories — see USER_GUIDE.md]

## 5. Edge Cases & State Transitions
- Boundary conditions: What if reserve is 0? Deposit is 1 wei?
- State transitions: Pause→Resume safety, Emergency→Normal consistency
- Reentrancy points: Where are external calls?
- Oracle manipulation: Staleness, zero price, extreme values

## 6. Catastrophic Failure Scenarios

### CF-1: Theft of Reserved Funds
Impact: Loss of beacon-chain deposit funds. CRITICAL.
Must Never Happen: Owner withdrawal uses staged ETH.
Affected: withdraw(), stage(), depositToBeaconChain()

### CF-2: Insolvency
Impact: User claims exceed protocol assets. CRITICAL.
Must Never Happen: Total liabilities > total assets.
Affected: borrow(), deposit(), withdraw(), liquidate()

### CF-3: Share Inflation
Impact: Attacker obtains disproportionate ownership. CRITICAL.
Must Never Happen: Shares minted exceed value deposited.
Affected: deposit(), mint(), convertToShares()

### CF-4: Permanent Lockup
Impact: Funds become unrecoverable. HIGH.
Must Never Happen: All withdrawal paths become inaccessible.
Affected: withdraw(), emergencyWithdraw(), pause()

### CF-5: Unauthorized Control
Impact: Attacker gains privileged authority. CRITICAL.
Must Never Happen: Non-owner performs owner-only action.
Affected: pause(), setFees(), upgradeTo()

## 7. Assumptions & Limitations

### 7.1 ERC20 / Token Assumptions

These assumptions apply to all external token contracts interacted with by this protocol. If any of these do not hold, findings related to token behavior may not apply.

```
ASSUMED TRUE for all tokens in scope:
- Transfer returns a boolean true on success (standard ERC20)
- transferFrom returns a boolean true on success
- No transfer fees or tax on send/receive
- No callback on transfer (not ERC777, not ERC1155)
- Not a rebasing token (balance does not change autonomously)
- Not pausable by an external party mid-transaction
- balanceOf() is accurate and not manipulable within a single transaction
- No malicious return values or silent failures

NOT ASSUMED:
- Token cannot be upgraded (upgradeable tokens are possible)
- Token cannot be paused by its own governance (possible)
- Token total supply is fixed (may be mintable)
```

If any token in the protocol violates these assumptions, note it explicitly:

```
EXCEPTION: [token name] is known to [e.g., have transfer fees].
This affects [which functions] and is treated as follows: [handling].
```

### 7.2 External Contract Assumptions

For every external contract this protocol calls, state what you assume about its behavior.

```
ORACLE (e.g., Chainlink):
- Returns a price within a reasonable range (> 0, < type(uint128).max)
- Is not permanently stale (updated within the last [X] hours)
- Is not malicious or manipulated by the oracle operator
NOT ASSUMED: Oracle cannot be flash-loan manipulated within a block

DEPOSIT CONTRACT (e.g., Beacon Chain):
- Accepts valid deposits and reverts on invalid ones
- Does not re-enter this contract during deposit execution
NOT ASSUMED: Beacon chain behavior after deposit is in scope

EXTERNAL PROTOCOL (e.g., VaultHub, Lido):
- Calls into this contract only through documented interfaces
- Does not drain this contract's balance through callbacks
```

### 7.3 Attacker Model

This defines what an attacker CAN and CANNOT do. AIComposer uses this to scope property verification. Judges use this to assess whether a finding is exploitable.

```
ATTACKER CAN:
✓ Call any public or external function with any valid parameters
✓ Deploy new contracts and use them as intermediaries
✓ Send ETH directly to the contract (if it has receive() or fallback())
✓ Front-run or sandwich any transaction
✓ Call functions in any order across multiple blocks
✓ Use flash loans (borrow large amounts within one transaction)
✓ Observe all on-chain state before deciding what to call

ATTACKER CANNOT:
✗ Modify deployed contract bytecode
✗ Break EVM rules (cannot modify storage of contracts they don't own)
✗ Break cryptographic primitives (hash collisions, signature forgery)
✗ Manipulate consensus (no reorgs, no block timestamp beyond ±15 sec)
✗ Intercept or modify transactions mid-execution
✗ Read private state before deciding to call (no telepathy)

ADMIN / OWNER TRUST MODEL:
[CHOOSE ONE AND STATE IT EXPLICITLY]

Option A — Admin is fully trusted:
  The owner and admin roles are assumed to be honest.
  We do not verify properties that require admin to be malicious.
  Findings related to admin rug-pull are out of scope.

Option B — Admin is partially trusted:
  The owner is trusted not to steal, but may make mistakes.
  We verify that even a mistaken admin cannot break core invariants.
  Deliberate malicious admin actions are out of scope.

Option C — Admin is untrusted (adversarial):
  We assume admin may be compromised or malicious.
  We verify that core user protections hold even if admin acts maliciously.
  This is the most conservative and highest-value trust model.
```

### 7.4 Blockchain / Environment Assumptions

```
ASSUMED TRUE:
- block.timestamp is within ±15 seconds of real time
- block.number increases monotonically
- msg.sender is accurate (no metatransactions unless protocol supports them)
- ETH transfers via .call() succeed unless the recipient reverts
- Solidity version is [X.X.X] — no compiler bugs assumed

NOT IN SCOPE:
- Chain reorg attacks
- Miner/validator extractable value (MEV) beyond front-running
- Layer 1 consensus failures
- Gas griefing (unless protocol has gas-sensitive logic)
- EIP changes that alter EVM semantics after deployment
```

### 7.5 Formal Verification Scope Limitations

These are things Certora Prover / CVL structurally cannot verify. Stating them prevents judges from asking "why didn't you check X?"

```
CVL CANNOT VERIFY:
- Off-chain events or actions (API calls, keeper behavior, etc.)
- Hash collisions or preimage attacks
- Economic incentives (whether an attack is profitable after gas costs)
- Behavior of contracts deployed AFTER the audit
- Cross-chain interactions or bridge logic on the other chain
- Probabilistic security properties
- Gas exhaustion attacks unless modeled explicitly
- Social engineering or key compromise

CVL CAN VERIFY (what this audit covers):
- All on-chain state transitions reachable through the contract's functions
- Mathematical invariants over all possible inputs
- Access control for all functions
- Pre/postconditions for all functions listed in Section 5
- Interaction between functions in the same transaction
```

### 7.6 Protocol-Specific Assumptions

These are assumptions unique to THIS protocol that affect verification scope. Fill these in per audit.

```
EXAMPLE FOR STAKINGVAULT:

ASSUMED:
- The DEPOSIT_CONTRACT (Ethereum Beacon Chain) behaves as specified
  in EIP-2982. Its internal behavior is not verified here.
- The VaultHub contract that manages locked balances is correct and
  non-malicious. Its behavior is summarized, not verified.
- The initial owner is set correctly at deployment time.
- ossify() is only called intentionally — we verify its effects
  but not the decision to call it.
- ETH sent to the contract via receive() is always legitimate funding,
  not an attack vector (no ETH refusal needed).

NOT ASSUMED:
- Validators will actually use the withdrawal credentials correctly
  (beacon chain behavior post-deposit is out of scope)
- The node operator will perform duties (off-chain behavior)
```

### 7.7 Out of Scope

Explicitly list what you are NOT auditing or verifying. This is critical — it prevents judges from marking your submission as incomplete.

```
OUT OF SCOPE FOR THIS AUDIT:

Contract level:
- [Any contracts not listed in Section 2]
- Deployment scripts and migration contracts
- Test contracts and mocks

Behavior level:
- Off-chain keeper or bot behavior
- Validator performance after beacon deposit
- Governance decisions (what parameters are set, not whether
  the mechanism enforces them)

Attack types:
- Social engineering / key compromise
- Miner/validator collusion at consensus level
- Attacks requiring > [X] ETH of capital to execute (state if relevant)

Known issues:
- [List any known issues already acknowledged by the protocol team]
- [Any issues disclosed in prior audit reports]
```

### 7.8 What This Means for Findings

Use this framing when you submit findings to make the scope explicit:

```
FINDING IS IN SCOPE IF:
✓ It violates a property in Section 4 or 6
✓ It is executable by an attacker within the attacker model (Section 7.3)
✓ It does not rely on assumptions we have already excluded (Sections 7.1–7.7)
✓ It affects contracts listed in Section 2

FINDING IS OUT OF SCOPE IF:
✗ It requires admin to be malicious (if Option A or B chosen in 7.3)
✗ It requires breaking cryptography
✗ It only affects contracts not listed in Section 2
✗ It was already disclosed in a prior audit report
```

**Why each subsection matters:**

| Subsection | Why It's Needed |
|-----------|-----------------|
| 7.1 Token Assumptions | Tells Certora how to summarize token calls (DISPATCHER vs NONDET) |
| 7.2 External Contracts | Prevents false positives from unmodeled external behavior |
| 7.3 Attacker Model | Judges use this to assess exploitability of findings |
| 7.4 Environment | Scopes out consensus-level attacks that are always out of scope |
| 7.5 FV Limitations | Prevents judges asking "why didn't Certora check X?" |
| 7.6 Protocol-Specific | Documents protocol knowledge that affects what you can verify |
| 7.7 Out of Scope | Stops judges from marking submission incomplete |
| 7.8 Finding Criteria | Makes it clear which of your findings are actually submittable |
```

**Size**: 2000-5000 words minimum.

---

## Threat Model Template

```markdown
# Threat Model: [Attack Surface]

## Attacker Profile
- Has wallet with tokens
- Can call any public function
- Cannot modify smart contracts
- Objectives: [profit, drain, break invariants]

## Attack Surface: [Area]

### Vector 1: [Specific Attack]
- What if [condition]?
- What if [extreme value]?
- What if [missing check]?

### Vector 2: [Specific Attack]
- ...

## Expected Findings
- We expect: [vulnerability type]
- We DON'T expect: [out of scope]

## Properties to Extract
1. "[Property description]"
2. "[Property description]"
```

---

## Claude Prompt Templates (Manual AutoSetup Replacement)

### Template 1: Full Contract Analysis (Impact-Focused)

```
Analyze this [DeFi/Governance/Token] contract for ALL 10 business logic categories.

Don't just look for bugs — look for IMPACT.
For each category, ask: "Can someone steal, print, freeze, or brick the protocol?"

CONTRACT CODE:
[paste code]

ANALYZE THESE 10 CATEGORIES:
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

FOR EACH CATEGORY:
- Is there a vulnerability? YES/NO
- If YES: Where in code? What function?
- Specific exploit steps
- Financial impact ($ estimate)
- How to fix

ALSO PROVIDE:
- Storage layout with byte offsets
- Function analysis (visibility, mutability, external calls, state mutations)
- External dependency classification
```

### Template 2: Storage & Compiler Analysis

```
Analyze this Solidity contract for:
1. Storage layout (all state variables with byte offsets)
2. Function visibility and mutability
3. All external/internal calls
4. State mutations per function
5. Potential reentrancy points

[Paste contract code]

Format as:
## Storage Layout
- variable_name: type (slot X, size Y bytes)

## Function Analysis
- function_name(params) → visibility, mutability, external_calls, state_mutations

## Reentrancy Points
- [location, impact, mitigation]
```

### Template 3: External Contract Classification

```
Classify all external contracts in this project:

[Paste all contract files in project]

For each external/imported contract:
1. Type (ERC20, interface, abstract, concrete)
2. Key functions used
3. Standard compliance (if applicable)
4. Potential issues
5. Suggested CVL summary

Format:
## External Contract: [Name]
- Type: ERC20 / Interface / Abstract / Concrete
- Import: [import statement]
- Used By: [which contracts]
- Key Functions: [critical functions]
- Issues: [known vulnerabilities, non-standard behavior]
- CVL Summary Template: [suggested CVL code]
```

### Template 4: Harness & Config Generation

```
Generate a Certora prover configuration (.conf file) for [ContractName].

Contract: [paste contract]
External Contracts: [list them]
Key Properties: [list what you're testing]
Compilation Target: Solidity 0.8.29

Generate a complete .conf file with:
1. Compilation rules
2. Method summaries for externals
3. Verification rules
4. Resource limits
5. Rule filters if needed
```

### Template 5: DeFi Vault Focus

```
Find business logic bugs in this vault/LST.

[paste code]

Critical vault properties:
1. Share price cannot be inflated (donation attack?)
2. Cannot deposit and get disproportionate shares
3. First depositor cannot manipulate future prices
4. Vault balance always ≥ share accounting
5. Redemptions always possible
6. Exchange rate stays consistent

Show specific attack if any exist.
```

### Template 6: Governance Focus

```
Find business logic bugs in this governance system.

[paste code]

Critical governance properties:
1. Voting power cannot be changed after voting
2. Voting power snapshot at proposal creation
3. Quorum requirement enforced
4. Voting weight calculated fairly
5. Proposal execution only after voting ends
6. Parameters cannot be changed mid-vote

Show governance takeover attacks if possible.
```

### Template 7: Counter-Example Analysis (PhD-Level)

*Use this prompt when the Certora Prover returns FAILED on a rule. Based on
the actual AIComposer `cex_instructions.j2` template. The instructions below
are critical — the LLM must not suggest making ghosts persistent or use tools
in its response.*

```
A Certora Prover run produced a FAILED result for the following rule.
Analyze the counterexample and produce a structured diagnosis.

RULE THAT FAILED:
[Paste the exact CVL rule that failed]

COUNTEREXAMPLE DATA:
[Paste the prover output or cex-analyzer output for this rule]

CONTRACT CODE (for context):
[Paste the relevant contract code]

ANALYSIS INSTRUCTIONS:

1. ROOT CAUSE:
   Summarize the root cause of the failure in the implementation.
   Keep the original specification for this rule in mind.
   If you are uncertain about the exact cause, state this explicitly.
   If you have multiple theories, list each theory.

2. CODE CHANGES NEEDED:
   Based on your analysis, identify the specific code changes
   required to fix the underlying defect identified by the prover.
   Reference exact function names and line-level changes.

3. RESPONSE FORMAT:
   - Respond in natural language ONLY. Do NOT use any tools or code blocks.
   - Phrase your analysis in the SECOND PERSON:
     "You learned that the original implementation had..."
     "You must make the following changes..."
   - This response will be fed back to the developer agent for action.

CRITICAL RULES:
- If the failure is due to ghosts being HAVOCed by an unresolved external
  call, NEVER suggest making the ghosts persistent. Instead, suggest
  alternative fixes: better method summaries, linking changes, dispatch
  resolution, or adding the unresolved contract to the verification scope.
- Do not suggest weakening the specification to make it pass.
- If the counterexample is a false positive (prover over-approximation),
  explain why and suggest how to constrain the rule's preconditions.
```

---

## CVL Guidelines Quick Reference

These are the 8 most critical rules from AIComposer's actual 23 CVL guidelines
(`composer/templates/cvl_guidelines.j2`). The CVL Judge enforces all 23.

| # | Rule | Example |
|---|------|---------|
| 1 | Every rule must end with `assert` or `satisfy` — not a conditional containing them | `assert x > 0;` NOT `if (cond) assert x > 0;` |
| 3 | Use `mathint` for all numeric variables by default | `mathint total;` not `uint256 total;` |
| 4 | Narrow `mathint` to `uintK`/`intK` only when passing to contracts or storing results | `uint256 amount = require_uint256(total);` |
| 5 | Every contract function implicitly gets an `env` parameter unless `envfree` | `rule foo(env e, uint256 x)` |
| 9 | Values are immutable — use `require` to constrain, not assignment | `require x == oldX + 1;` not `x = x + 1;` |
| 13 | Quantifier bodies must NOT contain contract calls | No `forall address a. foo(a)` where foo() calls a contract |
| 18 | `preserved` blocks add preconditions ON TOP of the invariant | Don't manually `require` the invariant in preserved blocks |
| 23 | Use direct storage access instead of mirroring state in ghosts via hooks | `balanceOf(user)` not `ghostBalance[user]` |

### Full CVL Guidelines (All 23)

Full guidelines live in `composer/templates/cvl_guidelines.j2`. Key additions beyond the 8 above:

- Rule 6b: Expression summaries on `_.method(params)` MUST include an expect clause
- Rule 6c: The operand of ALWAYS must be a constant
- Rule 6g: A method block entry without a contract identifier is implicitly the contract under verification
- Rule 10: Bitwise operations can be over-approximated — avoid if possible
- Rule 12: `method` variables must be declared as rule parameters, not in the rule body
- Rule 14: All CVL functions must end in a `return` statement
- Rule 15: Void functions should omit the returns clause — do NOT use `returns void`
- Rule 16: The return value of `@withrevert` calls is undefined if the function reverted
- Rule 17: Method block entries without `envfree` or summary serve no purpose
- Rule 19: CVL auto-promotes comparison operands to `mathint` — explicit `to_mathint` rarely needed
- Rule 20: `persistent` ghosts should rarely be used — never for state that mirrors contract storage
- Rule 21: Use `definition` for meaningful numerical constants
- Rule 22: Use invariant parameters instead of `forall` inside invariants

---

## CVL Review Checklist

When reviewing generated CVL, scan for:

```
RULE: [rule_name]
[ ] Rule name matches property title
[ ] All parameters correctly initialized
[ ] Preconditions realistic (not overly constraining)
[ ] Assertions directly test the property
[ ] No pointless bounds (uint256 >= 0)
[ ] No NONDET on critical behavior
[ ] No unsummarized external calls
[ ] Rule ends with assert or satisfy (Guidelines Rule 1)
[ ] Uses mathint by default, narrowed only when needed (Rules 3-4)
[ ] All contract functions receive env parameter or are envfree (Rule 5)
[ ] No mutation after declaration — use require (Rule 9)

Status: ✓ PASSES / ✗ FAILS / ⏱ TIMEOUT / ⊘ SKIPPED
```

### Red Flags

| Flag | Example | Fix | CVL Rule |
|------|---------|-----|----------|
| Tautology | `assert x == x` | Remove | — |
| Pointless bound | `require uint256 >= 0` | Remove | — |
| NONDET on critical | `transfer(...) NONDET` | Use DISPATCHER(true) | — |
| Missing precondition | No `require amount > 0` before `assert after > before` | Add precondition | — |
| Overconstrained | `require msg.sender == owner` on public function | Relax | — |
| SKIPPED property | No justification, or weak justification | Challenge | — |
| Mutable variable | `x = x + 1` in rule body | Use `require x == oldX + 1` | Rule 9 |
| Missing env param | `rule foo(uint256 x)` without `env` | Add `env e` | Rule 5 |

---

## Prover Results Reference

| Result | Icon | Meaning | Action |
|--------|------|---------|--------|
| **VERIFIED** | ✅ | Property holds for all inputs | No bug for this property |
| **FAILED** | ❌ | Counterexample found | **INVESTIGATE** — likely real bug |
| **TIMEOUT** | ⏱ | Prover couldn't finish | Simplify property or increase resources |
| **SKIPPED** | ⊘ | Not formalizable | May indicate false negative |

---

## Property Types Reference

| Type | Description | CVL Keyword | Example |
|------|-------------|-------------|---------|
| **Invariant** | Always true, no matter what | `invariant` | "Sum of balances = total supply" |
| **Safety Property** | Expected behavior under conditions | `rule` | "Deposit increases balance" |
| **Attack Vector** | Potential exploit path | `rule` (adversarial) | "Oracle can't manipulate price >5%" |

---

## Severity & Bounty Framework

Standardized per Immunefi, Sherlock, and HackenProof:

| Severity | Bounty Range | Requirements |
|----------|-------------|-------------|
| **CRITICAL** | $10k-$50k+ | Direct theft, permanent freezing, protocol insolvency, unauthorized minting |
| **HIGH** | $5k-$20k | Temporary freezing >1 month, unclaimed yield theft, broken accounting |
| **MEDIUM** | $1k-$5k | DoS >1 week, state inconsistency, broken returns |
| **LOW** | <$1k | Gas issues, informational |

---

## Business Logic Categories Quick Reference

| # | Category | Severity | Key CVL Pattern |
|---|----------|----------|-----------------|
| 1 | Asset Conservation | CRITICAL | `assert total_claimed <= total_actual` |
| 2 | Double Withdrawal | CRITICAL | `withdraw(claim); withdraw(1)@withrevert(); assert lastReverted` |
| 3 | Reserved Funds | CRITICAL | `assert stagedBalance() == stagedBefore` |
| 4 | Solvency | CRITICAL | `assert totalLiabilities <= totalAssets` |
| 5 | Share Price | CRITICAL | `assert priceAfter >= priceBefore - 1` |
| 6 | Access Control | CRITICAL | `pause@withrevert(e); assert lastReverted` |
| 7 | State Machine | HIGH | `assert isValidTransition(before, after)` |
| 8 | Accounting Sync | CRITICAL | `assert vaultBalance >= accountedAssets` |
| 9 | Reward Integrity | HIGH | `assert earned2 == 0` |
| 10 | Fund Lockup | HIGH | `emergencyWithdraw@withrevert(); assert !lastReverted` |
| 11 | Upgrade & Proxy Safety | CRITICAL | `upgradeToAndCall@withrevert(e,_,_); assert lastReverted` (non-authority) |
| 12 | Cross-Chain / Bridge Trust | CRITICAL | `processMessage@withrevert(e, usedId, _); assert lastReverted` |

See `MASTER_AUDIT_GUIDE_V2.md` Part 4 for the full Claim/Property/CVL treatment of each category and its
audit-workflow integration.

---

## Recommended Security Primitives

Diagnosing a bug category is only half the job — this maps each category to the audited,
battle-tested primitive that prevents it, so findings translate directly into safe code rather
than a bespoke re-implementation (which is itself a common source of *new* vulnerabilities).

| Category | Recommended primitive |
|---|---|
| Access Control (#6) | OpenZeppelin `AccessControl` / `Ownable2Step` (prefer two-step ownership transfer over single-step `Ownable`) |
| Reentrancy (classic, not separately numbered above) | OpenZeppelin `ReentrancyGuard`; Solmate's `ReentrancyGuard` for a gas-optimized equivalent |
| Token transfers / non-standard ERC20s | OpenZeppelin `SafeERC20` (handles missing return values, reverts consistently) |
| Solvency / Accounting Sync (#4, #8) | OpenZeppelin `ERC4626` as a reference vault-accounting implementation rather than hand-rolled share math |
| Upgrade & Proxy Safety (#11) | OpenZeppelin `UUPSUpgradeable` / `TransparentUpgradeableProxy`, with `_disableInitializers()` called in the implementation constructor; OpenZeppelin Upgrades plugin (or equivalent storage-layout-diff tooling) run before every upgrade |
| Cross-Chain / Bridge Trust (#12) | Prefer canonical, audited bridge infrastructure (a rollup's native bridge, LayerZero, Axelar, Wormhole's guardian design) over a hand-rolled trusted-relayer bridge wherever the protocol's design allows it |
| Signature-based authorization | OpenZeppelin `ECDSA` + `EIP712` (domain-separated signatures with an explicit nonce, to prevent cross-contract and replay attacks) |
| Pausability / Fund Lockup (#10) | OpenZeppelin `Pausable`, paired with an emergency-withdrawal path that explicitly remains callable while paused (see Category 10's CVL rule) |

This list is deliberately not exhaustive — it's the set of primitives that most directly map to
the 12 categories above, not a general Solidity library survey.

---

## Testing & Tooling Complement

This tool performs SMT-backed formal verification via the Certora Prover — it proves properties
hold for *all* inputs, which is a different and complementary guarantee to what dynamic testing
tools provide. A production security process typically layers several of these together, since
they catch different bug classes and have different false-negative profiles:

| Tool class | Example tools | What it catches that formal verification (as configured here) may not |
|---|---|---|
| Static analysis | Slither, MythX | Fast, cheap first-pass pattern detection across an entire codebase before investing in property-writing |
| Fuzz testing | Foundry (`forge test --fuzz`), Echidna | Concrete counterexamples without needing a hand-written CVL rule first; good at finding edge cases in arithmetic and boundary conditions |
| Invariant testing | Foundry invariant tests, Echidna | Stateful, multi-call sequences explored via random exploration rather than exhaustive proof — a useful cross-check against the properties formalized here |
| Symbolic execution | Manticore, `hevm symbolic` | Path-exhaustive exploration of a single function's behavior, complementary to whole-protocol CVL invariants |

None of these substitute for the others. A property that verifies here has a mathematical
guarantee that fuzzing can't offer (fuzzing can only fail to find a counterexample, never prove
its absence); conversely, fuzzing and static analysis catch classes of implementation-level bugs
(e.g. straightforward off-by-one errors) faster and cheaper than writing a CVL rule for them.

---

## Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `AUTOSETUP_PATH not set` | AutoSetup not installed | Add `--cloud` flag |
| `ANTHROPIC_API_KEY not set` | Missing API key | `export ANTHROPIC_API_KEY=sk-...` |
| `CERTORAKEY not set` | Missing certora key | `export CERTORAKEY=...` (cloud mode) |
| `PostgreSQL connection failed` | Docker not running | `cd scripts && docker compose start` |
| `No solc compiler found` | Solidity compiler missing | Install with name format `solcX.Y` (e.g., `solc8.29`) |
| `CERTORA not set` | Prover path not configured | `export CERTORA=/path/to/CertoraProver/target` |
| `rag_db connection failed` | RAG DB not populated | Run `./scripts/gen_docs.sh && ./scripts/populate_rag.sh` |
| `No properties extracted` | Design doc too vague | Expand design doc to 2000+ words, add specific edge cases |
| `All rules TIMEOUT` | Properties too complex | Simplify properties, reduce `--max-concurrent` |
| `ImportError: graphcore` | graphcore not on path | Install with `uv sync --extra ml` |

---

## Database Reference

| Database | Purpose | Default Connection String |
|----------|---------|--------------------------|
| `rag_db` | CVL manual search (pgvector) | `postgresql://rag_user:rag_password@localhost:5432/rag_db` |
| `langgraph_store_db` | LangGraph document/index store | Managed by docker compose |
| `langgraph_checkpoint_db` | Workflow checkpoints | Managed by docker compose |
| `memory_tool_db` | LLM context memory | Managed by docker compose |
| `audit_db` | Execution history, prover results | `postgresql://audit_db_user:audit_db_password@localhost:5432/audit_db` |

Override with environment variables:
```bash
export CERTORA_AI_COMPOSER_PGHOST=myhost
export CERTORA_AI_COMPOSER_PGPORT=5433
```
