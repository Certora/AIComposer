from typing import TypedDict, Any
import pathlib
import sys

composer_dir = str(pathlib.Path(__file__).parent.parent.absolute())

if composer_dir not in sys.path:
    sys.path.append(composer_dir)

from composer.workflow.services import get_indexed_store
from composer.kb.knowledge_base import DefaultEmbedder, KnowledgeBaseArticle

store = get_indexed_store(DefaultEmbedder())

class KBMessage(TypedDict):
    title: str
    body: str
    symptom: str

CVL_HELP_MESSAGES : list[KBMessage] = [
    {
        "title": "Summary not being applied",
        "symptom": "You wrote a summary like `function Foo.whatever() external => NONDET;` but the Prover reports the call as unresolved, or the summary doesn't seem to take effect.",
        "body": (
            "**Likely cause:** When you specify an explicit receiver contract (e.g. `Foo.whatever`), "
            "the Prover only applies the summary when it can *prove* that the receiver of the call is "
            "definitely the `Foo` contract. For external calls through an interface or an unknown address "
            "(e.g. `IERC20(addr).whatever()`), the Prover typically *cannot* prove the receiver, so the "
            "summary is silently skipped.\n\n"
            "**Fix:** Use the wildcard receiver `_` to match calls on any contract:\n"
            "```cvl\n"
            "function _.whatever() external => NONDET;\n"
            "```\n\n"
            "**Related:** Catch-all summaries (`function Token._ external => NONDET;`) have the same "
            "limitation — they are only applied when the Prover can definitively prove the target is "
            "`Token`. Check the \"Rule Call Resolution\" panel in the web report to see whether a summary "
            "was actually applied."
        ),
    },
    {
        "title": "Summary not applied to calls from CVL",
        "symptom": "You have a summary for function `f`, but when you call `f(e, args)` directly from your rule, the summary is ignored and the original implementation runs.",
        "body": (
            "**Cause:** Functions called *directly from CVL* are never summarized. This is by design.\n\n"
            "**Workaround:** If you need the summarized behavior, call the function through contract code "
            "(e.g. via another Solidity function that calls `f`), or restructure your rule so it doesn't "
            "depend on the summary for direct CVL calls."
        ),
    },
    {
        "title": "`public` function summary not working as expected",
        "symptom": "You declared a summary for a `public` Solidity function but it doesn't apply in the cases you expect.",
        "body": (
            "**Cause:** The Solidity compiler splits a `public` function into an internal implementation "
            "and an external wrapper. Your `internal` vs `external` annotation in the methods block "
            "determines *which* one is summarized:\n\n"
            "- **`internal` summary:** Effectively summarizes both internal and external calls, because "
            "the external wrapper delegates to the internal implementation. This is usually what you want.\n"
            "- **`external` summary:** Only summarizes external calls from Solidity code (`this.f()` or "
            "`c.f()`). Does NOT summarize direct CVL calls (which bypass summaries entirely), and does NOT "
            "affect internal calls.\n\n"
            "**Fix:** For `public` functions, prefer `internal` summaries unless you specifically need to "
            "only affect the external dispatch path."
        ),
    },
    {
        "title": "Wildcard summary needs `expect` clause",
        "symptom": "Type error when writing `function _.foo() external => myGhostOrFunction();`",
        "body": (
            "**Cause:** Wildcard entries (`_.foo`) must not declare a return type (since they may match "
            "methods returning different types). To compensate, any ghost or function summary on a "
            "wildcard entry *must* include an `expect` clause.\n\n"
            "**Fix:**\n"
            "```cvl\n"
            "function _.foo() external => myFunction() expect uint256 ALL;\n"
            "```\n"
            "Use `expect void` if the summarized function has no return value.\n\n"
            "**Warning:** The Prover cannot always verify your `expect` clause is correct. If the actual "
            "contract method returns `address` but you wrote `expect uint256`, the value will be "
            "misinterpreted, causing undefined behavior the Prover may not detect."
        ),
    },
    {
        "title": "Default summary application policy is `UNRESOLVED` for wildcards",
        "symptom": "You wrote `function _.foo(uint) external => NONDET;` but the summary isn't applied for a call where the Prover *knows* the target contract (e.g. a linked contract).",
        "body": (
            "**Cause:** For wildcard (`_`) external summaries, the default application policy is "
            "`UNRESOLVED`, meaning the summary only applies when the target contract is unknown. If the "
            "contract is linked or is `currentContract`, the resolved code runs instead.\n\n"
            "**Fix:** Append `ALL` to force the summary on all matching calls, even resolved ones:\n"
            "```cvl\n"
            "function _.foo(uint) external => NONDET ALL;\n"
            "```\n\n"
            "For exact contract entries and internal summaries, the default is `ALL`."
        ),
    },
    {
        "title": "`lastReverted` is overwritten by subsequent calls",
        "symptom": "A rule checking revert behavior is vacuous or gives unexpected results.",
        "body": (
            "**Cause:** `lastReverted` is updated after *every* contract call, even calls not marked "
            "`@withrevert`. This is a very common source of bugs. Example of a vacuous rule:\n"
            "```cvl\n"
            "rule revert_if_paused() {\n"
            "    withdraw@withrevert();\n"
            "    assert isPaused() => lastReverted; // BUG: isPaused() overwrites lastReverted!\n"
            "}\n"
            "```\n"
            "The call to `isPaused()` resets `lastReverted` to `false` (since it didn't revert), "
            "destroying the information from `withdraw`.\n\n"
            "**Fix:** Capture state before making additional calls:\n"
            "```cvl\n"
            "rule revert_if_paused() {\n"
            "    bool paused = isPaused();\n"
            "    withdraw@withrevert();\n"
            "    assert paused => lastReverted;\n"
            "}\n"
            "```"
        ),
    },
    {
        "title": "Missing `env` argument when calling a contract function",
        "symptom": "Type error or unexpected behavior when calling a Solidity function from CVL.",
        "body": (
            "**Cause:** All calls from CVL to contract functions require an `env` as the first argument, "
            "unless the function is declared `envfree` in the methods block. The `env` encapsulates "
            "`msg.sender`, `msg.value`, `block.timestamp`, etc.\n\n"
            "**Fix:** Either pass an env:\n"
            "```cvl\n"
            "rule example {\n"
            "    env e;\n"
            "    uint result = someFunction(e, arg1, arg2);\n"
            "}\n"
            "```\n"
            "Or declare it `envfree` if the function truly doesn't depend on environment variables:\n"
            "```cvl\n"
            "methods {\n"
            "    function someFunction(uint, uint) external returns (uint) envfree;\n"
            "}\n"
            "```\n"
            "The Prover will verify that `envfree` functions really don't depend on `msg.*`/`block.*` values."
        ),
    },
    {
        "title": "`^` means exponentiation in CVL, not XOR",
        "symptom": "Bitwise XOR computation gives wildly wrong results.",
        "body": (
            "**Cause:** CVL diverges from Solidity's operator meanings:\n"
            "- In **Solidity**: `^` = bitwise XOR, `**` = exponentiation\n"
            "- In **CVL**: `^` = exponentiation, `xor` = bitwise XOR\n\n"
            "**Fix:** Use the keyword `xor` for bitwise exclusive-or in CVL:\n"
            "```cvl\n"
            "uint result = a xor b;  // Correct CVL for bitwise XOR\n"
            "```"
        ),
    },
    {
        "title": "Invariant passes but contract still violates it",
        "symptom": "An invariant is verified but you discover the contract can actually violate it.",
        "body": (
            "**Cause:** Invariant proofs have several potential sources of unsoundness:\n\n"
            "1. **Preserved blocks:** `require` statements in preserved blocks can mask violations. Use "
            "`requireInvariant` for other invariants instead of raw `require` where possible (this is "
            "sound if you also prove the required invariant).\n"
            "2. **Filters:** Filtering out methods means the invariant is not checked for those methods.\n"
            "3. **Reverting functions:** If the invariant expression itself reverts, the check is vacuously true.\n"
            "4. **Parametric rules on secondary contracts (pre-v5):** Before CLI v5, parametric "
            "rules/invariants only checked `currentContract`'s methods. Now they check all contracts by "
            "default — be aware if you're on an older version.\n\n"
            "**Fix:** Always run with `\"rule_sanity\": \"basic\"` to catch vacuity. Review preserved "
            "blocks carefully — every `require` is a potential unsoundness."
        ),
    },
    {
        "title": "Parametric rule is checked on unexpected contracts",
        "symptom": "Starting with CLI v5, your parametric rule or invariant fails because it's being checked against methods on secondary contracts (not just `currentContract`).",
        "body": (
            "**Cause:** Since certora-cli 5.0, parametric rules and invariants are checked on methods of "
            "*all* contracts by default, not just the primary contract. This catches more bugs (e.g. an "
            "invariant relating `totalSupply` to an underlying token's balance can now be broken by "
            "calling the underlying token directly).\n\n"
            "**Fix:** If you need the old behavior, add a filter:\n"
            "```cvl\n"
            "rule myRule(method f) filtered {\n"
            "    f -> f.contract == currentContract\n"
            "} { ... }\n"
            "```\n"
            "However, you're encouraged to instead write additional invariants and preserved blocks to "
            "handle the new counterexamples, as they may reveal real security issues."
        ),
    },
    {
        "title": "Ghost variable has unexpected value after an external call",
        "symptom": "A non-persistent ghost has a surprising value after a call to an unresolved (havoced) external function.",
        "body": (
            "**Cause:** When the Prover havocs storage (e.g. due to an unresolved external call with "
            "`HAVOC_ECF` or `HAVOC_ALL`), **non-persistent ghosts are also havoced** — their values "
            "become completely unconstrained. This means even if your hook sets the ghost correctly, a "
            "havoc can overwrite it.\n\n"
            "**Fix:** If the ghost needs to survive havoc (e.g. tracking whether reentrancy occurred), "
            "declare it as a `persistent ghost`:\n"
            "```cvl\n"
            "persistent ghost bool reentrancy_happened {\n"
            "    init_state axiom !reentrancy_happened;\n"
            "}\n"
            "```\n\n"
            "**Rule of thumb:** Use regular ghosts for tracking storage-like state. Use persistent ghosts "
            "for tracking *events* (like \"a CALL happened\") that should never be forgotten."
        ),
    },
    {
        "title": "Hook is not triggered recursively",
        "symptom": "A store hook that calls a function which triggers another store to the same variable only fires once, not recursively.",
        "body": (
            "**Cause:** Hooks are **not recursively applied**. If a hook's body triggers another store to "
            "the same variable, the hook does not fire again. For example:\n"
            "```cvl\n"
            "hook Sstore x uint v {\n"
            "    xStoreCount = xStoreCount + 1;\n"
            "    if (xStoreCount < 5) { updateX(); } // This does NOT trigger the hook again\n"
            "}\n"
            "```\n\n"
            "**Workaround:** Be aware of this limitation. Design your ghost tracking to not depend on "
            "recursive hook invocations."
        ),
    },
    {
        "title": "Hook is not triggered by CVL code",
        "symptom": "You access a Solidity storage variable from CVL but your hook doesn't fire.",
        "body": (
            "**Cause:** Hooks are only triggered by *contract* code (Solidity). Accessing storage from "
            "CVL (e.g. direct storage access in a spec) does not trigger hooks."
        ),
    },
    {
        "title": "Struct field assignment not supported in CVL",
        "symptom": "`s.f = x;` gives a syntax error.",
        "body": (
            "**Cause:** CVL does not support assignment to struct fields.\n\n"
            "**Fix:** Use a `require` statement instead:\n"
            "```cvl\n"
            "require s.f == x;\n"
            "```\n"
            "But be careful — `require` can introduce vacuity if there are conflicting constraints."
        ),
    },
    {
        "title": "User-defined types must be qualified with the contract name",
        "symptom": "Error like \"type not found\" when referencing a struct, enum, or user-defined value type.",
        "body": (
            "**Cause:** All user-defined type names must be explicitly qualified by the contract that "
            "defines them.\n\n"
            "**Fix:**\n"
            "```cvl\n"
            "// Wrong:\n"
            "MyStruct s;\n"
            "\n"
            "// Correct:\n"
            "MyContract.MyStruct s;\n"
            "```"
        ),
    },
    {
        "title": "Methods block entry for inherited function — use the defining contract",
        "symptom": "Methods block entry doesn't match, or you get an error about the method not being found.",
        "body": (
            "**Cause:** The receiver contract in a methods block entry must be the contract where the "
            "method is *defined*. If a contract inherits a method from a supercontract, you must use the "
            "supercontract name, not the inheriting contract.\n\n"
            "**Fix:**\n"
            "```cvl\n"
            "// If Child inherits `foo` from Parent:\n"
            "// Wrong:\n"
            "function Child.foo(uint) external returns (uint);\n"
            "// Correct:\n"
            "function Parent.foo(uint) external returns (uint);\n"
            "```"
        ),
    },
    {
        "title": "`optional` keyword to avoid silent skipping",
        "symptom": "A rule seems to never run, or you suspect a methods block entry silently fails because the method doesn't exist.",
        "body": (
            "**Cause:** In CVL 2, if you declare a methods block entry for a method that doesn't exist "
            "in the contract, you get an error. If you *want* the old behavior (skip rules that reference "
            "a non-existent method), you must explicitly add `optional`.\n\n"
            "**Fix:**\n"
            "```cvl\n"
            "methods {\n"
            "    function mint(address, uint256) external optional;\n"
            "}\n"
            "```\n"
            "Without `optional`, a typo in a method name will produce an error (which is usually what you want)."
        ),
    },
    {
        "title": "`DISPATCHER(true)` fails if method doesn't exist in scene",
        "symptom": "Hard failure at type-checking when using `DISPATCHER(true)`.",
        "body": (
            "**Cause:** Since CLI v7.7.0, `_.someFunc() => DISPATCHER(true)` will fail if no contract "
            "in the scene implements `someFunc()`. Before this version, it would silently produce vacuous "
            "results.\n\n"
            "**Fix:** Ensure at least one contract in the scene implements the method, or use "
            "`DISPATCHER(false)` / `AUTO` if you need to handle the case where no implementation exists."
        ),
    },
    {
        "title": "`CONSTANT` / `PER_CALLEE_CONSTANT` summary causes vacuity",
        "symptom": "Rule passes trivially or counterexamples disappear when using `CONSTANT` or `PER_CALLEE_CONSTANT` summaries.",
        "body": (
            "**Cause:** These summaries assume *all* calls return the same value. For functions with "
            "variable-sized outputs (e.g. returning dynamic arrays), this assumption can be vacuously "
            "unsatisfiable.\n\n"
            "**Fix:** Prefer `NONDET` summaries when possible. `NONDET` makes no assumptions about return "
            "values and avoids this vacuity trap."
        ),
    },
    {
        "title": "`NONDET` summary may return wrong number of values",
        "symptom": "Unexpected reverts when a `NONDET`-summarized function is called.",
        "body": (
            "**Cause:** `NONDET` does *not* assume the number of returned values matches what the caller "
            "expects, unless `--optimisticReturnsize` is set. This mismatch often causes the calling code "
            "to revert on ABI decode.\n\n"
            "**Fix:** Add `--optimisticReturnsize` to your certoraRun config if you want the Prover to "
            "assume correct return sizes."
        ),
    },
    {
        "title": "Internal summary location annotations required",
        "symptom": "Error about missing `calldata`, `memory`, or `storage` annotations.",
        "body": (
            "**Cause:** Methods block entries for `internal` functions must include location annotations "
            "for all reference-type arguments (arrays, strings, structs, bytes).\n\n"
            "**Fix:**\n"
            "```cvl\n"
            "methods {\n"
            "    // Wrong:\n"
            "    function MyContract.foo(uint[] arr) internal returns (uint);\n"
            "    // Correct:\n"
            "    function MyContract.foo(uint[] memory arr) internal returns (uint);\n"
            "}\n"
            "```\n"
            "For `external` entries, location annotations must be *omitted* (unless it's a `storage` "
            "argument on an external library function)."
        ),
    },
    {
        "title": "Expression summary can't access `storage` parameters",
        "symptom": "Error when trying to use a `storage`-typed parameter in a CVL function summary.",
        "body": (
            "**Cause:** CVL functions cannot accept `storage` variables. If the function being summarized "
            "takes a `storage` parameter, you cannot reference it in an expression summary.\n\n"
            "**Fix:** Use a **rerouting summary** instead — route the call to an external library "
            "function written in Solidity that can access storage:\n"
            "```cvl\n"
            "methods {\n"
            "    function Bank.computeInterest(Bank.Vault storage v, address depositor, uint principal)\n"
            "        internal returns (uint)\n"
            "        => VaultHarness.computeInterestHarness(v, depositor, principal);\n"
            "}\n"
            "```\n"
            "Where `VaultHarness` is an external library in the scene."
        ),
    },
    {
        "title": "`with(env e)` — `e.msg.sender` may not be what you expect",
        "symptom": "In an expression summary using `with(env e)`, `e.msg.sender` doesn't match the original caller.",
        "body": (
            "**Cause:** In the `with(env e)` clause, `e.msg.sender` and `e.msg.value` refer to the "
            "sender/value from the *most recent non-library external call*, not necessarily the original "
            "CVL caller. For example, if your CVL rule calls contract C which calls contract D, and D's "
            "method is summarized, `e.msg.sender` will be C (not the CVL env's sender).\n\n"
            "`e.tx.origin` and `e.block.timestamp` will match the outermost call's environment."
        ),
    },
    {
        "title": "`calledContract` vs `executingContract` in summaries",
        "symptom": "Confusion about which contract address to use in a wildcard summary.",
        "body": (
            "**Cause:** These are distinct:\n"
            "- `calledContract`: The receiver of the summarized call (equivalent to `address(this)` in "
            "the called function's context).\n"
            "- `executingContract`: The contract that *made* the call to the summarized function.\n\n"
            "For internal, delegate, and library calls, they are the same. They differ only for "
            "non-delegate external calls.\n\n"
            "**Example use:** When summarizing `_.transferFrom()` across multiple tokens, use "
            "`calledContract` to identify *which* token:\n"
            "```cvl\n"
            "function _.transferFrom(address from, address to, uint256 amount) external\n"
            "    => cvlTransferFrom(calledContract, from, to, amount) expect void;\n"
            "```"
        ),
    },
    {
        "title": "`DISPATCHER` summaries cannot summarize library calls",
        "symptom": "`DISPATCHER` summary seems to have no effect on a library function call.",
        "body": (
            "**Cause:** This is a known limitation — `DISPATCHER` summaries do not work for library calls.\n\n"
            "**Fix:** Use a different summary type (e.g. `NONDET`, expression summary, or rerouting "
            "summary) for library functions."
        ),
    },
    {
        "title": "Vacuity — rule passes but `assert false` also passes",
        "symptom": "A rule verifies successfully, but something feels wrong. You add `assert false;` and it also verifies.",
        "body": (
            "**Cause:** The rule is **vacuous** — the combination of `require` statements and/or implicit "
            "reverts makes it impossible for execution to reach the assertions. Common causes:\n"
            "1. Calling a function without `@withrevert` that always reverts for the constrained inputs.\n"
            "2. Contradictory `require` statements.\n"
            "3. Overflow/underflow causing implicit reverts in Solidity ≥0.8.\n\n"
            "**Fix:** Always run with `\"rule_sanity\": \"basic\"` (this is now the default in recent "
            "versions). This checks whether each rule and assert is reachable. If the sanity check fails, "
            "revisit your `require` statements."
        ),
    },
    {
        "title": "`require` cast vs `assert` cast",
        "symptom": "Need to convert between integer types but unsure which cast to use.",
        "body": (
            "**Cause:** CVL 2 provides two casting operators:\n"
            "- `assert_uint256(x)`: Reports a violation if `x` is out of range.\n"
            "- `require_uint256(x)`: Silently ignores counterexamples where `x` is out of range "
            "(potential vacuity!).\n\n"
            "**Fix:** Use `assert_*` casts unless you have a deliberate reason to ignore out-of-range "
            "values. `require_*` casts can silently mask bugs."
        ),
    },
    {
        "title": "`mathint` operations never overflow",
        "symptom": "Confusion about when integer overflow can occur in CVL.",
        "body": (
            "**Cause:** The `mathint` type has arbitrary precision — operations on `mathint` can never "
            "overflow or underflow. This is a key difference from Solidity's fixed-width types.\n\n"
            "**Best practice:** Do arithmetic in `mathint` and only cast back to fixed-width types at "
            "boundaries:\n"
            "```cvl\n"
            "mathint total = balanceOf(a) + balanceOf(b); // No overflow possible\n"
            "assert total <= max_uint256; // Check the property you care about\n"
            "```"
        ),
    },
    {
        "title": "Out-of-bounds array access in CVL returns undefined, not revert",
        "symptom": "The Prover considers impossible-seeming values for array elements.",
        "body": (
            "**Cause:** Unlike Solidity, out-of-bounds array accesses in CVL are treated as *undefined "
            "values* — the Prover considers every possible value for `a[i]` when `i >= a.length`. This "
            "does not revert.\n\n"
            "**Fix:** Always guard array accesses with bounds checks in your `require` statements."
        ),
    },
    {
        "title": "Bitwise operations are overapproximated by default",
        "symptom": "Counterexamples show incorrect results for bitwise operations (`&`, `|`, `>>`, `<<`, `xor`).",
        "body": (
            "**Cause:** By default, the Prover overapproximates bitwise operations. The results may be "
            "imprecise (but still sound — the Prover won't miss real violations, but may produce spurious "
            "counterexamples).\n\n"
            "**Fix:** Use the `--precise_bitwise_ops` flag for more precise bitwise reasoning (may "
            "increase solving time)."
        ),
    },
    {
        "title": "Strong vs Weak invariants",
        "symptom": "Invariant doesn't hold across unresolved external calls.",
        "body": (
            "**Cause:** By default, invariants are *weak* — they hold before and after method execution "
            "but not across unresolved external calls during execution. A *strong* invariant is "
            "additionally asserted before a havoced external call and assumed afterward.\n\n"
            "**Fix:**\n"
            "```cvl\n"
            "strong invariant myInvariant() ...;  // Holds even across unresolved external calls\n"
            "```\n"
            "Use `strong` when the invariant must hold at all times, including mid-execution during "
            "external calls."
        ),
    },
    {
        "title": "Generic preserved block is not applied for constructor",
        "symptom": "Invariant fails on the base case (constructor step).",
        "body": (
            "**Cause:** The generic (unnamed) preserved block only applies to the induction step. For "
            "the constructor base case, you need a separate `preserved constructor()` block.\n\n"
            "**Fix:**\n"
            "```cvl\n"
            "invariant myInvariant()\n"
            "    someProperty()\n"
            "{\n"
            "    preserved { /* applies to all methods in induction step */ }\n"
            "    preserved constructor() { /* applies to constructor base step */ }\n"
            "}\n"
            "```"
        ),
    },
    {
        "title": "`DELETE` summary removes method from parametric rules",
        "symptom": "A method disappears from parametric rule checks entirely.",
        "body": (
            "**Cause:** A `DELETE` summary not only summarizes the method — it removes it from the "
            "scene. Parametric rules won't be instantiated on deleted methods, and calling it from CVL "
            "produces a violation.\n\n"
            "**Use case:** `DELETE` is useful for complex methods that cause timeouts but are irrelevant "
            "to the property being checked."
        ),
    },
]

to_store : list[KnowledgeBaseArticle] = [
    {
        "title": d["title"],
        "symptom": d["symptom"],
        "body": d["body"]
    } for d in CVL_HELP_MESSAGES
]

kb_ns = ("cvl", "agent", "knowledge")

for s in to_store:
    r = store.get(kb_ns, s["title"])
    if r is not None:
        continue
    value : dict[str, Any] = s #type: ignore
    store.put(kb_ns, s["title"], value, index=["symptom"])
