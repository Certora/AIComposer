# Requirement Extraction & Counterexample Analysis — Deep Dive

This guide explains two of AI Composer's LLM prompt templates in detail, and how to get and read a Certora Prover counterexample (CEX) with or without AI Composer.

Everything below was checked directly against the code in this repository (paths and line references are current as of this writing); where the analysis notes a gap or discrepancy, that's called out explicitly rather than glossed over.

## Table of Contents

- [Part 1: `req_extraction_prompt.j2` Deep Dive](#part-1-req_extraction_promptj2-deep-dive)
- [Part 2: `cex_instructions.j2` Deep Dive](#part-2-cex_instructionsj2-deep-dive)
- [Part 3: How CEX Extraction Works Inside AI Composer](#part-3-how-cex-extraction-works-inside-ai-composer)
- [Part 4: How to Extract a CEX Without AI Composer](#part-4-how-to-extract-a-cex-without-ai-composer)
- [Known gaps](#known-gaps)

---

## Part 1: `req_extraction_prompt.j2` Deep Dive

**File:** [`composer/templates/req_extraction_prompt.j2`](composer/templates/req_extraction_prompt.j2)
**Purpose:** Get Claude to find implementation requirements that a system design doc implies but that the *formal* CVL spec doesn't cover — turning informal prose into a set of natural-language directives that enrich the spec without rewriting it.
**Invoked from:** [`composer/natreq/extractor.py`](composer/natreq/extractor.py), as the `initial_prompt` for the requirement-extraction agent. It's paired with a system prompt loaded from `req_role_prompt.j2` (below), and is one leg of the NatSpec pipeline.

### 1.1 The system prompt: `req_role_prompt.j2`

Before `req_extraction_prompt.j2` is even shown to Claude, `req_role_prompt.j2` sets the persona: an expert systems architect who's spent years turning high-level descriptions ("the servers will exchange credentials") into individually-checkable requirements, with domain depth in DeFi economics and smart-contract implementation. This primes Claude to decompose prose rather than just paraphrase it.

### 1.2 The prompt itself, section by section

**Input framing** establishes a two-input model: a system document (informal, "varying levels of formalism/rigor") describing one or more protocol components, plus a *formal* CVL spec for one specific component. This is a deliberate scope-bounding move — Claude is told up front it's working with one formal artifact and one informal one, not reconciling two formal ones.

**Task definition** is the actual instruction:

> Analyze both the system document and the specification to identify any implementation requirements/invariants/properties implied by the system document which are *NOT* covered by the provided specification. Focus *only* on properties/invariants/requirements which can be stated in terms of the component that is the focus of the specification. Do *NOT* consider interactions between components, as they are out of scope.

This is a **delta-based** instruction — find what's in the doc but missing from the spec — with two scope limiters: stick to the one component the spec targets, and ignore cross-component interactions entirely (those are presumably someone else's spec's problem). Output is constrained to a specific phrasing pattern ("The implementation must ensure that...", "When X happens, the implementation must..."), which matters because those directives get consumed downstream as structured natural-language requirements, not free-form commentary.

**Safety & human-in-the-loop enforcement** — three `IMPORTANT` blocks:

1. If Claude can't tell which component the spec targets, it should ask for help.
2. Claude *must* consult a human via the `human_in_the_loop` tool if uncertain about requirements or meaning — this isn't optional guidance, it's a hard requirement.
3. An explicit non-goal: Claude is extracting gaps, **not** improving or rewriting the spec file. This boundary matters for soundness — an agent that "improves" a spec while extracting requirements could silently weaken or narrow it.

> **Fixed:** the file previously had a typo in directive 1 — *"...as the user for help"* instead of *"...ask the user for help."* This has been corrected.

**Tools & reasoning** — the last block wires in two shared includes:

```jinja
<tools>
{% with cvl_manual_full = false, cvl_kb = false, cvl_researcher = false %}
{% include "cvl_tools_system_prompt.j2" %}
{% endwith %}
</tools>

<reasoning>
{% with draft_subject = "your extracted requirements", rough_draft_enforced = true, result_tool_name = "reqs" %}
{% include "rough_draft_protocol.j2" %}
{% endwith %}
</reasoning>
```

- `cvl_tools_system_prompt.j2` is included with the full CVL manual, the CVL knowledge base, and the CVL researcher tool all turned **off** (`false`) — this agent doesn't need deep CVL tooling, since it's reasoning about natural language, not writing CVL.
- `rough_draft_protocol.j2` is a shared "think before you answer" macro (see [`composer/templates/rough_draft_protocol.j2`](composer/templates/rough_draft_protocol.j2)): Claude must call `write_rough_draft`, then `read_rough_draft` to review its own draft, before submitting via the `reqs` completion tool. Because `rough_draft_enforced = true` is passed here, this isn't just a suggestion — [`extractor.py`](composer/natreq/extractor.py)'s result validator (`_extraction_res_checker`) actively **rejects** a submission if the agent never read its draft back, forcing the two-step draft → review → submit flow rather than a one-shot answer.

### 1.3 Where the human-in-the-loop consultation actually goes

The `human_in_the_loop` tool doesn't necessarily mean "ping a person right now." In `extractor.py`, human interaction is routed through an `OracleHandler` wrapper: if an `extraction_question`-type interrupt fires and an **oracle** callback is configured, the question and context get sent to the oracle instead of blocking on a live human — otherwise it falls through to the underlying `IOHandler`. In interactive use (the TUI/console front ends) that's a real prompt to you; in automated/test contexts it can be answered programmatically.

### 1.4 Quality checks (fixed)

The instructions previously left four things entirely to Claude's judgment (and the human-in-the-loop channel), with no explicit instruction covering them:

- **No contradiction check** — nothing told Claude to flag it if a doc-derived requirement conflicts with the existing spec rather than just supplementing it.
- **No CVL-expressibility check** — a requirement could be extracted even if it wasn't actually formalizable (e.g. "the protocol must feel responsive to users").
- **No deduplication guidance** — nothing prevented re-deriving a requirement the spec already covers, beyond the general "focus on what's not covered" instruction.
- **No priority/ranking guidance** — all extracted requirements came back as an undifferentiated list.

The prompt now includes an explicit `IMPORTANT` block instructing Claude to apply four checks to each candidate requirement before finalizing the list:

1. **Contradiction check** — an apparent conflict with the spec must be raised via `human_in_the_loop`, not silently included or silently dropped.
2. **Expressibility check** — only requirements that could in principle be a CVL rule or invariant over contract state/behavior are included; purely qualitative claims are excluded.
3. **Deduplication check** — a requirement already enforced by the spec (even under different phrasing or a different rule name) is excluded, even if the system document also mentions it.
4. **Prioritization** — the final list is ordered most-to-least critical (fund safety / access control / core invariants first), expressed *only* through list order, with no inline tag or label in the requirement text.

That last constraint isn't arbitrary: the extraction result's `reqs` field is a plain `list[str]` (see [`composer/natreq/extractor.py`](composer/natreq/extractor.py)), and `req_text` is explicitly typed downstream as "the requirement text (without the numeric label)" — it's rendered as-is in the TUI/console front ends (`composer/ui/codegen_rich.py`, `composer/ui/console.py`). A bracketed priority tag baked into the string would leak into that display. `req_num`, a separate field, already drives ordering (`run_requirements` is read back `ORDER BY req_num ASC`), so expressing priority through list position — rather than inventing a new field — was the only change that didn't require touching the schema or any downstream consumer.

---

## Part 2: `cex_instructions.j2` Deep Dive

**File:** [`composer/templates/cex_instructions.j2`](composer/templates/cex_instructions.j2)
**Purpose:** Once the Prover reports a `VIOLATED` rule, this tells Claude how to read the counterexample and propose a fix.
**Invoked from:** [`composer/prover/analysis.py`](composer/prover/analysis.py)'s `analyze_cex_raw`, which is called by `DefaultCexHandler.analyze_cex` in [`composer/prover/core.py`](composer/prover/core.py), inside the main verify loop, only for rules whose `status == "VIOLATED"`.

### 2.1 The prompt, section by section

**Analysis task:**

> Please analyze the previous counterexample. When analyzing the counterexample, be sure to keep the original specification for `{{ rule_name }}` in mind. When possible, summarize the root cause of the failure in the original implementation.

This assumes the CEX is already in the conversation as prior context (it is — see [Part 3](#part-3-how-cex-extraction-works-inside-ai-composer)) and explicitly ties the analysis back to the spec being verified, rather than analyzing the trace in isolation.

**Uncertainty handling:**

> If you are uncertain as the exact cause, indicate this. If you have multiple theories, list those theories.

This is a hedge against false confidence — the prompt would rather get "here are two plausible root causes" than a single overconfident (and possibly wrong) diagnosis. That principle also shows up elsewhere in the codebase's LLM guidance around formal-verification analysis: getting a confidently wrong explanation is worse than admitting uncertainty, because a wrong fix can make an unsound spec look like it passes.

**Output format constraint:**

> IMPORTANT: Respond to this message with a natural language response. Do NOT use any tools. Phrase your analysis in the second person, e.g., "You learned that the original implementation had the defect ..." and "You must make the following changes ..."

Two things bundled here: (1) this is meant to be a terminal, non-tool-calling turn — the analysis is the deliverable, not a springboard for more tool use — and (2) second-person phrasing ("You learned...", "You must...") reads as direct, actionable guidance rather than a passive bug report, which matters because this text is often what gets surfaced straight to a human via `rule_feedback.j2`.

**The HAVOC/ghost anti-pattern warning:**

> IMPORTANT: If the failure is due to ghosts being HAVOCed due to an unresolved call, *NEVER* suggest making the ghosts persistent as a fix. Instead, suggest alternative ways to address the unresolved call from summarization, linking changes, etc.

This is the most CVL-specific instruction in the file, and it's guarding against a real trap. When the Prover hits a call it can't resolve (e.g. a call into an external contract with no known implementation), it HAVOCs the affected state — i.e. assumes it could be anything — which routinely causes otherwise-correct rules to fail. The naive "fix" is to make the relevant ghost variable `persistent`, which makes the failure go away by construction rather than by actually modeling the call. That produces a spec that verifies but no longer means anything for that code path — a soundness hole disguised as a fix. The prompt steers Claude toward the real remedies instead: linking the external contract to a concrete implementation (`--link`), summarizing the unresolved function's behavior in `methods {}`, or adjusting the linking strategy so the call resolves in the first place.

### 2.2 The exact message sequence Claude sees

From `analyze_cex_raw`:

```python
new_messages.append(
    ToolMessage(
        tool_call_id=tool_call_id,
        content=f"""
The Certora Prover found a violation for the rule {rule.name}, with the following counter example:
{rule.cex_dump}
"""
    )
)
new_messages.append(
    HumanMessage(
        content=load_jinja_template("cex_instructions.j2", rule_name=rule.name)
    )
)

res = await llm.ainvoke(new_messages)
```

So concretely: (1) a `ToolMessage` carrying the raw XML counterexample dump for that rule, followed immediately by (2) a `HumanMessage` containing the instructions above with `{{ rule_name }}` filled in, and (3) Claude's response is the natural-language analysis — returned as plain text (`res.text`), not a tool call.

### 2.3 Non-code root causes (fixed)

The file previously covered only **one** anti-pattern (HAVOCed ghosts), with no instruction to also consider whether the root cause lies in the *rule* rather than the implementation. It now includes an additional `IMPORTANT` block asking Claude to check for:

- **Unreachable preconditions** — the rule may never actually trigger on any input, which can look like "verified" for the wrong reason (vacuity) rather than a real guarantee.
- **Implicit assumptions** baked into the rule (e.g. "assumes the contract is already initialized") that the CEX is really exposing, not a code defect.
- **External-state assumptions** (e.g. "assumes an oracle price is stable") that the counterexample violates by construction.
- **Timelock/time-based constraints** the rule depends on but doesn't state.

In each of those cases, the right conclusion may be "the rule needs to be relaxed or made more precise," not "the implementation is buggy" — the prompt now explicitly asks Claude to make that call rather than defaulting straight to a code fix.

---

## Part 3: How CEX Extraction Works Inside AI Composer

### 3.1 End-to-end flow

```
Certora Prover run
        │
        ▼
Reports/treeView/treeViewStatus_*.json   (raw tree-view status)
        │
        ▼
composer/prover/results.py
  get_final_treeview()        — finds the highest-numbered treeViewStatus file
  read_and_format_run_result() — validates + flattens it into RuleResult objects
  flatten_tree_view()          — walks the tree, extracts CEX JSON for VIOLATED rules
  calltrace_to_xml()           — converts the CEX call trace to XML for the LLM
        │
        ▼
dict[str, RuleResult]   (one entry per rule, keyed by name)
        │
        ▼
composer/prover/core.py  (the verify loop)
  for each VIOLATED rule → DefaultCexHandler.analyze_cex()
        │
        ▼
composer/prover/analysis.py :: analyze_cex_raw()
  appends CEX as a ToolMessage + cex_instructions.j2 as a HumanMessage, calls the LLM
        │
        ▼
Claude's natural-language root-cause analysis
        │
        ▼
rule_feedback.j2  →  final Markdown report per rule
```

### 3.2 Parsing the tree view

The Prover writes its results as a `treeViewStatus_N.json` tree of `RuleNodeModel` nodes (name, status, `nodeType`, child nodes, and — for violated leaves — an `output` list pointing at the CEX file). `get_final_treeview` just finds the numerically-latest status file in `Reports/treeView/` and validates it against the `TreeViewStatus` Pydantic model.

`flatten_tree_view` walks this tree recursively, and it's more than a simple VIOLATED/not-VIOLATED switch — the actual status space is `VERIFIED | VIOLATED | TIMEOUT | ERROR | SANITY_FAILED | SKIPPED`, and each is handled differently (errors collect child error messages, timeouts with no children terminate directly, verified nodes with only sanity children collapse into a single result, and so on). The part relevant to CEX handling is this branch:

```python
violated_assert_children = any([ c.nodeType == "VIOLATED_ASSERT" for c in r.children])
if violated_assert_children:
    assert stat == "VIOLATED" and len(r.output) > 0
    output_file = r.output[0]
    dump_model = json.loads((context / output_file).read_text())
    cex_dump : None | str = None
    if "callTrace" in dump_model:
        cex_node = CallTraceModel.model_validate(dump_model["callTrace"])
        cex_dump = "<counterexample>" + calltrace_to_xml(cex_node) + "</counterexample>"
    return [RuleResult(path=effective_path, cex_dump=cex_dump, status=stat)]
```

`r.output[0]` is the pointer to the raw CEX JSON file (relative to the tree-view directory); it's loaded, validated against `CallTraceModel`, and converted to an XML string that becomes `RuleResult.cex_dump`.

### 3.3 CEX → XML conversion

`calltrace_to_xml` walks the call-trace tree recursively:

```python
def calltrace_to_xml(node: CallTraceModel) -> str:
    formatted_message = node.message.text
    for i, arg in enumerate(node.message.arguments):
        placeholder = f"{{{i}}}"
        formatted_message = formatted_message.replace(placeholder, arg.value)

    xml_parts = [f"<message>{formatted_message}</message>"]
    for child in node.childrenList:
        if child.message.text in ("Setup", "Global State", "Evaluate branch condition", "unknown loop source code"):
            continue  # skip this, avoid confusing the llm
        xml_parts.append(f"<child>{calltrace_to_xml(child)}</child>")
    return "".join(xml_parts)
```

Two things worth noting: it substitutes `{0}`, `{1}`, ... placeholders in each message with the corresponding argument's `value` field (the Prover emits templated messages, not pre-formatted strings), and it explicitly drops a small set of node types that are noisy but not informative for an LLM reader (`Setup`, `Global State`, `Evaluate branch condition`, `unknown loop source code`).

### 3.4 The `RuleResult` / `RulePath` data model

From [`composer/prover/ptypes.py`](composer/prover/ptypes.py):

```python
@dataclass
class RulePath:
    rule: str
    contract: Optional[str] = None
    method: Optional[str] = None
    sanity: bool = False

@dataclass
class RuleResult:
    path: RulePath
    cex_dump: Optional[str]
    status: StatusCodes
    error_messages: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.pprint()
```

`cex_dump` is only populated when `status == "VIOLATED"` and the Prover actually produced a call trace; `name` is derived from `RulePath.pprint()`, which renders something like `"transfer_succeeds for deposit(uint256)"` when contract/method context is present, or just the bare rule name otherwise.

### 3.5 Triggering analysis

Back in `core.py`, once `read_and_format_run_result` returns the parsed `dict[str, RuleResult]`, every rule is analyzed **concurrently**:

```python
async def _analyze(rule: RuleResult) -> tuple[RuleResult, str | None]:
    if rule.status != "VIOLATED":
        return (rule, None)
    await callbacks.on_analysis_start(rule)
    res = await cex.analyze_cex(rule, tool_call_id)
    ...
    return (rule, res)

jobs = [_analyze(res) for res in parsed.values()]
results_with_analysis = await asyncio.gather(*jobs)
```

Only `VIOLATED` rules get an LLM call; everything else passes through with `analysis = None`. The final per-rule results (including any analysis text) are rendered into a Markdown report via `rule_feedback.j2`.

---

## Part 4: How to Extract a CEX Without AI Composer

You have two real options in this repo, depending on how much infrastructure you want to stand up.

### Option A — the standalone script (recommended, no setup)

[`extract_and_analyze_cex.py`](extract_and_analyze_cex.py) already implements exactly this, with no Docker/Postgres/RAG requirement — just Python and (for the `analyze` command) an Anthropic API key. It's the tool the [Quick Start in the main README](README.md#quick-start) walks through, so if you just want the CEX and Claude's take on it, use that directly:

```bash
python extract_and_analyze_cex.py list /path/to/prover_results
python extract_and_analyze_cex.py extract /path/to/prover_results transfer_succeeds
python extract_and_analyze_cex.py analyze /path/to/prover_results transfer_succeeds \
  --spec ./spec/Token.spec --output analysis.md
```

Internally it mirrors the same shape as the in-Composer pipeline (`CertoraResultsParser` finds and loads `treeViewStatus_*.json`, `CEXExtractor` locates the violated rule and reads its CEX file via `output[0]`, formats it as an indented tree instead of XML, and `ClaudeAnalyzer` sends it to the Claude API directly via the `anthropic` SDK with its own prompt — it does not use `cex_instructions.j2` or go through AI Composer's LangGraph workflow at all, so it has no RAG grounding in the CVL manual). If you want RAG-grounded analysis without the full AI Composer setup, the packaged `cex-analyzer` command (`analyzer/analysis.py`) is the middle ground — same idea, backed by the `rag_db` built in the [full installation](README.md#full-installation) steps.

### Option B — fully manual, no script at all

If you want to understand the raw shape of the data (or you're extracting it for something other than Claude), here's what's actually on disk after a Prover run:

**1. Locate the tree view.**

```
<output_dir>/Reports/treeView/treeViewStatus_0.json
<output_dir>/Reports/treeView/treeViewStatus_1.json   ← use the highest-numbered file
```

**2. Find your rule and its status.** The JSON is a `rules` list of nodes, each with `name`, `status`, `nodeType`, `output` (a list of file pointers), and `children`. Walk it recursively looking for `status == "VIOLATED"` and a non-empty `output`:

```python
import json
from pathlib import Path

def find_rule(node, rule_name):
    if node.get("name") == rule_name:
        return node
    for child in node.get("children", []):
        found = find_rule(child, rule_name)
        if found:
            return found
    return None

treeview_dir = Path("/path/to/prover_results/Reports/treeView")
latest = sorted(treeview_dir.glob("treeViewStatus_*.json"))[-1]
tree = json.loads(latest.read_text())

rule_node = next(
    (r for rule in tree["rules"] if (r := find_rule(rule, "transfer_succeeds"))),
    None,
)
```

**3. Load the CEX file.** `rule_node["output"][0]` is a path *relative to the tree-view directory* pointing at the call-trace JSON:

```python
cex_path = treeview_dir / rule_node["output"][0]
cex_data = json.loads(cex_path.read_text())
call_trace = cex_data["callTrace"]
```

**4. Render the call trace.** Each node has `message: {text, arguments: [{value}, ...]}` and `childrenList`. `{0}`, `{1}`, ... in `text` are placeholders for the corresponding `arguments[i].value`:

```python
SKIP = {"Setup", "Global State", "Evaluate branch condition", "unknown loop source code"}

def render(node, depth=0):
    text = node["message"]["text"]
    for i, arg in enumerate(node["message"].get("arguments", [])):
        text = text.replace(f"{{{i}}}", arg.get("value", "?"))
    lines = ["  " * depth + "→ " + text]
    for child in node.get("childrenList", []):
        if child["message"]["text"] not in SKIP:
            lines.append(render(child, depth + 1))
    return "\n".join(lines)

print(render(call_trace))
```

That's the same shape `calltrace_to_xml` and `extract_and_analyze_cex.py`'s `format_cex_tree` both build on — this is just the un-abstracted version of it. From here, pasting the rendered trace (plus your `.spec` file) into Claude, or any other LLM, manually reproduces what `cex_instructions.j2` automates — you're just doing the prompt assembly by hand instead of letting `analyze_cex_raw` do it.

### When each option makes sense

| Situation | Use |
|---|---|
| Just want the CEX + a fix suggestion, minimal setup | `extract_and_analyze_cex.py` |
| Want RAG-grounded analysis against the CVL manual, no full AI Composer run | `cex-analyzer` |
| Debugging the pipeline itself, or scripting something custom | Option B (manual) |
| Running the full spec-generation/verification loop | AI Composer / AutoProve (see [Part 3](#part-3-how-cex-extraction-works-inside-ai-composer)) |

---

## Known gaps

**Fixed:**

- `cex_instructions.j2` now asks Claude to consider vacuity, implicit assumptions, external-state assumptions, and timelock constraints as possible root causes, not just HAVOCed-ghost handling. See [§2.3](#23-non-code-root-causes-fixed).
- The `req_extraction_prompt.j2` typo ("as the user for help" → "ask the user for help") is corrected. See [§1.2](#12-the-prompt-itself-section-by-section).
- `req_extraction_prompt.j2` now includes contradiction, CVL-expressibility, deduplication, and prioritization checks for extracted requirements, instead of leaving all four entirely to model judgment. See [§1.4](#14-quality-checks-fixed).
- `tui-autoprove`, `console-autoprove`, and `sanity-analyzer` are now registered under `[project.scripts]` in `pyproject.toml`, matching what `AUTOPROVE.md` and `sanity_analyzer/README.md` already documented as installable commands.

**Deliberately deferred** (not a prompt/doc fix — needs a decision from whoever maintains the dependency, not just an edit):

- `AUTOPROVE.md`'s documented install command references a `pou` extra that doesn't exist in `pyproject.toml`. It's tied to the `AutoSetup`-backed auto-setup component (a separate private Certora repo); inventing an empty `pou = []` would hide the problem rather than solve it. See [`GETTING_STARTED.md`](GETTING_STARTED.md) step B7 for the workaround in the meantime.
