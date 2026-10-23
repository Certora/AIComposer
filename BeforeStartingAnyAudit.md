## Before Starting Any Audit

*Read Everything*
      **Gather all documentation:**
      
- ✓ Whitepaper/design docs
- ✓ README files
- ✓ Code comments and NatSpec
- ✓ Previous audit reports
- ✓ Any blog posts or technical articles

---

- [ ] Target protocol selected (has public docs/whitepaper)
- [ ] VS Code open with contract file and Copilot Pro active
- [ ] Notepad/file ready for extracted claims (Step 2)

## Same Sources, Different Extraction

```
SOURCE MATERIAL
───────────────
✓ Whitepaper/design docs
✓ README files
✓ Code comments and NatSpec
✓ Previous audit reports
✓ Blog posts / technical articles
✓ The contract code itself
        │
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
  system_doc.txt                              design_doc.md
  ──────────────                              ──────────────
  Extract: DESCRIPTIVE                        Extract: SECURITY
  "What does it do?"                          "What does it guarantee?"
  "How does it work?"                         "What breaks it?"
  "Who are the actors?"                       "What must never happen?"
  ```

## The Complete Picture

```
SOURCES                    system_doc.txt        design_doc.md
─────────────────────────────────────────────────────────────────
Whitepaper         →       Architecture          Guarantees claimed
README             →       Overview              Stated invariants
NatSpec            →       Function descriptions Claimed behavior
Code comments      →       Design decisions      Properties to verify
Previous audits    →       Context               Known risk areas
Blog posts         →       How it works          Edge cases noted
Contract code      →       Components & actors   Claims & pre/postconds

─────────────────────────────────────────────────────────────────
QUESTION ANSWERED:         "What IS it?"         "What must HOLD?"
TONE:                      Descriptive           Adversarial
LENGTH:                    Short (500-1000 words) Long (2000-5000 words)
USED WITH:                 python main.py        tui_autoprove.py
```

## How It's Organized

```
composer/
  ├── workflow/        Async state machine (executor.py, services.py) + LLM orchestration
  ├── core/            State definitions, validation rules, user context
  ├── spec/            CVL generation, property inference, system modeling, refinement
  ├── prover/          Certora Prover invocation, result parsing, cloud integration
  ├── rag/             PostgreSQL-backed RAG with pgvector embeddings
  ├── audit/           Execution history & checkpoint persistence
  ├── tools/           LLM tool implementations (search, relaxation, memory)
  ├── templates/       Jinja2 prompts for system/synthesis/refinement
  ├── io/              Protocol handlers for tool I/O
  └── cli/             CLI entry points

analyzer/              CEX (counterexample) analysis: rule violations → English explanations
sanity_analyzer/       Unsat core analysis: identifies why specs are unsatisfiable
scripts/               RAG population, database setup, visualization
tests/ test_scenarios/ Test fixtures & example inputs
```

### Implement a formal specification validator:

```
1. Parse generated CVL syntax
2. Type-check against contract interface
3. Check for vacuity traps (rules always true, rules always false)
4. Validate that all requirements are covered
5. Flag under-constrained specifications
```

## During Property Extraction (7 Steps)

- [ ] Step 1: Copilot summarized contract (understand before extracting)
- [ ] Step 2: YOU manually extracted claims from comments/require/modifiers
- [ ] Step 3: Copilot mapped 3–6 components
- [ ] Step 4: YOU converted each claim to a formal property (plain English)
- [ ] Step 5: Pre/postconditions defined for 3+ critical functions
- [ ] Step 6: design_doc.md drafted — ZERO code in it
- [ ] Step 7: Copilot reviewed doc — all 12 categories addressed (or the condensed 10-category quick-start variant, at minimum — see `MASTER_AUDIT_GUIDE_V2.md` Part 4)

## Before Running AIComposer

- [ ] design_doc.md is 2000+ words
- [ ] design_doc.md has NO CVL or Solidity code
- [ ] All 10 business logic categories have entries
- [ ] Edge cases listed (zero, max, empty, paused states)
- [ ] ANTHROPIC_API_KEY set in WSL
- [ ] CERTORAKEY set in WSL
- [ ] AUTOSETUP_PATH set in WSL (after access granted)
- [ ] Docker PostgreSQL running: docker ps | grep pgvector
- [ ] RAG database populated (one-time setup done)

## After Getting Prover Results

- [ ] All FAILED rules investigated with cex_analyzer
- [ ] Each failure verified as real (not a proof artifact)
- [ ] Finding classified as Vulnerability / Design Flaw / Spec Violation / Correctness
- [ ] PoC test written and passing (using Part 11 template)
- [ ] Impact quantified in dollars
- [ ] Severity justified using Immunefi/Sherlock criteria
- [ ] Report written using Part 9 template
- [ ] Finding ties back to documented protocol claim
- [ ] Submission package prepared (report + specs + PoC + README)

**Step 2: Prompt To Extract Documented Claims from Whitepaper/design docs (Copilot Claude Guided)**

Ask Copilot to find all claims/guarantees from documentation:

Based on the whitepaper, README, and code comments, list ALL explicit claims about what the protocol guarantees or maintains. Format each as:

CLAIM: [exact quote or paraphrase] 
SOURCE: [file/section] 
AFFECTED_FUNCTIONS: [which functions implement this]

Copilot generates structured list of all claims 