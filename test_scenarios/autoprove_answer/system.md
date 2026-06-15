# Answer System

A deliberately trivial smoke-test system: one contract, one pure function, one
property. The point is to exercise the auto-prove pipeline end-to-end on the
smallest possible input — not to demonstrate creative specification work.

## Answer Contract

The `Answer` contract is a singleton with a single component, `Answer`, that
exposes one external entry point. There are no external contracts and no
external actors in this system, and the contract holds no state.

### Answer Component

- External entry points: `theAnswer()`
- State variables: none.
- Interactions: none.

Requirements:

- `theAnswer()` always returns `42`.

Do not extract additional properties, invariants, edge cases, security
implications, gas considerations, or composability concerns — none apply to
this contract.
