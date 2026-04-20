# Counter System

A minimal on-chain counter contract with a single component responsible for
incrementing a shared count and a per-caller tally of how many times each
sender has invoked it.

## Counter Contract

The `Counter` contract is a singleton and has a single component, `Increment`,
that handles all count updates. There are no external contracts and no
external actors in this system.

### Increment Component

- External entry points: `increment()`
- State variables:
  - `uint256 count` — the total number of increments across all senders.
  - `mapping(address => uint256) increments` — per-address tally of
    invocations.
- Interactions: none (no interactions with other contracts or external
  actors).

Requirements:

- Each call to `increment()` must increase `count` by exactly 1.
- Each call to `increment()` must increase `increments[msg.sender]` by
  exactly 1.
- `increment()` must not revert under normal operation.
