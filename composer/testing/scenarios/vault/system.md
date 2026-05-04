# Vault system

The `Vault` contract is a minimal single-asset accounting primitive. Users
deposit an amount against their own address and may later withdraw up to their
recorded balance.

## Entities

- `bal(address who)` — the running recorded balance of `who`.
- `total()` — the sum of all recorded balances; equal to the sum of `bal(x)`
  for every `x` that has ever deposited.

## Operations

### `deposit(uint256 amount)`

Credits `amount` to `bal[msg.sender]` and adds `amount` to `total`.

Zero-address senders are implicitly disallowed by the EVM. The contract does
not need to take an explicit `msg.sender != 0` precondition.

### `withdraw(uint256 amount)`

Debits `amount` from `bal[msg.sender]` and subtracts `amount` from `total`.
Reverts if `bal[msg.sender] < amount`.

## Invariants

- `bal(x)` only changes by the exact amount of a deposit or withdrawal by `x`.
- No user can withdraw more than their recorded balance.
- `total` tracks the sum of balances under all operations.

## Out of scope

- ERC-20 integration, pausing, role-based access, fee schedules, interest.

## Notes for the scenario

This document is intentionally a little ambiguous: it does not spell out
whether `total` underflow protection is required beyond Solidity 0.8 default
checked arithmetic, nor whether `deposit(0)` should be a no-op or revert.
Those ambiguities exist so the scripted tape has something plausible to ask
about via the `human_in_the_loop` tool during requirements extraction.
