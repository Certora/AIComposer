# Counter

A single-contract component. `Counter` stores a single `uint256` count
initialized to zero at deployment. `value()` returns the current count.
`increment()` adds exactly one to the count and is callable by anyone.

The count is monotonically non-decreasing across any call to any method —
no method may reduce it.
