# Counter (from-source)

A `Counter` contract to be added to an existing workspace. The workspace
already contains `ICounter.sol` (the interface the agent must implement) and
a small `Math` library that the agent may use but does not need to modify.

`Counter` stores a single `uint256` count initialized to zero at deployment.
`value()` returns the current count. `increment()` adds exactly one to the
count and is callable by anyone. The count is monotonically non-decreasing.
