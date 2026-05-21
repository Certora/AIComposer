// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.29;

contract Counter {
    uint256 public count;
    mapping(address => uint256) public increments;

    function increment() external {
        require(msg.sender != address(0));
        count += 1;
        increments[msg.sender] += 1;
    }

    /// Credit a different address with an increment. The caller bumps the
    /// global ``count`` and the per-address tally for the *target* address.
    ///
    /// BUG: the implementation credits ``msg.sender`` instead of ``other``.
    /// The structural invariant ``count == sum(increments)`` still holds
    /// (both sides grow by exactly 1) and ``increments[address(0)]`` is
    /// never written, so the structural-invariant phase still verifies.
    /// The per-method correctness rule for ``incrementOther`` is what
    /// surfaces this bug.
    function incrementOther(address other) external {
        require(msg.sender != address(0));
        require(other != address(0));
        count += 1;
        increments[msg.sender] += 1;
    }
}
