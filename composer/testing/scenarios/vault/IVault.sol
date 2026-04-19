// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

// Interface the LLM is being asked to implement.
// The scenario tape drives the LLM through one buggy implementation
// (caught by the prover), one rejected spec proposal, one accepted
// spec proposal, a working-spec round-trip, and finally a correct
// implementation that passes all rules.

interface IVault {
    function bal(address who) external view returns (uint256);
    function total() external view returns (uint256);
    function deposit(uint256 amount) external;
    function withdraw(uint256 amount) external;
}
