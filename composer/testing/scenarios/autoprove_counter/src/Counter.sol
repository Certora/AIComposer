// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.29;

contract Counter {
    uint256 public count;
    mapping(address => uint256) public increments;

    function increment() external {
<<<<<<< HEAD
        require(msg.sender != address(0));
=======
>>>>>>> e78e6e4 (test scenarios)
        count += 1;
        increments[msg.sender] += 1;
    }
}
