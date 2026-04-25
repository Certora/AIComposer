methods {
    function value() external returns (uint256) envfree;
    function increment() external;
}

// The count never decreases across any call to any method.
rule value_is_monotonic {
    uint256 before = value();
    method f;
    env e;
    calldataarg args;
    f(e, args);
    assert value() >= before;
}
