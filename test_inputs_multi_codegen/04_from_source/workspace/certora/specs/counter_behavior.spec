methods {
    function value() external returns (uint256) envfree;
    function increment() external;
}

rule increment_advances_by_one {
    uint256 before = value();
    env e;
    increment(e);
    assert value() == before + 1;
}
