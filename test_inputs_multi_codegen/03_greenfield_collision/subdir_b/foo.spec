methods {
    function value() external returns (uint256) envfree;
}

rule trivially_true_b {
    assert value() >= 0;
}
