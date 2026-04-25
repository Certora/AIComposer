methods {
    function value() external returns (uint256) envfree;
}

rule trivially_true_a {
    assert value() >= 0;
}
