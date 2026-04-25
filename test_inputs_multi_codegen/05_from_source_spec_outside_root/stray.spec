methods {
    function value() external returns (uint256) envfree;
}

rule stray {
    assert value() >= 0;
}
