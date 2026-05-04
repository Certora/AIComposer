// CVL spec fed in as rules.spec. The tape later causes the LLM to call
// propose_spec_change a couple of times; one of those is expected to be
// ACCEPTED, and it replaces this file's contents inside the VFS. See the
// tape in composer/testing/ui_harness.py for the exact replacement text.

methods {
    function bal(address) external returns uint256 envfree;
    function total() external returns uint256 envfree;
    function deposit(uint256) external;
    function withdraw(uint256) external;
}

rule depositIncreasesBalance(uint256 amount) {
    env e;
    require e.msg.sender != 0;
    mathint before = bal(e.msg.sender);
    deposit(e, amount);
    assert bal(e.msg.sender) == assert_uint256(before + to_mathint(amount));
}

rule withdrawDecreasesBalance(uint256 amount) {
    env e;
    require e.msg.sender != 0;
    mathint before = bal(e.msg.sender);
    require before >= to_mathint(amount);
    withdraw(e, amount);
    assert bal(e.msg.sender) == assert_uint256(before - to_mathint(amount));
}
