use cvlr::prelude::*;

/// Verifies that `add` correctly computes the sum of two numbers.
#[rule]
pub fn rule_add_is_correct() {
    let x: u64 = nondet();
    let y: u64 = nondet();
    let result = add(x, y);
    cvlr_assert_eq!(result, x + y);
}
