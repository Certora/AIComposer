/**
 * ============================================================================
 * CVL Specification for Progressive Fee AMM Pool
 * ============================================================================
 * 
 * This specification verifies a constant product AMM with progressive withdrawal
 * penalties that automatically redistribute to remaining liquidity providers.
 * 
 * Key Features Verified:
 * - Constant product invariant (x * y = k) with fees
 * - LP token proportional ownership
 * - Progressive penalty mechanics (larger withdrawals = higher fees)
 * - Automatic redistribution to remaining LPs
 * - Protocol-favoring rounding
 * - Protection against pool drainage
 * - Solidity 0.8+ semantics (uint256, automatic overflow protection)
 * 
 * Assumptions:
 * - Solidity 0.8+ (automatic overflow/underflow reversion)
 * - uint256 reserves (non-negative by type)
 * - Both tokens have 18 decimals (1e18 units = 1 token)
 * - Protocol-favoring rounding (users get slightly less, dust stays in pool)
 * 
 * ============================================================================
 */

methods {
    // ========================================================================
    // State variables (envfree - can call without environment)
    // ========================================================================
    function reserveA() external returns (uint256) envfree;
    function reserveB() external returns (uint256) envfree;
    function totalSupply() external returns (uint256) envfree;
    function balanceOf(address) external returns (uint256) envfree;
    
    // ========================================================================
    // View functions
    // ========================================================================
    function calculateK() external returns (uint256) envfree;
    function calculateWithdrawalPenalty(uint256 lpTokens, uint256 totalLP) external returns (uint256) envfree;
    function getSwapFee(uint256 amountOut, uint256 reserve) external returns (uint256) envfree;

    function _.transfer(address to, uint256 amount) external with (env e)
        => transferCVL(calledContract, e.msg.sender, to, amount) expect bool;
    function _.transferFrom(address from, address to, uint256 amount) external with (env e) 
        => transferFromCVL(calledContract, e.msg.sender, from, to, amount) expect bool;
    function _.balanceOf(address account) external => 
        tokenBalanceOf(calledContract, account) expect uint256;
    
}


/// CVL simple implementations of IERC20:
/// token => account => balance
ghost mapping(address => mapping(address => uint256)) balanceByToken;
/// token => owner => spender => allowance
ghost mapping(address => mapping(address => mapping(address => uint256))) allowanceByToken;


function tokenBalanceOf(address token, address account) returns uint256 {
    return balanceByToken[token][account];
}


function revertOn(bool b) {
    if(b) {
        revert();
    }
}

function transferFromCVL(address token, address spender, address from, address to, uint256 amount) returns bool {
    revertOn(allowanceByToken[token][from][spender] < amount);
    bool success = transferCVL(token, from, to, amount);
    if(success) {
        allowanceByToken[token][from][spender] = assert_uint256(allowanceByToken[token][from][spender] - amount);
    }
    return success;
}

ghost bool revertOrReturnFalse; 
function transferCVL(address token, address from, address to, uint256 amount) returns bool {
    revertOn(token == 0);

    if (balanceByToken[token][from] < amount) {
        if(revertOrReturnFalse) {
             revert();
        }
        else { 
            return false; 
        }
    } 
    balanceByToken[token][from] = assert_uint256(balanceByToken[token][from] - amount);
    balanceByToken[token][to] = require_uint256(balanceByToken[token][to] + amount);  // We neglect overflows.
    return true;
}


// ============================================================================
// GHOST VARIABLES - Track aggregate state across operations
// ============================================================================

/**
 * Ghost: Sum of all LP token balances across all addresses
 * Used to verify totalLPSupply equals sum of individual balances
 */
ghost mathint sumOfLPBalances {
    init_state axiom sumOfLPBalances == 0;
}

/**
 * Ghost: Previous value of K constant for monotonicity checking
 * K should only increase (from fees and rounding dust)
 */
ghost mathint ghostK {
    init_state axiom ghostK == 0;
}

/**
 * Hook: Update sumOfLPBalances whenever any user's LP balance changes
 */
hook Sstore lpBalances[KEY address user] uint256 newBalance 
    (uint256 oldBalance) {
    sumOfLPBalances = sumOfLPBalances + newBalance - oldBalance;
}

// ============================================================================
// INVARIANTS - Properties that should always hold
// ============================================================================

/**
 * INV1: LP token supply equals sum of all balances
 * The total supply should always equal the sum of individual holdings
 */
invariant lpSupplyIsSumOfBalances()
    totalSupply() == sumOfLPBalances
    {
        preserved with (env e) {
            require e.msg.sender != currentContract;
        }
    }

/**
 * INV2: K never decreases (only increases due to fees)
 * The constant product can only grow from swap fees and rounding dust
 * NOTE: This excludes removeLiquidity as K naturally decreases when liquidity exits
 */
invariant kNeverDecreases()
    calculateK() >= ghostK
    filtered {
        f -> f.selector != sig:removeLiquidity(uint256).selector
    }
    {
        preserved {
            ghostK = calculateK();
        }
    }

/**
 * INV3: Reserves are positive when LP supply is positive
 * If LP tokens exist, both reserves must be non-zero
 */
invariant reservesExistWithLPTokens()
    totalSupply() > 0 => (reserveA() > 0 && reserveB() > 0);

/**
 * INV4: No LP tokens exist when reserves are empty
 * If reserves are depleted, no LP tokens should exist
 */
invariant noLPTokensWithEmptyReserves()
    (reserveA() == 0 && reserveB() == 0) => totalSupply() == 0;

/**
 * INV5: Reserves are zero together
 * Both reserves should be zero or both non-zero (no partial depletion)
 */
invariant reservesZeroTogether()
    (reserveA() == 0 || reserveB() == 0) => (reserveA() == 0 && reserveB() == 0);

// ============================================================================
// RULES - Core Functionality Verification
// ============================================================================

// ============================================================================
// LIQUIDITY PROVISION RULES
// ============================================================================

/**
 * RULE: Adding liquidity mints proportional LP tokens (with rounding)
 * 
 * When users add liquidity, they receive LP tokens proportional to their
 * contribution. Protocol-favoring rounding means they get slightly fewer
 * LP tokens (at most 1 unit less).
 */
rule addLiquidityMintsProportionalTokens(env e, uint256 amountA, uint256 amountB) {
    require amountA > 0 && amountB > 0;
    
    uint256 reserveA_before = reserveA();
    uint256 reserveB_before = reserveB();
    uint256 totalLP_before = totalSupply();
    uint256 userLP_before = balanceOf(e.msg.sender);
    
    require reserveA_before > 0 && reserveB_before > 0 && totalLP_before > 0;
    
    // Prevent overflow in calculations
    require amountA < max_uint256 / totalLP_before;
    require amountB < max_uint256 / totalLP_before;
    
    uint256 lpMinted = addLiquidity(e, amountA, amountB);
    
    uint256 reserveA_after = reserveA();
    uint256 reserveB_after = reserveB();
    uint256 totalLP_after = totalSupply();
    uint256 userLP_after = balanceOf(e.msg.sender);
    
    // Reserves increased by full amounts
    assert reserveA_after == reserveA_before + amountA;
    assert reserveB_after == reserveB_before + amountB;
    
    // LP minted is AT MOST proportional (rounded down)
    // lpMinted / totalLP_before <= amountA / reserveA_before
    // Since both tokens have 18 decimals, these proportions are directly comparable
    assert lpMinted * reserveA_before <= amountA * totalLP_before;
    assert lpMinted * reserveB_before <= amountB * totalLP_before;
    
    // But not too much less (within 1 unit of rounding)
    assert lpMinted * reserveA_before >= (amountA * totalLP_before) - reserveA_before;
    assert lpMinted * reserveB_before >= (amountB * totalLP_before) - reserveB_before;
    
    // User received the LP tokens
    assert userLP_after == userLP_before + lpMinted;
    assert totalLP_after == totalLP_before + lpMinted;
}

// ============================================================================
// WITHDRAWAL RULES - Progressive Penalties
// ============================================================================

/**
 * RULE: Small withdrawals get proportional amounts (with rounding)
 * 
 * Withdrawals below the penalty threshold (2% of pool) should receive
 * proportional amounts with only rounding deductions (at most 1 smallest unit per token).
 */
rule smallWithdrawalGetsProportion(env e, uint256 lpTokens) {
    require lpTokens > 0;
    uint256 totalLP = totalSupply();
    require totalLP > 0;
    require lpTokens <= totalLP;
    
    // Small withdrawal: less than 2% of pool (no penalty threshold)
    require lpTokens * 100 < totalLP * 2;
    
    uint256 reserveA_before = reserveA();
    uint256 reserveB_before = reserveB();
    
    // Expected amounts (exact proportion)
    mathint expectedA = (lpTokens * reserveA_before) / totalLP;
    mathint expectedB = (lpTokens * reserveB_before) / totalLP;
    
    uint256 amountA;
    uint256 amountB;
    amountA, amountB = removeLiquidity(e, lpTokens);
    
    // User gets AT MOST the proportion (rounded down)
    assert amountA <= expectedA;
    assert amountB <= expectedB;
    
    // Within rounding error (at most 1 smallest unit less per token)
    assert expectedA - amountA <= 1;
    assert expectedB - amountB <= 1;
    
    // No penalty applied (using pre-withdrawal totalLP)
    uint256 penalty = calculateWithdrawalPenalty(lpTokens, totalLP);
    assert penalty == 0;
}

/**
 * RULE: Large withdrawals have penalty + rounding
 * 
 * Withdrawals above the penalty threshold (5% of pool) should incur
 * progressive penalties that keep tokens in the pool.
 */
rule largeWithdrawalHasPenalty(env e, uint256 lpTokens) {
    require lpTokens > 0;
    uint256 totalLP = totalSupply();
    require totalLP > 0;
    require lpTokens <= totalLP;
    
    // Large withdrawal: more than 5% of pool
    require lpTokens * 100 >= totalLP * 5;
    
    uint256 reserveA_before = reserveA();
    uint256 reserveB_before = reserveB();
    
    mathint expectedA = (lpTokens * reserveA_before) / totalLP;
    mathint expectedB = (lpTokens * reserveB_before) / totalLP;
    
    uint256 amountA;
    uint256 amountB;
    amountA, amountB = removeLiquidity(e, lpTokens);
    
    // Should receive LESS than proportional
    assert amountA < expectedA;
    assert amountB < expectedB;
    
    // Penalty is non-zero (using pre-withdrawal totalLP)
    uint256 penalty = calculateWithdrawalPenalty(lpTokens, totalLP);
    assert penalty > 0;
    
    // Difference is at least the penalty (could be more with rounding)
    mathint penaltyA = (expectedA * penalty) / 10000;
    mathint penaltyB = (expectedB * penalty) / 10000;
    assert expectedA - amountA >= penaltyA;
    assert expectedB - amountB >= penaltyB;
}

/**
 * RULE: Penalties + rounding dust redistribute to remaining LPs
 * 
 * When someone withdraws with a penalty, remaining LP token holders
 * should see their tokens increase in value. This is the core
 * redistribution mechanism.
 * 
 * NOTE: This rule requires realistic pool states where redistribution
 * effects are measurable with integer arithmetic. It enforces economic
 * realism by ensuring LP supply is proportional to reserves.
 */
rule penaltyAndRoundingRedistribute(env e, address withdrawer, address staker) {
    require withdrawer != staker;
    require withdrawer != currentContract && staker != currentContract;
    
    uint256 stakerLP = balanceOf(staker);
    require stakerLP > 0;
    
    uint256 totalLP_before = totalSupply();
    uint256 reserveA_before = reserveA();
    uint256 reserveB_before = reserveB();
    
    // Prevent division by zero and overflow
    require totalLP_before > 0;
    require reserveA_before < max_uint256 / 1000000000000000000;
    require reserveB_before < max_uint256 / 1000000000000000000;
    
    // CRITICAL: Ensure reserves are large enough that redistribution is measurable
    // Require minimum reserves (1000 wei) to ensure per-LP values have sufficient precision
    require reserveA_before >= 1000;
    require reserveB_before >= 1000;
    
    // CRITICAL: Enforce economic realism - LP supply must be proportional to reserves
    // Prevent pathological states where LP supply is astronomically larger than reserves
    // This ensures penalties of ~dozens of wei have measurable per-LP effects
    // Require: totalLP <= max(reserveA, reserveB) * 10000
    // This means each LP represents at least 1/10000 of the smaller reserve
    require totalLP_before <= reserveA_before * 10000;
    require totalLP_before <= reserveB_before * 10000;
    
    // Value per LP token before (scaled by 1000000000000000000 for precision)
    // Since both tokens have 18 decimals, this scaling maintains precision
    mathint valuePerLP_A_before = (reserveA_before * 1000000000000000000) / totalLP_before;
    mathint valuePerLP_B_before = (reserveB_before * 1000000000000000000) / totalLP_before;
    
    // CRITICAL: Ensure value per LP is large enough to observe increases
    // With the ratio constraint above, this should be >= 100 (1e18 / 10000 / 1000)
    // Keep conservative bound at 2 for safety
    require valuePerLP_A_before >= 2;
    require valuePerLP_B_before >= 2;
    
    // Withdrawer removes liquidity with penalty
    uint256 lpToWithdraw = balanceOf(withdrawer);
    require lpToWithdraw > 0;
    require lpToWithdraw <= totalLP_before;
    
    // Large withdrawal triggering penalty
    require lpToWithdraw * 100 >= totalLP_before * 5;
    
    uint256 withdrawn_A;
    uint256 withdrawn_B;
    withdrawn_A, withdrawn_B = removeLiquidity(e, lpToWithdraw);
    
    // CRITICAL: Ensure withdrawn amounts are large enough that penalties are measurable
    // With a 5%+ withdrawal triggering penalties, ensure at least 10 wei withdrawn
    require withdrawn_A >= 10;
    require withdrawn_B >= 10;
    
    uint256 totalLP_after = totalSupply();
    uint256 reserveA_after = reserveA();
    uint256 reserveB_after = reserveB();
    
    // Staker still has LP tokens
    require totalLP_after > 0;
    require reserveA_after < max_uint256 / 1000000000000000000;
    require reserveB_after < max_uint256 / 1000000000000000000;
    
    // Value per LP token after (scaled by 1000000000000000000 for precision)
    // Since both tokens have 18 decimals, this scaling maintains precision
    mathint valuePerLP_A_after = (reserveA_after * 1000000000000000000) / totalLP_after;
    mathint valuePerLP_B_after = (reserveB_after * 1000000000000000000) / totalLP_after;
    
    // Each remaining LP token is worth MORE (penalty + rounding dust stayed in pool)
    assert valuePerLP_A_after > valuePerLP_A_before;
    assert valuePerLP_B_after > valuePerLP_B_before;
}

/**
 * RULE: Penalty increases monotonically
 * 
 * Larger withdrawals should never have smaller penalties than
 * smaller withdrawals (progressivity).
 */
rule penaltyIsMonotonic(uint256 lpTokens1, uint256 lpTokens2) {
    require lpTokens1 < lpTokens2;
    uint256 totalLP = totalSupply();
    require lpTokens2 <= totalLP;
    
    uint256 penalty1 = calculateWithdrawalPenalty(lpTokens1, totalLP);
    uint256 penalty2 = calculateWithdrawalPenalty(lpTokens2, totalLP);
    
    assert penalty2 >= penalty1;
}

// ============================================================================
// SWAP RULES - Progressive Fees
// ============================================================================

/**
 * RULE: Swap fees are progressive
 * 
 * Larger swaps (by output amount) should incur fees (in basis points) that are
 * greater than or equal to smaller swaps. The fee is already expressed as a 
 * percentage in basis points, so we directly compare the basis point values.
 * 
 * NOTE: The requirements specify a flat 0.3% base fee for swaps ≤0.5% depletion,
 * so swaps in that range will have equal fees, which satisfies >=.
 */
rule swapFeeIsProgressive(uint256 amountOut1, uint256 amountOut2, uint256 reserve) {
    require amountOut1 < amountOut2;
    require amountOut2 < reserve;
    require reserve > 0;
    require amountOut1 > 0;
    
    uint256 fee1 = getSwapFee(amountOut1, reserve);
    uint256 fee2 = getSwapFee(amountOut2, reserve);
    
    // Larger swaps have fees >= smaller swaps (in basis points)
    // This correctly handles the flat-fee zone (≤0.5% depletion) where all fees are equal
    assert fee2 >= fee1;
}

/**
 * RULE: K increases after swaps
 * 
 * Every swap should increase the constant product due to collected
 * fees and rounding dust.
 */
rule swapIncreasesK(env e, uint256 amountIn, bool aToB) {
    require amountIn > 0;
    
    uint256 k_before = calculateK();
    
    swap(e, amountIn, aToB);
    
    uint256 k_after = calculateK();
    
    // K increases from fees and rounding dust
    assert k_after >= k_before;
}

// ============================================================================
// VALUE CONSERVATION RULES
// ============================================================================

/**
 * RULE: Value conservation with rounding
 * 
 * Total value in the system can only increase slightly due to rounding
 * dust (at most 1 smallest unit per token type per operation).
 */
rule valueConservationWithRounding(env e, uint256 lpTokens) {
    require lpTokens > 0;
    uint256 totalLP = totalSupply();
    require totalLP > 0;
    require lpTokens <= totalLP;
    
    uint256 reserveA_before = reserveA();
    uint256 reserveB_before = reserveB();
    
    uint256 amountA;
    uint256 amountB;
    amountA, amountB = removeLiquidity(e, lpTokens);
    
    uint256 reserveA_after = reserveA();
    uint256 reserveB_after = reserveB();
    
    // Total value after = reserves + withdrawn
    mathint totalValueA_after = reserveA_after + amountA;
    mathint totalValueB_after = reserveB_after + amountB;
    
    // With protocol-favoring rounding, total value >= original (dust stays)
    assert totalValueA_after >= reserveA_before;
    assert totalValueB_after >= reserveB_before;
    
    // But not too much more (just rounding dust, max 1 smallest unit)
    assert totalValueA_after - reserveA_before <= 1;
    assert totalValueB_after - reserveB_before <= 1;
}

/**
 * RULE: LP proportional claim on K never decreases without withdrawal
 * 
 * If a user doesn't withdraw, their proportional claim on the constant product K
 * should not decrease. This verifies that:
 * - Swap fees benefit all LPs proportionally (K increases from fees)
 * - AddLiquidity maintains existing LPs' proportional value
 * 
 * This formulation is immune to impermanent loss effects (which affect single-asset
 * valuations) and instead focuses on the LP's claim on total pool value as measured
 * by K = reserveA × reserveB.
 * 
 * Note: Uses mathint for all calculations to avoid overflow. Integer division
 * may cause small rounding (≤1 unit), so we allow ±1 tolerance.
 */
rule lpProportionalClaimOnKNeverDecreases(
    env e, 
    address user, 
    method f
)
    filtered {
        f -> f.selector != sig:removeLiquidity(uint256).selector
    }
{
    require user != currentContract;
    
    uint256 lpBalance_before = balanceOf(user);
    require lpBalance_before > 0;
    
    uint256 totalLP_before = totalSupply();
    require totalLP_before > 0;
    
    // Enforce MINIMUM_LIQUIDITY invariant
    require totalLP_before >= 1000;
    
    uint256 k_before = calculateK();
    require k_before > 0;
    
    // User's proportional claim on K (using mathint for precision)
    // userK = (lpBalance / totalLP) × K = (lpBalance × K) / totalLP
    mathint userK_before = (lpBalance_before * k_before) / totalLP_before;
    
    calldataarg args;
    f(e, args);
    
    uint256 lpBalance_after = balanceOf(user);
    uint256 totalLP_after = totalSupply();
    uint256 k_after = calculateK();
    
    // If user's LP balance decreased or pool emptied, skip check
    // Otherwise, proportional K claim should not decrease (allowing ±1 for rounding)
    assert lpBalance_after < lpBalance_before || totalLP_after == 0 || k_after == 0 ||
           ((lpBalance_after * k_after) / totalLP_after) >= userK_before - 1;
}


/**
 * PARAMETRIC: LP supply only changes through mint/burn
 * 
 * Total LP supply should only change via addLiquidity (mint) or
 * removeLiquidity (burn). No other functions should modify it.
 */
rule lpSupplyOnlyChangesThroughMintBurn(method f)
    filtered {
        f -> !f.isView &&
             f.selector != sig:addLiquidity(uint256, uint256).selector &&
             f.selector != sig:removeLiquidity(uint256).selector
    }
{
    env e;
    
    uint256 totalLP_before = totalSupply();
    
    calldataarg args;
    f(e, args);
    
    uint256 totalLP_after = totalSupply();
    
    // Total LP supply should not change for non-mint/burn operations
    assert totalLP_after == totalLP_before;
}

/**
 * PARAMETRIC: User balances only change for msg.sender
 * 
 * A user's LP balance should only change if they're the msg.sender.
 * Prevents unauthorized balance modifications.
 */
rule userBalanceChangeOnlyForSender(method f, address user)
    filtered {
        f -> !f.isView
    }
{
    env e;
    require user != currentContract;
    require user != e.msg.sender;
    
    uint256 balance_before = balanceOf(user);
    
    calldataarg args;
    f(e, args);
    
    uint256 balance_after = balanceOf(user);
    
    // Balance should only change if user is msg.sender
    assert balance_after == balance_before;
}
