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
    // Core operations
    // ========================================================================
    function addLiquidity(uint256 amountA, uint256 amountB) external returns (uint256);
    function removeLiquidity(uint256 lpTokens) external returns (uint256, uint256);
    function swap(uint256 amountIn, bool aToB) external returns (uint256);
    
    // ========================================================================
    // View functions
    // ========================================================================
    function calculateK() external returns (uint256) envfree;
    function calculateWithdrawalPenalty(uint256 lpTokens) external returns (uint256) envfree;
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
ghost uint256 sumOfLPBalances {
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
    (uint256 oldBalance) STORAGE {
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
 */
invariant kNeverDecreases()
    calculateK() >= ghostK
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
    uint256 expectedA = (lpTokens * reserveA_before) / totalLP;
    uint256 expectedB = (lpTokens * reserveB_before) / totalLP;
    
    uint256 amountA;
    uint256 amountB;
    amountA, amountB = removeLiquidity(e, lpTokens);
    
    // User gets AT MOST the proportion (rounded down)
    assert amountA <= expectedA;
    assert amountB <= expectedB;
    
    // Within rounding error (at most 1 smallest unit less per token)
    assert expectedA - amountA <= 1;
    assert expectedB - amountB <= 1;
    
    // No penalty applied
    uint256 penalty = calculateWithdrawalPenalty(lpTokens);
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
    
    uint256 expectedA = (lpTokens * reserveA_before) / totalLP;
    uint256 expectedB = (lpTokens * reserveB_before) / totalLP;
    
    uint256 amountA;
    uint256 amountB;
    amountA, amountB = removeLiquidity(e, lpTokens);
    
    // Should receive LESS than proportional
    assert amountA < expectedA;
    assert amountB < expectedB;
    
    // Penalty is non-zero
    uint256 penalty = calculateWithdrawalPenalty(lpTokens);
    assert penalty > 0;
    
    // Difference is at least the penalty (could be more with rounding)
    uint256 penaltyA = (expectedA * penalty) / 10000;
    uint256 penaltyB = (expectedB * penalty) / 10000;
    assert expectedA - amountA >= penaltyA;
    assert expectedB - amountB >= penaltyB;
}

/**
 * RULE: Penalties + rounding dust redistribute to remaining LPs
 * 
 * When someone withdraws with a penalty, remaining LP token holders
 * should see their tokens increase in value. This is the core
 * redistribution mechanism.
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
    require reserveA_before < max_uint256 / 1e18;
    require reserveB_before < max_uint256 / 1e18;
    
    // Value per LP token before (scaled by 1e18 for precision)
    // Since both tokens have 18 decimals, this scaling maintains precision
    uint256 valuePerLP_A_before = (reserveA_before * 1e18) / totalLP_before;
    uint256 valuePerLP_B_before = (reserveB_before * 1e18) / totalLP_before;
    
    // Withdrawer removes liquidity with penalty
    uint256 lpToWithdraw = balanceOf(withdrawer);
    require lpToWithdraw > 0;
    require lpToWithdraw <= totalLP_before;
    
    // Large withdrawal triggering penalty
    require lpToWithdraw * 100 >= totalLP_before * 5;
    
    uint256 withdrawn_A;
    uint256 withdrawn_B;
    withdrawn_A, withdrawn_B = removeLiquidity(e, lpToWithdraw);
    
    uint256 totalLP_after = totalSupply();
    uint256 reserveA_after = reserveA();
    uint256 reserveB_after = reserveB();
    
    // Staker still has LP tokens
    require totalLP_after > 0;
    require reserveA_after < max_uint256 / 1e18;
    require reserveB_after < max_uint256 / 1e18;
    
    // Value per LP token after (scaled by 1e18 for precision)
    // Since both tokens have 18 decimals, this scaling maintains precision
    uint256 valuePerLP_A_after = (reserveA_after * 1e18) / totalLP_after;
    uint256 valuePerLP_B_after = (reserveB_after * 1e18) / totalLP_after;
    
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
    require lpTokens2 <= totalSupply();
    
    uint256 penalty1 = calculateWithdrawalPenalty(lpTokens1);
    uint256 penalty2 = calculateWithdrawalPenalty(lpTokens2);
    
    assert penalty2 >= penalty1;
}

// ============================================================================
// SWAP RULES - Progressive Fees
// ============================================================================

/**
 * RULE: Swap fees are progressive
 * 
 * Larger swaps (as % of reserve) should incur higher fee percentages
 * to discourage pool drainage via swaps.
 */
rule swapFeeIsProgressive(uint256 amountOut1, uint256 amountOut2, uint256 reserve) {
    require amountOut1 < amountOut2;
    require amountOut2 < reserve;
    require reserve > 0;
    require amountOut1 > 0;
    
    uint256 fee1 = getSwapFee(amountOut1, reserve);
    uint256 fee2 = getSwapFee(amountOut2, reserve);
    
    // Fee as percentage of output (scaled to prevent rounding issues)
    uint256 feePercent1 = (fee1 * 10000) / amountOut1;
    uint256 feePercent2 = (fee2 * 10000) / amountOut2;
    
    // Larger swap has higher fee percentage
    assert feePercent2 >= feePercent1;
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
    uint256 totalValueA_after = reserveA_after + amountA;
    uint256 totalValueB_after = reserveB_after + amountB;
    
    // With protocol-favoring rounding, total value >= original (dust stays)
    assert totalValueA_after >= reserveA_before;
    assert totalValueB_after >= reserveB_before;
    
    // But not too much more (just rounding dust, max 1 smallest unit)
    assert totalValueA_after - reserveA_before <= 1;
    assert totalValueB_after - reserveB_before <= 1;
}

/**
 * RULE: LP value never decreases without withdrawal
 * 
 * If a user doesn't withdraw, their LP tokens should maintain or increase
 * value due to fees from others' activity.
 */
rule lpValueNeverDecreasesWithoutWithdrawal(
    env e, 
    address user, 
    method f
) {
    require user != currentContract;
    require f.selector != sig:removeLiquidity(uint256).selector;
    
    uint256 lpBalance_before = balanceOf(user);
    require lpBalance_before > 0;
    
    uint256 totalLP_before = totalSupply();
    require totalLP_before > 0;
    
    uint256 valueA_before = (lpBalance_before * reserveA()) / totalLP_before;
    uint256 valueB_before = (lpBalance_before * reserveB()) / totalLP_before;
    
    calldataarg args;
    f(e, args);
    
    uint256 lpBalance_after = balanceOf(user);
    
    // If user didn't lose LP tokens, value per token shouldn't decrease
    if (lpBalance_after >= lpBalance_before) {
        uint256 totalLP_after = totalSupply();
        require totalLP_after > 0;
        
        uint256 valueA_after = (lpBalance_after * reserveA()) / totalLP_after;
        uint256 valueB_after = (lpBalance_after * reserveB()) / totalLP_after;
        
        // Value increases or stays same (from fees and dust)
        assert valueA_after >= valueA_before;
        assert valueB_after >= valueB_before;
    }
}


/**
 * PARAMETRIC: No privilege escalation
 * 
 * Users cannot gain more value than they should through any function.
 * Value can only increase from earning fees, not from exploits.
 */
rule noPrivilegeEscalation(method f, address user)
    filtered {
        f -> !f.isView
    }
{
    env e;
    require user != currentContract;
    require e.msg.sender == user;
    
    uint256 lpBefore = balanceOf(user);
    uint256 totalLPBefore = totalSupply();
    
    // Calculate value before (only if pool has LP tokens)
    uint256 valueA_before;
    uint256 valueB_before;
    if (totalLPBefore > 0) {
        require reserveA() < max_uint256 / lpBefore;
        require reserveB() < max_uint256 / lpBefore;
        valueA_before = (lpBefore * reserveA()) / totalLPBefore;
        valueB_before = (lpBefore * reserveB()) / totalLPBefore;
    } else {
        valueA_before = 0;
        valueB_before = 0;
    }
    
    calldataarg args;
    f(e, args);
    
    uint256 lpAfter = balanceOf(user);
    uint256 totalLPAfter = totalSupply();
    
    // Calculate value after
    uint256 valueA_after;
    uint256 valueB_after;
    if (totalLPAfter > 0 && lpAfter > 0) {
        require reserveA() < max_uint256 / lpAfter;
        require reserveB() < max_uint256 / lpAfter;
        valueA_after = (lpAfter * reserveA()) / totalLPAfter;
        valueB_after = (lpAfter * reserveB()) / totalLPAfter;
    } else {
        valueA_after = 0;
        valueB_after = 0;
    }
    
    // If LP balance increased or stayed same (and not withdrawing),
    // value per LP shouldn't decrease
    assert f.selector == sig:removeLiquidity(uint256).selector || 
           lpAfter >= lpBefore => 
           (valueA_after >= valueA_before && valueB_after >= valueB_before);
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

/**
 * PARAMETRIC: Reserves only change through core operations
 * 
 * Reserves should only be modified by addLiquidity, removeLiquidity,
 * and swap. Administrative or helper functions shouldn't touch reserves.
 */
rule reservesOnlyChangeThroughCoreOps(method f)
    filtered {
        f -> !f.isView &&
             f.selector != sig:addLiquidity(uint256, uint256).selector &&
             f.selector != sig:removeLiquidity(uint256).selector &&
             f.selector != sig:swap(uint256, bool).selector
    }
{
    env e;
    
    uint256 reserveA_before = reserveA();
    uint256 reserveB_before = reserveB();
    
    calldataarg args;
    f(e, args);
    
    uint256 reserveA_after = reserveA();
    uint256 reserveB_after = reserveB();
    
    // Reserves should not change for non-core operations
    assert reserveA_after == reserveA_before;
    assert reserveB_after == reserveB_before;
}
