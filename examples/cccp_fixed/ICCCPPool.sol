// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface ICCCPPool {

    function lpBalanceOf(address user) external returns (uint256);
    /**
    @dev Returns the current K, the product of reserveA() and reserveB()
     */
    function calculateK() external returns (uint256);

    /**
    @dev previews the swap fee for amountOut given reserve supply
     */
    function getSwapFee(uint256 amountOut, uint256 reserve) view external returns (uint256);

    /**
    @dev Calculates the withdrawal penalty for burning lpTokens given a total LP supply
    @param lpTokens The number of LP tokens being withdrawn
    @param totalLP The total LP supply to use for the calculation (typically pre-withdrawal supply)
    @return penaltyBps The penalty in basis points (0-10000)
     */
    function calculateWithdrawalPenalty(uint256 lpTokens, uint256 totalLP) pure external returns (uint256);

    /**
    @dev swaps amountIn from reserve A to B or B to A, depending on aToB.
    @notice The caller must approve transferring amountIn of the relevant token to the pool
    @return amountOut The amount of reserve A or B resulting from the swap, minus fees
    */
    function swap(uint256 amountIn, bool aToB) external returns (uint256 amountOut);

    /**
    @dev Add reserves to the pool in exchange for liquidity tokens. Guaranteed to provide as much
    liquidity as possible without breaking the pool invariant.
    @param amountA The maximum amount of reserve A to contribute
    @param amountB The maximum amount of reserve B to contribute
    @return lpMinted the amount of lp minted.
    @notice The user must have approved transferring amountA from reserve A to the pool,
    and ditto amountB
     */
    function addLiquidity(uint256 amountA, uint256 amountB) external returns (uint256 lpMinted);


    /**
    @dev Burns tokens in exchange for underlying liquidity, less withdrawal penalties
    @param lpTokens The number of tokens to burn, the user balance must have at least this amount.
     */
    function removeLiquidity(uint256 lpTokens) external returns (uint256, uint256);

    /**
    @dev returns the total circulating supply of LP tokens
    @return supply the total supply of LP */
    function totalSupply() external returns (uint256 supply);

    /**
    @dev returns the balance LP of the given user
    */
    function balanceOf(address user) external returns (uint256);
}