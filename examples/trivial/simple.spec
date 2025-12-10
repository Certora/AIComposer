methods {
    function the_magic_function() external returns (uint256);
}

rule my_test {
   env e;
   assert the_magic_function(e) % 4 == 0;
}
