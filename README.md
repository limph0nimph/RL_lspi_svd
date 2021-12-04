# RL_lspi_svd
RL SVD feature policy iteration

For 50 iterations or until A_next is identical to A
1)Sample according to policy
2) Crate matrix A, A_next for eac action
3) Calculte low rank approximation
4) Calculate by Woodbery Identity invers e for calculation of weights
5) Update for agent our weight 
6) Sample according to agent policy
