# RL_lspi_svd
RL SVD feature policy iteration

For 50 iterations or until A_next is identical to A

1)Sample according to policy Yana

K=number of fetures

weights_list=np.zeros(n_actions,K)


2) For each action:
  2.1) Crate matrix A, A_next for each action
   
  2.2) Calculte low rank approximation

  2.3) Calculate by Woodbery Identity invers e for calculation of weights
  
  Lera
  
  weights_list[action,:]=weight

3) Update for agent our weight 
 
agent.set_weights(weights_list)

4) Sample according to agent policy

We should use flattened env.rander('rdg_array') saved as np.array as observation

Policy is agent.predict(obs)
Back to step1

Experiments with Bellman error -> Vlad
