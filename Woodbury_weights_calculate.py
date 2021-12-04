import numpy as np


def Woodbury(gamma, US, V):
    '''
    gamma : scalar
           discounted infinite-horizon return
    US : matrix k x r
        U @ Sigma from SVD(Prob_trans) = U @ Sigma @ V
    V : matrix r x k
        V from SVD(Prob_trans)

    Return : matrix k x k
             inverse(I - gamma * u @ v)
    '''
    k = US.shape[0]
    r = US.shape[1]
    I_k = np.eye(k)
    I_r = np.eye(r)
    return I_k + gamma * US @ np.linalg.inv(I_r - gamma * V @ US) @ V


def calculate_weights(Inv_Woodbury, r):
    '''
    Inv_Woodbury : matrix k x k
                   Inv(I - gamma* US @ V) calculated by Woodbury
    r : vector k x 1
        reward vector

    Return : vector k x 1
             weights of the importance of features
    '''
    return Inv_Woodbury @ r
