import numpy as np
import scipy.linalg as la

def LRA(A, r, method='svd', max_iter=16):
    if method=='svd':
        U, s, V = la.svd(A)
        return U[:,:r]*s[:r].reshape(1, -1), V[:r,:]
    if method=='eigen':
        w, vr = la.eig(A, right=True)
        ind = np.argpartition(np.abs(w), w.shape[0]-r)[-r:]
        return vr[:, ind]*w[ind], la.inv(vr)[ind, :]
    if method=='lfd':
        D = np.random.rand(r, A.shape[1])
        E = np.random.rand(A.shape[1], r)
        diff, i = np.inf, 0
        while la.norm(E@D - A) < diff and i < max_iter:
            diff = la.norm(E@D - A)
            E = A @ la.pinv(D)
            D = la.pinv(E) @ A
            i += 1
        return E, D
    raise ValueError('Unidentified method')
