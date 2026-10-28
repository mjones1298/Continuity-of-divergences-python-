import numpy as np, cvxpy as cp
from ad_fid import *
np.set_printoptions(precision=5, suppress=True, linewidth=150)
g, h = 0.5, 0.4
f = np.sqrt(g*h)+np.sqrt((1-g)*(1-h))
for n in [1, 2, 3, 4]:
    KM, LM = kraus_mat(kraus(g), n), kraus_mat(kraus(h), n)
    D = 4**n; r = KM.shape[1]
    Ky = [KM.reshape(D//2, 2, r)[:, y, :] for y in range(2)]
    Ly = [LM.reshape(D//2, 2, r)[:, y, :] for y in range(2)]
    C = cp.Variable((r, r)); t = cp.Variable()
    Hraw = sum(Ky[y] @ C.T @ Ly[y].T for y in range(2)); H = (Hraw+Hraw.T)/2
    cons = [cp.sigma_max(C) <= 1]
    Zs = {j: cp.Variable((4**(j-1), 4**(j-1)), symmetric=True) for j in range(2, n+1)}
    if n == 1: cons += [H - t*np.eye(2) >> 0]
    else:
        cons += [H - cp.kron(Zs[n], np.eye(2)) >> 0]
        for j in range(n, 2, -1):
            cons += [cp.partial_trace(Zs[j], [4**(j-1)//2, 2], axis=1) - cp.kron(Zs[j-1], np.eye(2)) >> 0]
        cons += [cp.partial_trace(Zs[2], [2, 2], axis=1) - t*np.eye(2) >> 0]
    pr = cp.Problem(cp.Maximize(t), cons)
    pr.solve(solver='CLARABEL', tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
    print("n=%d  t=%.14f  f^n=%.14f  diff=%.2e  status=%s" % (n, pr.value, f**n, pr.value-f**n, pr.status))
    if n <= 2:
        print(" C=\n", C.value)
        if n == 2:
            # is C a product?  reshape (l1 l2),(k1 k2) -> (l1 k1),(l2 k2) and check rank
            Cv = C.value.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
            print(" operator-Schmidt singular values of C:", np.linalg.svd(Cv, compute_uv=False))
