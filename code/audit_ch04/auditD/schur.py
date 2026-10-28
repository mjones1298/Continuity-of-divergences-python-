import numpy as np, cvxpy as cp

def schur_norm(M, symmetric=None, eps=1e-9):
    """Schur multiplier norm of real matrix M via Haagerup SDP:
       min t s.t. [[Y1,M],[M^T,Y2]]>=0, diag(Y1),diag(Y2)<=t."""
    M = np.array(M, dtype=float)
    n, m = M.shape
    if symmetric is None:
        symmetric = (n == m) and np.allclose(M, M.T)
    t = cp.Variable()
    if symmetric:
        Y = cp.Variable((n, n), symmetric=True)
        cons = [Y + M >> 0, Y - M >> 0, cp.diag(Y) <= t]
    else:
        Y1 = cp.Variable((n, n), symmetric=True)
        Y2 = cp.Variable((m, m), symmetric=True)
        B = cp.bmat([[Y1, M], [M.T, Y2]])
        cons = [B >> 0, cp.diag(Y1) <= t, cp.diag(Y2) <= t]
    prob = cp.Problem(cp.Minimize(t), cons)
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
    except Exception:
        prob.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
    return float(t.value)

def divdiff(g, gp, pts):
    pts = np.asarray(pts, float)
    n = len(pts)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if abs(pts[i]-pts[j]) < 1e-13:
                M[i, j] = gp(pts[i])
            else:
                M[i, j] = (g(pts[i])-g(pts[j]))/(pts[i]-pts[j])
    return M

if __name__ == "__main__":
    print(schur_norm([[1,1],[1,-1]]), np.sqrt(2))
    A = np.random.randn(5,5); P = A@A.T
    print(schur_norm(P), P.diagonal().max())
    print(schur_norm(np.outer([1,2,3],[1,-1,.5])), 3.0)
