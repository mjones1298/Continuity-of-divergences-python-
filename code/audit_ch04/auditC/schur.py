import numpy as np, cvxpy as cp

def schur_norm(M, solver=None):
    n = M.shape[0]
    t = cp.Variable()
    Y1 = cp.Variable((n,n), symmetric=True); Y2 = cp.Variable((n,n), symmetric=True)
    X = cp.bmat([[Y1, M],[M.T, Y2]])
    cons = [X >> 0, cp.diag(Y1) <= t, cp.diag(Y2) <= t]
    prob = cp.Problem(cp.Minimize(t), cons)
    prob.solve(solver=solver or cp.CLARABEL)
    return t.value

def divdiff(g, dg, lam, tol=1e-12):
    n = len(lam); M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if abs(lam[i]-lam[j]) < tol: M[i,j] = dg(lam[i])
            else: M[i,j] = (g(lam[i]) - g(lam[j]))/(lam[i]-lam[j])
    return M

def lip_grid(g, dg, a, b, n=40):
    lam = np.linspace(a, b, n)
    return schur_norm(divdiff(g, dg, lam))
