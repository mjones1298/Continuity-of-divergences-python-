import numpy as np, cvxpy as cp
from lib import *

def diamond_herm(J, dY, dX, solver=cp.CLARABEL):
    n = dY*dX
    Y0 = cp.Variable((n,n), hermitian=True); Y1 = cp.Variable((n,n), hermitian=True)
    def ptr(Y):
        # partial trace over Y of operator on Y (x) X : sum_a Y[a*dX:(a+1)*dX, a*dX:(a+1)*dX]
        return sum(Y[a*dX:(a+1)*dX, a*dX:(a+1)*dX] for a in range(dY))
    t0 = cp.Variable(); t1 = cp.Variable()
    cons = [cp.bmat([[Y0, -J],[-J.conj().T, Y1]]) >> 0, Y0 >> 0, Y1 >> 0,
            cp.lambda_max(ptr(Y0)) <= t0, cp.lambda_max(ptr(Y1)) <= t1]
    prob = cp.Problem(cp.Minimize((t0+t1)/2), cons)
    prob.solve(solver=solver)
    return prob.value

def choi_of_PsiK(W, K):
    return W @ K @ W.conj().T

def nu_est(Ks, restarts=40, rng=np.random.default_rng(0)):
    from scipy.optimize import minimize
    r = len(Ks)
    def obj(x):
        c = x[:r] + 1j*x[r:]; c = c/np.linalg.norm(c)
        V = sum(ci*K for ci,K in zip(c,Ks))
        return np.linalg.norm(V, 2)
    best = 1e9
    for _ in range(restarts):
        x0 = rng.normal(size=2*r)
        res = minimize(obj, x0, method='Powell', options={'xtol':1e-10,'ftol':1e-14,'maxiter':4000})
        best = min(best, res.fun)
    return best   # upper bound on nu (>= true nu)
