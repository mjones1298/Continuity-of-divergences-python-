# double-precision helpers + diamond-norm SDP (Watrous) for trace-annihilating differences
import numpy as np, cvxpy as cp
def vec(A): return A.reshape(-1)
def choi(Ks):
    return sum(np.outer(vec(K), vec(K).conj()) for K in Ks)
def ptrace_Y(J, dY, dX):
    return np.einsum('yayb->ab', J.reshape(dY, dX, dY, dX))
def diamond_ta(D, dY, dX):
    """|||Delta|||_diamond for Hermitian-preserving, trace-annihilating Delta with Choi D (on Y(x)X)."""
    n = dY*dX
    W = cp.Variable((n, n), hermitian=True); rho = cp.Variable((dX, dX), hermitian=True)
    cons = [W >> 0, cp.kron(np.eye(dY), rho) - W >> 0, rho >> 0, cp.real(cp.trace(rho)) == 1]
    pr = cp.Problem(cp.Maximize(cp.real(cp.trace(D @ W))), cons)
    pr.solve(solver=cp.CLARABEL)
    return 2*pr.value
def rand_iso(dX, dY, k, rng):
    G = rng.normal(size=(k*dY, dX)) + 1j*rng.normal(size=(k*dY, dX))
    Q, _ = np.linalg.qr(G)
    return [Q[i*dY:(i+1)*dY, :] for i in range(k)]
