import numpy as np, sys, cvxpy as cp
sys.path.insert(0,'.')
from lib import *
from scipy.optimize import minimize
rng = np.random.default_rng(11)
geo = make_conns()['geo']

def rand_psd(n, rank, rng):
    G = rng.normal(size=(n, rank)) + 1j*rng.normal(size=(n, rank)); return G@G.conj().T

# ---------------- A.3 Ando maximality ----------------
print("== Lemma A.3 (Ando max) ==")
worst = 0
for trial in range(30):
    n = 4
    ra, rb = rng.integers(1, 5, size=2)
    A = rand_psd(n, ra, rng); B = rand_psd(n, rb, rng)
    G = ka_frame_conn(A, B, geo)
    # feasibility of A#B: min eig of block
    blk = np.block([[A, G], [G, B]])
    me = np.linalg.eigvalsh(herm(blk))[0]
    # SDP: maximise v^* X v over feasible X
    v = rng.normal(size=n) + 1j*rng.normal(size=n); v /= np.linalg.norm(v)
    Xv = cp.Variable((n, n), hermitian=True)
    prob = cp.Problem(cp.Maximize(cp.real(v.conj() @ Xv @ v)), [cp.bmat([[A, Xv], [Xv, B]]) >> 0])
    prob.solve(solver=cp.SCS, eps=1e-9, max_iters=100000)
    worst = max(worst, abs(prob.value - np.real(v.conj() @ G @ v)), -me)
print("max |SDP max v*Xv - v*(A#B)v| and block-infeasibility:", worst)

# --------------- A.6 block CS in ui norms ----------------
print("== Lemma A.6 (block Cauchy-Schwarz) ==")
def sv(M): return np.linalg.svd(M, compute_uv=False)
def kyfan(M, k): return sv(M)[:k].sum()
def schat(M, p): s = sv(M); return s.max() if p == np.inf else (s**p).sum()**(1/p)
def uis(M, n):
    out = {('KF', k): kyfan(M, k) for k in range(1, n+1)}
    for p in [1, 1.5, 2, 3, 6, np.inf]: out[('S', p)] = schat(M, p)
    # (p,k)-norms
    for p in [2, 3]:
        for k in range(1, n+1): out[('pk', p, k)] = (sv(M)[:k]**p).sum()**(1/p)
    return out
worst = 0
for trial in range(300):
    n = rng.integers(2, 6)
    A = rand_psd(n, rng.integers(1, n+1), rng); B = rand_psd(n, rng.integers(1, n+1), rng)
    C = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)); C = C/np.linalg.norm(C, 2)*rng.uniform(0.2, 1)
    X = mpow(A, .5) @ C @ mpow(B, .5)
    blk = np.block([[A, X], [X.conj().T, B]]); assert np.linalg.eigvalsh(herm(blk))[0] > -1e-9
    nx, na, nb = uis(X, n), uis(A, n), uis(B, n)
    for key in nx:
        worst = max(worst, nx[key]**2/(na[key]*nb[key]))
    G = ka_frame_conn(A, B, geo); ng = uis(G, n)
    for key in ng:
        worst = max(worst, ng[key]**2/(na[key]*nb[key]))
print("max ratio ||X||^2/(||A|| ||B||) over ui norms (should be <= 1):", worst)
# adversarial: A=B and X=A (C=I)
A = rand_psd(4, 4, rng); X = A
print("equality case A=B=X:", max(uis(X, 4)[k]**2/(uis(A, 4)[k]**2) for k in uis(X, 4)))
