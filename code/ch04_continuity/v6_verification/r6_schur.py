# Probe the conjecture  ||[sin(psi_i+psi_j)]||_Schur <= max_i |sin 2 psi_i|  (would give 2*gamma*||phi||) by SDP.
import numpy as np, cvxpy as cp
rng = np.random.default_rng(3)
worst = 0
for trial in range(300):
    n = rng.integers(2, 6)
    g = rng.uniform(0, 0.5)
    lam = 0.5 + g*rng.uniform(-1, 1, n); lam[0] = 0.5 + g*rng.choice([-1, 1])
    psi = np.pi/4 - np.arccos(np.sqrt(lam))
    K = np.sin(psi[:, None] + psi[None, :])
    # Haagerup: ||K||_S = min t s.t. [[X, K],[K^T, Y]] >= 0, diag(X), diag(Y) <= t
    X = cp.Variable((n, n), symmetric=True); Y = cp.Variable((n, n), symmetric=True); t = cp.Variable()
    cons = [cp.bmat([[X, K], [K.T, Y]]) >> 0, cp.diag(X) <= t, cp.diag(Y) <= t]
    cp.Problem(cp.Minimize(t), cons).solve(solver=cp.CLARABEL)
    worst = max(worst, t.value/np.max(np.abs(np.sin(2*psi))))
print('max ||K||_S / max|sin 2psi| =', worst)
