import numpy as np
from lib import *
rng = np.random.default_rng(7)
fs = {
 'BS t log t': lambda t: np.where(t>0, t*np.log(np.where(t>0,t,1)), 0.0),
 't^1.5': lambda t: t**1.5,
 't^2': lambda t: t**2,
 'chi2 (t-1)^2': lambda t: (t-1)**2,
}
worst = 0
for name, f in fs.items():
    h = h_of(f)
    for trial in range(40):
        dX = rng.integers(1,4); dY = rng.integers(1,4)
        rmax = dX*dY
        r = rng.integers(1, rmax+1)
        Ks = rand_kraus(dX, dY, r)
        W = W_of(Ks)
        if np.linalg.matrix_rank(W) < r: continue
        eps = 0.05
        spec = rng.uniform(0, 1-eps, size=r)
        if trial % 3 == 0: spec[0] = 0.0
        T = rand_T(r, spec=spec)
        M = M_of(W, T, lambda t: h(np.asarray(t, float)), dY, dX)
        # random invertible A
        A = rng.normal(size=(dX,dX)) + 1j*rng.normal(size=(dX,dX)); A /= np.linalg.norm(A)
        P1, P2 = phi_state(W, T, A, dY, dX)
        # direct (eta-regularised)
        vals = [Dhat_reg(P1, P2, f, eta) for eta in (1e-6, 1e-8)]
        sup, res = Dhat_support(P1, P2, f)
        xi = (A @ A.conj().T).T
        pred = np.trace(xi @ M).real
        err = max(abs(vals[-1]-pred), abs(sup-pred))
        worst = max(worst, err/ max(1,abs(pred)))
        if err/max(1,abs(pred)) > 1e-5:
            print('MISMATCH', name, dX, dY, r, vals, sup, pred, res)
print('worst rel err', worst)
