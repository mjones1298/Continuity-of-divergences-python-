# Sharp constant c in ||J(Delta)||_inf <= c |||Delta|||_diamond
import numpy as np
from npl import *
rng = np.random.default_rng(2)
print('validation: U=diag(1,e^{ia}), a=0.7: SDP', diamond_ta(choi([np.eye(2)])-choi([np.diag([1,np.exp(0.7j)])]),2,2), ' exact', 2*np.sin(0.35))
for dX in [2, 3, 4]:
    best = 0
    # unitary examples: spectrum split +-alpha
    for k in range(1, dX):
        for a in [0.05, 0.3, 1.0]:
            U = np.diag([np.exp(1j*a)]*k + [np.exp(-1j*a)]*(dX-k))
            D = choi([np.eye(dX)]) - choi([U])
            r = np.linalg.norm(D, 2)/diamond_ta(D, dX, dX)
            best = max(best, r)
    # random channel pairs (various Kraus ranks) and small perturbations
    rbest = 0
    for _ in range(40):
        dY = rng.integers(2, 4) if dX < 4 else 2
        kmin = -(-dX // dY); k1, k2 = rng.integers(kmin, kmin+3, size=2)
        D = choi(rand_iso(dX, dY, k1, rng)) - choi(rand_iso(dX, dY, k2, rng))
        rbest = max(rbest, np.linalg.norm(D, 2)/diamond_ta(D, dY, dX))
    print(f'dX={dX}: unitary-split examples max ratio {best:.6f}; random pairs max ratio {rbest:.4f}; '
          f'proved upper bound dX/2={dX/2}; sqrt(dX^2-1)/2={np.sqrt(dX**2-1)/2:.6f}')
