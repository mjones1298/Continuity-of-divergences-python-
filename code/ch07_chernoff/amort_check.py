import numpy as np, scipy.linalg as sla
from scipy.optimize import minimize
from amort_test import kraus, state, apply, ratio
import mpmath as mp
rng = np.random.default_rng(3)
g, h, dR = 0.5, 0.2, 2; d = 2*dR
best = None
for trial in range(20):
    r = minimize(ratio, rng.normal(size=4*d*d), args=(g, h, dR), method='L-BFGS-B', options={'maxiter': 3000})
    if best is None or r.fun < best.fun: best = r
x = best.x; rho = state(x[:2*d*d], d); sig = state(x[2*d*d:], d)
def rootF_mp(A, B):
    A = mp.matrix(A.tolist()); B = mp.matrix(B.tolist())
    sA = mp.sqrtm(A); M = sA*B*sA; M = (M + M.H)/2
    ev = mp.eig(M)[0]; return sum(mp.sqrt(max(mp.re(e), 0)) for e in ev)
mp.mp.dps = 40
Fin = rootF_mp(rho, sig); Fout = rootF_mp(apply(kraus(g), rho, dR), apply(kraus(h), sig, dR))
f = np.sqrt(g*h)+np.sqrt((1-g)*(1-h))
print("sqrtF_in=%s sqrtF_out=%s ratio=%s f=%.10f" % (mp.nstr(Fin, 12), mp.nstr(Fout, 12), mp.nstr(Fout/Fin, 12), f))
print("eig rho", np.round(np.linalg.eigvalsh(rho), 4), "eig sig", np.round(np.linalg.eigvalsh(sig), 4))
def psd_sqrt(A):
    A = mp.matrix(A.tolist()); A = (A + A.H)/2
    E, Q = mp.eighe(A)
    D = mp.diag([mp.sqrt(max(mp.re(e), 0)) for e in E])
    return Q*D*Q.H
def rootF_eig(A, B):
    X = psd_sqrt(A)*psd_sqrt(B)
    s = mp.svd_c(X, compute_uv=False)
    return sum(s)
Fin2 = rootF_eig(rho, sig); Fout2 = rootF_eig(apply(kraus(g), rho, dR), apply(kraus(h), sig, dR))
print("eig-based: sqrtF_in=%s sqrtF_out=%s ratio=%s" % (mp.nstr(Fin2, 15), mp.nstr(Fout2, 15), mp.nstr(Fout2/Fin2, 15)))
