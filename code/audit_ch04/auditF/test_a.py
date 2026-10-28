import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
from diamond import diamond_hp
from scipy.optimize import minimize
rng = np.random.default_rng(2)
dY=dX=2
geo = make_conns()['geo']

def Jof(Ks): return sum(np.outer(vec(K), vec(K).conj()) for K in Ks)
def Fhat(JP, JQ):
    M = ka_frame_conn(JP, JQ, geo, tol=1e-12)
    return np.linalg.eigvalsh(herm(ptrace_Y(M, dY, dX)))[0]
def nu_of(JO):
    # nu(Omega) = min over unit c of ||sum c_k V_k||_inf ; Kraus operators from eigendecomposition of J
    w, V = np.linalg.eigh(JO)
    Ks = [np.sqrt(w[i]) * V[:, i].reshape(dY, dX) for i in range(len(w)) if w[i] > 1e-12]
    r = len(Ks)
    best = 9
    for s in range(60):
        x0 = rng.normal(size=2*r)
        def fn(x):
            c = x[:r] + 1j*x[r:]; c = c/np.linalg.norm(c)
            return np.linalg.norm(sum(ci*K for ci, K in zip(c, Ks)), 2)
        res = minimize(fn, x0, method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-12, maxiter=3000))
        best = min(best, res.fun)
    return best
def Omstar1(J):  # ||Omega^*(1_Y)||_inf = ||Tr_Y J||
    return np.linalg.norm(ptrace_Y(J, dY, dX), 2)

worst = 0
for trial in range(25):
    KP = random_kraus(dY, dX, 2, rng); KQ = random_kraus(dY, dX, 2, rng)
    JP, JQ = Jof(KP), Jof(KQ)
    JO = JP + JQ
    assert np.linalg.eigvalsh(JO)[0] > 1e-3
    nu = nu_of(JO)
    for eps in [0.01, 0.1, 0.5, 2.0]:
        KP2 = [K + eps*(rng.normal(size=K.shape)+1j*rng.normal(size=K.shape)) for K in KP]
        KQ2 = [K + eps*(rng.normal(size=K.shape)+1j*rng.normal(size=K.shape)) for K in KQ]
        JP2, JQ2 = Jof(KP2), Jof(KQ2)
        dcb = max(diamond_hp(JP - JP2, dY, dX), diamond_hp(JQ - JQ2, dY, dX))
        Obar = 0.5*(JO + JP2 + JQ2)
        rhs = 11/4*Omstar1(Obar)/nu*np.sqrt(dcb)
        lhs = abs(Fhat(JP, JQ) - Fhat(JP2, JQ2))
        worst = max(worst, lhs/rhs)
    if trial % 5 == 4: print("trial", trial, "worst lhs/rhs so far", worst)
print("worst ratio", worst)
