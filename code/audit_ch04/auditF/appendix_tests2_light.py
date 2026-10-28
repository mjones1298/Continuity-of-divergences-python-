import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
from scipy.optimize import minimize
rng = np.random.default_rng(12)
conns = make_conns()

def kraus_from_choi(J, dY, dX, tol=1e-10):
    w, V = np.linalg.eigh(herm(J))
    return [np.sqrt(w[i]) * V[:, i].reshape(dY, dX) for i in range(len(w)) if w[i] > tol]

def apply(Ks, X): return sum(K @ X @ K.conj().T for K in Ks)
def schp(M, p):
    s = np.linalg.svd(M, compute_uv=False)
    return s.max() if p == np.inf else (s**p).sum()**(1/p)

def pnorm_cp(Ks, dY, dX, p, restarts=4):
    if not Ks: return 0.0
    if p == 1: return np.linalg.norm(sum(K.conj().T @ K for K in Ks), 2)
    if p == np.inf: return np.linalg.norm(sum(K @ K.conj().T for K in Ks), 2)
    best = 0
    def fn(x):
        B = (x[:dX*dX] + 1j*x[dX*dX:]).reshape(dX, dX); X = B @ B.conj().T
        nx = schp(X, p)
        return -schp(apply(Ks, X), p)/nx
    for s in range(restarts):
        x0 = rng.normal(size=2*dX*dX)
        res = minimize(fn, x0, method='BFGS')
        best = max(best, -res.fun)
    return best

def pnorm_general_2(Ks, dY, dX):
    # superoperator on vec(X) with Hilbert-Schmidt norm
    S = sum(np.kron(K, K.conj()) for K in Ks)   # acts on row-major vec: vec(KXK^*) = (K (x) conj(K)) vec(X)
    return np.linalg.norm(S, 2)

def cb_Ks(Ks, n): return [np.kron(K, np.eye(n)) for K in Ks]

ps = [1, 1.5, 2, 3, np.inf]
dY, dX = 2, 3
worstA1 = 0; worstA7 = 0; worstA7cb = 0
for trial in range(5):
    r1, r2 = rng.integers(1, 4, size=2)
    KP = random_kraus(dY, dX, r1, rng); KQ = random_kraus(dY, dX, r2, rng)
    JP = sum(np.outer(vec(K), vec(K).conj()) for K in KP); JQ = sum(np.outer(vec(K), vec(K).conj()) for K in KQ)
    JO = JP + JQ
    # ---- A.7 : ||Phi#Psi||^2 <= ||Phi|| ||Psi|| , and cb (n=2) version
    G = ka_frame_conn(JP, JQ, conns['geo']); KG = kraus_from_choi(G, dY, dX)
    for p in ps:
        a = pnorm_cp(KP, dY, dX, p); b = pnorm_cp(KQ, dY, dX, p); g = pnorm_cp(KG, dY, dX, p)
        worstA7 = max(worstA7, g**2/(a*b))
    for p in [1.5, 2, 3]:
        n = 2
        a = pnorm_cp(cb_Ks(KP, n), dY*n, dX*n, p, restarts=3) ; b = pnorm_cp(cb_Ks(KQ, n), dY*n, dX*n, p, restarts=3); g = pnorm_cp(cb_Ks(KG, n), dY*n, dX*n, p, restarts=3)
        # note pnorm_cp with dX*n for X dims: need dX*n-dim input: handled by passing dX*n
        worstA7cb = max(worstA7cb, g**2/(a*b))
    # ---- A.1 : ||Phi sigma Psi||_{p->p} <= ||g_sigma(T)||_inf ||Omega||_{p->p}
    S = JO; w, V = np.linalg.eigh(S); keep = w > 1e-10
    Wf = V[:, keep]*np.sqrt(w[keep]); Wi = V[:, keep]/np.sqrt(w[keep])
    T = herm(Wi.conj().T @ JP @ Wi); wt = np.linalg.eigvalsh(T)
    for name in ['geo', 'harm', 'alpha0.3', 'alpha0.8', 'log', 'mix']:
        cn = conns[name]
        M = ka_frame_conn(JP, JQ, cn)
        KM = kraus_from_choi(M, dY, dX); KO = kraus_from_choi(JO, dY, dX)
        gmax = cn.phi(np.clip(wt, 0, 1)).max()
        for p in ps:
            lhs = pnorm_cp(KM, dY, dX, p) if KM else 0
            rhs = gmax * pnorm_cp(KO, dY, dX, p)
            worstA1 = max(worstA1, lhs/rhs)
print("A.1 worst ||PhiσPsi||/(max g * ||Omega||) :", worstA1, " (<=1 expected)")
print("A.7 worst ||Phi#Psi||^2/(||Phi|| ||Psi||) :", worstA7, " (<=1 expected)")
print("A.7 cb(n=2) worst:", worstA7cb)
