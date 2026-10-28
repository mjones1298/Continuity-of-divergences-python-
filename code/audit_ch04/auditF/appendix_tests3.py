import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
rng = np.random.default_rng(15)
conns = make_conns()
def kraus_from_choi(J, dY, dX, tol=1e-10):
    w, V = np.linalg.eigh(herm(J))
    return [np.sqrt(w[i]) * V[:, i].reshape(dY, dX) for i in range(len(w)) if w[i] > tol]
def p1(Ks): return np.linalg.norm(sum(K.conj().T @ K for K in Ks), 2) if Ks else 0.0   # ||.||_{1->1} = ||Xi^*(1)||
def pinf(Ks): return np.linalg.norm(sum(K @ K.conj().T for K in Ks), 2) if Ks else 0.0 # ||.||_{inf->inf} = ||Xi(1)||
def p2(Ks):  # Hilbert-Schmidt 2->2: largest singular value of superoperator
    if not Ks: return 0.0
    return np.linalg.norm(sum(np.kron(K, K.conj()) for K in Ks), 2)
w1 = w2 = wA = 0
for trial in range(400):
    dY = rng.integers(1, 4); dX = rng.integers(1, 4)
    KP = random_kraus(dY, dX, rng.integers(1, dY*dX+1), rng); KQ = random_kraus(dY, dX, rng.integers(1, dY*dX+1), rng)
    JP = sum(np.outer(vec(K), vec(K).conj()) for K in KP); JQ = sum(np.outer(vec(K), vec(K).conj()) for K in KQ); JO = JP + JQ
    KO = kraus_from_choi(JO, dY, dX)
    w, V = np.linalg.eigh(JO); keep = w > 1e-10
    Wf = V[:, keep]*np.sqrt(w[keep]); Wi = V[:, keep]/np.sqrt(w[keep]); T = herm(Wi.conj().T@JP@Wi); wt = np.clip(np.linalg.eigvalsh(T), 0, 1)
    # A.7
    G = ka_frame_conn(JP, JQ, conns['geo'], tol=1e-12); KG = kraus_from_choi(G, dY, dX)
    for nm, nf in (('1', p1), ('inf', pinf), ('2', p2)):
        a, b, g = nf(KP), nf(KQ), nf(KG)
        if a*b > 0: w2 = max(w2, g**2/(a*b))
    # A.1
    for name in ['geo', 'harm', 'alpha0.3', 'alpha0.8', 'log', 'mix', 'arith', 'skew']:
        cn = conns[name]
        M = ka_frame_conn(JP, JQ, cn, tol=1e-12); KM = kraus_from_choi(M, dY, dX)
        gmax = cn.phi(wt).max()
        for nf in (p1, pinf, p2):
            rhs = gmax*nf(KO)
            if rhs > 0: w1 = max(w1, nf(KM)/rhs)
    # formula for p=1: ||PhiσPsi||_{1->1} = max_rho tr(g(T)^T Omega^c(rho)) = lambda_max( sum_kl g_kl V_l^* V_k ) in the Kraus frame of Omega
    cn = conns['geo']
    if KO:
        Vs = KO  # orthonormal-ish Kraus operators from eigen-decomp: W columns are sqrt(w_i) vec^{-1}(v_i)
        Wk = np.array([vec(k) for k in KO]).T
        Tk = herm(np.linalg.pinv(Wk)@JP@np.linalg.pinv(Wk).conj().T)
        wk, Uk = np.linalg.eigh(Tk); g = (Uk*cn.phi(np.clip(wk, 0, 1)))@Uk.conj().T
        Mop = sum(g[k, l]*Vs[l].conj().T@Vs[k] for k in range(len(Vs)) for l in range(len(Vs)))
        lhs = np.linalg.eigvalsh(herm(Mop))[-1]
        wA = max(wA, abs(lhs - p1(kraus_from_choi(ka_frame_conn(JP, JQ, cn, tol=1e-12), dY, dX))))
print("A.1 worst ratio ||PhiσPsi||_{p->p}/(max g * ||Omega||_{p->p}) over p in {1,2,inf}, 8 connections, 400 pairs:", w1)
print("A.7 worst ratio ||Phi#Psi||^2/(||Phi|| ||Psi||):", w2)
print("A.1 trace formula max_rho tr(g(T)^T Omega^c(rho)) vs ||Phi#Psi||_{1->1}: max discrepancy", wA)
