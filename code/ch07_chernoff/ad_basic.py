"""Basic quantities for AD_g vs AD_g': level decomposition, single-letter converses,
mu_* (pure-program / scalar-overlap constant), product-input Chernoff exponents."""
import numpy as np, scipy.linalg as sla
from scipy.optimize import minimize_scalar, minimize
import cvxpy as cp

def kraus(g):
    return [np.array([[1,0],[0,np.sqrt(1-g)]]), np.array([[0,np.sqrt(g)],[0,0]])]

def vec(K):  # vec in Y (x) X, column stacking convention: vec(|y><x|) = |y>|x>
    return K.reshape(-1)          # row-major: index y*dX + x  -> |y>|x>

def choi(Ks):
    return sum(np.outer(vec(K), vec(K).conj()) for K in Ks)

def apply(Ks, rho):
    return sum(K@rho@K.conj().T for K in Ks)

def mpow(A, s):
    w, U = np.linalg.eigh((A+A.conj().T)/2); w = np.clip(w, 0, None)
    ws = np.where(w > 1e-14, w**s, 0.0)
    return (U*ws)@U.conj().T

def qc_Q(A, B, s):
    return np.real(np.trace(mpow(A, s)@mpow(B, 1-s)))

def xi_QC(A, B):
    r = minimize_scalar(lambda s: qc_Q(A, B, s), bounds=(0, 1), method='bounded', options={'xatol':1e-12})
    return -np.log(r.fun), r.x

def levels(KP, KQ):
    """ARN derivative of Phi wrt Phi+Psi via Kraus frame of Omega."""
    Vs = KP + KQ
    W = np.array([vec(V) for V in Vs]).T      # columns vec(V_k)
    # reduce to linearly independent frame
    U, s, Vh = np.linalg.svd(W, full_matrices=False)
    r = int((s > 1e-12).sum())
    Wr = U[:, :r]*s[:r]                       # J(Omega) = Wr Wr^*
    JP = choi(KP)
    Wp = np.linalg.pinv(Wr)
    T = Wp@JP@Wp.conj().T
    T = (T+T.conj().T)/2
    t, F = np.linalg.eigh(T)
    out = []
    for i in range(r):
        c = F[:, i]
        Kv = Wr@c                              # vec of K_i (common diag Kraus)
        K = Kv.reshape(2, 2)
        out.append((t[i], K.conj().T@K))
    return T, out

def Chat_info(KP, KQ):
    T, lv = levels(KP, KQ)
    amb = [(t, S) for t, S in lv if 1e-9 < t < 1-1e-9]
    def lam(s):
        B = sum(t**s*(1-t)**(1-s)*S for t, S in amb) if amb else np.zeros((2, 2))
        return np.linalg.eigvalsh(B)[0]
    vals = [lam(s) for s in np.linspace(0, 1, 101)]
    return lv, min(vals), lam(0.5)

def mu_star_closed(g, h):
    a = np.sqrt((1-g)*(1-h)); return min(1.0, np.sqrt(g*h)/(1-a))

def mu_star_sdp(KP, KQ, dE=None):
    """max mu s.t. sum_{lk} C_{lk} L_l^* K_k = mu*1, ||C||<=1 (C: E_Phi -> E_Psi)."""
    m, k = len(KQ), len(KP)
    C = cp.Variable((m, k), complex=True); mu = cp.Variable()
    M = sum(C[l, j]*(KQ[l].conj().T@KP[j]) for l in range(m) for j in range(k))
    cons = [M == mu*np.eye(KP[0].shape[1]), cp.sigma_max(C) <= 1]
    pr = cp.Problem(cp.Maximize(mu), cons); pr.solve(solver='CLARABEL')
    return mu.value, C.value

def bern_chernoff(p, q):
    f = lambda s: p**s*q**(1-s)+(1-p)**s*(1-q)**(1-s)
    r = minimize_scalar(f, bounds=(0, 1), method='bounded', options={'xatol':1e-12})
    return -np.log(r.fun)

def output_pair(g, h, sigma):
    """purification u of sigma (on X (x) R) -> (Phi (x) id)(uu*), (Psi (x) id)(uu*)"""
    w, U = np.linalg.eigh(sigma)
    A = U@np.diag(np.sqrt(np.clip(w, 0, None)))   # A A^* = sigma, u = sum_x,r A_{xr}|x>|r>
    u = A.reshape(-1)
    rho = np.outer(u, u.conj())
    def ch(Ks):
        Kt = [np.kron(K, np.eye(2)) for K in Ks]
        return apply(Kt, rho)
    return ch(kraus(g)), ch(kraus(h))

def best_product(g, h):
    """optimize xi_QC over input sigma (Bloch params) with purification."""
    def neg(x):
        th, ph, r = x
        r = 1/(1+np.exp(-r))
        n = r*np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
        sig = 0.5*(np.eye(2)+n[0]*np.array([[0,1],[1,0]])+n[1]*np.array([[0,-1j],[1j,0]])+n[2]*np.diag([1,-1]))
        A, B = output_pair(g, h, sig)
        return -xi_QC(A, B)[0]
    best = None
    rng = np.random.default_rng(1)
    for _ in range(12):
        x0 = [rng.uniform(0, np.pi), rng.uniform(0, 2*np.pi), rng.normal()*2]
        r = minimize(neg, x0, method='Nelder-Mead', options={'xatol':1e-9, 'fatol':1e-14, 'maxiter':4000})
        if best is None or r.fun < best.fun: best = r
    th, ph, r = best.x
    return -best.fun, (th, ph, 1/(1+np.exp(-r)))

def bloch_z_scan(g, h, ns=201):
    """sigma = diag(1-p, p) with purification (Schmidt basis computational)."""
    res = []
    for p in np.linspace(0, 1, ns):
        A, B = output_pair(g, h, np.diag([1-p, p]))
        res.append((p, xi_QC(A, B)[0]))
    return res

if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    g = 0.5
    print("== structure check gamma=0.5, gamma'=0.4 ==")
    lv, cmin, c12 = Chat_info(kraus(0.5), kraus(0.4))
    for t, S in lv: print(" level tau=%.6f  S=\n" % t, S)
    print(" min_s lambda_min(B_s) =", cmin, "  Qhat_1/2 =", c12, " -> Chat = inf, fidelity converse = inf")
    JP, JQ = choi(kraus(0.5)), choi(kraus(0.4))
    # range intersection dimension
    def ran(J):
        w, U = np.linalg.eigh(J); return U[:, w > 1e-12]
    RP, RQ = ran(JP), ran(JQ)
    sv = np.linalg.svd(RP.conj().T@RQ, compute_uv=False)
    print(" principal cosines between Choi ranges:", sv, "(one =1 => intersection dim 1, span{|0>_Y|1>_X})")
    print()
    print("== table: gamma=0.5, gamma'=0.5-Delta ==")
    print("%6s %12s %12s %12s %12s %12s %12s %12s" % ("Delta","bern(|1>)","best prod","Choi prod","-2ln mu*","mu*closed","mu*SDP","-ln f^2"))
    for D in [0.1, 0.03, 0.01]:
        h = g-D
        b = bern_chernoff(g, h)
        bp, arg = best_product(g, h)
        A, B = output_pair(g, h, np.eye(2)/2); xc = xi_QC(A, B)[0]
        mc = mu_star_closed(g, h); ms, C = mu_star_sdp(kraus(g), kraus(h))
        f = np.sqrt(g*h)+np.sqrt((1-g)*(1-h))
        print("%6.3f %12.4e %12.4e %12.4e %12.4e %12.8f %12.8f %12.4e   argbest(th,ph,r)=%s" % (D, b, bp, xc, -2*np.log(mc), mc, ms, -2*np.log(f), np.round(arg, 4)))
        print("      ratios to D^2: bern %.4f  bestprod %.4f  mu-bound %.4f   [predicted 1/(8g(1-g))=%.4f, 1/(4g^2(1-g))=%.4f]" % (b/D**2, bp/D**2, -2*np.log(mc)/D**2, 1/(8*g*(1-g)), 1/(4*g*g*(1-g))))
