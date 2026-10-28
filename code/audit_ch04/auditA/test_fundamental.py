"""Test of the 'fundamental principle' (Thm 2.15 / Rem 2.16):  J(Phi) sigma J(Psi) = W g_sigma(T) W^*
for several Kubo-Ando connections, with SINGULAR J(Phi+Psi), in mpmath high precision.

Reference value: independent regularisation  (A+eta 1) sigma (B+eta 1)  (Kubo-Ando formula on PD matrices)
computed WITHOUT using W or T at all, for decreasing eta.
Second test (finite eta, exact): transfer identity  (W A W^*) sigma (W B W^*) = W (A sigma B) W^*  for PD A,B,
computed via the pseudo-inverse Kubo-Ando formula on range(W).
"""
import numpy as np, mpmath as mp, sys
mp.mp.dps = 80
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)

def mpm(a):  # numpy complex -> mp matrix
    M = mp.zeros(a.shape[0], a.shape[1])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            M[i, j] = mp.mpc(float(a[i, j].real), float(a[i, j].imag))
    return M

def herm(M): return (M + M.H) / 2

def hfun(H, fn):
    E, Q = mp.eigh(herm(H))
    D = mp.diag([fn(e) for e in E])
    return Q * D * Q.H

def vec(K):
    d_y, d_x = K.rows, K.cols
    v = mp.zeros(d_y * d_x, 1)
    for j in range(d_x):
        for i in range(d_y):
            v[i * d_x + j] = K[i, j]       # vec(K) = sum_j K e_j (x) e_j, index (i,j)->i*dX+j
    return v

def choi(kraus):
    n = kraus[0].rows * kraus[0].cols
    J = mp.zeros(n, n)
    for K in kraus:
        v = vec(K); J += v * v.H
    return J

# connections: (name, f, f0=f(0+), finf=f'(inf), f(1))
def f_log(t):
    t = mp.mpf(t)
    return mp.mpf(1) if abs(t - 1) < mp.mpf(10) ** (-60) else (t - 1) / mp.log(t)
CONN = [
    ("geometric", lambda t: mp.sqrt(t), 0, 0),
    ("geo alpha=.3", lambda t: t ** mp.mpf('0.3'), 0, 0),
    ("geo alpha=.8", lambda t: t ** mp.mpf('0.8'), 0, 0),
    ("harmonic", lambda t: 2 * t / (1 + t), 0, 0),
    ("arithmetic", lambda t: (1 + t) / 2, mp.mpf(1) / 2, mp.mpf(1) / 2),
    ("logarithmic", f_log, 0, 0),
    ("left trivial (f=1)", lambda t: mp.mpf(1), 1, 0),
    ("right trivial (f=t)", lambda t: t, 0, 1),
    ("parallel sum t/(1+t)", lambda t: t / (1 + t), 0, 0),
    ("mixed .3+.5t/(1+t)+.2t", lambda t: mp.mpf('0.3') + mp.mpf('0.5') * t / (1 + t) + mp.mpf('0.2') * t, mp.mpf('0.3'), mp.mpf('0.2')),
    ("power mean p=1/2", lambda t: ((1 + mp.sqrt(t)) / 2) ** 2, mp.mpf(1) / 4, mp.mpf(1) / 4),
]

def ka_pd(A, B, f):
    """A^{1/2} f(A^{-1/2} B A^{-1/2}) A^{1/2} for PD A (via eigh)."""
    Ah = hfun(A, mp.sqrt); Aih = hfun(A, lambda e: 1 / mp.sqrt(e))
    return Ah * hfun(Aih * B * Aih, f) * Ah

def ka_pinv_range(A, B, f, tol=mp.mpf(10) ** (-40)):
    """same, on range(A) (assumes range(B) subset range(A) and A>0 there)."""
    E, Q = mp.eigh(herm(A))
    keep = [i for i in range(len(E)) if E[i] > tol]
    Qr = Q[:, keep] if False else mp.matrix(Q.rows, len(keep))
    for c, i in enumerate(keep):
        for r in range(Q.rows): Qr[r, c] = Q[r, i]
    Er = [E[i] for i in keep]
    Ah = Qr * mp.diag([mp.sqrt(e) for e in Er]) * Qr.H
    Aih = Qr * mp.diag([1 / mp.sqrt(e) for e in Er]) * Qr.H
    inner = Aih * B * Aih
    # inner acts on range(A): restrict
    Ei, Qi = mp.eigh(herm(inner))
    # eigenvectors in range(A): those not in kernel of A... use projection
    P = Qr * Qr.H
    inner_r = Qr.H * inner * Qr
    Ef, Qf = mp.eigh(herm(inner_r))
    F = Qf * mp.diag([f(e if e > 0 else mp.mpf(0)) if e > mp.mpf(10) ** (-60) else f(mp.mpf(10) ** (-60)) for e in Ef]) * Qf.H
    return Ah * Qr * F * Qr.H * Ah

def make_pair(dx, dy, kind):
    def rk(n):
        return [mpm(rng.normal(size=(dy, dx)) + 1j * rng.normal(size=(dy, dx))) for _ in range(n)]
    n = dx * dy
    if kind == "random":
        k1 = int(rng.integers(1, max(2, n // 2))); k2 = int(rng.integers(1, max(2, n // 2)))
        return rk(k1), rk(k2)
    if kind == "shared":
        K = rk(3)
        return [K[0], K[1]], [K[1] * mp.mpf('0.7'), K[2]]
    if kind == "orthogonal":
        K = rk(2)
        return [K[0]], [K[1]]
    if kind == "proportional":
        K = rk(2)
        return [K[0], K[1]], [K[0] * mp.sqrt(mp.mpf('0.4')), K[1] * mp.sqrt(mp.mpf('0.4'))]
    if kind == "Psi_zero":
        K = rk(2)
        return [K[0], K[1]], []
    if kind == "rotated":     # nearly orthogonal supports
        K = rk(2)
        return [K[0]], [K[0] * mp.mpf('0.3') + K[1] * mp.mpf('0.1')]

def arn(Jp, Jq):
    S = Jp + Jq
    E, Q = mp.eigh(herm(S))
    keep = [i for i in range(len(E)) if E[i] > mp.mpf(10) ** (-40)]
    n = S.rows; r = len(keep)
    W = mp.matrix(n, r); Wp = mp.matrix(r, n)
    for c, i in enumerate(keep):
        for rr in range(n):
            W[rr, c] = Q[rr, i] * mp.sqrt(E[i]); Wp[c, rr] = mp.conj(Q[rr, i]) / mp.sqrt(E[i])
    T = herm(Wp * Jp * Wp.H)
    return W, Wp, T

def snap(t, tol=mp.mpf(10) ** (-40)):
    if abs(t) < tol: return mp.mpf(0)
    if abs(1 - t) < tol: return mp.mpf(1)
    return t

def g_of_T(T, f, f0, finf):
    def g(t):
        t = snap(t)
        if t == 0: return mp.mpf(finf)
        if t == 1: return mp.mpf(f0)
        return t * f((1 - t) / t)
    return hfun(T, g)

def fro(M): return mp.sqrt(sum(abs(M[i, j]) ** 2 for i in range(M.rows) for j in range(M.cols)))

worst = {}
for (dx, dy) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
    for kind in ["random", "shared", "orthogonal", "proportional", "Psi_zero", "rotated"]:
        kp, kq = make_pair(dx, dy, kind)
        n = dx * dy
        Jp = choi(kp)
        Jq = choi(kq) if kq else mp.zeros(n, n)
        W, Wp, T = arn(Jp, Jq)
        r = W.cols
        spec = [mp.nstr(e, 5) for e in mp.eigh(herm(T), eigvals_only=True)]
        # sanity: W T W^* = J(Phi), W (1-T) W^* = J(Psi)
        chk1 = fro(W * T * W.H - Jp) + fro(W * (mp.eye(r) - T) * W.H - Jq)
        for (name, f, f0, finf) in CONN:
            G = W * g_of_T(T, f, f0, finf) * W.H
            errs = []
            for eta in [mp.mpf(10) ** (-10), mp.mpf(10) ** (-30), mp.mpf(10) ** (-60)]:
                Ae = Jp + eta * mp.eye(n); Be = Jq + eta * mp.eye(n)
                M = ka_pd(Ae, Be, f)
                errs.append(fro(M - G))
            # finite-eta transfer identity test
            eta = mp.mpf('0.05')
            Ae = T + eta * mp.eye(r); Be = mp.eye(r) - T + eta * mp.eye(r)
            lhs = ka_pinv_range(W * Ae * W.H, W * Be * W.H, f)
            rhs = W * hfun(Ae, lambda e: e) * 0  # placeholder
            # (A sigma B) with A,B commuting PD:
            ge = hfun(T, lambda t: (t + eta) * f((1 - t + eta) / (t + eta)))
            rhs = W * ge * W.H
            terr = fro(lhs - rhs)
            key = name
            worst.setdefault(key, [0, 0, 0, 0])
            worst[key][0] = max(worst[key][0], float(errs[0])); worst[key][1] = max(worst[key][1], float(errs[1]))
            worst[key][2] = max(worst[key][2], float(errs[2])); worst[key][3] = max(worst[key][3], float(terr))
        print(f"dX={dx} dY={dy} {kind:12s} r={r} spec(T)={spec} |WTW*-J|={mp.nstr(chk1,3)}")
print()
print("max over instances of Frobenius error ||(A+eta)s(B+eta) - W g(T) W*||  for eta=1e-10,1e-30,1e-60 ; and finite-eta transfer error")
for k, v in worst.items():
    print(f"{k:26s} " + "  ".join(f"{x:9.2e}" for x in v))
