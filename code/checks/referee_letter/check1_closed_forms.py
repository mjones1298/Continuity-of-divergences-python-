"""Referee check 1: qubit closed forms, Table I coefficients, Choi floor, sharp constant.

Independent of the author's code: builds Choi matrices from Kraus operators in mpmath
and evaluates the channel quantities directly (no Pauli-block reduction).
Convention: J = sum_k vec(K_k) vec(K_k)^dag, vec row-major (output index first).
"""
import mpmath as mp

mp.mp.dps = 50
I2 = mp.eye(2)
X = mp.matrix([[0, 1], [1, 0]])
Y = mp.matrix([[0, -1j], [1j, 0]])
Z = mp.matrix([[1, 0], [0, -1]])


def kron(A, B):
    m, n = A.rows, A.cols
    p, q = B.rows, B.cols
    C = mp.matrix(m * p, n * q)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                for l in range(q):
                    C[i * p + k, j * q + l] = A[i, j] * B[k, l]
    return C


def vec(A):
    v = mp.matrix(A.rows * A.cols, 1)
    for a in range(A.rows):
        for b in range(A.cols):
            v[a * A.cols + b] = A[a, b]
    return v


def choi(kraus):
    d = kraus[0].rows * kraus[0].cols
    J = mp.matrix(d, d)
    for K in kraus:
        v = vec(K)
        J += v * v.H
    return J


def herm_fun(A, f):
    A = (A + A.H) / 2
    E, Q = mp.eighe(A)
    D = mp.diag([f(e) for e in E])
    return Q * D * Q.H


def msqrt(A):
    return herm_fun(A, lambda x: mp.sqrt(max(x, mp.mpf(0))))


def minv(A):
    return herm_fun(A, lambda x: 1 / x)


def mlog(A):
    return herm_fun(A, mp.log)


def gmean(A, B):
    As = msqrt(A)
    Ais = minv(As)
    return As * msqrt(Ais * B * Ais) * As


def ptrace_Y(J, dY, dX):
    R = mp.matrix(dX, dX)
    for a in range(dX):
        for b in range(dX):
            s = 0
            for y in range(dY):
                s += J[y * dX + a, y * dX + b]
            R[a, b] = s
    return R


def lmin(A):
    E, _ = mp.eighe((A + A.H) / 2)
    return min(E)


def lmax(A):
    E, _ = mp.eighe((A + A.H) / 2)
    return max(E)


def Rz(d):
    return mp.matrix([[mp.exp(-0.5j * d), 0], [0, mp.exp(0.5j * d)]])


def depol_kraus(p, V):
    ks = [mp.sqrt(1 - 3 * p / 4) * V]
    for P in (X, Y, Z):
        ks.append(mp.sqrt(p / 4) * P * V)
    return ks


def quantities(p, delta):
    J1 = choi(depol_kraus(p, Rz(delta)))
    J0 = choi(depol_kraus(p, I2))
    Fhat = lmin(ptrace_Y(gmean(J1, J0), 2, 2))
    S = msqrt(J1)
    DBS = lmax(ptrace_Y(S * mlog(S * minv(J0) * S) * S, 2, 2))
    rho, sig = J1 / 2, J0 / 2
    D = mp.re(sum((rho * (mlog(rho) - mlog(sig)))[i, i] for i in range(4)))
    # Uhlmann root fidelity ||sqrt(rho) sqrt(sig)||_1 = Tr sqrt( sqrt(rho) sig sqrt(rho) )
    sr = msqrt(rho)
    M = msqrt(sr * sig * sr)
    F = mp.re(sum(M[i, i] for i in range(4)))
    return mp.re(Fhat), mp.re(DBS), D, F, J0


def closed(p):
    FR = 2 * (1 - p) ** 2 * (2 - p) / (p * (4 - 3 * p))
    FB = mp.mpf(1) / 2 * (1 - p) * mp.log((4 - 3 * p) / p)
    FS = 2 * (1 - p) ** 2 / (2 - p)
    Finf = 2 * (1 - p) ** 2 / (p * (3 - 2 * p))
    return FR, FB, FS, Finf


print("=== Choi floor of D_p o Ad_U (qubit) ===")
for p in [mp.mpf('1e-2'), mp.mpf('1e-3')]:
    J0 = choi(depol_kraus(p, I2))
    E, _ = mp.eighe(J0)
    print(f"p={mp.nstr(p,3)} eig(J)={[mp.nstr(e,8) for e in E]}  expected p/2={mp.nstr(p/2,8)}, 2-3p/2={mp.nstr(2-3*p/2,8)}")

print("\n=== finite-delta closed form for Fhat (SM eq. before Table I) ===")
for p, d in [(mp.mpf('1e-2'), mp.mpf('0.2')), (mp.mpf('1e-3'), mp.mpf('0.05'))]:
    Fh, DBS, D, F, _ = quantities(p, d)
    x = 4 * (1 - p) ** 2 * mp.sin(d / 2) ** 2 / (p * (4 - 3 * p))
    Fh_cf = p / 2 + (1 - p / 2) / mp.sqrt(1 + x)
    phi = mp.sqrt(x / (1 + x)) * mp.asinh(mp.sqrt(x))
    DBS_cf = (2 - p) * phi
    D_cf = (1 - p) * mp.sin(d / 2) ** 2 * mp.log((4 - 3 * p) / p)
    F_cf = p / 2 + mp.sqrt((1 - p / 2) ** 2 - (1 - p) ** 2 * mp.sin(d / 2) ** 2)
    print(f"p={mp.nstr(p,3)} delta={mp.nstr(d,3)}")
    for name, a, b in [("Fhat", Fh, Fh_cf), ("D_BS", DBS, DBS_cf), ("D(maxent)", D, D_cf), ("F(maxent)", F, F_cf)]:
        print(f"   {name:10s} matrix={mp.nstr(a,15)}  closed={mp.nstr(b,15)}  rel.diff={mp.nstr(abs(a-b)/abs(b),3)}")

print("\n=== Table I: delta^2 coefficients (matrix evaluation at delta=1e-12) ===")
d = mp.mpf('1e-12')
for p in [mp.mpf('1e-1'), mp.mpf('1e-2'), mp.mpf('1e-3')]:
    Fh, DBS, D, F, _ = quantities(p, d)
    FR, FB, FS, Finf = closed(p)
    rows = [("1-Fhat", (1 - Fh) / d**2, FR / 8), ("D_BS", DBS / d**2, FR / 2),
            ("D", D / d**2, FB / 2), ("1-F", (1 - F) / d**2, FS / 8)]
    print(f"p={mp.nstr(p,3)}")
    for name, a, b in rows:
        print(f"   {name:7s} numeric coeff={mp.nstr(a,12)}  table={mp.nstr(b,12)}  ratio={mp.nstr(a/b,12)}")
    print(f"   p->0 forms: 1/(8p)={mp.nstr(1/(8*p),6)} 1/(2p)={mp.nstr(1/(2*p),6)} ln(4/p)/4={mp.nstr(mp.log(4/p)/4,6)} 1/8")

print("\n=== F_RLD/F_inf = (2-p)(3-2p)/(4-3p) and SM Table S(F_inf) ===")
for p in ['1e-1', '1e-2', '1e-3', '1e-4']:
    p = mp.mpf(p)
    FR, FB, FS, Finf = closed(p)
    print(f"p={mp.nstr(p,2)}: pFinf={mp.nstr(p*Finf,6)} pFRLD={mp.nstr(p*FR,6)} ratio={mp.nstr(FR/Finf,6)} "
          f"formula={mp.nstr((2-p)*(3-2*p)/(4-3*p),6)} pFinf/2={mp.nstr(p*Finf/2,6)}")
# sign of (2-p)(3-2p) - 1.5(4-3p) = p(2p-5/2)
print("(2-p)(3-2p)-(3/2)(4-3p) at p=0.5, 0.99, 1.2, 1.3:",
      [mp.nstr((2-q)*(3-2*q)-1.5*(4-3*q), 5) for q in map(mp.mpf, ['0.5', '0.99', '1.2', '1.3'])])

print("\n=== Sharp constant 1/(4 lt): d/de ln Fhat(Ad_W, Psi + e Delta), W=Z, Delta=(Ad_Z-Ad_X)/2 ===")
for p in [mp.mpf('1e-2'), mp.mpf('1e-3')]:
    J0 = choi(depol_kraus(p, I2))
    w = vec(Z)
    wp = vec(X)
    JD = (w * w.H - wp * wp.H) / 2

    def lnF(e):
        Je = J0 + e * JD
        return -mp.log(mp.re((w.H * minv(Je) * w)[0])) / 2

    # check rank-one formula against generic geometric mean with tiny regularisation
    h = mp.mpf('1e-20')
    slope = (lnF(h) - lnF(-h)) / (2 * h)
    print(f"p={mp.nstr(p,3)} slope={mp.nstr(slope,12)} 1/(4 lt)=1/p={mp.nstr(1/p,12)} ratio={mp.nstr(slope*p,12)}")
# cross-check rank-one formula against regularized geometric mean
p = mp.mpf('1e-2')
J0 = choi(depol_kraus(p, I2))
w = vec(Z)
Jphi = w * w.H
for eps in ['1e-20', '1e-30']:
    G = gmean(Jphi + mp.mpf(eps) * mp.eye(4), J0)
    print("   regularized Fhat(Ad_Z, N_p) eps=", eps, mp.nstr(lmin(ptrace_Y(G, 2, 2)), 15),
          " rank-one formula:", mp.nstr(1 / mp.sqrt(mp.re((w.H * minv(J0) * w)[0])), 15))

print("\n=== Prop L2: p*Sigma_unit and 2/(3 sqrt 3) ===")
for p in ['1e-2', '1e-3', '1e-4']:
    p = mp.mpf(p)
    beta, lam = 2 - 3 * p / 2, p / 2
    f = lambda t: mp.sqrt(2) / 4 * (mp.cos(t)**2 / beta**2 + mp.sin(t)**2 / lam**2) / (mp.cos(t)**2 / beta + mp.sin(t)**2 / lam) ** 1.5
    # maximize over theta: optimum near sin^2 theta = 2 lam/d = lam
    t0 = mp.asin(mp.sqrt(lam))
    tt = mp.findroot(lambda t: mp.diff(f, t), t0)
    print(f"p={mp.nstr(p,2)} p*Sigma_unit={mp.nstr(p*f(tt),8)}  (2/(3sqrt3)={mp.nstr(2/(3*mp.sqrt(3)),8)})")
