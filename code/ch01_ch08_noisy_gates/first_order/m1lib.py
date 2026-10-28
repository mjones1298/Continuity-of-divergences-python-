"""m1lib: high-precision helpers for the M1 analysis (first-order sense, Chernoff side).

Conventions (identical to the manuscript and to A_numerics/common.py):
  J(Phi) = sum_ij Phi(E_ij) (x) E_ij, output Y first, input X second; Tr J = d_X; Tr_Y J = 1 for channels.
  vec(A) = sum_ij A_ij |i>|j>  (row-major), so J(Ad_A) = vec(A) vec(A)^dagger.
All matrix functions are evaluated by Hermitian eigendecomposition in mpmath.
"""
import mpmath as mp

mp.mp.dps = 40

I2 = mp.matrix([[1, 0], [0, 1]])
X2 = mp.matrix([[0, 1], [1, 0]])
Y2 = mp.matrix([[0, -1j], [1j, 0]])
Z2 = mp.matrix([[1, 0], [0, -1]])
PAULI = {'I': I2, 'X': X2, 'Y': Y2, 'Z': Z2}


def herm(A):
    return (A + A.H) / 2


def eigh(A):
    E, Q = mp.eighe(herm(A))
    return [mp.re(e) for e in E], Q


def funm(A, f):
    E, Q = eigh(A)
    n = A.rows
    D = mp.zeros(n, n)
    for i in range(n):
        D[i, i] = f(E[i])
    return Q * D * Q.H


def eigvals(A):
    return sorted(eigh(A)[0])


def _thr(A):
    """eigenvalues below max|eig| * 10^-(dps-8) are treated as exact zeros (rank-deficient PSD inputs)."""
    E, _ = eigh(A)
    return max(abs(e) for e in E) * mp.mpf(10) ** (-(mp.mp.dps - 8))


def sqrtm_psd(A, tol=None):
    t = _thr(A)
    return funm(A, lambda x: mp.sqrt(x) if x > t else mp.mpf(0))


def powm(A, s):
    """A^s for PSD A with 0^s = 0 (s > 0); round-off eigenvalues are thresholded, which matters for small s."""
    t = _thr(A)
    return funm(A, lambda x: x ** s if x > t else mp.mpf(0))


def logm(A):
    return funm(A, mp.log)


def tr(A):
    return mp.fsum(A[i, i] for i in range(A.rows))


def kron(*Ms):
    C = Ms[0]
    for B in Ms[1:]:
        m, n, p, q = C.rows, C.cols, B.rows, B.cols
        D = mp.zeros(m * p, n * q)
        for i in range(m):
            for j in range(n):
                a = C[i, j]
                if a == 0:
                    continue
                for k in range(p):
                    for l in range(q):
                        D[i * p + k, j * q + l] = a * B[k, l]
        C = D
    return C


def vec(A):
    n, m = A.rows, A.cols
    v = mp.matrix(n * m, 1)
    for i in range(n):
        for j in range(m):
            v[i * m + j] = A[i, j]
    return v


def outer(u, v):
    return u * v.H


def inner(u, v):
    return (u.H * v)[0]


def ptrace_Y(J, dY, dX):
    R = mp.zeros(dX, dX)
    for x in range(dX):
        for z in range(dX):
            R[x, z] = mp.fsum(J[y * dX + x, y * dX + z] for y in range(dY))
    return R


def pauli_string(s):
    return kron(*[PAULI[c] for c in s])


def pauli_labels(n):
    import itertools
    return [''.join(t) for t in itertools.product('IXYZ', repeat=n)]


def choi_pauli(q, U=None):
    """Choi of P_q o Ad_U, q: dict label -> rate (labels all of one length n)."""
    n = len(next(iter(q)))
    d = 2 ** n
    if U is None:
        U = mp.eye(d)
    J = mp.zeros(d * d, d * d)
    for lab, w in q.items():
        v = vec(pauli_string(lab) * U)
        J += w * outer(v, v)
    return J


def depol_rates(p, n):
    p = mp.mpf(p)
    q = {}
    for lab in pauli_labels(n):
        w = sum(1 for c in lab if c != 'I')
        q[lab] = (1 - 3 * p / 4) ** (n - w) * (p / 4) ** w
    return q


def rand_herm(n, rng):
    A = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            A[i, j] = mp.mpf(rng.standard_normal()) + 1j * mp.mpf(rng.standard_normal())
    return herm(A)


def rand_unitary(n, rng):
    H = rand_herm(n, rng)
    return funm(H, lambda x: mp.mpf(1)) if False else expi_herm(H)


def expi_herm(H):
    E, Q = eigh(H)
    n = H.rows
    D = mp.zeros(n, n)
    for i in range(n):
        D[i, i] = mp.exp(1j * E[i])
    return Q * D * Q.H


def traceless_Y_direction(dY, dX, rng):
    """Choi of a random Hermitian-preserving, trace-annihilating map: Tr_Y X = 0."""
    H = rand_herm(dY * dX, rng)
    T = ptrace_Y(H, dY, dX) / dY
    return herm(H - kron(mp.eye(dY), T))


# ---------------- channel divergences ----------------
def gmean_s(A, B, s):
    """A #_s B = A^{1/2}(A^{-1/2} B A^{-1/2})^s A^{1/2}, A > 0, B >= 0."""
    Ah = funm(A, mp.sqrt)
    Aih = funm(A, lambda x: 1 / mp.sqrt(x))
    M = herm(Aih * B * Aih)
    return herm(Ah * powm(M, s) * Ah)


def Fhat(JPhi, JPsi, dY, dX):
    """geometric fidelity lambda_min Tr_Y[J_Psi # J_Phi], J_Psi > 0 (symmetric mean, J_Phi may be singular)."""
    return eigvals(ptrace_Y(gmean_s(JPsi, JPhi, mp.mpf(1) / 2), dY, dX))[0]


def Qhat_s(JPhi, JPsi, s, dY, dX):
    """lambda_min Tr_Y[J_Psi #_s J_Phi] (manuscript ordering), J_Psi > 0."""
    return eigvals(ptrace_Y(gmean_s(JPsi, JPhi, s), dY, dX))[0]


def Chat(JPhi, JPsi, dY, dX, smin=mp.mpf('1e-12')):
    """-ln inf_{s in (0,1)} Qhat_s, by golden-section on [smin, 1-smin] after a coarse scan."""
    f = lambda s: Qhat_s(JPhi, JPsi, s, dY, dX)
    grid = [smin] + [mp.mpf(k) / 20 for k in range(1, 20)] + [1 - smin]
    vals = [f(s) for s in grid]
    i = min(range(len(vals)), key=lambda k: vals[k])
    lo = grid[max(i - 1, 0)]; hi = grid[min(i + 1, len(grid) - 1)]
    g = (mp.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, dd = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(dd)
    for _ in range(60):
        if fc < fd:
            b, dd, fd = dd, c, fc
            c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, dd, fd
            dd = a + g * (b - a); fd = f(dd)
    best = min(vals + [fc, fd])
    return -mp.log(best)


def DBS(JPhi, JPsi, dY, dX):
    """lambda_max Tr_Y[J_Phi^{1/2} ln(J_Phi^{1/2} J_Psi^{-1} J_Phi^{1/2}) J_Phi^{1/2}], J_Psi > 0, J_Phi >= 0
    (log taken on the support of the middle operator, which is the support of J_Phi)."""
    s = sqrtm_psd(JPhi)
    M = herm(s * mp.inverse(JPsi) * s)
    E, Q = eigh(M)
    thr = max(E) * mp.mpf(10) ** (-(mp.mp.dps - 5))
    n = M.rows
    D = mp.zeros(n, n)
    for i in range(n):
        D[i, i] = mp.log(E[i]) if E[i] > thr else mp.mpf(0)
    L = Q * D * Q.H
    G = herm(s * L * s)
    return eigvals(ptrace_Y(G, dY, dX))[-1]


def Dmax(JPhi, JPsi):
    Pih = funm(JPsi, lambda x: 1 / mp.sqrt(x))
    return mp.log(eigvals(herm(Pih * JPhi * Pih))[-1])


def thompson(J1, J2):
    """d_T between two positive definite matrices = max |ln eig(J1^{-1/2} J2 J1^{-1/2})|."""
    Pih = funm(J1, lambda x: 1 / mp.sqrt(x))
    ev = eigvals(herm(Pih * J2 * Pih))
    return max(abs(mp.log(ev[0])), abs(mp.log(ev[-1])))


def relent(r, s):
    """Umegaki D(r||s), r >= 0, s > 0 (0 log 0 = 0)."""
    E, Q = eigh(r)
    n = r.rows
    D = mp.zeros(n, n)
    for i in range(n):
        D[i, i] = E[i] * mp.log(E[i]) if E[i] > 0 else mp.mpf(0)
    return mp.re(tr(Q * D * Q.H) - tr(r * logm(s)))


def chernoff_states(r, s, smin=mp.mpf(0), smax=mp.mpf(1)):
    """quantum Chernoff -ln min_{s in [0,1]} Tr r^s s^{1-s} (both full rank), golden section (log-convex in s)."""
    def f(t):
        return mp.re(tr(powm(r, t) * powm(s, 1 - t)))
    a, b = mp.mpf('1e-30') if smin == 0 else smin, smax - mp.mpf('1e-30')
    g = (mp.sqrt(5) - 1) / 2
    c, dd = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(dd)
    for _ in range(90):
        if fc < fd:
            b, dd, fd = dd, c, fc
            c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, dd, fd
            dd = a + g * (b - a); fd = f(dd)
    return -mp.log(min(fc, fd)), (c + dd) / 2


def classical_chernoff(P, Q):
    """-ln min_{s in [0,1]} sum P^s Q^{1-s} for strictly positive vectors (golden section; log-convex in s)."""
    def f(t):
        return mp.fsum(a ** t * b ** (1 - t) for a, b in zip(P, Q))
    a, b = mp.mpf(0), mp.mpf(1)
    g = (mp.sqrt(5) - 1) / 2
    c, dd = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(dd)
    for _ in range(120):
        if fc < fd:
            b, dd, fd = dd, c, fc
            c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, dd, fd
            dd = a + g * (b - a); fd = f(dd)
    return -mp.log(min(fc, fd, f(mp.mpf(0)), f(mp.mpf(1)))), (c + dd) / 2


def Rz(delta):
    delta = mp.mpf(delta)
    return mp.matrix([[mp.exp(-1j * delta / 2), 0], [0, mp.exp(1j * delta / 2)]])


def nstr(x, n=12):
    return mp.nstr(x, n)
