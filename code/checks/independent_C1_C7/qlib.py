"""Independent checker library (written from scratch).
Conventions: J(Phi) = sum_ij Phi(E_ij) (x) E_ij, output Y first, input X second.
vec(K) = sum_i K|i> (x) |i>, so J = sum_k vec(K_k) vec(K_k)^*.
"""
import numpy as np
import mpmath as mp
import cvxpy as cp

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def vec(K):
    dout, din = K.shape
    v = np.zeros(dout * din, dtype=complex)
    for i in range(din):
        e = np.zeros(din); e[i] = 1
        v += np.kron(K[:, i], e)
    return v


def choi_from_kraus(Ks):
    return sum(np.outer(vec(K), vec(K).conj()) for K in Ks)


def choi_from_map(Phi, din):
    J = 0
    for i in range(din):
        for j in range(din):
            E = np.zeros((din, din), dtype=complex); E[i, j] = 1
            J = J + np.kron(Phi(E), E)
    return J


def ptrace_Y(J, dy, dx):
    return np.trace(J.reshape(dy, dx, dy, dx), axis1=0, axis2=2)


def apply_choi(J, rho, dy, dx):
    # Phi(rho) = Tr_X[J (1 (x) rho^T)]
    M = J @ np.kron(np.eye(dy), rho.T)
    return np.trace(M.reshape(dy, dx, dy, dx), axis1=1, axis2=3)


# ---------------- geometric mean ----------------
def gm_np(A, B):
    w, V = np.linalg.eigh(A)
    Ah = V @ np.diag(np.sqrt(w)) @ V.conj().T
    Aih = V @ np.diag(1 / np.sqrt(w)) @ V.conj().T
    M = Aih @ B @ Aih; M = (M + M.conj().T) / 2
    w2, V2 = np.linalg.eigh(M)
    Mh = V2 @ np.diag(np.sqrt(np.clip(w2, 0, None))) @ V2.conj().T
    return Ah @ Mh @ Ah


def _mpsqrtm(A, inv=False):
    w, V = mp.eighe(A) if False else mp.eigh(A)
    n = A.rows
    D = mp.zeros(n, n)
    for k in range(n):
        x = w[k]
        if x < 0:
            x = mp.mpf(0)
        D[k, k] = (1 / mp.sqrt(x)) if inv else mp.sqrt(x)
    return V * D * V.H


def gm_mp(A, B, eps, dps=120):
    with mp.workdps(dps):
        n = A.rows if isinstance(A, mp.matrix) else A.shape[0]
        Am = (A if isinstance(A, mp.matrix) else mp.matrix(A.tolist())) + mp.mpf(eps) * mp.eye(n)
        Bm = (B if isinstance(B, mp.matrix) else mp.matrix(B.tolist())) + mp.mpf(eps) * mp.eye(n)
        Ah = _mpsqrtm(Am); Aih = _mpsqrtm(Am, inv=True)
        M = Aih * Bm * Aih
        M = (M + M.H) / 2
        return Ah * _mpsqrtm(M) * Ah


def fid_mp(JA, JB, dy, dx, eps, dps=120):
    """lambda_min Tr_Y((JA+eps)#(JB+eps)) in high precision; inputs given as exact-ish
    numpy arrays (or mp matrices via lists)."""
    with mp.workdps(dps):
        G = gm_mp(JA, JB, eps, dps)
        T = mp.zeros(dx, dx)
        for a in range(dx):
            for b in range(dx):
                s = 0
                for y in range(dy):
                    s += G[y * dx + a, y * dx + b]
                T[a, b] = s
        T = (T + T.H) / 2
        w, _ = mp.eigh(T)
        return min(w)


def fid_np(JA, JB, dy, dx):
    G = gm_np(JA, JB)
    T = ptrace_Y(G, dy, dx); T = (T + T.conj().T) / 2
    return np.linalg.eigvalsh(T)[0]


def fid_sdp(JA, JB, dy, dx, solver='CLARABEL'):
    n = JA.shape[0]
    Xv = cp.Variable((n, n), hermitian=True)
    t = cp.Variable()
    M = cp.bmat([[JA, Xv], [Xv, JB]])
    T = cp.partial_trace(Xv, [dy, dx], axis=0)
    cons = [(M + M.H) / 2 >> 0, (T + T.H) / 2 - t * np.eye(dx) >> 0]
    prob = cp.Problem(cp.Maximize(t), cons)
    try:
        prob.solve(solver=solver)
    except Exception:
        prob.solve(solver='SCS', eps=1e-9, max_iters=200000)
    return t.value


# ---------------- diamond norm (Watrous SDP) ----------------
def diamond(JD, dy, dx, solver='CLARABEL'):
    """Watrous: max Re tr(J^* Xv) s.t. [[1(x)rho0, Xv],[Xv^*, 1(x)rho1]] >= 0."""
    n = dy * dx
    Xv = cp.Variable((n, n), complex=True)
    r0 = cp.Variable((dx, dx), hermitian=True)
    r1 = cp.Variable((dx, dx), hermitian=True)
    M = cp.bmat([[cp.kron(np.eye(dy), r0), Xv], [Xv.H, cp.kron(np.eye(dy), r1)]])
    cons = [(M + M.H) / 2 >> 0, cp.trace(r0) == 1, cp.trace(r1) == 1, r0 >> 0, r1 >> 0]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(JD.conj().T @ Xv))), cons)
    prob.solve(solver=solver)
    return prob.value


# ---------------- hockey stick ----------------
def hockey_sdp(JA, JB, gamma, dy, dx, solver='CLARABEL'):
    n = dy * dx
    P = cp.Variable((n, n), hermitian=True)
    r = cp.Variable((dx, dx), hermitian=True)
    D = JA - gamma * JB
    cons = [P >> 0, cp.kron(np.eye(dy), r.T) - P >> 0, cp.trace(r) == 1, r >> 0]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(D @ P))), cons)
    prob.solve(solver=solver)
    return prob.value


# ---------------- random channels ----------------
def rand_channel_kraus(din, dout, r, rng):
    G = rng.normal(size=(r * dout, din)) + 1j * rng.normal(size=(r * dout, din))
    Q, _ = np.linalg.qr(G)  # isometry din -> r*dout
    return [Q[k * dout:(k + 1) * dout, :] for k in range(r)]


def kraus_from_choi(J, dy, dx, tol=1e-12):
    w, V = np.linalg.eigh(J)
    Ks = []
    for k in range(len(w)):
        if w[k] > tol:
            v = np.sqrt(w[k]) * V[:, k]
            Ks.append(v.reshape(dy, dx))  # since vec is row-major (y,x)
    return Ks


def mpkron(A, B):
    m, n = A.rows, A.cols; p, q = B.rows, B.cols
    C = mp.zeros(m * p, n * q)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                for l in range(q):
                    C[i * p + k, j * q + l] = A[i, j] * B[k, l]
    return C


def mpouter(v):
    n = len(v)
    M = mp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = v[i] * mp.conj(v[j])
    return M


def mp2np(M):
    return np.array([[complex(M[i, j]) for j in range(M.cols)] for i in range(M.rows)])


def _range(A, tol):
    w, V = np.linalg.eigh((A + A.conj().T) / 2)
    return V[:, w > tol * max(w.max(), 1e-300)]


def short_to(A, Pm, Pperp):
    a11 = Pm.conj().T @ A @ Pm
    if Pperp.shape[1] == 0:
        return a11
    a12 = Pm.conj().T @ A @ Pperp; a22 = Pperp.conj().T @ A @ Pperp
    return a11 - a12 @ np.linalg.pinv(a22, rcond=1e-12, hermitian=True) @ a12.conj().T


def gm_sub(A, B, tol=1e-10):
    """Exact-subspace geometric mean: A#B = A_M # B_M on M = ran A cap ran B."""
    n = A.shape[0]
    RA, RB = _range(A, tol), _range(B, tol)
    # intersection: vectors in ran A with unit cosine to ran B
    U, s, Vh = np.linalg.svd(RA.conj().T @ RB) if RA.shape[1] and RB.shape[1] else (None, np.array([]), None)
    k = int((s > 1 - 1e-9).sum())
    if k == 0:
        return np.zeros((n, n), complex)
    M = RA @ U[:, :k]
    Q, _ = np.linalg.qr(np.hstack([M, np.eye(n)]))
    Pm = M; Pp = Q[:, k:n]
    G = gm_np(short_to(A, Pm, Pp), short_to(B, Pm, Pp))
    return Pm @ G @ Pm.conj().T


def fid_sub(JA, JB, dy, dx, tol=1e-10):
    G = gm_sub(JA, JB, tol)
    T = ptrace_Y(G, dy, dx); T = (T + T.conj().T) / 2
    return np.linalg.eigvalsh(T)[0]
