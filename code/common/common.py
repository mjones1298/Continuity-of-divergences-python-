"""Shared helpers for A_numerics (fresh code; re-verified against check2/lib.py conventions).
Choi convention: J(Phi) = sum_ij Phi(E_ij) (x) E_ij  (output Y first, input X second), tr J = d_X.
vec row-major: J = sum_k vec(K_k) vec(K_k)^*.
"""
import numpy as np
import mpmath as mp

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [I2, X, Y, Z]


# ---------------- mpmath matrix utilities ----------------
def mpmat(A):
    A = np.asarray(A)
    return mp.matrix([[mp.mpc(complex(A[i, j])) for j in range(A.shape[1])] for i in range(A.shape[0])])


def herm(A):
    return (A + A.H) / 2


def eigh(A):
    E, Q = mp.eighe(herm(A)) if hasattr(mp, 'eighe') else mp.eigh(herm(A))
    return [mp.re(e) for e in E], Q


def funm(A, f):
    E, Q = eigh(A)
    n = A.rows
    D = mp.zeros(n, n)
    for i in range(n):
        D[i, i] = f(E[i])
    return Q * D * Q.H


def sqrtm(A):
    return funm(A, lambda x: mp.sqrt(x) if x > 0 else mp.mpf(0))


def isqrtm(A):
    return funm(A, lambda x: 1 / mp.sqrt(x))


def logm(A):
    return funm(A, mp.log)


def gmean(A, B):
    """A # B for A invertible (high precision)."""
    As = sqrtm(A); Ais = isqrtm(A)
    M = herm(Ais * B * Ais)
    return herm(As * sqrtm(M) * As)


def ptrace_Y(J, dY, dX):
    R = mp.zeros(dX, dX)
    for x in range(dX):
        for z in range(dX):
            R[x, z] = mp.fsum(J[y * dX + x, y * dX + z] for y in range(dY))
    return R


def eigvals(A):
    return sorted(eigh(A)[0])


def tr(A):
    return mp.fsum(A[i, i] for i in range(A.rows))


def kron(A, B):
    m, n = A.rows, A.cols
    p, q = B.rows, B.cols
    C = mp.zeros(m * p, n * q)
    for i in range(m):
        for j in range(n):
            a = A[i, j]
            if a == 0:
                continue
            for k in range(p):
                for l in range(q):
                    C[i * p + k, j * q + l] = a * B[k, l]
    return C


def relent(r, s):
    return mp.re(tr(r * (logm(r) - logm(s))))


# ---------------- channels ----------------
def choi_from_map_mp(f, dX):
    J = None
    for i in range(dX):
        for j in range(dX):
            E = mp.zeros(dX, dX); E[i, j] = 1
            out = f(E)
            dY = out.rows
            if J is None:
                J = mp.zeros(dY * dX, dY * dX)
            for a in range(dY):
                for b in range(dY):
                    J[a * dX + i, b * dX + j] += out[a, b]
    return J


def Rz_mp(delta):
    delta = mp.mpf(delta)
    return mp.matrix([[mp.exp(-1j * delta / 2), 0], [0, mp.exp(1j * delta / 2)]])


def choi_qubit(p, delta=0):
    """Choi of N_{p,delta} = D_p o Rz(delta) (U = 1 WLOG: input unitaries do not change any quantity here)."""
    p = mp.mpf(p)
    R = Rz_mp(delta)

    def f(E):
        out = (1 - p) * R * E * R.H
        out += p * tr(E) * mp.eye(2) / 2
        return out
    return choi_from_map_mp(f, 2)


def Fhat_qubit_closed(p, delta):
    p = mp.mpf(p); delta = mp.mpf(delta)
    x = 4 * (1 - p) ** 2 * mp.sin(delta / 2) ** 2 / (p * (4 - 3 * p))
    return p / 2 + (1 - p / 2) / mp.sqrt(1 + x)


def DBS_qubit_closed(p, delta):
    p = mp.mpf(p); delta = mp.mpf(delta)
    x = 4 * (1 - p) ** 2 * mp.sin(delta / 2) ** 2 / (p * (4 - 3 * p))
    return (2 - p) * mp.sqrt(x / (1 + x)) * mp.asinh(mp.sqrt(x))


def DBS_channel(J1, J2, dY, dX):
    """Belavkin-Staszewski channel divergence D_BS(Phi||Psi) = lambda_max Tr_Y[ J1^{1/2} log(J1^{1/2} J2^{-1} J1^{1/2}) J1^{1/2} ]
    (both invertible)."""
    s = sqrtm(J1)
    M = herm(s * mp.inverse(J2) * s)
    G = herm(s * logm(M) * s)
    return eigvals(ptrace_Y(G, dY, dX))[-1]


def Fhat_channel(J1, J2, dY, dX):
    return eigvals(ptrace_Y(gmean(J1, J2), dY, dX))[0]


def umegaki_maxent_qubit(p, delta):
    p = mp.mpf(p); delta = mp.mpf(delta)
    return (1 - p) * mp.sin(delta / 2) ** 2 * mp.log(1 + 4 * (1 - p) / p)
