"""Independent checker library (fresh code). Choi ordering: Y (output) first, X (input) second.
J(Phi) = sum_ij Phi(E_ij) (x) E_ij.
"""
import numpy as np
import mpmath as mp

mp.mp.dps = 60

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
PAULI = [I2, X, Y, Z]


# ---------------- numpy (double) side ----------------
def choi_from_kraus(Ks, dX):
    dY = Ks[0].shape[0]
    J = np.zeros((dY * dX, dY * dX), dtype=complex)
    for i in range(dX):
        for j in range(dX):
            Eij = np.zeros((dX, dX), dtype=complex); Eij[i, j] = 1
            out = sum(K @ Eij @ K.conj().T for K in Ks)
            J += np.kron(out, Eij)
    return J


def choi_from_map(f, dX):
    J = None
    for i in range(dX):
        for j in range(dX):
            Eij = np.zeros((dX, dX), dtype=complex); Eij[i, j] = 1
            out = f(Eij)
            if J is None:
                J = np.zeros((out.shape[0] * dX,) * 2, dtype=complex)
            J += np.kron(out, Eij)
    return J


def depol(p):
    return lambda r: (1 - p) * r + p * np.trace(r) * I2 / 2


def Rz(delta):
    return np.diag([np.exp(-1j * delta / 2), np.exp(1j * delta / 2)])


def Np_map(p, delta=0.0):
    V = Rz(delta) @ H
    D = depol(p)
    return lambda r: D(V @ r @ V.conj().T)


def J_Np(p, delta=0.0):
    return choi_from_map(Np_map(p, delta), 2)


def ptrace_Y(J, dY, dX):
    return np.einsum('yxyz->xz', J.reshape(dY, dX, dY, dX))


def diamond_sdp(Jdelta, dY, dX, solver='CLARABEL'):
    """Watrous: ||Delta||_diamond = 2 max tr(J W), 0<=W<=1_Y (x) rho, rho density on X.
    Valid for Delta = difference of trace-preserving CP maps (trace-annihilating, Hermiticity-preserving)."""
    import cvxpy as cp
    n = dY * dX
    W = cp.Variable((n, n), hermitian=True)
    rho = cp.Variable((dX, dX), hermitian=True)
    cons = [W >> 0, cp.kron(np.eye(dY), rho) - W >> 0, rho >> 0, cp.real(cp.trace(rho)) == 1]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(Jdelta @ W))), cons)
    prob.solve(solver=solver)
    return 2 * prob.value


def random_channel_kraus(dX, dY, r, rng):
    G = rng.normal(size=(dY * r, dX)) + 1j * rng.normal(size=(dY * r, dX))
    Q, _ = np.linalg.qr(G)  # isometry dX -> dY*r
    return [Q[k * dY:(k + 1) * dY, :] for k in range(r)]


# ---------------- mpmath (high precision) side ----------------
def to_mp(A):
    A = np.asarray(A)
    return mp.matrix([[mp.mpc(complex(A[i, j])) for j in range(A.shape[1])] for i in range(A.shape[0])])


def mp_herm(A):
    return (A + A.H) / 2


def mp_funm_h(A, f):
    A = mp_herm(A)
    E, Q = mp.eighe(A) if hasattr(mp, 'eighe') else mp.eigh(A)
    n = A.rows
    D = mp.zeros(n, n)
    for i in range(n):
        D[i, i] = f(E[i])
    return Q * D * Q.H


def mp_sqrtm(A):
    return mp_funm_h(A, lambda x: mp.sqrt(x) if x > 0 else mp.mpf(0))


def mp_isqrtm(A):
    return mp_funm_h(A, lambda x: 1 / mp.sqrt(x))


def mp_gmean(A, B):
    As = mp_sqrtm(A); Ais = mp_isqrtm(A)
    M = mp_herm(Ais * B * Ais)
    return mp_herm(As * mp_sqrtm(M) * As)


def mp_ptrace_Y(J, dY, dX):
    R = mp.zeros(dX, dX)
    for x in range(dX):
        for z in range(dX):
            R[x, z] = sum(J[y * dX + x, y * dX + z] for y in range(dY))
    return R


def mp_eigvals_h(A):
    E, Q = mp.eigh(mp_herm(A))
    return sorted([mp.re(e) for e in E])


def Fhat_mp(J1, J2, dY=2, dX=2):
    G = mp_gmean(J1, J2)
    return mp_eigvals_h(mp_ptrace_Y(G, dY, dX))[0]


def mp_choi_Np(p, delta=0):
    """Exact high-precision Choi of D_p o (Rz(delta) H)."""
    p = mp.mpf(p); delta = mp.mpf(delta)
    s = 1 / mp.sqrt(2)
    Hm = mp.matrix([[s, s], [s, -s]])
    R = mp.matrix([[mp.exp(-1j * delta / 2), 0], [0, mp.exp(1j * delta / 2)]])
    V = R * Hm
    J = mp.zeros(4, 4)
    for i in range(2):
        for j in range(2):
            E = mp.zeros(2, 2); E[i, j] = 1
            out = (1 - p) * V * E * V.H
            if i == j:
                out += p * mp.eye(2) / 2
            for a in range(2):
                for b in range(2):
                    J[a * 2 + i, b * 2 + j] += out[a, b]
    return J


def mp_choi_from_np(J):
    return to_mp(J)
