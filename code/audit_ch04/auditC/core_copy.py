import numpy as np
from scipy.linalg import sqrtm, eigh

def herm(A): return (A + A.conj().T) / 2

def psd_fun(A, f):
    w, V = eigh(herm(A))
    return (V * f(np.clip(w, 0, None))) @ V.conj().T

def mpow(A, p):
    w, V = eigh(herm(A))
    w = np.clip(w, 0, None)
    return (V * np.where(w > 1e-14, w ** p, 0.0)) @ V.conj().T

def gmean_pd(A, B):
    Ah = mpow(A, 0.5); Aih = mpow(A, -0.5)
    return herm(Ah @ psd_fun(Aih @ B @ Aih, np.sqrt) @ Ah)

def gmean_frame(A, B, tol=1e-12):
    """Exact Kubo-Ando geometric mean of PSD A,B via S=A+B frame: S^{1/2} g(T) S^{1/2}."""
    S = herm(A + B)
    w, V = eigh(S)
    keep = w > tol * max(1, w.max())
    Vr = V[:, keep]; wr = w[keep]
    Sh = Vr * np.sqrt(wr)                     # N x r frame W
    Sih = Vr / np.sqrt(wr)
    T = herm(Sih.conj().T @ A @ Sih)           # r x r, 0<=T<=1
    g = psd_fun(T, lambda t: np.sqrt(np.clip(t * (1 - t), 0, None)))
    return herm(Sh @ g @ Sh.conj().T), T, Sh

def ptrace_out(J, dY, dX):
    """Tr_Y of operator on Y (x) X."""
    return np.einsum('aiaj->ij', J.reshape(dY, dX, dY, dX))

def choi(kraus, dX):
    """J = sum_ij Phi(E_ij) (x) E_ij on Y (x) X."""
    dY = kraus[0].shape[0]
    J = np.zeros((dY * dX, dY * dX), complex)
    for K in kraus:
        v = sum(np.kron(K[:, j], np.eye(dX)[:, j]) for j in range(dX))  # vec(K)
        J += np.outer(v, v.conj())
    return J

def Fhat_from_mean(M, dY, dX):
    return np.linalg.eigvalsh(herm(ptrace_out(M, dY, dX)))[0]

def kyfan(rho, c):
    w = np.sort(np.linalg.eigvalsh(herm(rho)))[::-1]
    return w[:c].sum()

# Pauli matrices
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.]).astype(complex)
