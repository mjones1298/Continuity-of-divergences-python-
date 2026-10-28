import numpy as np
from scipy.linalg import eigh
from core import herm, psd_fun, mpow, ptrace_out, choi

rng = np.random.default_rng(1)

def rand_c(m, n):
    return rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))

def vec(K):
    # Watrous vec: sum_j K e_j (x) e_j  (column-stacking is e_j (x) K e_j ; Watrous: vec(E_ab)=e_a (x) e_b)
    dY, dX = K.shape
    return np.concatenate([K[i, :] for i in range(dY)])  # row-stacking: vec(e_a e_b^*) = e_a (x) e_b

def Wmat(kraus):
    return np.stack([vec(K) for K in kraus], axis=1)

def choi_w(kraus, dX):
    # Watrous: J(Phi) = sum_{ab} Phi(E_ab) (x) E_ab  on Y (x) X ; for Phi=K . K^*, J = vec_W(K) vec_W(K)^*, with
    # vec_W(K) = sum_b K e_b (x) e_b
    dY = kraus[0].shape[0]
    vs = [sum(np.kron(K[:, b], np.eye(dX)[:, b]) for b in range(dX)) for K in kraus]
    return np.stack(vs, axis=1)

def rand_kraus(r, dY, dX):
    return [rand_c(dY, dX) for _ in range(r)]

def effect(r, spec=None):
    Q, _ = np.linalg.qr(rand_c(r, r))
    if spec is None:
        spec = rng.uniform(0, 1, r)
    return herm(Q @ np.diag(spec) @ Q.conj().T)

def psd_fun2(A, f):
    w, V = eigh(herm(A))
    return (V * f(w)) @ V.conj().T

def opnorm(A):
    return np.linalg.norm(A, 2)

def perspective_pd(A, B, f):
    Bh = mpow(B, 0.5); Bih = mpow(B, -0.5)
    X = herm(Bih @ A @ Bih)
    return herm(Bh @ psd_fun2(X, f) @ Bh)

def Df_reg(A, B, f, eta=1e-9):
    n = A.shape[0]
    I = np.eye(n)
    return np.trace(perspective_pd(A + eta * I, B + eta * I, f)).real

def gmean_alpha_reg(A, B, alpha, eta=1e-10):
    n = A.shape[0]; I = np.eye(n)
    return perspective_pd(A + eta * I, B + eta * I, lambda x: np.sign(x)*np.abs(x) ** alpha) if False else None

def mean_pd(A, B, fr):
    """A sigma B = A^{1/2} f(A^{-1/2} B A^{-1/2}) A^{1/2}"""
    Ah = mpow(A, 0.5); Aih = mpow(A, -0.5)
    X = herm(Aih @ B @ Aih)
    return herm(Ah @ psd_fun2(X, fr) @ Ah)

def mean_reg(A, B, fr, eta=1e-10):
    n = A.shape[0]; I = np.eye(n)
    return mean_pd(A + eta * I, B + eta * I, fr)

def ptr_Y(M, dY, dX):
    return ptrace_out(M, dY, dX)
