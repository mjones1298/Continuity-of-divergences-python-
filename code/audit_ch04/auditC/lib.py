import numpy as np
from scipy.linalg import eigh, sqrtm
rng = np.random.default_rng(1)

def herm(A): return (A + A.conj().T) / 2

def fun(A, f):
    w, V = eigh(herm(A))
    return (V * f(w)) @ V.conj().T

def vec(K):  # row-major, Watrous: sum_x K e_x (x) e_x
    return K.reshape(-1)

def rand_kraus(dX, dY, r, rng=rng, scale=None):
    Ks = [rng.normal(size=(dY, dX)) + 1j * rng.normal(size=(dY, dX)) for _ in range(r)]
    return Ks

def W_of(Ks):
    return np.stack([vec(K) for K in Ks], axis=1)   # (dY dX) x r

def ptraceY(J, dY, dX):
    return np.einsum('aiaj->ij', J.reshape(dY, dX, dY, dX))

def rand_T(r, lo=0.0, hi=1.0, rng=rng, spec=None):
    Q, _ = np.linalg.qr(rng.normal(size=(r, r)) + 1j * rng.normal(size=(r, r)))
    if spec is None:
        spec = rng.uniform(lo, hi, size=r)
    return herm((Q * spec) @ Q.conj().T)

def h_of(f):
    return lambda t: (1 - t) * f(t / (1 - t))

def hfun_T(T, h):
    w, V = eigh(herm(T))
    w = np.clip(w, 0.0, 1.0)
    return (V * h(w)) @ V.conj().T

def M_of(W, T, h, dY, dX):
    H = hfun_T(T, h)
    return herm(ptraceY(W @ H @ W.conj().T, dY, dX))

# --- direct evaluation of maximal f-divergence for PSD pair via eta-regularisation
def Dhat_reg(A, B, f, eta):
    n = A.shape[0]
    Ae = A + eta * np.eye(n); Be = B + eta * np.eye(n)
    w, V = eigh(herm(Be))
    Bih = (V / np.sqrt(w)) @ V.conj().T
    Bh = (V * np.sqrt(w)) @ V.conj().T
    X = herm(Bih @ Ae @ Bih)
    P = Bh @ fun(X, f) @ Bh
    return np.trace(P).real

def Dhat_support(A, B, f, tol=1e-10):
    """Def 6.2 / Lemma 6.4(3): requires range(A) in range(B)."""
    w, V = eigh(herm(B))
    keep = w > tol * max(1.0, w.max())
    U = V[:, keep]
    A0 = U.conj().T @ A @ U; B0 = U.conj().T @ B @ U
    # check range inclusion
    Ares = A - U @ A0 @ U.conj().T
    res = np.linalg.norm(Ares)
    w0, V0 = eigh(herm(B0))
    Bih = (V0 / np.sqrt(w0)) @ V0.conj().T
    Bh = (V0 * np.sqrt(w0)) @ V0.conj().T
    X = herm(Bih @ A0 @ Bih)
    P = Bh @ fun(X, f) @ Bh
    return np.trace(P).real, res

def phi_state(W, T, A, dY, dX):
    """Phi_rho, Psi_rho for rho = vec(A)vec(A)^*/||A||^2 ; returns unnormalised (A as given)"""
    C = np.kron(np.eye(dY), A.T)
    J1 = W @ T @ W.conj().T
    J2 = W @ (np.eye(T.shape[0]) - T) @ W.conj().T
    return C @ J1 @ C.conj().T, C @ J2 @ C.conj().T
