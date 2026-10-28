import numpy as np
from scipy.linalg import eigh

def herm(A): return (A + A.conj().T) / 2
def psd_fun(A, f):
    w, V = eigh(herm(A)); return (V * f(w)) @ V.conj().T
def mpow(A, p, tol=1e-13):
    w, V = eigh(herm(A)); w = np.clip(w, 0, None)
    return (V * np.where(w > tol, w ** p, 0.0)) @ V.conj().T

def vec(K):
    # vec(K) = sum_j K e_j (x) e_j  (K: dX -> dY), vector in Y (x) X
    dY, dX = K.shape
    return sum(np.kron(K[:, j], np.eye(dX)[:, j]) for j in range(dX))

def unvec(v, dY, dX):
    V = v.reshape(dY, dX)   # v = sum_j (K e_j) (x) e_j  => v[a*dX+j] = K[a,j]
    return V

def choi_from_W(W): return W @ W.conj().T
def ptrace_Y(J, dY, dX): return np.einsum('aiaj->ij', J.reshape(dY, dX, dY, dX))

def kraus_to_W(kraus):
    return np.array([vec(K) for K in kraus]).T   # columns vec(K_k)

def random_kraus(dY, dX, r, rng):
    return [rng.normal(size=(dY, dX)) + 1j * rng.normal(size=(dY, dX)) for _ in range(r)]

# ---- Kubo-Ando connections: represented by (a, b, measure as list of (s, weight))
def phi_sigma(t, a, b, meas):
    """phi(t) = a t + b (1-t) + sum_k w_k * (1+s_k) t (1-t) / (1 + (s_k-1) t)"""
    t = np.asarray(t, float)
    out = a * t + b * (1 - t)
    for s, w in meas:
        den = 1 + (s - 1) * t
        out = out + w * (1 + s) * t * (1 - t) / den
    return out

def eta_sigma(t, meas):
    return phi_sigma(t, 0, 0, meas)

def f_sigma(x, a, b, meas):
    x = np.asarray(x, float)
    out = a + b * x
    for s, w in meas: out = out + w * x * (1 + s) / (x + s)
    return out

def ka_reg(A, B, a, b, meas, eta):
    """regularised Kubo-Ando connection: (A+eta) sigma (B+eta), by the PD formula."""
    n = A.shape[0]
    Ae = A + eta * np.eye(n); Be = B + eta * np.eye(n)
    Ah = psd_fun(Ae, np.sqrt); Aih = psd_fun(Ae, lambda w: 1 / np.sqrt(w))
    M = herm(Aih @ Be @ Aih)
    w, V = eigh(M)
    fw = f_sigma(np.clip(w, 1e-300, None), a, b, meas)
    return herm(Ah @ ((V * fw) @ V.conj().T) @ Ah)

def ka_frame(A, B, a, b, meas, tol=1e-11):
    """exact via S=A+B frame: W phi(T) W^*"""
    S = herm(A + B)
    w, V = eigh(S)
    keep = w > tol * max(1.0, w.max())
    Vr = V[:, keep]; wr = w[keep]
    Wf = Vr * np.sqrt(wr); Wi = Vr / np.sqrt(wr)
    T = herm(Wi.conj().T @ A @ Wi)
    wt, U = eigh(T)
    wt = np.clip(wt, 0, 1)
    ph = phi_sigma(wt, a, b, meas)
    return herm(Wf @ ((U * ph) @ U.conj().T) @ Wf.conj().T)

GEO = (0.0, 0.0, None)  # placeholder

# ---------- general connections by representing function f on (0,inf) ----------
class Conn:
    def __init__(self, name, f, a, b, mpos=True):
        self.name = name; self.f = f; self.a = a; self.b = b   # a=f(0+), b=f'(inf)
        self.mpos = mpos                                          # m_sigma != 0 ?
    def phi(self, t):
        t = np.asarray(t, float)
        out = np.empty_like(t)
        flat = t.ravel(); o = out.ravel()
        for i, x in enumerate(flat):
            if x <= 0: o[i] = self.b
            elif x >= 1: o[i] = self.a
            else: o[i] = x * self.f((1 - x) / x)
        return out
    def eta(self, t):
        t = np.asarray(t, float)
        return self.phi(t) - (self.a * t + self.b * (1 - t))

def make_conns():
    C = {}
    C['geo'] = Conn('geo', lambda x: np.sqrt(x), 0, 0)
    C['harm'] = Conn('harm', lambda x: 2 * x / (1 + x), 0, 0)
    C['arith'] = Conn('arith', lambda x: (1 + x) / 2, .5, .5, mpos=False)
    C['alpha0.3'] = Conn('#0.3', lambda x: x ** 0.3, 0, 0)
    C['alpha0.8'] = Conn('#0.8', lambda x: x ** 0.8, 0, 0)
    C['log'] = Conn('log', lambda x: (x - 1) / np.log(x) if abs(x - 1) > 1e-12 else 1.0, 0, 0)
    # mixture with arithmetic part: f = 0.3 + 0.2 x + 0.5*2x/(x+1)  -> mean (a+b+m =1)
    C['mix'] = Conn('mix', lambda x: 0.3 + 0.2 * x + 0.5 * 2 * x / (1 + x), 0.3, 0.2)
    C['skew'] = Conn('skew', lambda x: 0.4 + 0.6*np.sqrt(x), 0.4, 0.0)   # a=0.4,b=0 ; 0.6 sqrt: a connection (not nec mean normal.)
    return C

def ka_reg_conn(A, B, conn, eta):
    n = A.shape[0]
    Ae = A + eta * np.eye(n); Be = B + eta * np.eye(n)
    Ah = psd_fun(Ae, np.sqrt); Aih = psd_fun(Ae, lambda w: 1 / np.sqrt(w))
    M = herm(Aih @ Be @ Aih)
    w, V = eigh(M)
    fw = np.array([conn.f(max(x, 1e-300)) for x in w])
    return herm(Ah @ ((V * fw) @ V.conj().T) @ Ah)

def ka_frame_conn(A, B, conn, tol=1e-11):
    S = herm(A + B)
    w, V = eigh(S)
    keep = w > tol * max(1.0, w.max())
    Vr = V[:, keep]; wr = w[keep]
    Wf = Vr * np.sqrt(wr); Wi = Vr / np.sqrt(wr)
    T = herm(Wi.conj().T @ A @ Wi)
    wt, U = eigh(T)
    wt = np.clip(wt, 0, 1)
    ph = conn.phi(wt)
    return herm(Wf @ ((U * ph) @ U.conj().T) @ Wf.conj().T)
