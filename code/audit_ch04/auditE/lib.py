import numpy as np
from scipy.linalg import eigh
np.set_printoptions(precision=6, suppress=True, linewidth=150)

def herm(A): return (A + A.conj().T) / 2

def rand_psd(N, rank, rng, scale=1.0):
    G = rng.standard_normal((N, rank)) + 1j * rng.standard_normal((N, rank))
    return scale * (G @ G.conj().T) / rank

def fun_herm(A, f):
    w, V = eigh(herm(A))
    return (V * f(w)) @ V.conj().T

def frame(Jsum, tol=1e-11):
    w, V = eigh(herm(Jsum))
    keep = w > tol * max(1.0, w.max())
    return V[:, keep] * np.sqrt(w[keep])

def ptrace_Y(J, dY, dX):
    return np.einsum('aiaj->ij', J.reshape(dY, dX, dY, dX))

def lmin(M): return np.linalg.eigvalsh(herm(M))[0]
def lmax(M): return np.linalg.eigvalsh(herm(M))[-1]
def opn(M):
    if M.size == 0: return 0.0
    return np.linalg.norm(M, 2)

def gmean_reg(A, B, eta=1e-10):
    # regularised geometric mean (A+eta)#(B+eta), reference implementation, independent of frames
    N = A.shape[0]
    Ae = A + eta*np.eye(N); Be = B + eta*np.eye(N)
    Ah = fun_herm(Ae, lambda w: np.sqrt(w)); Aih = fun_herm(Ae, lambda w: 1/np.sqrt(w))
    M = herm(Aih @ Be @ Aih)
    return herm(Ah @ fun_herm(M, lambda w: np.sqrt(np.clip(w,0,None))) @ Ah)

def gmean_frame(A, B):
    W = frame(A + B)
    Wp = np.linalg.pinv(W)
    T = herm(Wp @ A @ Wp.conj().T)
    g = fun_herm(T, lambda t: np.sqrt(np.clip(t*(1-t), 0, None)))
    return herm(W @ g @ W.conj().T)

def Fhat(JP, JQ, dY, dX):
    return lmin(ptrace_Y(gmean_frame(JP, JQ), dY, dX))

class Setup:
    """Four CP maps given by Choi matrices; frame of Omega' = Omega1+Omega2."""
    def __init__(self, J1, K1, J2, K2, dY, dX, tol=1e-11):
        self.dY, self.dX = dY, dX
        self.JPhi = [J1, J2]; self.JPsi = [K1, K2]
        self.Om = [J1 + K1, J2 + K2]
        self.Omp = self.Om[0] + self.Om[1]
        self.Omb = self.Omp / 2
        self.W = frame(self.Omp, tol)
        W = self.W; self.Wp = np.linalg.pinv(W)
        self.r = W.shape[1]
        self.A = [herm(self.Wp @ J @ self.Wp.conj().T) for J in self.JPhi]
        self.B = [herm(self.Wp @ J @ self.Wp.conj().T) for J in self.JPsi]
        self.C = [self.A[i] + self.B[i] for i in range(2)]
        self.nOmb = opn(ptrace_Y(self.Omb, dY, dX))      # ||Omegabar^*(1)||
        self.nOm = [opn(ptrace_Y(O, dY, dX)) for O in self.Om]
        self.delta = max(opn(self.A[0]-self.A[1]), opn(self.B[0]-self.B[1]))
        self.gamma = opn(self.C[0] - 0.5*np.eye(self.r))
    def Delta(self, Z):
        return ptrace_Y(self.W @ Z @ self.W.conj().T, self.dY, self.dX)


def projectors_C1(C1, tol=1e-8):
    w, V = eigh(herm(C1))
    def P(mask):
        Vm = V[:, mask]
        return Vm @ Vm.conj().T
    Q1 = P(w > 1 - tol); Q2 = P(w < tol); QR = P((w >= tol) & (w <= 1 - tol))
    return Q1, QR, Q2

def sec11(S, tol=1e-8):
    """All quantities of Section 11 for a Setup S."""
    r = S.r
    C1, C2 = S.C
    Q1, QR, Q2 = projectors_C1(C1, tol)
    a1 = fun_herm(C1, lambda w: np.sqrt(np.clip(w, 0, None)))
    a2 = fun_herm(C2, lambda w: np.sqrt(np.clip(w, 0, None)))
    a1p = np.linalg.pinv(a1, rcond=1e-9, hermitian=True); a2p = np.linalg.pinv(a2, rcond=1e-9, hermitian=True)
    T1 = herm(a1p @ S.A[0] @ a1p); T2 = herm(a2p @ S.A[1] @ a2p)
    out = dict(Q1=Q1, QR=QR, Q2=Q2, a1=a1, a2=a2, T1=T1, T2=T2)
    out['tau_ac'] = opn(QR @ (T1 - T2) @ QR)
    out['kap1'] = opn(Q1 @ T1 @ QR); out['kap2'] = opn(Q2 @ T2 @ QR)
    out['gam_ac'] = opn(QR @ (C1 - 0.5*np.eye(r)) @ QR)
    out['tanh_ac'] = 2*out['gam_ac']
    out['ms'] = opn(S.Delta(Q1 + Q2))
    out['hac2'] = opn(S.Delta(QR @ (a1 - a2) @ (a1 - a2) @ QR))
    out['hac'] = np.sqrt(max(out['hac2'], 0))
    out['DQR'] = opn(S.Delta(QR))
    return out

def R_gsharp(S, q):
    nO = S.nOmb
    return (nO*np.sqrt(q['tau_ac']) + 0.5*q['ms'] + S.nOm[0]*np.sqrt(q['kap1']) + S.nOm[1]*np.sqrt(q['kap2'])
            + min(np.sqrt(2)*nO*q['tanh_ac'], np.sqrt(nO)*q['hac']))


from scipy.optimize import minimize as _minimize
def nu_of_frame(W, dY, dX, rng, starts=30):
    r = W.shape[1]
    Vs = [W[:, k].reshape(dY, dX) for k in range(r)]
    def f(x):
        c = x[:r] + 1j*x[r:]
        nc = np.linalg.norm(c)
        M = sum(c[k]*Vs[k] for k in range(r))
        return np.linalg.norm(M, 2)/nc
    best = np.inf
    for s in range(starts):
        x0 = rng.standard_normal(2*r)
        res = _minimize(f, x0, method='Nelder-Mead', options={'xatol':1e-10,'fatol':1e-13,'maxiter':4000})
        res = _minimize(f, res.x, method='BFGS')
        best = min(best, res.fun)
    return best
