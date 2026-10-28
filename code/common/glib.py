"""Minimal self-contained helpers for the proofs in pair_bound.md / open_problems.md.
Conventions (Watrous): J(Phi) on Y (x) X, vec row-major (vec(V)=V.reshape(-1)),
J = sum_k vec(V_k)vec(V_k)^*, Tr_Y J(Phi) = Phi^*(1)^T.  Fhat = lambda_min Tr_Y(J1 # J2)."""
import numpy as np, scipy.linalg as sla

def herm(A): return 0.5*(A+A.conj().T)
def vec(V): return np.asarray(V).reshape(-1)
def choi(Ks):
    return herm(sum(np.outer(vec(K), vec(K).conj()) for K in Ks))
def trY(M, dY, dX):
    return np.einsum('iaib->ab', M.reshape(dY, dX, dY, dX))
def fn(A, f):
    w, U = np.linalg.eigh(herm(A)); return (U*f(w))@U.conj().T
def sqrtm(A): return fn(A, lambda w: np.sqrt(np.clip(w, 0, None)))
def gmean_pd(A, B, t=0.5):
    """A #_t B for A>0 (A^{1/2}(A^{-1/2}BA^{-1/2})^t A^{1/2})."""
    Ah = sqrtm(A); Aih = fn(A, lambda w: w**-0.5)
    return herm(Ah@fn(Aih@B@Aih, lambda w: np.clip(w, 0, None)**t)@Ah)
def frame(JOm, tol=1e-11):
    w, U = np.linalg.eigh(herm(JOm)); k = w > tol*max(1, w.max())
    return U[:, k]*np.sqrt(w[k])
def arn(W, J):
    Wp = np.linalg.pinv(W); return herm(Wp@J@Wp.conj().T)
def gmean(A, B, t=0.5, tol=1e-11):
    """A #_t B for PSD A,B via the ARN frame of A+B: W g_t(T) W^*, g_t(x)=x^{1-t}(1-x)^t
    (A#_tB = A^{1/2}(A^{-1/2}BA^{-1/2})^tA^{1/2}; T = derivative of A)."""
    W = frame(A+B, tol); T = arn(W, A)
    g = lambda x: np.clip(x, 0, 1)**(1-t)*np.clip(1-x, 0, 1)**t
    return herm(W@fn(T, g)@W.conj().T)
def Fhat(J1, J2, dY, dX):
    return np.linalg.eigvalsh(trY(gmean(J1, J2), dY, dX)).min()
def dnorm(Jd, dY, dX):
    """diamond norm of a Hermitian-preserving map from its Choi matrix (Watrous SDP)."""
    import cvxpy as cp
    n = dY*dX
    W0 = cp.Variable((n, n), hermitian=True); rho = cp.Variable((dX, dX), hermitian=True)
    cons = [W0 >> 0, rho >> 0, cp.real(cp.trace(rho)) == 1,
            W0 << cp.kron(np.eye(dY), rho)]
    Jp = herm(Jd)
    # ||Delta||_dia = 2 max <Jd,W> s.t. 0<=W<=1(x)rho  (for trace-annihilating Delta)
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(Jp@W0))), cons)
    prob.solve(solver='CLARABEL')
    return 2*prob.value
def rand_kraus(rng, dY, dX, r):
    r = max(int(r), -(-dX//dY))   # need r*dY >= dX for a channel
    G = rng.normal(size=(r, dY, dX))+1j*rng.normal(size=(r, dY, dX))
    S = sum(K.conj().T@K for K in G); Si = fn(S, lambda w: w**-0.5)
    return [K@Si for K in G]
def rand_unitary(rng, d):
    Q, R = np.linalg.qr(rng.normal(size=(d, d))+1j*rng.normal(size=(d, d)))
    return Q*(np.diag(R)/abs(np.diag(R)))
