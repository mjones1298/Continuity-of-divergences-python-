import numpy as np, sys
from scipy.optimize import minimize
from core import *
rng = np.random.default_rng(7)

def rk(n, dy, dx, scale=1.0):
    return [scale*(rng.normal(size=(dy,dx))+1j*rng.normal(size=(dy,dx)))/np.sqrt(2) for _ in range(n)]

def apply_Id(kraus, rho, dx):   # (Phi (x) 1)(rho), rho on X(x)R (X first)
    out = 0
    for K in kraus:
        KK = np.kron(K, np.eye(dx))
        out = out + KK @ rho @ KK.conj().T
    return out

def F_of_A(A, kp, kq, dx):
    u = A.reshape(-1)    # vec(A) = sum_ij A_ij e_i (x) e_j  (row-major reshape)
    u = u/np.linalg.norm(u)
    rho = np.outer(u, u.conj())
    P = apply_Id(kp, rho, dx); Q = apply_Id(kq, rho, dx)
    G,_,_ = gmean_frame(P, Q)
    return np.trace(G).real

def lam_formula(kp, kq, dx, dy):
    Jp = choi(kp, dx); Jq = choi(kq, dx)
    G,_,_ = gmean_frame(Jp, Jq)
    M = ptrace_out(G, dy, dx)
    return np.linalg.eigvalsh((M+M.conj().T)/2)[0]

def direct(kp, kq, dx, nstart=12):
    best = 10
    for s in range(nstart):
        x0 = rng.normal(size=2*dx*dx)
        fun = lambda x: F_of_A((x[:dx*dx]+1j*x[dx*dx:]).reshape(dx,dx), kp, kq, dx)
        r = minimize(fun, x0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-10, fatol=1e-13))
        r = minimize(fun, r.x, method='BFGS')
        best = min(best, r.fun)
    # random sampling lower bound check
    samp = min(F_of_A(rng.normal(size=(dx,dx))+1j*rng.normal(size=(dx,dx)), kp, kq, dx) for _ in range(2000))
    return min(best, samp), samp

cases = []
for dx, dy in [(2,2),(2,3),(3,2)]:
    n = dx*dy
    # full-rank-ish channels (invertible sum)
    cases.append((f"rand full dx={dx} dy={dy}", dx, dy, rk(n,dy,dx), rk(n,dy,dx)))
    # singular sum
    cases.append((f"rand sing dx={dx} dy={dy}", dx, dy, rk(2,dy,dx), rk(1,dy,dx)))
    K = rk(3,dy,dx)
    cases.append((f"shared Kraus dx={dx} dy={dy}", dx, dy, [K[0],K[1]], [0.7*K[1],K[2]]))
# unitary-ish channels
def hrand(d):
    Z = rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)); Q,_ = np.linalg.qr(Z); return Q
U = hrand(2); U2 = hrand(2)
cases.append(("two unitaries dx=dy=2", 2, 2, [U], [U2]))
cases.append(("unitary vs depolarised-unitary mix", 2, 2, [np.sqrt(.9)*U, np.sqrt(.1)*U2], [U]))
cases.append(("Phi=Psi", 2, 2, rk(3,2,2), None))
for name, dx, dy, kp, kq in cases:
    if kq is None: kq = kp
    lam = lam_formula(kp, kq, dx, dy)
    dmin, samp = direct(kp, kq, dx)
    print(f"{name:40s} lam_min formula={lam:.8f}   direct min={dmin:.8f}  (diff {dmin-lam:+.2e}) ; min over 2000 random u = {samp:.6f}")
