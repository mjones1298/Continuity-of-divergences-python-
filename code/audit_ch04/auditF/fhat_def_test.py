import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
from scipy.optimize import minimize
rng = np.random.default_rng(3)
dY, dX = 2, 2
N = dY*dX
# a pair of CP maps with singular-ish but regular positive-definite Choi (random Kraus)
KP = random_kraus(dY, dX, 3, rng); KQ = random_kraus(dY, dX, 3, rng)
JP = sum(np.outer(vec(K), vec(K).conj()) for K in KP); JQ = sum(np.outer(vec(K), vec(K).conj()) for K in KQ)
M = ptrace_Y(ka_frame_conn(JP, JQ, make_conns()['geo']), dY, dX)
Ff = np.linalg.eigvalsh(herm(M))[0]
def apply_kraus_tensor(Ks, rho, dX):
    out = 0
    for K in Ks:
        KK = np.kron(K, np.eye(dX))
        out = out + KK @ rho @ KK.conj().T
    return out
def Fdef(Avec):
    A = (Avec[:dX*dX] + 1j*Avec[dX*dX:]).reshape(dX, dX)
    A = A/np.linalg.norm(A)
    u = vec(A)  # in X(x)X: sum_j A e_j (x) e_j
    rho = np.outer(u, u.conj())
    P = apply_kraus_tensor(KP, rho, dX); Q = apply_kraus_tensor(KQ, rho, dX)
    return np.trace(ka_reg_conn(P, Q, make_conns()['geo'], 1e-12)).real
best = 9
for s in range(30):
    x0 = rng.normal(size=2*dX*dX)
    res = minimize(Fdef, x0, method='Nelder-Mead', options=dict(xatol=1e-9, fatol=1e-12, maxiter=4000))
    best = min(best, res.fun)
print("lambda_min formula:", Ff, "  min over sampled input states of tr(Phi_rho # Psi_rho):", best)
