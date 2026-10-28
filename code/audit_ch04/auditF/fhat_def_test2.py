import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
rng = np.random.default_rng(3)
dY, dX = 2, 2
geo = make_conns()['geo']
KP = random_kraus(dY, dX, 3, rng); KQ = random_kraus(dY, dX, 3, rng)
JP = sum(np.outer(vec(K), vec(K).conj()) for K in KP); JQ = sum(np.outer(vec(K), vec(K).conj()) for K in KQ)
M = ptrace_Y(ka_frame_conn(JP, JQ, geo), dY, dX)
def tensor_apply(Ks, rho):
    return sum(np.kron(K, np.eye(dX)) @ rho @ np.kron(K, np.eye(dX)).conj().T for K in Ks)
worst = 0
vals = []
for s in range(200):
    A = rng.normal(size=(dX, dX)) + 1j*rng.normal(size=(dX, dX)); A /= np.linalg.norm(A)
    u = vec(A); rho = np.outer(u, u.conj())
    P = tensor_apply(KP, rho); Q = tensor_apply(KQ, rho)
    val = np.trace(ka_frame_conn(P, Q, geo)).real
    pred = np.trace((A @ A.conj().T).T @ M).real
    worst = max(worst, abs(val - pred)); vals.append(val)
print("max |tr(Phi_rho # Psi_rho) - tr((AA^*)^T M)| =", worst, "; min over samples =", min(vals), "  lambda_min(M)=", np.linalg.eigvalsh(M)[0])
