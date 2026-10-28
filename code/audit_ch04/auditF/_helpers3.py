import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
rng = np.random.default_rng(15)
conns = make_conns()
def kraus_from_choi(J, dY, dX, tol=1e-10):
    w, V = np.linalg.eigh(herm(J))
    return [np.sqrt(w[i]) * V[:, i].reshape(dY, dX) for i in range(len(w)) if w[i] > tol]
def p1(Ks): return np.linalg.norm(sum(K.conj().T @ K for K in Ks), 2) if Ks else 0.0   # ||.||_{1->1} = ||Xi^*(1)||
def pinf(Ks): return np.linalg.norm(sum(K @ K.conj().T for K in Ks), 2) if Ks else 0.0 # ||.||_{inf->inf} = ||Xi(1)||
def p2(Ks):  # Hilbert-Schmidt 2->2: largest singular value of superoperator
    if not Ks: return 0.0
    return np.linalg.norm(sum(np.kron(K, K.conj()) for K in Ks), 2)
