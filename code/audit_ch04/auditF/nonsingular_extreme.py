import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
rng = np.random.default_rng(9)
geo = make_conns()['geo']
dY=dX=2
e = np.eye(2)
y1 = np.array([1,0.]); y2 = np.array([0,1.]); y3 = np.array([1,1.])/np.sqrt(2)
# Omega Kraus: K1 = |y1><e1|, K2 = |y2><e2|, K3 = |y3><e2| ; T = diag(1/2, 1, 0)
K = [np.outer(y1, e[0]), np.outer(y2, e[1]), np.outer(y3, e[1])]
W = np.array([vec(k) for k in K]).T.astype(complex)
T = np.diag([0.5, 1.0, 0.0]).astype(complex)
JP = W@T@W.conj().T; JQ = W@(np.eye(3)-T)@W.conj().T
def DH(JP, JQ):
    G = ka_frame_conn(JP, JQ, geo, tol=1e-12)
    return np.linalg.eigvalsh(herm(ptrace_Y(JP+JQ-2*G, dY, dX)))[-1]
def Ef(JP, JQ): return np.linalg.eigvalsh(herm(ptrace_Y(JP+JQ, dY, dX)))[-1]
print("pair: Phi,Psi not mutually singular? T has eigenvalue 1/2.  D_H =", DH(JP,JQ), "  E_f =", Ef(JP,JQ), " (equal => theorem says D_H continuous here)")
# random CP perturbations
for eps in [1e-1,1e-2,1e-3,1e-4]:
    worst=0; 
    for trial in range(300):
        def pert(J):
            G = rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); G = G@G.conj().T
            # perturb Kraus-like: add eps-sized PSD plus eps-sized rotation of the range
            Wp = np.linalg.cholesky(J + 1e-14*np.eye(4)) if False else None
            return None
        # perturb the Kraus operators of each map (T-split): build CP maps near Phi, Psi
        PhiK = [np.sqrt(0.5)*K[0], 1.0*K[1]]; PsiK=[np.sqrt(0.5)*K[0], 1.0*K[2]]
        def jof(Ks): return sum(np.outer(vec(k), vec(k).conj()) for k in Ks)
        PhiK2 = [k + eps*(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))) for k in PhiK]
        PsiK2 = [k + eps*(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))) for k in PsiK]
        # sanity: unperturbed Choi matrices match
        JP2, JQ2 = jof(PhiK2), jof(PsiK2)
        worst = max(worst, abs(DH(JP2,JQ2)-2.0))
    print("eps=%.0e  max |D_H(Phi'||Psi') - E_f| over 300 random Kraus-perturbations = %.3e" % (eps, worst))
