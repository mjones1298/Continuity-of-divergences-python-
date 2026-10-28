"""P1 checks. (1) AD_gamma vs U o AD_gamma: Choi ranges intersect trivially for every non-diagonal U
(principal angles). (2) Random channels with singular Choi: limsup F(Phi, e^{-itH} Phi e^{itH}) bound
1 - lambda_max Tr_Y(W Q W^*) < 1 and numerics. (3) Generic rotational separation for pairs with c_Omega >= r."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
import numpy as np, glib as g
from scipy.linalg import expm, null_space, orth, subspace_angles
rng = np.random.default_rng(11)
def ad(gm): return [np.diag([1, np.sqrt(1-gm)]).astype(complex), np.sqrt(gm)*np.array([[0,1],[0,0]],complex)]
def rng_of(Ks): return orth(np.array([g.vec(K) for K in Ks]).T)
# (1)
mx = 0
for k in range(2000):
    gm = rng.uniform(0.01, 0.99); U = g.rand_unitary(rng, 2)
    if abs(U[1,0]) < 1e-3: continue
    R1 = rng_of(ad(gm)); R2 = rng_of([U@K for K in ad(gm)])
    sv = np.linalg.svd(R1.conj().T@R2, compute_uv=False)   # cosines of principal angles
    mx = max(mx, sv.max())
print("(1) AD vs U.AD, non-diagonal U: max cos(principal angle) =", mx, "(<1 means trivial intersection)")
for t in [1e-1, 1e-2, 1e-3]:
    U = expm(-1j*t*np.array([[0,-1j],[1j,0]])); R1 = rng_of(ad(0.3)); R2 = rng_of([U@K for K in ad(0.3)])
    print(f"    t={t}: smallest principal angle {subspace_angles(R1,R2).min():.3e}  (~ t: the ranges separate linearly)")
# (2)
def check_diag(dY, dX, r, trials=40):
    worst = 0; cnt = 0
    for _ in range(trials):
        Ks = g.rand_kraus(rng, dY, dX, r); J = g.choi(Ks); W = g.frame(J)
        Ups = null_space(W.conj().T)
        if Ups.shape[1] == 0: continue
        best_bound = 1
        for _h in range(5):
            H = rng.normal(size=(dY,dY))+1j*rng.normal(size=(dY,dY)); H = g.herm(H)
            Bt = Ups.conj().T@np.kron(H, np.eye(dX))@W
            Q = orth(Bt.conj().T)               # range of B~^*
            N = g.trY(W@Q@Q.conj().T@W.conj().T, dY, dX)
            bound = 1-np.linalg.eigvalsh(N).max()
            if bound < best_bound: best_bound, Hb = bound, H
        t = 1e-3; Ut = expm(-1j*t*Hb); Jt = g.choi([Ut@K for K in Ks])
        Fn = g.Fhat(J, Jt, dY, dX)
        worst = max(worst, Fn - best_bound); cnt += 1
        print(f"   dY={dY} dX={dX} r={r}: limsup bound {best_bound:.4f}, F at t=1e-3: {Fn:.4f}")
        if cnt >= 3: break
    return worst
print("(2) diagonal pairs (Phi,Phi), J(Phi) singular:")
for (dY,dX,r) in [(2,2,2),(2,2,3),(3,2,3),(2,3,4),(3,3,5)]:
    check_diag(dY,dX,r)
# (3) generic pairs Phi != Psi with r = rank J(Phi+Psi) <= dYdX/2: is B~ = Ups^*(H(x)1)W injective?
print("(3) rotational separation for random pairs:")
for (dY,dX,k1,k2) in [(2,2,1,1),(3,2,1,2),(3,2,2,1),(2,3,1,2),(3,3,2,2),(4,2,2,2)]:
    inj = 0; tot = 0; Fs = []
    for _ in range(30):
        K1 = g.rand_kraus(rng,dY,dX,k1); K2 = g.rand_kraus(rng,dY,dX,k2)
        J1, J2 = g.choi(K1), g.choi(K2); W = g.frame(J1+J2); r = W.shape[1]
        Ups = null_space(W.conj().T)
        H = g.herm(rng.normal(size=(dY,dY))+1j*rng.normal(size=(dY,dY)))
        Bt = Ups.conj().T@np.kron(H,np.eye(dX))@W
        tot += 1; inj += np.linalg.matrix_rank(Bt, tol=1e-9) == r
        t = 1e-3; Ut = expm(-1j*t*H)
        Fs.append((g.Fhat(J1,J2,dY,dX), g.Fhat(J1, g.choi([Ut@K for K in K2]), dY, dX)))
    Fs = np.array(Fs)
    print(f"   dY={dY} dX={dX} ranks {k1},{k2}: r={r}, codim={dY*dX-r}; B~ injective in {inj}/{tot}; "
          f"mean F(Phi,Psi)={Fs[:,0].mean():.3f}, mean F(Phi,U_t Psi)={Fs[:,1].mean():.2e}")
