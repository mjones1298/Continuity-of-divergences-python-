"""P1 off-diagonal: channel pairs with Kraus ops in a common 3-dim span S (so J(Phi+Psi) singular, codim 1).
Perturb Psi -> Ad_{e^{-itH}} o Psi.  Bound: limsup <= lambda_min Tr_Y W_Phi (A0 Pi_L A0^*)^{1/2} W_Phi^*,
L = {c in ker B0 : B1 c in ran B0}.  Compare with numerics at small t."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
import numpy as np, cvxpy as cp, glib as g
from scipy.linalg import expm, null_space, orth
rng=np.random.default_rng(2)
def rand_tp_in_span(Ms, dX):
    m=len(Ms); C=cp.Variable((m,m),hermitian=True)
    cons=[C>>0, sum(C[j,k]*(Ms[j].conj().T@Ms[k]) for j in range(m) for k in range(m))==np.eye(dX)]
    R=g.herm(rng.normal(size=(m,m))+1j*rng.normal(size=(m,m)))
    cp.Problem(cp.Maximize(cp.real(cp.trace(R@C))+0.0),cons).solve(solver='CLARABEL')
    Cv=g.herm(C.value); 
    # move to interior: mix with another solution
    return Cv
def bound(JPhi, WPsi, H, dY, dX):
    WPhi=g.frame(JPhi); Ups=null_space(WPhi.conj().T, rcond=1e-9)
    Hb=np.kron(H,np.eye(dX))
    A0=np.linalg.pinv(WPhi)@WPsi; B0=Ups.conj().T@WPsi; B1=-1j*Ups.conj().T@Hb@WPsi
    kB0=null_space(B0, rcond=1e-9)
    if kB0.shape[1]==0: return 0.0
    ranB0=orth(B0, rcond=1e-9); Pc=np.eye(B0.shape[0])-ranB0@ranB0.conj().T
    Lc=null_space(Pc@B1@kB0, rcond=1e-9)
    if Lc.shape[1]==0: return 0.0
    L=kB0@Lc; PiL=L@np.linalg.pinv(L)
    M=WPhi@g.sqrtm(A0@PiL@A0.conj().T)@WPhi.conj().T
    return np.linalg.eigvalsh(g.trY(M,dY,dX)).min()
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1.,-1]).astype(complex)
for trial in range(6):
    dY=dX=2
    Ms=[rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)) for _ in range(3)]
    Mm=np.array([g.vec(M) for M in Ms]).T
    C1=rand_tp_in_span(Ms,dX); C2=rand_tp_in_span(Ms,dX)
    C1=0.5*(C1+C2)  # interior-ish point
    J1=g.herm(Mm@C1@Mm.conj().T); J2=g.herm(Mm@C2@Mm.conj().T)
    WPsi=g.frame(J2)
    F0=g.Fhat(J1,J2,2,2)
    rk=[np.linalg.matrix_rank(J,tol=1e-8) for J in (J1,J2,J1+J2)]
    res=[]
    for Hn,H in [("X",X),("Y",Y),("Z",Z),("rand",g.herm(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))))]:
        b=bound(J1,WPsi,H,2,2)
        t=1e-4; Ut=expm(-1j*t*H); Jt=np.kron(Ut,np.eye(2))@J2@np.kron(Ut,np.eye(2)).conj().T
        res.append(f"{Hn}: bnd {b:.4f} num {g.Fhat(J1,Jt,2,2):.4f}")
    print(f"ranks {rk}, F(Phi,Psi)={F0:.4f} | "+" | ".join(res))
