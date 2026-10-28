"""General pair (Phi,Psi) of channels, J(Omega) singular. Perturb Psi -> Ad_{U_t} o Psi.
Exact: J(Phi)#J(Psi_t) = W [ T # (A_t Pi_t A_t^*) ] W^*, F=[W,Ups], A_t=W^+ (U_t(x)1) W (1-T)^{1/2},
B_t = Ups^*(U_t(x)1) W (1-T)^{1/2}, Pi_t = proj ker B_t.   (Lemma SG)
limsup <= F_N = lambda_min Tr_Y W [T # ((1-T)^{1/2} Pi_N (1-T)^{1/2})] W^*, N = ker(Bt~ (1-T)^{1/2})."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
import numpy as np, cvxpy as cp, glib as g
from scipy.linalg import expm, null_space
rng=np.random.default_rng(4)
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1.,-1]).astype(complex)
def tp_in_span(Ms, dX, R):
    m=len(Ms); C=cp.Variable((m,m),hermitian=True)
    cons=[C>>0, sum(C[j,k]*(Ms[j].conj().T@Ms[k]) for j in range(m) for k in range(m))==np.eye(dX)]
    cp.Problem(cp.Maximize(cp.real(cp.trace(R@C))),cons).solve(solver='CLARABEL'); return g.herm(C.value)
def analyse(J1, J2, H, dY=2, dX=2, ts=(1e-2,1e-3,1e-4)):
    W=g.frame(J1+J2); Ups=null_space(W.conj().T); T=g.arn(W,J1); r=W.shape[1]
    S=g.sqrtm(np.eye(r)-T); Hb=np.kron(H,np.eye(dX))
    Bt=-1j*Ups.conj().T@Hb@W@S; N=null_space(Bt, rcond=1e-9); PiN=N@N.conj().T
    FN=np.linalg.eigvalsh(g.trY(W@g.gmean(T, S@PiN@S)@W.conj().T,dY,dX)).min()
    out=[]
    for t in ts:
        Ut=np.kron(expm(-1j*t*H),np.eye(dX))
        At=np.linalg.pinv(W)@Ut@W@S; Btt=Ups.conj().T@Ut@W@S
        Nt=null_space(Btt, rcond=1e-12); Pit=Nt@Nt.conj().T
        Fexact=np.linalg.eigvalsh(g.trY(W@g.gmean(T, At@Pit@At.conj().T)@W.conj().T,dY,dX)).min()
        Fnum=g.Fhat(J1, Ut@J2@Ut.conj().T, dY, dX)
        out.append((t,Fexact,Fnum))
    return FN, out
for trial in range(5):
    Ms=[rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)) for _ in range(3)]
    Mm=np.array([g.vec(M) for M in Ms]).T
    Cs=[tp_in_span(Ms,2,g.herm(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))) for _ in range(4)]
    C1=0.5*(Cs[0]+Cs[1]); C2=0.5*(Cs[2]+Cs[3])
    J1=g.herm(Mm@C1@Mm.conj().T); J2=g.herm(Mm@C2@Mm.conj().T)
    F0=g.Fhat(J1,J2,2,2)
    print(f"trial {trial}: ranks {[np.linalg.matrix_rank(J,tol=1e-8) for J in (J1,J2,J1+J2)]}, F(Phi,Psi)={F0:.5f}")
    for Hn,H in [("X",X),("Y",Y),("Z",Z)]:
        FN,out=analyse(J1,J2,H)
        print(f"   H={Hn}: F_N={FN:.5f}; "+"; ".join(f"t={t:g}: exact {a:.5f} num {b:.5f}" for t,a,b in out))
