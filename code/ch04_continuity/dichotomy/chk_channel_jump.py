"""P1: channel-valued discontinuity at (Phi,Phi) for singular J(Phi).
Exact formula: J#J' = W (A Pi_{ker B} A^*)^{1/2} W^*, V = W A + Ups B (J'=VV^*, F=[W,Ups] invertible)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
import numpy as np, glib as g
from scipy.linalg import expm, null_space
def ad(gm): return [np.diag([1, np.sqrt(1-gm)]).astype(complex), np.sqrt(gm)*np.array([[0,1],[0,0]],complex)]
def ad_deph(gm, q):  # AD followed by dephasing with prob q
    Z=np.diag([1,-1]).astype(complex); out=[]
    for K in ad(gm): out += [np.sqrt(1-q)*K, np.sqrt(q)*Z@K]
    return out
def frame_from_kraus(Ks):
    J=g.choi(Ks); return g.frame(J), J
def formula(W, Vp, tol=1e-12):
    n=W.shape[0]; Ups=null_space(W.conj().T)
    A=np.linalg.pinv(W)@Vp; B=Ups.conj().T@Vp
    kerB=null_space(B, rcond=1e-9) if B.size else np.eye(A.shape[1])
    Pi=kerB@kerB.conj().T
    return W@g.sqrtm(A@Pi@A.conj().T)@W.conj().T
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1]).astype(complex)
for name, Ks in [("AD(0.3)", ad(0.3)), ("AD(0.3)+deph(0.2)", ad_deph(0.3,0.2)), ("AD(0.05)+deph(0.02)", ad_deph(0.05,0.02))]:
    W, J = frame_from_kraus(Ks); r=W.shape[1]; Ups=null_space(W.conj().T)
    print(f"--- {name}: Kraus rank {r}, codim {Ups.shape[1]}")
    for Hname, H in [("Y",Y),("X",X),("Z",Z)]:
        Hb=np.kron(H, np.eye(2)); Bt=-1j*Ups.conj().T@Hb@W
        kb=null_space(Bt, rcond=1e-9); Pi=kb@kb.conj().T
        pred=np.linalg.eigvalsh(g.trY(W@Pi@W.conj().T,2,2)).min()
        vals=[]
        for t in [1e-2,1e-3,1e-4]:
            Ut=expm(-1j*t*H); Kt=[Ut@K for K in Ks]; Jt=g.choi(Kt)
            Vp=np.kron(Ut,np.eye(2))@W
            Fform=np.linalg.eigvalsh(g.trY(formula(W,Vp),2,2)).min()
            Fnum=g.Fhat(J,Jt,2,2)   # ARN-frame numerics, independent route
            dn=2*np.sin(t)          # ||W_t - id||_dia upper bound on ||Phi_t-Phi||_dia
            vals.append((t,Fform,Fnum))
        print(f"  H={Hname}: ||B~||={np.linalg.norm(Bt,2):.3f}, predicted limsup<= {pred:.6f};",
              "  ".join(f"t={t:g}: F(formula)={a:.6f} F(num)={b:.6f}" for t,a,b in vals))
