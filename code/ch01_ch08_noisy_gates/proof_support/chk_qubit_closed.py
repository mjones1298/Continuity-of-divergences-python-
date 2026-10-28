import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
# Check closed forms for qubit depolarized gates: F_hat, D_BS (channel), Umegaki(max-ent)
import numpy as np, glib as g
X=np.array([[0,1],[1,0]],complex); Z=np.diag([1,-1]).astype(complex); Yp=np.array([[0,-1j],[1j,0]])
def Rn(th,n): # exp(-i th n.sigma)
    S=n[0]*X+n[1]*Yp+n[2]*Z; return np.cos(th)*np.eye(2)-1j*np.sin(th)*S
def depol_choi(U,p):
    v=g.vec(U); return (1-p)*np.outer(v,v.conj())+p/2*np.eye(4)
def bs_channel(J1,J2,dY=2,dX=2):
    J1h=g.sqrtm(J1); M=J1h@np.linalg.inv(J2)@J1h
    P=J1h@g.fn(M,np.log)@J1h
    return np.linalg.eigvalsh(g.trY(P,dY,dX)).max(), np.linalg.eigvalsh(g.trY(P,dY,dX))
def umeg(r,s): return np.trace(r@(g.fn(r,np.log)-g.fn(s,np.log))).real
rng=np.random.default_rng(1)
for trial in range(6):
    p=10**rng.uniform(-4,-0.5); th=rng.uniform(0,1.5)
    U=g.rand_unitary(rng,2); n=rng.normal(size=3); n/=np.linalg.norm(n)
    U2=Rn(th,n)@U
    J1,J2=depol_choi(U,p),depol_choi(U2,p)
    c=abs(np.trace(U.conj().T@U2))/2; s2=1-c**2
    x=4*(1-p)**2*s2/(p*(4-3*p))
    F_closed=p/2+(1-p/2)/np.sqrt(1+x)
    F_num=g.Fhat(J1,J2,2,2)
    bs_num,ev=bs_channel(J1,J2)
    bs_closed=(2-p)*np.sqrt(x/(1+x))*np.arcsinh(np.sqrt(x))
    um_closed=(1-p)*s2*np.log(1+4*(1-p)/p)
    print(f"p={p:.2e} th={th:.3f}  F {F_num:.10f} {F_closed:.10f} | BS {bs_num:.8f} {bs_closed:.8f} spreadTrY={ev.max()-ev.min():.1e} | Um {umeg(J1/2,J2/2):.8f} {um_closed:.8f}")
