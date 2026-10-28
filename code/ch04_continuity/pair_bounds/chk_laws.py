import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
import numpy as np, glib as g, scipy.linalg as sla
from scipy.optimize import brentq
rng=np.random.default_rng(9)
# (1) improved sandwich lemma: ||a1 phi(T1) a1 - a2 phi(T2) a2|| <= 1/2||phi(T1)-phi(T2)|| + 2 gam ||phi||
worst=0
for t in range(3000):
    r=rng.integers(2,6); U=g.rand_unitary(rng,r); lam=rng.uniform(0.02,0.98,size=r)
    C1=(U*lam)@U.conj().T; C2=np.eye(r)-C1; a1,a2=g.sqrtm(C1),g.sqrtm(C2)
    def rT():
        V=g.rand_unitary(rng,r); return (V*rng.uniform(0,1,size=r))@V.conj().T
    T1=rT(); T2=T1+0.3*rng.uniform()*g.herm(rng.normal(size=(r,r))) if t%2 else rT()
    w,V=np.linalg.eigh(T2); T2=(V*np.clip(w,0,1))@V.conj().T
    gam=np.abs(lam-0.5).max()
    for phi,nphi in [(lambda x: np.sqrt(np.clip(x,0,1)*np.clip(1-x,0,1)),0.5),(lambda x:np.cos(4*x),1.0),(lambda x:x,1.0),(lambda x:(x-.5)**2,0.25)]:
        L=np.linalg.norm(a1@g.fn(T1,phi)@a1-a2@g.fn(T2,phi)@a2,2)
        R=0.5*np.linalg.norm(g.fn(T1,phi)-g.fn(T2,phi),2)+2*gam*nphi
        worst=max(worst,L/R)
print("(1) improved Lemma 10.3: max LHS/RHS =",worst)
# constant optimisation for Cor 10.7 with improved lemma
def c_small(d0): return np.sqrt(2*np.sqrt(2)/np.sqrt(1-4*d0**2))+2*np.sqrt(d0)
d0=brentq(lambda d: c_small(d)-1/np.sqrt(d),1e-4,0.49)
print("    optimal threshold delta0=%.5f, constant=%.4f (v5: 11/4=2.75)"%(d0,1/np.sqrt(d0)))
# (2) diagonal law: chi/4 <= 1-F <= chi/2 ; chi <= min(dX,dY) d^2 / lambda_min^+(J(Phi+Psi))
w1=w2=0; wb=0
for t in range(150):
    dX,dY=rng.integers(2,4),rng.integers(2,4)
    K1=g.rand_kraus(rng,dY,dX,rng.integers(1,dX*dY+1))
    if t%2: K2=g.rand_kraus(rng,dY,dX,rng.integers(1,dX*dY+1))
    else:
        U=sla.expm(-1j*0.05*g.herm(rng.normal(size=(dY,dY)))); K2=[U@K for K in K1]
    J1,J2=g.choi(K1),g.choi(K2); F=g.Fhat(J1,J2,dY,dX); E=J1-J2; Om=J1+J2
    chi=np.linalg.eigvalsh(g.trY(E@np.linalg.pinv(Om,rcond=1e-10,hermitian=True)@E,dY,dX)).max()
    w1=max(w1,(chi/4-(1-F))); w2=max(w2,(1-F)-chi/2)
    lm=min(x for x in np.linalg.eigvalsh(Om) if x>1e-9)
    d=g.dnorm(E,dY,dX); wb=max(wb,chi/(min(dX,dY)*d**2/lm))
print("(2) max(chi/4-(1-F))=%.2e  max((1-F)-chi/2)=%.2e  max chi/(min(dX,dY)d^2/lmin+)=%.3f"%(w1,w2,wb))
# (3) depolarized gate: diagonal incoherent example and coherent
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1.,-1]).astype(complex)
def gate(p,delta):
    U=sla.expm(-0.5j*delta*X); v=g.vec(U); return (1-p)*np.outer(v,v.conj())+p/2*np.eye(4)
for p in [1e-2,1e-3,1e-4]:
    J=gate(p,0); PY=np.outer(g.vec(Y),g.vec(Y).conj())/2; PZ=np.outer(g.vec(Z),g.vec(Z).conj())/2
    for s in [0.1,0.5]:
        e=s*p; J2=J+e*(PY-PZ); d=g.dnorm(J2-J,2,2); F=g.Fhat(J,J2,2,2)
        print(f"(3) p={p:g} incoh Y<->Z eps={s}p: d={d:.3e}  (1-F)*p/d^2={(1-F)*p/d**2:.4f}")
    dl=0.1*np.sqrt(p); J2=gate(p,dl); d=g.dnorm(J2-J,2,2); F=g.Fhat(J,J2,2,2)
    print(f"    coherent delta=0.1sqrt(p): (1-F)*p/d^2={(1-F)*p/d**2:.4f}; bound d^2/(2*lmin)*2: {(1-F)/(2*d**2/(2*p)):.4f} of bound")
# (4) interior levels of the depolarized pair: epsilon = min spec T(1-T) endpoints
for p in [1e-2,1e-3,1e-4]:
    dl=2*np.sqrt(p); J1,J2=gate(p,dl),gate(p,0); W=g.frame(J1+J2); T=g.arn(W,J1); ev=np.linalg.eigvalsh(T)
    print(f"(4) p={p:g}, delta=2sqrt(p): spec T = {np.round(ev,4)}  eps={ev.min():.4f}  lambda_min J(Om)={np.linalg.eigvalsh(J1+J2).min():.2e}")
# (5) pure-state zero-fidelity rate: rho'=(1-e)aa*+e ee*, sigma=bb*
for th in [0.3,0.1,0.03]:
    a=np.array([1,0]);b=np.array([np.cos(th),np.sin(th)]); e_=np.array([0,1])
    for eps in [1e-4,1e-6]:
        r=(1-eps)*np.outer(a,a)+eps*np.outer(e_,e_); s=np.outer(b,b)
        F=np.trace(g.gmean(r,s)).real; d=2*eps
        print(f"(5) theta={th}, d={d:.0e}: F={F:.4e}, F*sin(th)/sqrt(d)={F*np.sin(th)/np.sqrt(d):.4f} (pred 1/sqrt2=0.7071); bound 2sqrt2 sqrt(d)/(sqrt2 sin(th/2)...)")
