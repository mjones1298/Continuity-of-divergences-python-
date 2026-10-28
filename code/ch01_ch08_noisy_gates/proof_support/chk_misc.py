import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
import numpy as np, glib as g, scipy.linalg as sla
from scipy.optimize import minimize_scalar
X=np.array([[0,1],[1,0]],complex)
def ad(gm): return [np.diag([1,np.sqrt(1-gm)]).astype(complex), np.sqrt(gm)*np.array([[0,1],[0,0]],complex)]
# (a) F(AD_g, AD_g') = 0 for g != g' ; =1 for equal
for (a,b) in [(0.3,0.3),(0.3,0.31),(0.9,0.99),(0.05,0.5)]:
    print("(a) F(AD%.2f,AD%.2f) = %.3e"%(a,b,g.Fhat(g.choi(ad(a)),g.choi(ad(b)),2,2)))
# (b) phi(x)/log(1+x) range, phi(x)/x on [0,1]
xs=np.geomspace(1e-8,1e8,20001); phi=np.sqrt(xs/(1+xs))*np.arcsinh(np.sqrt(xs))
r=phi/np.log1p(xs); print("(b) phi/log(1+x) in [%.4f, %.4f]"%(r.min(),r.max()))
xx=np.linspace(1e-9,1,10001); print("    min_{x<=1} phi/x = %.4f"%((np.sqrt(xx/(1+xx))*np.arcsinh(np.sqrt(xx))/xx).min()))
# (c) decreasing incoherent direction: move X-weight into I in the first gate
def gate(p,delta):
    U=sla.expm(-0.5j*delta*X); v=g.vec(U); return (1-p)*np.outer(v,v.conj())+p/2*np.eye(4)
for p in [1e-2,1e-3,1e-4]:
    d0=2*np.sqrt(p); R=sla.expm(-0.5j*d0*X); J1=gate(p,d0); J2=gate(p,0)
    PI=np.outer(g.vec(R),g.vec(R).conj())/2; PX=np.outer(g.vec(X@R),g.vec(X@R).conj())/2
    F0=g.Fhat(J1,J2,2,2)
    for k in [0.01,0.1]:
        e=k*p; J1e=J1+e*(PI-PX); assert np.linalg.eigvalsh(J1e).min()>-1e-12
        d=g.dnorm(J1e-J1,2,2); F1=g.Fhat(J1e,J2,2,2)
        print(f"(c) p={p:g} k={k}: F0-F1={F0-F1:.4e}, d={d:.3e}, (F0-F1)p/d={(F0-F1)*p/d:.4f}")
# (d) geometric Chernoff along the incoherent family: e^{-C} change per d
def chat(J1,J2):
    vals=[]
    for s in np.linspace(0.02,0.98,49):
        M=g.trY(g.gmean(J2,J1,s),2,2)    # J2 #_s J1
        vals.append(np.linalg.eigvalsh(M).min())
    return min(vals)
for p in [1e-2,1e-3,1e-4]:
    d0=2*np.sqrt(p); J1=gate(p,d0); J2=gate(p,0); XX=np.kron(X,np.eye(2))
    e=0.01*p; J1e=(1-e)*J1+e*XX@J1@XX; d=g.dnorm(J1e-J1,2,2)
    print(f"(d) p={p:g}: |Delta e^-C| p/d = {abs(chat(J1e,J2)-chat(J1,J2))*p/d:.4f}")
