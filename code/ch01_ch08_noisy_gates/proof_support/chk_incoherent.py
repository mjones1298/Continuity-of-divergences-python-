import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
# Finite (not first-order) check of the worst direction found by the SDP: incoherent X-Pauli mixing
import numpy as np, glib as g, scipy.linalg as sla
X=np.array([[0,1],[1,0]],complex)
def gate(p,delta):
    U=sla.expm(-0.5j*delta*X); v=g.vec(U); return (1-p)*np.outer(v,v.conj())+p/2*np.eye(4)
def mixX(J,e):
    XX=np.kron(X,np.eye(2)); return (1-e)*J+e*XX@J@XX
print(" p      eps/p   F(N_pd,N_p)  F(mixX(N_pd),N_p)  dF   diamond=2eps*(...)  dF/diamond*p")
for p in [1e-2,1e-3,1e-4]:
    d0=2*np.sqrt(p); J1=gate(p,d0); J2=gate(p,0)
    F0=g.Fhat(J1,J2,2,2)
    for s in [0.01,0.1,0.5,1,3]:
        e=s*p; J1e=mixX(J1,e); F1=g.Fhat(J1e,J2,2,2)
        dn=g.dnorm(J1e-J1,2,2)
        print(f"{p:6.0e} {s:6.2f}   {F0:.4f}       {F1:.4f}        {F0-F1:.4f}   {dn:.3e}        {(F0-F1)/dn*p:.4f}")
# coherent direction for comparison
for p in [1e-2,1e-3,1e-4]:
    d0=2*np.sqrt(p); J1=gate(p,d0); J2=gate(p,0); h=1e-3*p
    F0=g.Fhat(J1,J2,2,2); F1=g.Fhat(gate(p,d0+h),J2,2,2); dn=g.dnorm(gate(p,d0+h)-J1,2,2)
    print(f"coherent p={p:g}: rate*sqrt(p) = {(F0-F1)/dn*np.sqrt(p):.4f}")
# delta* where F=0.9 (exact closed form)
from scipy.optimize import brentq
for p in [1e-2,1e-3,1e-4,1e-5]:
    F=lambda d: p/2+(1-p/2)/np.sqrt(1+4*(1-p)**2*np.sin(d/2)**2/(p*(4-3*p)))-0.9
    ds=brentq(F,1e-9,3); print(f"p={p:g}: delta*(F=0.9)={ds:.5g} = {ds/np.sqrt(p):.4f} sqrt(p)")
