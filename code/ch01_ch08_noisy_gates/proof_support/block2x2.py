# check 2x2 block identities: tr(B#_s A)=T cosh((s-1/2)th)/cosh(th/2), D_BS=T th tanh(th/2), sinh(th/2)=sqrt(x)
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # glib.py in code/common
from glib import *
rng=np.random.default_rng(5)
for _ in range(5):
    a,b=rng.uniform(0.01,1,2); de=rng.uniform(0,3)
    R=np.array([[np.cos(de/2),-1j*np.sin(de/2)],[-1j*np.sin(de/2),np.cos(de/2)]])
    A=np.diag([a,b]).astype(complex); B=R@A@R.conj().T; T=a+b
    x=(a-b)**2*np.sin(de/2)**2/(4*a*b); th=2*np.arcsinh(np.sqrt(x))
    for s in [0.3,0.5,1.7]:
        lhs=np.trace(gmean_pd(B,A,s)).real; rhs=T*np.cosh((s-0.5)*th)/np.cosh(th/2)
        assert abs(lhs-rhs)<1e-10*T,(lhs,rhs)
    Ah=sqrtm(A); dbs=np.trace(Ah@fn(Ah@np.linalg.inv(B)@Ah,np.log)@Ah).real
    assert abs(dbs-T*th*np.tanh(th/2))<1e-10, (dbs,T*th*np.tanh(th/2))
    U=np.trace(B@(fn(B,np.log)-fn(A,np.log))).real; assert abs(U-np.sin(de/2)**2*(a-b)*np.log(a/b))<1e-10
    rf=np.sum(np.sqrt(np.clip(np.linalg.eigvalsh(herm(Ah@B@Ah)),0,None))); assert abs(rf-np.sqrt(T**2-(a-b)**2*np.sin(de/2)**2))<1e-10
print('2x2 block identities verified (geometric #_s incl. s>1, BS, Umegaki, Uhlmann)')
