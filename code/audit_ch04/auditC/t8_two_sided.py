import numpy as np
from diamond import *
rng = np.random.default_rng(21)
rows=[]
minratio_nu=1e9; minratio_lam=1e9; maxratio_up=0
for trial in range(25):
    dX=rng.integers(1,4); dY=rng.integers(1,4)
    r=rng.integers(1,min(dX*dY,4)+1)
    Ks=rand_kraus(dX,dY,r); W=W_of(Ks)
    if np.linalg.matrix_rank(W)<r: continue
    J=W@W.conj().T
    ev=np.linalg.eigvalsh(J); lam_min=ev[ev>1e-9].min()
    nu=nu_est(Ks,restarts=15)
    Om_star=sum(K.conj().T@K for K in Ks); on=np.linalg.norm(Om_star,2)
    for kind in range(4):
        if kind==0: K=rand_T(r,spec=rng.normal(size=r))
        elif kind==1:
            v=rng.normal(size=r)+1j*rng.normal(size=r); v/=np.linalg.norm(v); K=np.outer(v,v.conj())
        elif kind==2: K=np.eye(r,dtype=complex)
        else:
            spec=np.zeros(r); spec[0]=1; spec[1:]=-rng.uniform(0,1,size=r-1) if r>1 else 0
            K=rand_T(r,spec=spec)
        nK=np.linalg.norm(K,2)
        d=diamond_herm(W@K@W.conj().T,dY,dX)
        r_nu=d/(nu**2*nK); r_lam=d/(lam_min/min(dX,dY)*nK); r_up=d/(on*nK)
        minratio_nu=min(minratio_nu,r_nu); minratio_lam=min(minratio_lam,r_lam); maxratio_up=max(maxratio_up,r_up)
        if r_lam<1-1e-6 or r_up>1+1e-6: print('VIOLATION', dX,dY,r,kind,r_lam,r_up)
print('min d/(nu_est^2 |K|)=',minratio_nu,' min d/(lam/min dim |K|)=',minratio_lam,' max d/(|Om*| |K|)=',maxratio_up)
