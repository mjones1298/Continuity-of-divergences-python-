import numpy as np
from diamond import *
rng=np.random.default_rng(77)
def gmean_T(T):
    return fun(T, lambda t: np.sqrt(np.clip(t*(1-t),0,None)))
worst=0; worst_lb=0
for trial in range(14):
    dX=rng.integers(1,4); dY=rng.integers(1,4)
    if dX*dY<2: continue
    r=rng.integers(2,min(dX*dY,4)+1)
    Ks=rand_kraus(dX,dY,r); W=W_of(Ks)
    if np.linalg.matrix_rank(W)<r: continue
    nu=nu_est(Ks,restarts=10)
    J=W@W.conj().T; ev=np.linalg.eigvalsh(J); lm=ev[ev>1e-12].min(); nu_lb=np.sqrt(lm/min(dX,dY))
    on=np.linalg.norm(sum(K.conj().T@K for K in Ks),2)
    T1=rand_T(r,spec=rng.uniform(0,1,size=r)); 
    T2=herm(T1+0.1*rng.uniform(0,1)*herm(rng.normal(size=(r,r))+1j*rng.normal(size=(r,r))))
    w,V=np.linalg.eigh(T2); w=np.clip(w,0,1); T2=(V*w)@V.conj().T
    F1=np.linalg.eigvalsh(herm(ptraceY(W@gmean_T(T1)@W.conj().T,dY,dX)))[0]
    F2=np.linalg.eigvalsh(herm(ptraceY(W@gmean_T(T2)@W.conj().T,dY,dX)))[0]
    d=diamond_herm(W@(T1-T2)@W.conj().T,dY,dX)
    bound=on/nu*np.sqrt(d); bound_lb=on/nu_lb*np.sqrt(d)
    worst=max(worst,abs(F1-F2)/bound); worst_lb=max(worst_lb,abs(F1-F2)/bound_lb)
print('max |dF|/bound(nu_est)=',worst,' max |dF|/bound(nu_lb)=',worst_lb)
