import numpy as np
from lib import *
rng = np.random.default_rng(99)
def xlogx(t):
    t=np.asarray(t,float); return np.where(t>0, t*np.log(np.where(t>0,t,1)), 0.0)
hBS = lambda t: xlogx(t) - np.where(t>0, t*np.log(1-np.clip(t,None,1-1e-300)),0.0)
def hBSv(t):
    t=np.asarray(t,float); return np.where(t>0, t*np.log(np.where(t>0,t,1)/(1-t)),0.0)
def hal(alpha): return lambda t: np.asarray(t,float)**alpha*(1-np.asarray(t,float))**(1-alpha)
worstBS=0; worstR={1.2:0,1.5:0,2.0:0}
for trial in range(4000):
    dX=rng.integers(1,4); dY=rng.integers(1,4)
    r=rng.integers(1,min(dX*dY,4)+1)
    Ks=rand_kraus(dX,dY,r)
    if rng.random()<0.3:  # near-dependent Kraus
        Ks[-1]=Ks[0]*0.7+1e-3*Ks[-1] if r>1 else Ks[-1]
    W=W_of(Ks)
    if np.linalg.matrix_rank(W,tol=1e-12)<r: continue
    eps=10**rng.uniform(-4,-0.3)
    # T1, T2 in [0,1-eps]
    s1=rng.uniform(0,1-eps,size=r); 
    if rng.random()<0.5: s1[0]=1-eps
    if rng.random()<0.3: s1[-1]=0.0
    T1=rand_T(r,spec=s1)
    Hh=rng.normal(size=(r,r))+1j*rng.normal(size=(r,r)); Hh=herm(Hh); 
    d=10**rng.uniform(-6,np.log10(1-eps))
    Hh*=d/np.linalg.norm(Hh,2)
    T2=herm(T1+Hh)
    w2=np.linalg.eigvalsh(T2)
    if w2.min()<0 or w2.max()>1-eps: continue
    dd=np.linalg.norm(T1-T2,2)
    on=np.linalg.norm(sum(K.conj().T@K for K in Ks),2)
    D1=np.linalg.eigvalsh(M_of(W,T1,hBSv,dY,dX))[-1]; D2=np.linalg.eigvalsh(M_of(W,T2,hBSv,dY,dX))[-1]
    bnd=on*dd*(np.log(4/(eps*dd))+1/eps)
    worstBS=max(worstBS,abs(D1-D2)/bnd)
    for al in worstR:
        h=hal(al)
        Q1=np.linalg.eigvalsh(M_of(W,T1,h,dY,dX))[-1]; Q2=np.linalg.eigvalsh(M_of(W,T2,h,dY,dX))[-1]
        bnd=on*((4-al)*eps**(1-al)+(al-1)*eps**(-al))*dd
        worstR[al]=max(worstR[al],abs(Q1-Q2)/bnd)
print('max ratio actual/bound  BS:',worstBS,' Renyi:',worstR)
