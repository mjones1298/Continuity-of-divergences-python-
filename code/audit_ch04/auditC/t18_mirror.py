import numpy as np
from lib import *
rng=np.random.default_rng(14)
f=lambda t:-np.log(t)
def h(x):
    x=np.asarray(x,float); out=np.zeros_like(x)
    m=(x>0)&(x<1); out[m]=-(1-x[m])*np.log(x[m]/(1-x[m]))
    out[x<=0]=np.inf; out[x>=1]=0.0   # f(0+)=inf, f'(inf)=0
    return out
dX,dY,r=2,2,3
Ks=rand_kraus(dX,dY,r); W=W_of(Ks)
for spec in ([0.2,0.5,1.0],[0.3,0.6,0.9]):
    T=rand_T(r,spec=np.array(spec))
    A=rng.normal(size=(dX,dX))+1j*rng.normal(size=(dX,dX)); A/=np.linalg.norm(A)
    P1,P2=phi_state(W,T,A,dY,dX)
    vals=[Dhat_reg(P1,P2,f,e) for e in (1e-4,1e-6,1e-8,1e-10)]
    Wt=np.kron(np.eye(dY),A.T)@W
    w,V=np.linalg.eigh(T); 
    pred=sum(h(np.array([wi]))[0]*np.linalg.norm(Wt@V[:,i:i+1])**2 for i,wi in enumerate(w))
    print(spec,vals,'pred',pred)
