import numpy as np
from lib import *
rng = np.random.default_rng(3)
fJ = lambda t: (t-1)*np.log(t)
hJ = lambda x: (1-x)*fJ(x/(1-x)) if True else None
def hJv(x):
    x=np.asarray(x,float); return (1-x)*((x/(1-x))-1)*np.log(x/(1-x))
dX,dY,r=2,2,3
Ks=rand_kraus(dX,dY,r); W=W_of(Ks)
for spec in ([0.2,0.5,0.8],[0.0,0.5,0.8],[0.2,0.5,1.0]):
    T=rand_T(r,spec=np.array(spec))
    A=rng.normal(size=(dX,dX))+1j*rng.normal(size=(dX,dX)); A/=np.linalg.norm(A)
    P1,P2=phi_state(W,T,A,dY,dX)
    vals=[Dhat_reg(P1,P2,fJ,e) for e in (1e-3,1e-5,1e-7,1e-9)]
    print(spec, vals)
    if min(spec)>0 and max(spec)<1:
        Wt=np.kron(np.eye(dY),A.T)@W
        print('  pred', np.trace(Wt@hfun_T(T,hJv)@Wt.conj().T).real)
