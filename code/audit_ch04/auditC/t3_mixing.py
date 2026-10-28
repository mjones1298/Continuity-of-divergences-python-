import numpy as np
from lib import *
rng = np.random.default_rng(5)
# Lemma 7.4 with boundary eigenvalues, f bounded and unbounded
fs = {
 'LC (t-1)^2/(t+1)': (lambda t: (t-1)**2/(t+1), lambda x: np.where(x<=0,1.0,np.where(x>=1,1.0,(2*x-1)**2)) ),
 'JS': (lambda t: t*np.log(t)-(1+t)*np.log((1+t)/2), None),
}
def hLC(x):
    x = np.asarray(x,float); return (2*x-1)**2
def hJS(x):
    x = np.asarray(x,float)
    out = np.log(2)+ np.where(x>0, x*np.log(np.where(x>0,x,1)),0)+np.where(x<1,(1-x)*np.log(np.where(x<1,1-x,1)),0)
    return out
for name,(f,h) in [('LC',(lambda t:(t-1)**2/(t+1),hLC)),('JS',(lambda t: np.where(t>0,t*np.log(np.where(t>0,t,1)),0)-(1+t)*np.log((1+t)/2),hJS))]:
    for trial in range(10):
        dX,dY,r = 2,2,3
        Ks = rand_kraus(dX,dY,r); W=W_of(Ks)
        spec = np.array([0.0,1.0,rng.uniform(0.1,0.9)])
        T = rand_T(r,spec=spec)
        A = rng.normal(size=(dX,dX))+1j*rng.normal(size=(dX,dX)); A/=np.linalg.norm(A)
        P1,P2 = phi_state(W,T,A,dY,dX)
        vals=[Dhat_reg(P1,P2,f,e) for e in (1e-5,1e-7,1e-9)]
        Wt = np.kron(np.eye(dY),A.T)@W
        pred = np.trace(Wt@hfun_T(T,h)@Wt.conj().T).real
        print(name, vals, pred)
