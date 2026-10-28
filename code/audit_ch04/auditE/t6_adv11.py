import numpy as np, sys
from scipy.optimize import minimize
from lib import *
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)

def build(x,dY,dX,ns,ne1,ne2):
    N=dY*dX;k=ns+ne1+ne2
    p=0
    def take(n):
        nonlocal p
        v=x[p:p+n];p+=n;return v
    M=(take(N*k)+1j*take(N*k)).reshape(N,k)
    idx1=list(range(ns+ne1));idx2=list(range(ns))+list(range(ns+ne1,k))
    Js=[]
    for idx in (idx1,idx2):
        m=len(idx)
        Mi=M[:,idx]
        for _ in range(2):
            L=(take(m*m)+1j*take(m*m)).reshape(m,m)
            Js.append(Mi@(L@L.conj().T)@Mi.conj().T)
    return Js
def nparams(dY,dX,ns,ne1,ne2):
    N=dY*dX;k=ns+ne1+ne2
    return 2*N*k+2*(2*(ns+ne1)**2*2//2*1)+0
def nparam(dY,dX,ns,ne1,ne2):
    N=dY*dX;k=ns+ne1+ne2
    m1=ns+ne1;m2=ns+ne2
    return 2*N*k+2*2*m1*m1+2*2*m2*m2

def obj(x,cfg):
    dY,dX,ns,ne1,ne2=cfg
    J=build(x,dY,dX,ns,ne1,ne2)
    if min(np.trace(j).real for j in J)<1e-9: pass
    try:
        S=Setup(*J,dY,dX)
        q=sec11(S)
        R=R_gsharp(S,q)
        F1=Fhat(J[0],J[1],dY,dX);F2=Fhat(J[2],J[3],dY,dX)
    except Exception as e:
        return 0.0
    if R<1e-10: return 0.0
    return -abs(F1-F2)/R

cfgs=[(2,2,1,1,1),(2,2,2,1,1),(2,2,1,2,1),(2,2,0+1,0,1),(2,2,2,0,1),(1,3,1,1,1),(2,3,2,2,1),(2,3,1,2,2)]
best=0
for rep in range(int(sys.argv[2]) if len(sys.argv)>2 else 10):
    for cfg in cfgs:
        n=nparam(*cfg)
        x0=rng.standard_normal(n)
        res=minimize(obj,x0,args=(cfg,),method='Powell',options={'maxiter':20000,'xtol':1e-8,'ftol':1e-12})
        if -res.fun>best:
            best=-res.fun;print(rep,cfg,'ratio',best,flush=True)
print('best',best)
