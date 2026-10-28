import numpy as np
from scipy.optimize import minimize
from lib import fun, herm
rng = np.random.default_rng(2)
def xlogx(t):
    t=np.clip(t,0,None); return np.where(t>1e-300, t*np.log(np.where(t>1e-300,t,1)),0.0)
def mk(q,n):
    k=n*(n+1)//2
    H=np.zeros((n,n),complex); iu=np.triu_indices(n); H[iu]=q[:k]+1j*q[k:2*k]*(iu[0]!=iu[1])
    return np.triu(H)+np.triu(H,1).conj().T
def val(p,n,delta):
    k=n*(n+1)//2
    A=mk(p[:2*k],n); D=mk(p[2*k:4*k],n)
    w,V=np.linalg.eigh(A); w=0.5*(1+np.tanh(w)); A=(V*w)@V.conj().T
    D=D/np.linalg.norm(D,2)*delta
    B=A+D
    wB=np.linalg.eigvalsh(B)
    pen=1e3*max(0,-wB.min())+1e3*max(0,wB.max()-1)
    dd=fun(A,xlogx)-fun(herm(B),lambda t:xlogx(np.clip(t,0,None)))
    return -np.linalg.norm(dd,2)+pen
for n in (2,3):
    for delta in (0.5,0.1,0.01):
        best=0
        k=n*(n+1)//2
        for rs in range(6):
            p0=rng.normal(size=4*k)*2
            res=minimize(val,p0,args=(n,delta),method='Powell',options={'maxiter':3000,'xtol':1e-8,'ftol':1e-12})
            best=max(best,-res.fun)
        bound=delta*np.log(1/delta)+(1+delta)*np.log(1+delta)
        print(n,delta,'best found',best,'bound',bound,'scalar sup', delta*np.log(1/delta),flush=True)
