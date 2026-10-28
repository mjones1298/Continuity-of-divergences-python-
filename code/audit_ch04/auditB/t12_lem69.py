import sys; sys.path.insert(0,'.')
from lib import *
from scipy.optimize import minimize
def xlogx(x): return np.where(x>1e-300,x*np.log(np.where(x>1e-300,x,1)),0.)
def sq(H):
    H=herm(H); w,V=eigh(H); return (V*(1/(1+np.exp(-np.clip(w,-60,60)))))@V.conj().T
def mk(x,n):
    m=n*n
    return sq((x[:m]+1j*x[m:2*m]).reshape(n,n)), sq((x[2*m:3*m]+1j*x[3*m:4*m]).reshape(n,n))
def bound(d): return d*np.log(1/d)+(1+d)*np.log(1+d)
def ratio3(x,n):
    A,B=mk(x,n); d=opnorm(A-B)
    if d<1e-9 or d>1: return 0
    return opnorm(psd_fun2(A,xlogx)-psd_fun2(B,xlogx))/bound(d)
def ratio4(x,n,b):
    A,B=mk(x,n); d=opnorm(A-B)
    if d<1e-9: return 0
    f=lambda t: np.clip(t,0,None)**(1+b)
    return opnorm(psd_fun2(A,f)-psd_fun2(B,f))/((3-b)*d)
import time; t0=time.time()
r3=0;r4={0.2:0,0.5:0,1.0:0}
for n in [2,3]:
    for s in range(8):
        res=minimize(lambda x:-ratio3(x,n),rng.normal(size=4*n*n)*2,method='Nelder-Mead',options={'maxiter':2500})
        r3=max(r3,-res.fun)
    for b in r4:
        for s in range(5):
            res=minimize(lambda x:-ratio4(x,n,b),rng.normal(size=4*n*n)*2,method='Nelder-Mead',options={'maxiter':2500})
            r4[b]=max(r4[b],-res.fun)
    print(n,r3,r4,time.time()-t0,flush=True)
