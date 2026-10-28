import sys; sys.path.insert(0,'.')
from lib import *
from scipy.optimize import minimize
g=lambda t: np.sqrt(np.clip(t*(1-t),0,None))
def sq(H):
    H=herm(H); w,V=eigh(H); return (V*(1/(1+np.exp(-w))))@V.conj().T
def mk(x,n):
    m=n*n
    A=(x[:m]+1j*x[m:2*m]).reshape(n,n); B=(x[2*m:3*m]+1j*x[3*m:4*m]).reshape(n,n)
    return sq(A),sq(B)
def ratio(x,n):
    E,F=mk(x,n); d=opnorm(E-F)
    if d<1e-9: return 0
    return opnorm(psd_fun2(E,g)-psd_fun2(F,g))/np.sqrt(d)
best={}
for n in [1,2,3]:
    b=0
    for s in range(10):
        res=minimize(lambda x:-ratio(x,n),rng.normal(size=4*n*n)*3,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-9,'fatol':1e-12})
        b=max(b,-res.fun)
    best[n]=b
print(best)
# Also E,E' with sup at fixed small delta: scalar sup sqrt(d(1-d)) vs matrices
for d in [0.5,0.2,0.05,0.01]:
    bb=0
    for n in [2,3]:
        for s in range(6):
            def obj(x):
                E,F=mk(x,n)
                # rescale F towards E to have distance exactly d: F' = E + d*(F-E)/||F-E||
                D=F-E; nd=opnorm(D)
                if nd<1e-9: return 0
                F2=E+d*D/nd
                w=np.linalg.eigvalsh(F2)
                if w[0]<0 or w[-1]>1: return 0
                return -opnorm(psd_fun2(E,g)-psd_fun2(F2,g))
            res=minimize(obj,rng.normal(size=4*n*n)*3,method='Nelder-Mead',options={'maxiter':3000})
            bb=max(bb,-res.fun)
    print("delta",d,"matrix sup",bb,"scalar sqrt(d(1-d))",np.sqrt(d*(1-d)),"bound sqrt(d)",np.sqrt(d))
