import sys; sys.path.insert(0,'.')
from lib import *
from scipy.optimize import minimize
g=lambda t: np.sqrt(np.clip(t*(1-t),0,None))
dX,dY,r=2,2,3
def sq(H):
    H=herm(H); w,V=eigh(H); return (V*(1/(1+np.exp(-np.clip(w,-50,50)))))@V.conj().T
def unpack(x):
    k=0; kr=[]
    for i in range(r):
        n=dY*dX
        K=(x[k:k+n]+1j*x[k+n:k+2*n]).reshape(dY,dX); k+=2*n; kr.append(K)
    Ts=[]
    for j in range(2):
        H=(x[k:k+r*r]+1j*x[k+r*r:k+2*r*r]).reshape(r,r); k+=2*r*r; Ts.append(sq(H))
    return kr,Ts
def ratio(x):
    kr,(T1,T2)=unpack(x)
    W=choi_w(kr,dX)
    if np.linalg.svd(W,compute_uv=False)[-1]<1e-6: return 0
    d=opnorm(T1-T2)
    if d<1e-10: return 0
    Om=opnorm(sum(K.conj().T@K for K in kr))
    f=lambda T: np.linalg.eigvalsh(herm(ptrace_out(W@psd_fun2(T,g)@W.conj().T,dY,dX)))[0]
    return abs(f(T1)-f(T2))/(Om*np.sqrt(d))
n=2*r*dX*dY+4*r*r
best=0
for s in range(12):
    x0=rng.normal(size=n)*2
    res=minimize(lambda x:-ratio(x),x0,method='Nelder-Mead',options={'maxiter':6000,'xatol':1e-9,'fatol':1e-12})
    best=max(best,-res.fun); print(s,-res.fun,flush=True)
print("best",best)
