import numpy as np, warnings
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
rng=np.random.default_rng(8)
L=np.log
def mf(A,h):
    w,V=np.linalg.eigh(A); return (V*h(w))@V.conj().T
def herm_from(v,n,lo,hi):
    # v: n real eigen-params + n*n complex unitary params
    ev=lo+(hi-lo)/(1+np.exp(-v[:n]))
    G=(v[n:n+n*n]+1j*v[n+n*n:n+2*n*n]).reshape(n,n)
    Q,_=np.linalg.qr(G)
    return (Q*ev)@Q.conj().T
def search(h,lo,hi,n=3,tries=25):
    best=0
    def neg(v):
        X=herm_from(v[:n+2*n*n],n,lo,hi); Y=herm_from(v[n+2*n*n:],n,lo,hi)
        d=np.linalg.norm(X-Y,2)
        if d<1e-9: return 0
        return -np.linalg.norm(mf(X,h)-mf(Y,h),2)/d
    for _ in range(tries):
        v0=rng.normal(size=2*(n+2*n*n))*2
        r=minimize(neg,v0,method='BFGS',options={'maxiter':400})
        best=max(best,-r.fun)
    return best
tests={
 'Hellinger [0.2,0.8] claim 1.5': (lambda t:1-2*np.sqrt(t*(1-t)),0.2,0.8,1.5),
 'BS [0.2,0.8] claim 6.386294': (lambda t:t*L(t/(1-t)),0.2,0.8,6.386294),
 'JS [0.1,0.9] claim 2.197225': (lambda t:np.log(2)+t*L(t)+(1-t)*L(1-t),0.1,0.9,np.log(9)),
 'k_z z=0.05 on [0,1] Lip=1.3?': (lambda t:t+2*0.05**2/(t+0.05),0.0,1.0,None),
}
for name,(h,lo,hi,claim) in tests.items():
    b=search(h,lo,hi,n=3,tries=12)
    print(name,"-> best operator ratio found:",b)
