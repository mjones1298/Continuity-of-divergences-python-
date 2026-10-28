import numpy as np, warnings
from scipy.optimize import minimize
from schur import schur_norm
warnings.filterwarnings("ignore")
rng=np.random.default_rng(2)
def best(M,cplx,tries=40):
    n=M.shape[0]; b=0
    for _ in range(tries):
        def f(v):
            if cplx: Z=(v[:n*n]+1j*v[n*n:]).reshape(n,n)
            else: Z=v[:n*n].reshape(n,n)
            Z=Z/np.linalg.norm(Z,2)
            return -np.linalg.norm(M*Z,2)
        v0=rng.normal(size=2*n*n if cplx else n*n)
        r=minimize(f,v0,method='BFGS')
        b=max(b,-r.fun)
    return b
for M in [np.array([[-1,1],[1,1.]]), np.array([[-1,1,0.3],[1,1,-1],[0.3,-1,0.5]]), rng.normal(size=(3,3))]:
    M=(M+M.T)/2
    print("SDP:",schur_norm(M)," real-Z sup:",best(M,False)," complex-Z sup:",best(M,True))
