import numpy as np
from schur import *
g = lambda t: (2*t-1)**2/(1-t)
dg = lambda t: (2*t-1)*(3-2*t)/(1-t)**2
print('eps, Lip(grid n=12 incl endpoints), 2-point endpoint Schur, eps^-2-4, max|k\'|')
for eps in (0.45,0.4,0.36,1/3,0.3,0.2,0.1):
    lam=np.linspace(0,1-eps,12)
    L=schur_norm(divdiff(g,dg,lam))
    L2=schur_norm(divdiff(g,dg,np.array([0,1-eps])))
    ts=np.linspace(0,1-eps,4001)
    print(round(eps,4),L,L2,eps**-2-4,np.max(np.abs(dg(ts))),flush=True)
