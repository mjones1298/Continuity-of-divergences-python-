import numpy as np
from lib import *
rng=np.random.default_rng(3)
lo=1e9;hi=0
for trial in range(40):
    dY,dX=[(2,2),(2,3),(3,2),(3,3)][rng.integers(4)];N=dY*dX
    m=rng.integers(1,N+1)
    J=rand_psd(N,m,rng,np.exp(rng.standard_normal()))
    W=frame(J)
    nu2=nu_of_frame(W,dY,dX,rng,starts=6)**2
    w=np.linalg.eigvalsh(J);lm=w[w>1e-10*w.max()][0]
    lo=min(lo,nu2/(lm/min(dX,dY)));hi=max(hi,nu2/lm)
print('min nu2/(lam/min d) (should be>=1):',lo,' max nu2/lam (should be<=1):',hi)
