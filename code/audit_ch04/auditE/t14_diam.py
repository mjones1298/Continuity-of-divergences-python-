import numpy as np
from lib import *
from diamond import diamond
rng=np.random.default_rng(5)
worst=0
for trial in range(25):
    dY,dX=[(2,2),(2,3),(3,2)][rng.integers(3)];N=dY*dX
    m=rng.integers(2,N+1)
    W=(rng.standard_normal((N,m))+1j*rng.standard_normal((N,m)))
    Z=herm(rng.standard_normal((m,m))+1j*rng.standard_normal((m,m)))
    J=W@Z@W.conj().T
    d=diamond(J,dY,dX)
    ub=opn(ptrace_Y(W@W.conj().T,dY,dX))*opn(Z)
    worst=max(worst,d/ub)
print('max diamond/upper bound',worst)
