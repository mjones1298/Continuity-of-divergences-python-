import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
rng=np.random.default_rng(5)
conns=make_conns(); worst=0
for trial in range(40):
    d=int(rng.integers(2,4)); m=int(rng.integers(1,4))
    ra=int(rng.integers(1,d+1)); rb=int(rng.integers(1,d+1)); rc=int(rng.integers(1,m+1))
    def psd(n,r):
        B=rng.normal(size=(n,r))+1j*rng.normal(size=(n,r)); return B@B.conj().T
    A,B,C=psd(d,ra),psd(d,rb),psd(m,rc)
    for name in ['geo','harm','mix','alpha0.3']:
        cn=conns[name]
        L=ka_frame_conn(np.kron(A,C),np.kron(B,C),cn,tol=1e-12)
        R=np.kron(ka_frame_conn(A,B,cn,tol=1e-12),C)
        worst=max(worst,np.linalg.norm(L-R)/max(1,np.linalg.norm(R)))
print("max rel |(A(x)C) sigma (B(x)C) - (A sigma B)(x)C| over 40 random (singular) triples, 4 means:",worst)
