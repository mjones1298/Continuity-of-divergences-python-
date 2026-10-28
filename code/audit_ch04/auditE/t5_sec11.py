import numpy as np
from lib import *
rng=np.random.default_rng(11)

def struct_instance(dY,dX,rng,n_shared,n_e1,n_e2,scale_small=1.0):
    N=dY*dX
    k=n_shared+n_e1+n_e2
    M=rng.standard_normal((N,k))+1j*rng.standard_normal((N,k))
    idx1=list(range(n_shared))+list(range(n_shared,n_shared+n_e1))
    idx2=list(range(n_shared))+list(range(n_shared+n_e1,k))
    def pair(idx):
        Mi=M[:,idx]
        out=[]
        for _ in range(2):
            G=rand_psd(len(idx),len(idx),rng,scale=np.exp(rng.standard_normal()))
            # random scaling of extra directions
            out.append(Mi@G@Mi.conj().T)
        return out
    P1=pair(idx1);P2=pair(idx2)
    return P1[0],P1[1],P2[0],P2[1]

worst=0;cnt=0
for trial in range(3000):
    dY,dX=[(2,2),(3,2),(2,3),(1,3),(3,1)][rng.integers(5)]
    N=dY*dX
    ns=rng.integers(0,N); 
    ne1=rng.integers(0,N-ns+1); ne2=rng.integers(0,N-ns+1)
    if ns+ne1+ne2>N or ns+ne1==0 or ns+ne2==0: continue
    J=struct_instance(dY,dX,rng,ns,ne1,ne2)
    S=Setup(*J,dY,dX)
    q=sec11(S)
    F1=Fhat(J[0],J[1],dY,dX);F2=Fhat(J[2],J[3],dY,dX)
    lhs=abs(F1-F2); R=R_gsharp(S,q)
    # also compare via delta,Delta(N)
    ratio=lhs/R if R>1e-12 else 0
    worst=max(worst,ratio);cnt+=1
    if ratio>1: print('VIOLATION',dY,dX,ns,ne1,ne2,lhs,R)
print(cnt,'worst ratio lhs/R',worst)
