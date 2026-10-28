import numpy as np
from lib import *
def struct_instance(dY,dX,rng,n_shared,n_e1,n_e2):
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
            out.append(Mi@G@Mi.conj().T)
        return out
    P1=pair(idx1);P2=pair(idx2)
    return P1[0],P1[1],P2[0],P2[1]
