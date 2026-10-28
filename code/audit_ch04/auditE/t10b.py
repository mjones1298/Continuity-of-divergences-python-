import numpy as np
from lib import *
from t5_gen import struct_instance
from t10_leb import shorted
rng=np.random.default_rng(41)
worst=(0,None)
for trial in range(300):
    dY,dX=[(2,2),(3,2),(2,3)][rng.integers(3)]
    N=dY*dX
    ns=rng.integers(0,N); ne1=rng.integers(0,N-ns+1); ne2=rng.integers(0,N-ns+1)
    if ns+ne1+ne2>N or ns+ne1==0 or ns+ne2==0: continue
    J=struct_instance(dY,dX,rng,ns,ne1,ne2)
    S=Setup(*J,dY,dX);q=sec11(S)
    W=S.W
    G=gmean_frame(S.Om[0],S.Om[1])
    G2=W@(q['a1']@q['a2'])@W.conj().T
    d=np.abs(G-G2).max()
    if d>worst[0]: worst=(d,(dY,dX,ns,ne1,ne2,np.linalg.eigvalsh(S.C[0])))
print(worst)
