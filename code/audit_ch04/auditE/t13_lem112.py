import numpy as np
from lib import *
from t5_gen import struct_instance
rng=np.random.default_rng(77)
mk=0;m1=0;m2=0;cnt=0
for trial in range(1500):
    dY,dX=[(2,2),(3,2),(2,3),(1,3),(3,1)][rng.integers(5)]
    N=dY*dX
    ns=rng.integers(0,N); ne1=rng.integers(0,N-ns+1); ne2=rng.integers(0,N-ns+1)
    if ns+ne1+ne2>N or ns+ne1==0 or ns+ne2==0: continue
    J=struct_instance(dY,dX,rng,ns,ne1,ne2)
    S=Setup(*J,dY,dX);q=sec11(S,tol=1e-7)
    cnt+=1
    mk=max(mk,q['kap1'],q['kap2'])
    assert q['gam_ac']<0.5
    a=q['hac2']
    b=opn(S.Delta(q['QR']@fun_herm(S.C[0]-S.C[1],np.abs)@q['QR']))
    c=4*q['gam_ac']**2*q['DQR']
    assert a<=min(b,c)+1e-8,(a,b,c)
    assert c<=8*q['gam_ac']**2*S.nOmb+1e-8
    # also: sums of ranks / Q consistency
    assert opn(q['Q1']+q['QR']+q['Q2']-np.eye(S.r))<1e-8
print(cnt,'max kappa',mk)
