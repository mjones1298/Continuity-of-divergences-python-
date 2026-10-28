import numpy as np, mpmath as mp
from qlib import *
mp.mp.dps=100
def P(v): v=np.asarray(v,dtype=complex); return np.outer(v,v.conj())
for th in ['0.3','0.1','0.01','0.001']:
    t=mp.mpf(th); thf=float(t)
    k0=[mp.mpf(1),mp.mpf(0)]; kt=[mp.cos(t),mp.sin(t)]
    I=mp.eye(2)
    J1=mpkron(mpouter(k0),I); Jt=mpkron(mpouter(kt),I); J2=(J1+Jt)/2
    f1=[fid_mp(J1,Jt,2,2,eps,dps=100) for eps in ['1e-10','1e-20','1e-40']]
    f2=fid_mp(J2,J2,2,2,'1e-40',dps=100)
    J1n,Jtn,J2n=mp2np(J1),mp2np(Jt),mp2np(J2)
    s1=fid_sdp(J1n,Jtn,2,2); s2=fid_sdp(J2n,J2n,2,2)
    dn=diamond(J1n-J2n,2,2)
    W=np.column_stack([vec(K) for K in kraus_from_choi(J1n+Jtn,2,2)])
    lm=np.linalg.eigvalsh(W.conj().T@W)[0]
    print(f"th={th}: F(Phi1,Psi1) eps=1e-10,1e-20,1e-40: {[mp.nstr(x,4) for x in f1]} SDP={s1:.2e} | F(Phi2,Psi2)={mp.nstr(f2,15)} SDP={s2:.8f}")
    print(f"   diamond={dn:.10f} sin={np.sin(thf):.10f} | lmin(W*W)={lm:.8e} th^2/2={thf**2/2:.8e} 1-cos={1-np.cos(thf):.8e}")
