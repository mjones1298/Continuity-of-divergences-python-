import numpy as np
from lib import *
# Example 11.7 (coupling): dim X=1, Y=C^2
for t in [1e-2,1e-4,1e-6]:
    P=0.5*np.array([[1,1],[1,1.]])
    Zt=np.diag([np.sqrt(t),1.])
    JPhi=Zt@P@Zt;JPsi=Zt@(np.eye(2)-P)@Zt
    J0=0.5*np.diag([0,1.])
    S=Setup(JPhi,JPsi,J0,J0,2,1)
    q=sec11(S,tol=1e-9)
    F1=Fhat(JPhi,JPsi,2,1);F2=Fhat(J0,J0,2,1)
    print('t',t,'F1',F1,'F2',F2,'tau_ac',q['tau_ac'],'kap1',q['kap1'],'kap2',q['kap2'],'ms',q['ms'],'gam_ac',q['gam_ac'],'hac',q['hac'],'R',R_gsharp(S,q),'term',S.nOm[0]*np.sqrt(q['kap1']),(1+t)/np.sqrt(2))
# example after Cor 11.4 (dim X=1 rank one)
rng=np.random.default_rng(0)
th=0.01
y1=np.array([1,0],complex);y2=np.array([np.cos(th),np.sin(th)],complex)
P1=np.outer(y1,y1.conj());P2=np.outer(y2,y2.conj())
S=Setup(P1,P1,P1,P2,2,1)
q=sec11(S,tol=1e-9)
print('F1',Fhat(P1,P1,2,1),'F2',Fhat(P1,P2,2,1),'T1',q['T1'],'T2',q['T2'],'C1',np.linalg.eigvalsh(S.C[0]),'ms',q['ms'],'tau_ac',q['tau_ac'])
