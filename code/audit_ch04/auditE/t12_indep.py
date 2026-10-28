import numpy as np
from lib import *
rng=np.random.default_rng(2)
bad=0
for trial in range(300):
    dY,dX=[(2,2),(3,2),(2,3)][rng.integers(3)];N=dY*dX
    k1=rng.integers(1,N);k2=rng.integers(1,N-k1+1)
    M=rng.standard_normal((N,k1+k2))+1j*rng.standard_normal((N,k1+k2))
    M1=M[:,:k1];M2=M[:,k1:]
    mk=lambda Mi:rand_psd(Mi.shape[1],Mi.shape[1],rng,np.exp(rng.standard_normal()))
    Jp1=M1@mk(M1)@M1.conj().T;Jq1=M1@mk(M1)@M1.conj().T
    Jp2=M2@mk(M2)@M2.conj().T;Jq2=M2@mk(M2)@M2.conj().T   # Theta_Phi, Theta_Psi
    F1=Fhat(Jp1,Jq1,dY,dX);FT=Fhat(Jp2,Jq2,dY,dX);F2=Fhat(Jp1+Jp2,Jq1+Jq2,dY,dX)
    nTh=opn(ptrace_Y(Jp2+Jq2,dY,dX))
    ok=(F1+FT<=F2+1e-9) and (F2<=F1+0.5*nTh+1e-9)
    # identity of means
    G=gmean_frame(Jp1+Jp2,Jq1+Jq2)-gmean_frame(Jp1,Jq1)-gmean_frame(Jp2,Jq2)
    # frame-based gmean is exact, compare with regularised version for the first, loose
    if not ok or opn(G)>1e-6: bad+=1;print('BAD',F1,FT,F2,nTh,opn(G))
    # theorem 11.3 consistency: ms equals ||Theta^*(1)||
    S=Setup(Jp1,Jq1,Jp1+Jp2,Jq1+Jq2,dY,dX);q=sec11(S,tol=1e-8)
    if abs(q['ms']-nTh)>1e-6 or q['tau_ac']>1e-6 or q['kap1']>1e-6 or q['kap2']>1e-6 or q['gam_ac']>1e-6:
        bad+=1;print('BADq',q['ms'],nTh,q['tau_ac'],q['kap1'],q['kap2'],q['gam_ac'])
print('bad',bad)
