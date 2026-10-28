import numpy as np, mpmath as mp
from lib import *
mp.mp.dps = 60
def mp_gmean_reg(A,B,eta):
    N=A.shape[0]
    Am=mp.matrix(A.tolist()); Bm=mp.matrix(B.tolist())
    I=mp.eye(N)
    Ae=Am+eta*I; Be=Bm+eta*I
    # geometric mean via A # B = A^{1/2}(A^{-1/2}BA^{-1/2})^{1/2}A^{1/2}; use eigh in mp
    def msqrt(M, inv=False):
        E,Q=mp.eigh(M)
        D=mp.diag([ (1/mp.sqrt(e) if inv else mp.sqrt(e)) for e in E])
        return Q*D*Q.H
    Ah=msqrt(Ae); Aih=msqrt(Ae,True)
    M=Aih*Be*Aih
    M=(M+M.H)/2
    return Ah*msqrt(M)*Ah
rng=np.random.default_rng(1)
N=4
for t in range(3):
    A=rand_psd(N,2,rng);B=rand_psd(N,3,rng)
    G=gmean_frame(A,B)
    for eta in [mp.mpf('1e-6'),mp.mpf('1e-10'),mp.mpf('1e-14')]:
        R=mp_gmean_reg(A,B,eta)
        Rn=np.array(R.tolist(),dtype=complex)
        print(t,eta,np.abs(Rn-G).max())
