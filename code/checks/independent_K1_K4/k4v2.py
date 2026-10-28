from pathlib import Path
import mpmath as mp, numpy as np
from lib import *
mp.mp.dps=50
def mpK(p,d):
    p=mp.mpf(p); s=1/mp.sqrt(2)
    V=mp.matrix([[mp.exp(-1j*d/2),0],[0,mp.exp(1j*d/2)]])*mp.matrix([[s,s],[s,-s]])
    P=[to_mp(X),to_mp(Y),to_mp(Z)]
    return [mp.sqrt(1-3*p/4)*V]+[mp.sqrt(p/4)*Pk*V for Pk in P]
def kron(A,B):
    C=mp.zeros(A.rows*B.rows,A.cols*B.cols)
    for i in range(A.rows):
        for j in range(A.cols):
            if A[i,j]!=0:
                for k in range(B.rows):
                    for l in range(B.cols): C[i*B.rows+k,j*B.cols+l]=A[i,j]*B[k,l]
    return C
def out2(p,d,psi):
    rho=psi*psi.H; Ks=mpK(p,d); I8=mp.eye(8); I2=mp.eye(2)
    for k in range(2):
        new=mp.zeros(16,16)
        for K in Ks:
            Kf=kron(K,mp.eye(8)) if k==0 else kron(kron(I2,K),mp.eye(4))
            new+=Kf*rho*Kf.H
        rho=new
    return rho
def logm(A): return mp_funm_h(A,lambda x: mp.log(x))
def relent(r,s): return mp.re(sum((r*(logm(r)-logm(s)))[i,i] for i in range(r.rows)))
for p in ['0.01','0.001']:
    x=np.load(Path(__file__).resolve().parent / f'k4_best_{float(p)}_2.npy'); v=x[:16]+1j*x[16:]; v/=np.linalg.norm(v)
    psi=mp.matrix([mp.mpc(complex(c)) for c in v]); psi=psi/mp.norm(psi)
    de=mp.mpf('0.01')*mp.mpf(p)
    D=relent(out2(p,0,psi),out2(p,de,psi))
    print('p',p,'n=2 exact (1/2)D/δ² =',mp.nstr(D/2/de**2,10),' *p =',mp.nstr(D/2/de**2*mp.mpf(p),6), ' Schmidt/ancilla rank of psi:',np.linalg.matrix_rank(v.reshape(4,4),1e-6))
