import numpy as np
from lib import *
rng=np.random.default_rng(4)
def short(A, S):
    """shorted operator of PSD A to subspace with orthonormal basis columns S: max C<=A, ran C in S"""
    n=A.shape[0]
    # complement basis
    Q,_=np.linalg.qr(np.hstack([S,rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))]))
    Sc=Q[:,S.shape[1]:n]
    # A in block form wrt [S, Sc]
    B=np.hstack([S,Sc]); Ab=B.conj().T@A@B; k=S.shape[1]
    A11=Ab[:k,:k]; A12=Ab[:k,k:]; A22=Ab[k:,k:]
    C=np.zeros_like(Ab); C[:k,:k]=A11-A12@np.linalg.pinv(A22)@A12.conj().T
    return B@C@B.conj().T
dX,dY,r=2,2,4
Ks=rand_kraus(dX,dY,r); W=W_of(Ks)
spec=np.array([0.0,1.0,1.0,0.4])
T=rand_T(r,spec=spec)
J1=W@T@W.conj().T; J2=W@(np.eye(r)-T)@W.conj().T
w,V=np.linalg.eigh(J2); S=V[:,w>1e-9]
ac=short(J1,S)
w,V=np.linalg.eigh(T); P1=(V[:,np.abs(w-1)<1e-9])@(V[:,np.abs(w-1)<1e-9]).conj().T
pred_ac=W@T@(np.eye(r)-P1)@W.conj().T
print('ac err',np.linalg.norm(ac-pred_ac),' s err',np.linalg.norm((J1-ac)-W@P1@W.conj().T))
