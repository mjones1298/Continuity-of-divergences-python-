import numpy as np
from lib import *
rng=np.random.default_rng(12)
f=lambda t:(t-1)**2/(t+1)  # f(0+)=1, f'(inf)=1
h=lambda x:(2*np.asarray(x,float)-1)**2
dX,dY,r=2,2,4
Ks=rand_kraus(dX,dY,r); W=W_of(Ks)
spec=np.array([0.0,1.0,0.3,0.8]); T=rand_T(r,spec=spec)
w,V=np.linalg.eigh(T)
P0=V[:,np.abs(w)<1e-9]@V[:,np.abs(w)<1e-9].conj().T
P1=V[:,np.abs(w-1)<1e-9]@V[:,np.abs(w-1)<1e-9].conj().T
Pa=np.eye(r)-P0-P1
Jac1=W@T@Pa@W.conj().T; Jac2=W@(np.eye(r)-T)@Pa@W.conj().T
# P_f on common range via Def 6.2
ww,VV=np.linalg.eigh(Jac2); U=VV[:,ww>1e-9]
A0=U.conj().T@Jac1@U; B0=U.conj().T@Jac2@U
w0,V0=np.linalg.eigh(B0); Bh=(V0*np.sqrt(w0))@V0.conj().T; Bih=(V0/np.sqrt(w0))@V0.conj().T
X=herm(Bih@A0@Bih); P=U@(Bh@fun(X,f)@Bh)@U.conj().T
lhs=ptraceY(W@hfun_T(T,h)@W.conj().T,dY,dX)
rhs=ptraceY(P,dY,dX)+1.0*ptraceY(W@P1@W.conj().T,dY,dX)+1.0*ptraceY(W@P0@W.conj().T,dY,dX)
print(np.linalg.norm(lhs-rhs))
