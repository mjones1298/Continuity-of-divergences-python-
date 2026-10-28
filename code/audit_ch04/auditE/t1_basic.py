import numpy as np
from lib import *
rng=np.random.default_rng(7)

def same_support_instance(dY,dX,m,rng,spread=1.0):
    N=dY*dX
    U,_=np.linalg.qr(rng.standard_normal((N,N))+1j*rng.standard_normal((N,N)))
    U=U[:,:m]
    Js=[]
    for _ in range(4):
        G=rand_psd(m,m,rng,scale=np.exp(spread*rng.standard_normal()))
        Js.append(U@G@U.conj().T)
    return Js

def dmax(JA,JB):
    # D_max(A||B) = log min{c: A<=cB}; +inf if range not contained
    w,V=eigh(herm(JB)); keep=w>1e-12*max(1,w.max())
    Vr=V[:,keep]; 
    # check range containment
    P=Vr@Vr.conj().T
    if np.abs(JA-P@JA@P).max()>1e-9*max(1,np.abs(JA).max()): return np.inf
    S=Vr/np.sqrt(w[keep])
    M=herm(S.conj().T@JA@S)
    return np.log(lmax(M))

worst=0
for trial in range(300):
    dY,dX=rng.choice([(2,2),(3,2),(2,3),(3,3)])
    N=dY*dX; m=rng.integers(2,N+1)
    Js=same_support_instance(dY,dX,m,rng,spread=rng.choice([0.2,1,2]))
    S=Setup(*Js,dY,dX)
    dT=max(dmax(S.Om[0],S.Om[1]),dmax(S.Om[1],S.Om[0]))
    lhs=np.tanh(dT/2); rhs=2*S.gamma
    worst=max(worst,abs(lhs-rhs))
    # lemma inpair
    for i in range(2):
        pass
    C1=S.C[0]; w,V=eigh(herm(C1))
    a1=fun_herm(C1,np.sqrt); a2=fun_herm(np.eye(S.r)-C1,np.sqrt)
    assert opn(a1-a2)<=2*S.gamma+1e-9
    assert opn(a1+a2)<=np.sqrt(2)+1e-9
    # check C1+C2=1
    assert opn(S.C[0]+S.C[1]-np.eye(S.r))<1e-8
    # gamma<=delta
    assert S.gamma<=S.delta+1e-9
print('tanh formula worst abs err',worst)
