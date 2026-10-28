import numpy as np
from lib import *
from scipy.optimize import minimize
rng=np.random.default_rng(31)
def nu_exact_r2(Ks, ng=400):
    # c=(cos a, sin a e^{ib}) global phase irrelevant
    best=1e9
    A=np.linspace(0,np.pi/2,ng); B=np.linspace(0,2*np.pi,ng,endpoint=False)
    # coarse
    vals=np.empty((ng,ng))
    for i,a in enumerate(A):
        for j,b in enumerate(B):
            V=np.cos(a)*Ks[0]+np.sin(a)*np.exp(1j*b)*Ks[1]
            vals[i,j]=np.linalg.norm(V,2)
    i,j=np.unravel_index(vals.argmin(),vals.shape)
    f=lambda x: np.linalg.norm(np.cos(x[0])*Ks[0]+np.sin(x[0])*np.exp(1j*x[1])*Ks[1],2)
    res=minimize(f,[A[i],B[j]],method='Nelder-Mead',options={'xatol':1e-12,'fatol':1e-15})
    return res.fun
viol=0
for trial in range(60):
    dX=rng.integers(1,4); dY=rng.integers(1,4)
    if dX*dY<2: continue
    Ks=rand_kraus(dX,dY,2)
    if trial%3==0: Ks[1]=Ks[0]*rng.normal()+0.05*Ks[1]
    W=W_of(Ks); J=W@W.conj().T; ev=np.linalg.eigvalsh(J); lm=ev[ev>1e-12*ev.max()].min()
    nu2=nu_exact_r2(Ks,ng=60)**2
    lo=lm/min(dX,dY); hi=lm
    if not (lo-1e-9<=nu2<=hi+1e-9): viol+=1; print('VIOL Lemma 8.2(2)',dX,dY,lo,nu2,hi)
    # Lemma 8.2(1): formula
    # Lemma 8.2(3): Omega' >= Omega with same range
    Kp=[K for K in Ks]  # Omega' = Omega + extra with range inside range(J)
    # add a CP perturbation within the range: Omega' = Omega + eps*Psi_S with S psd
    S=rand_T(2,spec=rng.uniform(0,1,size=2))
    Wc=W
    # Kraus ops of Psi_S: columns of W S^{1/2}
    w,V=np.linalg.eigh(S); Sh=(V*np.sqrt(np.clip(w,0,None)))@V.conj().T
    Ws=W@Sh
    Kp=[Ks[k]+0 for k in range(2)]
    # Omega' has J' = W(1+S)W^*, Kraus: W (1+S)^{1/2}
    M=np.linalg.cholesky(np.eye(2)+S)  # (1+S)=M M^*
    Wp=W@M
    Kp=[Wp[:,k].reshape(dY,dX) for k in range(2)]
    nu2p=nu_exact_r2(Kp,ng=60)**2
    if nu2p < nu2-1e-7: viol+=1; print('VIOL Lemma 8.2(3)',nu2,nu2p)
print('violations',viol)
