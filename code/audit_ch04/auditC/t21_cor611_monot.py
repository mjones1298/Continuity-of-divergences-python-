"""F5: replacing delta=||T1-T2|| by delta'=|||Phi1-Phi2|||_1/nu^2 > 1 in Cor 6.11 (BS bound) is false: the RHS
psi(d)=d*(log(4/(eps d))+1/eps) is increasing only on d<=1 and becomes negative for d>(4/eps)e^{1/eps}; LHS>=0."""
import numpy as np
for eps in (0.5,0.25,0.1):
    psi=lambda d: d*(np.log(4/(eps*d))+1/eps)
    dstar=(4/eps)*np.exp(1/eps)
    ds=np.linspace(1e-6,1,200001)
    mono=np.all(np.diff(psi(ds))>0)
    print(f'eps={eps}: psi increasing on (0,1]: {mono};  psi(d)<0 for d>{dstar:.4g};  psi(2*dstar)={psi(2*dstar):.4g}')
# also unitarily-invariant version of Lemma 8.1 (F3)
rng=np.random.default_rng(0)
def herm(n):
    a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); return (a+a.conj().T)/2
worst=0
for _ in range(200):
    dY,dX=3,3; r=int(rng.integers(1,dX*dY+1))
    Ks=[rng.normal(size=(dY,dX))+1j*rng.normal(size=(dY,dX)) for _ in range(r)]
    W=np.stack([K.reshape(-1) for K in Ks],axis=1)
    K=herm(r)
    Jo=W@W.conj().T; Jk=W@K@W.conj().T
    w,V=np.linalg.eigh(Jo); wi=np.where(w>1e-10,w**-0.5,0); R=(V*wi)@V.conj().T
    X=R@Jk@R
    for p in (1,2,np.inf):
        a=np.linalg.norm(np.linalg.eigvalsh(X),p) if p!=np.inf else np.abs(np.linalg.eigvalsh(X)).max()
        b=np.linalg.norm(np.linalg.eigvalsh(K),p) if p!=np.inf else np.abs(np.linalg.eigvalsh(K)).max()
        worst=max(worst,abs(a-b))
print('Lemma 8.1 holds for Schatten 1,2,inf norms too: max discrepancy',worst)
