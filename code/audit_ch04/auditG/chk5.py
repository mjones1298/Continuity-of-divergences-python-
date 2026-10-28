# Independent numerical check of abstract/intro claim: |F1-F2| <= ||Omega^*(1)|| ||T1-T2||^{1/2}, Omega a channel-sum, via regularised geometric mean of Choi matrices
import numpy as np
from scipy.linalg import sqrtm
rng=np.random.default_rng(7)
def psd_sqrt(A):
    w,v=np.linalg.eigh(A); w=np.clip(w,0,None); return (v*np.sqrt(w))@v.conj().T
def gm(A,B,eps=1e-10):
    n=A.shape[0]; A=A+eps*np.eye(n); B=B+eps*np.eye(n)
    Ah=psd_sqrt(A); w,v=np.linalg.eigh(A); Aih=(v/np.sqrt(w))@v.conj().T
    M=Aih@B@Aih; M=(M+M.conj().T)/2
    return Ah@psd_sqrt(M)@Ah
def ptrace_Y(J,dY,dX):  # J on Y⊗X (Y first), trace out Y
    return np.einsum('ayby->ab' if False else 'iaib->ab', J.reshape(dY,dX,dY,dX))
worst=0
for trial in range(300):
    dX=int(rng.integers(1,4)); dY=int(rng.integers(2,4)); n=dX*dY
    r=int(rng.integers(1,n+1))
    # random Kraus ops V_k: X->Y, normalised so Omega is a channel (sum V*V = 1)
    V=[rng.normal(size=(dY,dX))+1j*rng.normal(size=(dY,dX)) for _ in range(r)]
    S=sum(v.conj().T@v for v in V); Si=np.linalg.inv(psd_sqrt(S))
    V=[v@Si for v in V]
    # (r may exceed what is possible for linear independence only if r<=n, fine)
    W=np.stack([v.reshape(-1,order='F') for v in V],axis=1)  # vec column-stacking: vec(V)= columns of V stacked; Choi = sum vec(V)vec(V)^* with Y⊗X ordering?
    # Choi in Y⊗X ordering: J = sum_{ij} Phi(E_ij) ⊗ E_ij -> for Phi(X)=V X V^*, J = vec(V)vec(V)^* with vec row-major? build directly:
    def choi(Phi):
        J=np.zeros((n,n),dtype=complex)
        for i in range(dX):
            for j in range(dX):
                E=np.zeros((dX,dX));E[i,j]=1
                J+=np.kron(Phi(E),E)
        return J
    def Phi_T(T):
        return lambda X: sum(T[k,l]*V[k]@X@V[l].conj().T for k in range(r) for l in range(r))
    def rndT():
        G=rng.normal(size=(r,r))+1j*rng.normal(size=(r,r)); P=G@G.conj().T
        w,v=np.linalg.eigh(P); w=w/w.max()*rng.uniform(0.5,1)
        # sometimes make a near-projection
        if rng.random()<0.3: w=np.where(w>np.median(w),1.0,0.0)
        return (v*w)@v.conj().T
    T1=rndT(); T2=rndT()
    if rng.random()<0.5: T2=T1+ (T2-T1)*rng.uniform(0,0.2); 
    def F(T):
        Jp=choi(Phi_T(T)); Jq=choi(Phi_T(np.eye(r)-T))
        M=gm(Jp,Jq); return np.linalg.eigvalsh(ptrace_Y(M,dY,dX)).min()
    F1,F2=F(T1),F(T2)
    OmegaStar1=np.linalg.norm(sum(v.conj().T@v for v in V),2)  # = 1 (Omega channel)
    lhs=abs(F1-F2); rhs=OmegaStar1*np.linalg.norm(T1-T2,2)**0.5
    worst=max(worst,lhs/max(rhs,1e-12) if rhs>1e-9 else 0)
print("max |F1-F2| / (||Omega*(1)|| ||T1-T2||^(1/2)) over random trials (Omega a channel; should be <=1):",worst)
