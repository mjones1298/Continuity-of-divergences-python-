import numpy as np
from core import *
rng=np.random.default_rng(3)
def crand(*s): return rng.normal(size=s)+1j*rng.normal(size=s)
dx,dy,r=3,2,4
V=[crand(dy,dx) for _ in range(r)]
# W = [vec(V_k)], vec(K)=sum_j K e_j (x) e_j  -> index (i,j) -> i*dx+j  (row-major flatten of K)
W=np.stack([v.reshape(-1) for v in V],axis=1)
J=choi(V,dx)
print("J = W W* :",np.abs(J-W@W.conj().T).max())
# Psi_T(X)=sum T_kl V_k X V_l^*, J(Psi_T)=W T W^*
A=crand(r,r); T=A@A.conj().T; T/=np.linalg.eigvalsh(T)[-1]*1.3   # 0<=T<=1 Hermitian
def PsiT(X, T): return sum(T[k,l]*V[k]@X@V[l].conj().T for k in range(r) for l in range(r))
Jpsi=sum(np.kron(PsiT(np.outer(np.eye(dx)[i],np.eye(dx)[j]),T),np.outer(np.eye(dx)[i],np.eye(dx)[j])) for i in range(dx) for j in range(dx))
print("J(Psi_T) = W T W*:",np.abs(Jpsi-W@T@W.conj().T).max())
# effect thm
S=sum(np.kron(V[k],np.eye(r)[:,[k]]) for k in range(r))   # S x = sum V_k x (x) e_k : (dy*r) x dx
X=crand(dx,dx)
SXS=S@X@S.conj().T
def ptrace_Z(M): return np.einsum('iaja->ij',M.reshape(dy,r,dy,r))
def ptrace_Y(M): return np.einsum('aiaj->ij',M.reshape(dy,r,dy,r))
lhs=ptrace_Z(np.kron(np.eye(dy),T.T)@SXS)
print("(1):",np.abs(lhs-PsiT(X,T)).max())
print("(2):",abs(np.trace(PsiT(X,T))-np.trace(T.T@ptrace_Y(SXS))))
Pstar1=sum(T[k,l]*V[l].conj().T@V[k] for k in range(r) for l in range(r))   # Psi_T^*(1)=sum T_kl V_l^* V_k
print("(3):",np.abs(Pstar1-S.conj().T@np.kron(np.eye(dy),T.T)@S).max())
# Phi^c adjoint
# diamond norm of CP map = ||Psi^*(1)||: check against max over pure states of ||(Psi(x)1)(uu*)||_1
def apply_Id_T(u):
    rho=np.outer(u,u.conj())
    out=np.zeros((dy*dx,dy*dx),complex)
    for k in range(r):
        for l in range(r):
            KK=np.kron(V[k],np.eye(dx)); LL=np.kron(V[l],np.eye(dx))
            out+=T[k,l]*KK@rho@LL.conj().T
    return out
best=0
for _ in range(2000):
    u=crand(dx*dx); u/=np.linalg.norm(u)
    best=max(best,np.abs(np.linalg.eigvalsh(apply_Id_T(u))).sum())
print("diamond(PsiT) lower bound from sampling",best," ||Psi^*(1)||=",np.linalg.eigvalsh(Pstar1)[-1])
# max_rho tr(T^T Phi^c(rho))
def Phic(rho): return ptrace_Y(S@rho@S.conj().T)
M=sum(np.eye(dx)[:,[i]]@np.eye(dx)[[i],:]*0 for i in range(dx))
# top eigenvector
w,Q=np.linalg.eigh(Pstar1); rho=np.outer(Q[:,-1],Q[:,-1].conj())
print("tr(T^T Phi^c(rho*)) =",np.trace(T.T@Phic(rho)).real)
# unitary freedom remark: V'_j=sum_k U_jk V_k ; T'=conj(U) T U^T
Uq,_=np.linalg.qr(crand(r,r))
Vp=[sum(Uq[j,k]*V[k] for k in range(r)) for j in range(r)]
Wp=np.stack([v.reshape(-1) for v in Vp],axis=1)
Tp=Uq.conj()@T@Uq.T
print("unitary remark: W' T' W'^* - W T W^* =",np.abs(Wp@Tp@Wp.conj().T-W@T@W.conj().T).max())
# nu is invariant
