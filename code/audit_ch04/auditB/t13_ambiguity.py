# Literal-reading counterexample to Thm 4.6 if T1, T2 may be taken w.r.t. DIFFERENT Kraus decompositions of Omega.
import sys; sys.path.insert(0,'.')
from lib import *
from core import ptrace_out
dX=dY=2
V1=np.array([[1,0],[0,0]],complex)      # |0><0|
V2=np.eye(2,dtype=complex)              # identity
kr=[V1,V2]
W=choi_w(kr,dX)
print("singular values of W (independence):",np.linalg.svd(W,compute_uv=False))
g=lambda t: np.sqrt(t*(1-t))
a,b=0.5,0.1
T1=np.diag([a,b]).astype(complex); T2=np.diag([b,a]).astype(complex)
def Fhat(T):
    M=ptrace_out(W@psd_fun2(T,g)@W.conj().T,dY,dX)
    return np.linalg.eigvalsh(herm(M))[0]
F1,F2=Fhat(T1),Fhat(T2)
Om=opnorm(sum(K.conj().T@K for K in kr))
print("F1,F2",F1,F2,"|diff|",abs(F1-F2))
print("fixed decomposition: bound",Om*np.sqrt(opnorm(T1-T2)),"(holds)")
# derivative of Phi2 = Psi_{T2} w.r.t. the permuted decomposition (V2,V1): swap indices
P=np.array([[0,1],[1,0]]); T2perm=P@T2@P.T
print("T2 in decomposition (V2,V1):",np.diag(T2perm).real,"; ||T1-T2perm||=",opnorm(T1-T2perm))
print("literal RHS with mixed decompositions:",Om*np.sqrt(opnorm(T1-T2perm)),"<  LHS",abs(F1-F2))
