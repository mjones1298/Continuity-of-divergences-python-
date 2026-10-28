import numpy as np
from core import *
rng=np.random.default_rng(11)
def crand(*s): return rng.normal(size=s)+1j*rng.normal(size=s)
for (dx,dy,k1,k2) in [(2,2,2,1),(3,2,2,3),(2,3,3,2),(3,3,5,2)]:
    kp=[crand(dy,dx) for _ in range(k1)]; kq=[crand(dy,dx) for _ in range(k2)]
    Jp,Jq=choi(kp,dx),choi(kq,dx)
    G,_,_=gmean_frame(Jp,Jq)
    A=crand(dx,dx); A/=np.linalg.norm(A)
    u=A.reshape(-1); rho=np.outer(u,u.conj())
    def app(kr):
        return sum(np.kron(K,np.eye(dx))@rho@np.kron(K,np.eye(dx)).conj().T for K in kr)
    C=np.kron(np.eye(dy),A.T)
    e1=np.abs(app(kp)-C@Jp@C.conj().T).max()
    P,Q=app(kp),app(kq)
    Gu,_,_=gmean_frame(P,Q)
    lhs=np.trace(Gu).real
    rhs=np.trace((A@A.conj().T).T@ptrace_out(G,dy,dx)).real
    print(dx,dy,"eq(3.1) err",e1,"  tr(P#Q) - tr((AA*)^T Tr_Y(J#J)) =",lhs-rhs)
