import numpy as np
from core import *
rng=np.random.default_rng(2)
def crand(*s): return rng.normal(size=s)+1j*rng.normal(size=s)
for dx,dy in [(1,2),(2,2),(3,2),(2,3)]:
    th=0.05
    y1=np.zeros(dy,complex); y1[0]=1
    y2=np.zeros(dy,complex); y2[0]=np.cos(th); y2[1]=np.sin(th)*np.exp(0.7j)
    phi=np.outer(y1,y1.conj()); psi=np.outer(y2,y2.conj())
    # replacement channel tr(.) rho  has Kraus ops  sqrt-free: |y><e_j|
    def repl(y): return [np.outer(y,np.eye(dx)[j]) for j in range(dx)]
    K1,K2=repl(y1),repl(y2)
    J1,J2=choi(K1,dx),choi(K2,dx)
    print(f"dx={dx} dy={dy} J(Phi1)=phi(x)1:", np.abs(J1-np.kron(phi,np.eye(dx))).max())
    f1=lam=np.linalg.eigvalsh(ptrace_out(gmean_frame(J1,J2)[0],dy,dx))[0]
    Om=J1+J2; J3=Om/2
    f2=np.linalg.eigvalsh(ptrace_out(gmean_frame(J3,J3)[0],dy,dx))[0]
    # diamond norm of Phi1-Phi2 = (Phi1-Psi1)/2 : max over u of trace norm
    best=0
    for _ in range(300):
        u=crand(dx*dx); u/=np.linalg.norm(u); rho=np.outer(u,u.conj())
        def app(kr): return sum(np.kron(K,np.eye(dx))@rho@np.kron(K,np.eye(dx)).conj().T for K in kr)
        D=(app(K1)-app(K2))/2
        best=max(best,np.abs(np.linalg.eigvalsh(D)).sum())
    print(f"   Fhat(Phi1,Psi1)={f1:.2e}  Fhat(Phi2,Psi2)={f2:.8f}  ||Phi1-Phi2||_dia(sampled)={best:.6f}  sin(theta)={np.sin(th):.6f}")
