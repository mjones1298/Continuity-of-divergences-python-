import numpy as np
from scipy.linalg import sqrtm
exec(open('test_fhat.py').read().split("def direct")[0])
rng = np.random.default_rng(21)
def hrand(d):
    Z = rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)); Q,_ = np.linalg.qr(Z); return Q
cases=[]
for dx,dy in [(2,2),(2,3),(3,2),(3,3)]:
    n=dx*dy
    cases.append((f"full-rank dx={dx} dy={dy}",dx,dy,rk(n,dy,dx),rk(n,dy,dx)))
    cases.append((f"singular sum dx={dx} dy={dy}",dx,dy,rk(2,dy,dx),rk(2,dy,dx)))
    K=rk(3,dy,dx); cases.append((f"shared Kraus dx={dx} dy={dy}",dx,dy,[K[0],K[1]],[.7*K[1],K[2]]))
    P_=rk(n,dy,dx); cases.append((f"Phi=Psi dx={dx} dy={dy}",dx,dy,P_,P_))
    cases.append((f"Phi=2 Psi (CP, non-channel) dx={dx} dy={dy}",dx,dy,[np.sqrt(2)*k for k in P_],P_))
U=hrand(2);U2=hrand(2)
cases.append(("two unitaries",2,2,[U],[U2]))
for name,dx,dy,kp,kq in cases:
    lam=lam_formula(kp,kq,dx,dy)
    vals=[]
    for _ in range(1500):
        A=rng.normal(size=(dx,dx))+1j*rng.normal(size=(dx,dx))
        if rng.random()<.3:
            U_,s,Vh=np.linalg.svd(A); s[-1]=0; A=(U_*s)@Vh
        vals.append(F_of_A(A,kp,kq,dx))
    lb=min(vals)
    Jp,Jq=choi(kp,dx),choi(kq,dx); G,_,_=gmean_frame(Jp,Jq); M=ptrace_out(G,dy,dx); M=(M+M.conj().T)/2
    w,Q=np.linalg.eigh(M); v=Q[:,0]
    out=[]
    for s in [1e-1,1e-2,1e-3]:
        D=(1-s)*np.outer(v,v.conj())+s*np.eye(dx)/dx
        A=sqrtm(D.T)
        out.append(F_of_A(A,kp,kq,dx)-lam)
    print(f"{name:48s} lam={lam:10.6f}  min(1500 random u)-lam={lb-lam:+.2e} ; F(u_s)-lam, s=.1,.01,.001: "+" ".join(f"{x:.1e}" for x in out))
