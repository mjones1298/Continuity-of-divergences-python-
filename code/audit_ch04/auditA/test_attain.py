import numpy as np
from scipy.linalg import sqrtm
exec(open('test_fhat.py').read().split("def direct")[0])
rng=np.random.default_rng(31)
for dx,dy in [(2,2),(2,3),(3,2)]:
    n=dx*dy
    kp=rk(n,dy,dx); kq=rk(n,dy,dx)
    lam=lam_formula(kp,kq,dx,dy)
    Jp,Jq=choi(kp,dx),choi(kq,dx); G,_,_=gmean_frame(Jp,Jq); M=ptrace_out(G,dy,dx); M=(M+M.conj().T)/2
    w,Q=np.linalg.eigh(M); v=Q[:,0]
    A0=np.outer(v.conj(),v.conj()) # A A^* = vbar vbar^*  -> (AA^*)^T = v v^*
    A0=np.outer(v.conj(),np.ones(dx))/np.sqrt(dx)  # rank-one with (AA*)^T = v v*
    print(dx,dy,"lam_min =",lam," F at the singular (rank-one) A_0 with (A A*)^T=vv* :",F_of_A(A0,kp,kq,dx)," (>= lam by transformer ineq.)")
    for s in [1e-1,1e-2,1e-3,1e-4]:
        D=(1-s)*np.outer(v,v.conj())+s*np.eye(dx)/dx
        print("    s=%g F=%.8f"%(s,F_of_A(sqrtm(D.T),kp,kq,dx)))
