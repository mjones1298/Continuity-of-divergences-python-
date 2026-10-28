import numpy as np
from scipy.optimize import minimize
exec(open('test_fhat.py').read().split("cases = []")[0])
rng = np.random.default_rng(7)
dx,dy=2,2
kp=rk(4,dy,dx); kq=rk(4,dy,dx)
lam=lam_formula(kp,kq,dx,dy)
Jp,Jq=choi(kp,dx),choi(kq,dx)
G,_,_=gmean_frame(Jp,Jq)
M=ptrace_out(G,dy,dx)
print("lam",lam, "eig M",np.linalg.eigvalsh((M+M.conj().T)/2))
best=None
for s in range(6):
    x0=rng.normal(size=2*dx*dx)
    fun=lambda x:F_of_A((x[:dx*dx]+1j*x[dx*dx:]).reshape(dx,dx),kp,kq,dx)
    r=minimize(fun,x0,method='Nelder-Mead',options=dict(maxiter=4000,xatol=1e-10,fatol=1e-13))
    A=(r.x[:dx*dx]+1j*r.x[dx*dx:]).reshape(dx,dx); A/=np.linalg.norm(A)
    sv=np.linalg.svd(A,compute_uv=False)
    pred=np.trace((A@A.conj().T).T@M).real
    print("F direct",r.fun,"pred tr((AA*)^T M)",pred,"sv",sv)
