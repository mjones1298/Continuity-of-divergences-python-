import sys; sys.path.insert(0,'.')
from lib import *
g = lambda t: np.sqrt(np.clip(t*(1-t),0,None))
# Theorem 2.15 check + Theorem 4.6 random tests + Prop 4.3
worst = 0; worst_ratio46=0; worst43=0; worst43a=0
for trial in range(300):
    dX=rng.integers(1,4); dY=rng.integers(1,4)
    r=rng.integers(1,dX*dY+1)
    kr=rand_kraus(r,dY,dX); W=choi_w(kr,dX)
    # random normalization of Omega
    # derivative T1, T2 with various spectra incl. endpoints
    def rT():
        mode=rng.integers(0,4)
        if mode==0: spec=rng.uniform(0,1,r)
        elif mode==1: spec=rng.choice([0,1],r).astype(float)
        elif mode==2: spec=np.clip(rng.normal(0.5,0.5,r),0,1)
        else: spec=rng.choice([0,1,0.5],r)
        return effect(r,spec)
    T1,T2=rT(),rT()
    J1=W@T1@W.conj().T; J2=W@(np.eye(r)-T1)@W.conj().T
    M=mean_reg(J1,J2,np.sqrt,1e-11)
    M2=W@psd_fun2(T1,g)@W.conj().T
    err=np.abs(M-M2).max()/max(1,np.abs(M2).max())
    worst=max(worst,err)
    F1=np.linalg.eigvalsh(herm(ptrace_out(M2,dY,dX)))[0]
    Mb=W@psd_fun2(T2,g)@W.conj().T
    F2=np.linalg.eigvalsh(herm(ptrace_out(Mb,dY,dX)))[0]
    Om=np.linalg.norm(sum(K.conj().T@K for K in kr),2)
    d=opnorm(T1-T2)
    if d>1e-9:
        worst_ratio46=max(worst_ratio46,abs(F1-F2)/(Om*np.sqrt(d)))
    # prop 4.3
    K=psd_fun2(T1,g)-psd_fun2(T2,g)
    if d>1e-9:
        worst43=max(worst43,opnorm(K)/np.sqrt(d))
    A1=T1-T1@T1;A2=T2-T2@T2
    if d>1e-9: worst43a=max(worst43a,opnorm(A1-A2)/d)
print("Thm2.15 err",worst,"Thm4.6 max ratio",worst_ratio46,"4.3(2) max ratio",worst43,"4.3(1) max ratio",worst43a)
