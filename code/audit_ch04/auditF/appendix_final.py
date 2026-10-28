"""Pointwise (optimisation-free) tests of Prop A.1 and Thm A.7 with snapped T-spectrum."""
import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
from scipy.optimize import minimize
rng = np.random.default_rng(2026)
conns = make_conns()

def frame(A,B,tol=1e-11):
    S=herm(A+B); w,V=np.linalg.eigh(S); keep=w>tol*max(1.0,w.max())
    Vr=V[:,keep]; wr=w[keep]
    return Vr*np.sqrt(wr), Vr/np.sqrt(wr)
def snapped_T(A,B):
    Wf,Wi=frame(A,B); T=herm(Wi.conj().T@A@Wi); wt,U=np.linalg.eigh(T); wt=np.clip(wt,0,1)
    wt=np.where(wt<1e-9,0.0,wt); wt=np.where(wt>1-1e-9,1.0,wt)
    return Wf,wt,U
def mean_choi(A,B,conn):
    Wf,wt,U=snapped_T(A,B); ph=conn.phi(wt); return herm(Wf@((U*ph)@U.conj().T)@Wf.conj().T), ph.max()
def kraus_from_choi(J,dY,dX,tol=1e-11):
    w,V=np.linalg.eigh(herm(J)); return [np.sqrt(w[i])*V[:,i].reshape(dY,dX) for i in range(len(w)) if w[i]>tol]
def apply(Ks,X): return sum(K@X@K.conj().T for K in Ks) if Ks else 0*X[:1,:1]
def sp(M,p):
    s=np.linalg.svd(M,compute_uv=False); return s.max() if p==np.inf else (s**p).sum()**(1/p)
def randpsd(d,rank=None):
    r=rank or d; B=rng.normal(size=(d,r))+1j*rng.normal(size=(d,r)); return B@B.conj().T
ps=[1,1.5,2,3,6,np.inf]
worst1=0; worst7=0; worst7cb=0; nonvac=0
inst=[]
for trial in range(60):
    dY=int(rng.integers(2,4)); dX=int(rng.integers(2,4)); d=dY*dX
    r1=int(rng.integers(1,d+1)); r2=int(rng.integers(1,d+1))
    if trial%2==0: r1=r2=d      # full-rank / overlapping
    KP=random_kraus(dY,dX,r1,rng); KQ=random_kraus(dY,dX,r2,rng)
    JP=sum(np.outer(vec(K),vec(K).conj()) for K in KP); JQ=sum(np.outer(vec(K),vec(K).conj()) for K in KQ)
    JO=JP+JQ; KO=kraus_from_choi(JO,dY,dX)
    # A.1: pointwise ||M(X)||_p <= gmax ||Omega(X)||_p on PSD X, and on general X
    for name,cn in conns.items():
        M,gmax=mean_choi(JP,JQ,cn); KM=kraus_from_choi(M,dY,dX)
        if not KM or gmax<1e-6: continue
        for k in range(6):
            X=randpsd(dX,int(rng.integers(1,dX+1)))
            for p in ps:
                a=sp(apply(KM,X),p); b=gmax*sp(apply(KO,X),p)
                if b>1e-12: worst1=max(worst1,a/b)
            Xg=rng.normal(size=(dX,dX))+1j*rng.normal(size=(dX,dX))   # general X: norm of M(X) vs ||Omega|| restricted sup <= (via PSD rest.)
    # A.7 pointwise on PSD X (and cb n=2 via K tensor I_n on PSD X in L(X⊗C^n)) with optimisation of X
    G,_=mean_choi(JP,JQ,conns['geo']); KG=kraus_from_choi(G,dY,dX)
    if KG: nonvac+=1
    for n in (1,2):
        KPn=[np.kron(K,np.eye(n)) for K in KP]; KQn=[np.kron(K,np.eye(n)) for K in KQ]; KGn=[np.kron(K,np.eye(n)) for K in KG]
        D=dX*n
        for p in [1,1.5,2,3,np.inf]:
            def fn(x):
                B=(x[:D*D]+1j*x[D*D:]).reshape(D,D); X=B@B.conj().T
                num=sp(apply(KGn,X),p)**2 if KGn else 0.0
                den=sp(apply(KPn,X),p)*sp(apply(KQn,X),p)
                return -(num/den) if den>1e-14 else 0.0
            best=0
            for s in range(3 if n==2 else 4):
                if D*D*2>40 and s>=2: break
                res=minimize(fn,rng.normal(size=2*D*D),method='L-BFGS-B',options={'maxiter':150})
                best=max(best,-res.fun)
            if n==1: worst7=max(worst7,best)
            else: worst7cb=max(worst7cb,best)
print("nonvacuous A.7 instances (geometric mean nonzero):",nonvac,"of 60")
print("A.1 pointwise worst ||M(X)||_p/(gmax ||Omega(X)||_p), 8 connections, p in",ps,":",worst1)
print("A.7 optimised-X worst ||(Phi#Psi)(X)||_p^2/(||Phi(X)||_p ||Psi(X)||_p), n=1:",worst7)
print("A.7 optimised-X worst, cb n=2:",worst7cb)
