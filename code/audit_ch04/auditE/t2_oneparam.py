import numpy as np
from scipy.optimize import minimize
from lib import *
rng=np.random.default_rng(3)

def four_from_params(x, r, cplx=True):
    # x -> four PSD G_i (r x r) via G = L L^*, then normalise so that sum = 1
    n=r*r*(2 if cplx else 1)
    Gs=[]
    k=0
    for i in range(4):
        L=x[k:k+r*r].reshape(r,r).astype(complex); k+=r*r
        if cplx:
            L=L+1j*x[k:k+r*r].reshape(r,r); k+=r*r
        Gs.append(L@L.conj().T)
    S=sum(Gs)
    w,V=eigh(herm(S)); Sih=(V/np.sqrt(w))@V.conj().T
    return [herm(Sih@G@Sih) for G in Gs]

def stats(P):
    A1,B1,A2,B2=P
    C1=A1+B1;C2=A2+B2
    r=A1.shape[0]
    gam=opn(C1-0.5*np.eye(r)); dl=max(opn(A1-A2),opn(B1-B2))
    if gam>=0.5-1e-9: return None
    c1h=fun_herm(C1,lambda w:1/np.sqrt(w)); c2h=fun_herm(C2,lambda w:1/np.sqrt(w))
    T1=herm(c1h@A1@c1h);T2=herm(c2h@A2@c2h)
    tau=opn(T1-T2)
    return tau,dl,gam

def ratio(x,r):
    P=four_from_params(x,r)
    s=stats(P)
    if s is None: return 0.0
    tau,dl,gam=s
    return tau/dl

best=0
for r in [1,2,3]:
    for rep in range(40):
        n=4*r*r*2
        x0=rng.standard_normal(n)
        res=minimize(lambda x:-ratio(x,r),x0,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-9,'fatol':1e-12})
        res=minimize(lambda x:-ratio(x,r),res.x,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-10,'fatol':1e-13})
        P=four_from_params(res.x,r)
        tau,dl,gam=stats(P)
        bound=2*np.sqrt(2)*dl/np.sqrt(1-4*gam**2)
        best=max(best,tau/bound)
        if rep%10==0: print(r,rep,'tau/delta',tau/dl,'delta',dl,'gamma',gam,'tau/bound',tau/bound)
print('max tau/bound',best)
