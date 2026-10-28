# Is the exponent 1/2 of the coupling term kappa^{1/2} in Thm 11.3(b) sharp for g_#?  (dimX=1, Y=C^2, Q_1 weight w->0, c=1/2, tau_ac=0)
import numpy as np
from scipy.optimize import minimize
def gsharp(T):
    w,v=np.linalg.eigh(T); w=np.clip(w,0,1); return (v*np.sqrt(w*(1-w)))@v.conj().T
rng=np.random.default_rng(1)
def sup_ratio(kappa,trials=4000):
    best=0
    for _ in range(trials):
        # T = [[s,x],[x,t]] with |x| = kappa exactly, 0<=T<=1
        s,t=rng.random(2)
        if rng.random()<0.5: s=rng.random()*kappa*10
        x=kappa
        T=np.array([[s,x],[x,t]])
        ev=np.linalg.eigvalsh(T)
        if ev[0]<0 or ev[1]>1: continue
        g=gsharp(T)
        diff=abs(g[1,1]-np.sqrt(t*(1-t)))      # c=1/2 : F1-F2 = w g11 + 1/2 (g22 - g#(t));  w->0
        best=max(best,0.5*diff)
    return best
for k in [1e-1,1e-2,1e-3,1e-4]:
    b=sup_ratio(k)
    print(f"kappa={k:g}: sup |dF| (w->0) ~ {b:.3e}   ratio to kappa={b/k:.3f}   ratio to sqrt(kappa)={b/np.sqrt(k):.3f}")
