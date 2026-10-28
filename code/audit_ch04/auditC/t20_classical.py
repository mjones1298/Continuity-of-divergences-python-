"""F1: classical formula printed after Thm 7.5 (source lines 1272-1274) double counts f(0) Q_s(S).
Direct definition: D_f(P||Q) = sum_{q>0} q f(p/q) + f'(inf) P(q=0) + f(0) Q(p=0)  [Q(p=0) is already inside the first sum].
Printed:  int f(dP_ac/dQ) dQ + f'(inf) P_s(S) + f(0) Q_s(S).
Corrected: int f(dP_ac/dQ_ac) dQ_ac + f'(inf) P_s(S) + f(0) Q_s(S)   (P_ac, Q_ac mutually a.c.)."""
import numpy as np
rng=np.random.default_rng(1)
f=lambda x: 2.0*(x==0)*1.0+ (x>0)*(x*np.log(np.maximum(x,1e-300))-x+1+2.0*np.exp(-x)*0)  # f(0)=2 (jump placeholder, see below)
# use a genuinely convex f with f(0+)=2, f'(inf)=3, f(1)=0:  f(t)=2(1-t)+ 3 t -... build: f(t)=2*(1-t)^+ part is convex piecewise linear
def f(x): return np.maximum(2.0*(1-x),0)+3.0*np.maximum(x-1,0)*0+ (x>1)*0  # f(0)=2, f(1)=0, f(x)=0 beyond 1 -> f'(inf)=0
# choose instead f(x)=2/(1+x)+ (3-0)*... keep simple: f(x)=2*max(1-x,0)+3*max(x-1,0): f(0)=2, f'(inf)=3, f(1)=0 convex
def f(x): return 2.0*np.maximum(1-x,0)+3.0*np.maximum(x-1,0)
f0,finf=2.0,3.0
worst=0
for trial in range(2000):
    n=6
    p=rng.random(n)*(rng.random(n)<0.7); q=rng.random(n)*(rng.random(n)<0.7)
    if p.sum()==0 or q.sum()==0: continue
    p/=p.sum(); q/=q.sum()
    pos=q>0
    true=np.sum(q[pos]*f(p[pos]/q[pos]))+finf*p[~pos].sum()
    Ps=p[~pos].sum(); Qs=q[p==0].sum()
    printed=np.sum(q[pos]*f(p[pos]/q[pos]))+finf*Ps+f0*Qs
    both=(p>0)&(q>0)
    corrected=np.sum(q[both]*f(p[both]/q[both]))+finf*Ps+f0*Qs
    assert abs(true-corrected)<1e-12
    worst=max(worst,printed-true)
print('max (printed - true) over random pairs =',worst,'  (zero only if Q_s=0)')
P=np.array([1.,0.]);Q=np.array([0.,1.])
pos=Q>0
true=np.sum(Q[pos]*f(P[pos]/Q[pos]))+finf*P[~pos].sum()
printed=true+f0*Q[P==0].sum()
print('S={1,2}, P=(1,0), Q=(0,1): true',true,'printed',printed)
