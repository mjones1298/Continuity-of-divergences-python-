import numpy as np, warnings
from schur import schur_norm
from t_lem95 import schur2_formula
warnings.filterwarnings("ignore")
rng=np.random.default_rng(3)

def kappa(s):
    if np.isinf(s):
        k = lambda t: (2*t-1)**2/(1-t)
        kp = lambda t: (2*t-1)*(3-2*t)/(1-t)**2
    else:
        k = lambda t: (2*t-1)**2/(s*(1-t)+t)
        def kp(t):
            w = s*(1-t)+t
            return (2*t-1)*(2/w+(1+s)/w**2)
    return k,kp

def Mdd(g,gp,Lam):
    return np.array([[ (g(x)-g(y))/(x-y) if abs(x-y)>1e-12 else gp(x) for y in Lam] for x in Lam])

# check kappa' formula numerically
for s in [0,0.3,1,2,50,np.inf]:
    k,kp=kappa(s)
    for t in [0.2,0.5,0.9]:
        if s==0 and t==0: continue
        h=1e-6
        num=(k(t+h)-k(t-h))/(2*h)
        assert abs(num-kp(t))<1e-4*max(1,abs(kp(t))), (s,t,num,kp(t))
print("kappa' formula ok")

# (a): symmetric I -> Lip = max |kappa'| = bound ; general I -> Lip <= bound
def bound(s,I):
    lo,hi=I
    ts=np.linspace(lo,hi,2001)
    if np.isinf(s): vp=1-ts
    else: vp=s*(1-ts)+ts
    d=vp.min(); rho=np.max(abs(2*ts-1))
    if np.isinf(s): return rho*(2/d+1/d**2), d
    return rho*(2/d+(1+s)/d**2), d
viol=0
for trial in range(60):
    s = rng.choice([0.0, rng.uniform(0,1), 1.0, rng.uniform(1,5), rng.uniform(5,100), np.inf])
    lo = rng.uniform(0 if s!=0 else 0.01,0.9); hi = rng.uniform(lo+0.02,1.0)
    if rng.random()<0.4:  # symmetric
        e=rng.uniform(0.01 if True else 0,0.49); lo,hi=e,1-e
    if np.isinf(s) and hi>=1-1e-3: hi=1-0.02
    k,kp=kappa(s)
    B,d=bound(s,(lo,hi))
    best=0
    for j in range(5):
        n=rng.integers(3,8)
        Lam=np.sort(rng.uniform(lo,hi,n))
        if j==0: Lam=np.linspace(lo,hi,10)
        if j==1: Lam=np.concatenate([[lo,hi],rng.uniform(lo,hi,4)])
        best=max(best,schur_norm(Mdd(k,kp,Lam)))
    if best>B*(1+1e-6)+1e-8: viol+=1; print("(a) bound violated",s,lo,hi,best,B)
    if abs((lo+hi)/2-0.5)<1e-12:
        ts=np.linspace(lo,hi,4001); mx=max(abs(kp(ts)))
        if abs(mx-B)>1e-5*B: print("sym: max|k'| != bound",s,lo,hi,mx,B)
        # equality check: best vs B (best may be below B if Lambda grid doesn't include endpoints)
        e2=schur_norm(Mdd(k,kp,np.array([lo,hi])))
        if abs(e2-B)>1e-6*B: print("sym: endpoint Schur != bound",s,lo,hi,e2,B)
print("(a) violations:",viol)

# (b) Lip^{[0,1-eps]}(kappa_inf)
k,kp=kappa(np.inf)
for eps in [0.5,0.45,0.4,1/3,0.3,0.2,0.1,0.01]:
    lo,hi=0.0,1-eps
    ends=Mdd(k,kp,np.array([lo,hi]))
    # Lip estimated as max over grid Schur norms and endpoint
    gridv=max(schur_norm(Mdd(k,kp,np.linspace(lo,hi,n))) for n in (3,6,12))
    print("eps=%.4f endpoint Schur=%.6f grid sup=%.6f eps^-2-4=%.6f max|k'|=%.6f"%(eps,schur_norm(ends),gridv,eps**-2-4,max(abs(kp(lo)),abs(kp(hi)))))
