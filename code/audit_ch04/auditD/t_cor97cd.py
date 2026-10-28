import numpy as np, warnings
from schur import schur_norm
from t_cor97 import Mdd
warnings.filterwarnings("ignore")
rng=np.random.default_rng(5)
print("--- (c) eta_s")
for s in [0.01,0.1,0.5,1,1.5,3,10,100]:
    eta=lambda t,s=s:(1+s)*t*(1-t)/(1+(s-1)*t)
    def etap(t,s=s):
        D=1+(s-1)*t
        return (1+s)*((1-2*t)*D-(t-t*t)*(s-1))/D**2
    claim=(1+s)*max(1,1/s)
    ends=schur_norm(Mdd(eta,etap,np.array([0.,1.])))
    grid=max(schur_norm(Mdd(eta,etap,np.linspace(0,1,n))) for n in (4,9,17))
    rnd=max(schur_norm(Mdd(eta,etap,np.sort(np.concatenate([[0,1],rng.uniform(0,1,6)])))) for _ in range(10))
    mx=max(abs(etap(np.linspace(0,1,10001))))
    print("s=%g claim=%.6f endpoints=%.6f grid=%.6f rand=%.6f max|eta'|=%.6f"%(s,claim,ends,grid,rnd,mx))
print("--- (d) g_s")
for s in [0.01,0.1,0.5,1,3,10,100]:
    g=lambda t,s=s: t/(1+s)-t/(t+s)
    gp=lambda t,s=s: 1/(1+s)-s/(t+s)**2
    claim=1/(s*(1+s))
    ends=schur_norm(Mdd(g,gp,np.array([0.,1.])))
    grid=max(schur_norm(Mdd(g,gp,np.linspace(0,1,n))) for n in (4,9,17))
    ts=np.linspace(0,1,100001); vals=g(ts)
    print("s=%g claim=%.6f endpoints=%.6f grid=%.6f ; range of g_s [%.5f,%.5f] vs [-1/(1+s),0]=[%.5f,0]"%(s,claim,ends,grid,vals.min(),vals.max(),-1/(1+s)))
# (e)
bad=0
for _ in range(10000):
    s=rng.choice([0,rng.uniform(0,5),rng.uniform(5,500)]); lo=rng.uniform(0,0.9); hi=rng.uniform(lo,1)
    ts=np.linspace(lo,hi,50)
    d=(s*(1-ts)+ts).min(); rhs=(1+s)*np.minimum(ts,1-ts).min()
    if d<rhs-1e-12: bad+=1
print("(e) violations",bad)
