import numpy as np, warnings, sys
from schur import schur_norm
from lib import Mdd, kappa
warnings.filterwarnings("ignore")
rng=np.random.default_rng(21)
# test hypothesis of Prop 9.10: g = beta t + w1 kappa_s1 + w2 kappa_s2 ; Lip^I(g) == ||endpoint matrix||_S ?
def test(g,gp,lo,hi,ngrid=(5,9,15)):
    ends=schur_norm(Mdd(g,gp,np.array([lo,hi])))
    grid=max(schur_norm(Mdd(g,gp,np.linspace(lo,hi,n))) for n in ngrid)
    return ends,grid
worst=0
rows=[]
for trial in range(40):
    eps=rng.choice([0.3,0.1,0.02])
    lo,hi=(eps,1-eps) if rng.random()<0.5 else (rng.uniform(0.02,0.4), rng.uniform(0.5,0.98))
    s1=rng.choice([0.0,0.2,0.5,1.0,2.0,5.0,np.inf]); s2=rng.choice([0.0,0.2,0.5,1.0,2.0,5.0,np.inf])
    w1,w2=rng.uniform(0,3,2); beta=rng.normal()*rng.choice([0.1,1,5])
    k1,k1p=kappa(s1); k2,k2p=kappa(s2)
    g=lambda t:beta*t+w1*k1(t)+w2*k2(t); gp=lambda t:beta+w1*k1p(t)+w2*k2p(t)
    e,gr=test(g,gp,lo,hi)
    rel=(gr-e)/e
    rows.append((rel,s1,s2,w1,w2,beta,lo,hi,e,gr))
    if rel>1e-6: print("HYP FAILS? rel=%.3e s1=%s s2=%s w=(%.2f,%.2f) beta=%.2f I=[%.3f,%.3f] ends=%.6f grid=%.6f"%(rel,s1,s2,w1,w2,beta,lo,hi,e,gr)); sys.stdout.flush()
print("max rel excess", max(r[0] for r in rows))
