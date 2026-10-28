import numpy as np, warnings, sys
from schur import schur_norm
from lib import Mdd, kappa
warnings.filterwarnings("ignore")
rng=np.random.default_rng(1234)
worst=0
cases=[(0.0,np.inf),(0.0,1e3),(1e-3,np.inf),(0.5,3.0),(1e-2,1e2)]
for (s1,s2) in cases:
    for (w1,w2,beta) in [(1,1,0),(1,10,0),(10,1,0),(1,1,-3),(1,1,3),(1,0.1,-8),(1,3,5)]:
        for (lo,hi) in [(0.01,0.99),(0.02,0.9),(0.2,0.7)]:
            k1,k1p=kappa(s1); k2,k2p=kappa(s2)
            g=lambda t:beta*t+w1*k1(t)+w2*k2(t); gp=lambda t:beta+w1*k1p(t)+w2*k2p(t)
            e=schur_norm(Mdd(g,gp,np.array([lo,hi])))
            gr=max(schur_norm(Mdd(g,gp,np.linspace(lo,hi,n))) for n in (6,11))
            # also random perturbed Lambda
            gr=max(gr,max(schur_norm(Mdd(g,gp,np.sort(np.concatenate([[lo,hi],rng.uniform(lo,hi,7)])))) for _ in range(3)))
            rel=(gr-e)/e; worst=max(worst,rel)
            if rel>1e-6: print("EXCESS",s1,s2,w1,w2,beta,lo,hi,e,gr); sys.stdout.flush()
print("worst relative excess of grid over endpoint-norm:",worst)
