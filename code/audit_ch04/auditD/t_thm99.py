import numpy as np, warnings
from schur import schur_norm
from lib import Mdd, kappa
warnings.filterwarnings("ignore")
rng=np.random.default_rng(11)
L=np.log
fam = {
 'Hellinger': (lambda t:1-2*np.sqrt(t*(1-t)), lambda t:-(1-2*t)/np.sqrt(t*(1-t)), lambda e:(1-2*e)/np.sqrt(e*(1-e))),
 'JS':        (lambda t:np.log(2)+t*L(t)+(1-t)*L(1-t), lambda t:L(t/(1-t)), lambda e:L((1-e)/e)),
 'Jeffreys':  (lambda t:(2*t-1)*L(t/(1-t)), lambda t:2*L(t/(1-t))+(2*t-1)/(t*(1-t)), lambda e:2*L((1-e)/e)+(1-2*e)/(e*(1-e))),
 'symchi2':   (lambda t:1/(t*(1-t))-4, lambda t:(2*t-1)/(t*(1-t))**2, lambda e:e**-2-(1-e)**-2),
 'LeCam':     (lambda t:(2*t-1)**2, lambda t:4*(2*t-1), lambda e:4*(1-2*e)),
 'Pearson':   (lambda t:(2*t-1)**2/(1-t), lambda t:(2*t-1)*(3-2*t)/(1-t)**2, lambda e:e**-2-4),
 'Neyman':    (lambda t:(2*t-1)**2/t, lambda t:4-1/t**2, lambda e:e**-2-4),
}
for name,(h,hp,claim) in fam.items():
    for eps in [0.4,0.2,0.05,0.01]:
        lo,hi=eps,1-eps
        ends=schur_norm(Mdd(h,hp,np.array([lo,hi])))
        grid=max(schur_norm(Mdd(h,hp,np.linspace(lo,hi,n))) for n in (5,11,21,41))
        rnd=max(schur_norm(Mdd(h,hp,np.sort(np.concatenate([[lo,hi],rng.uniform(lo,hi,10)])))) for _ in range(6))
        print("%-10s eps=%.2f claim=%.6f endpoints=%.6f gridsup=%.6f rand=%.6f"%(name,eps,claim(eps),ends,grid,rnd))
