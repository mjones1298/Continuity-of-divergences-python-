import numpy as np, warnings
from schur import schur_norm
from lib import Mdd
warnings.filterwarnings("ignore")
rng=np.random.default_rng(4)
for al in [1.1,1.2,1.3,1.38,1.42,1.5,1.7,1.9,2.0]:
    g=lambda t:t**al; gp=lambda t:al*t**(al-1)
    best=0;arg=None
    for n in (3,4,6,9):
        for trial in range(4 if n>3 else 1):
            Lam=np.concatenate([[1.0],np.sort(rng.uniform(0,1,n-1))]) if trial>0 else np.linspace(0,1,n)
            v=schur_norm(Mdd(g,gp,Lam))
            if v>best: best=v
    print("alpha=%.2f  max Schur norm found=%.6f  (alpha=%.2f, 3-beta=%.2f)  excess=%.2e"%(al,best,al,3-(al-1),best-al))
