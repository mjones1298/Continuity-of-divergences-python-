import numpy as np
from schur import *
print('alpha, Lip^{[0,1]}(t^alpha) (grid lower bound), alpha (max deriv), 3-beta')
for al in (1.1,1.2,1.3,1.4,1.42,1.5,1.7,2.0,2.5):
    g=lambda t,al=al: t**al
    dg=lambda t,al=al: al*t**(al-1)
    best=0
    for n in (12,24):
        lam=np.linspace(0,1,n)
        best=max(best,schur_norm(divdiff(g,dg,lam)))
    print(al,best,al,3-(al-1),flush=True)
