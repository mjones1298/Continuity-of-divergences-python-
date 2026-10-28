import numpy as np
from schur import *
print('alpha<0: Lip^{[0.05,1]}(t^alpha) vs max|alpha t^(alpha-1)|')
for al in (-0.5,-1.0,-1.5,-2.0,-3.0):
    g=lambda t,al=al: t**al
    dg=lambda t,al=al: al*t**(al-1)
    lam=np.linspace(0.05,1,12)
    L=schur_norm(divdiff(g,dg,lam))
    print(al,L,abs(al)*0.05**(al-1),flush=True)
