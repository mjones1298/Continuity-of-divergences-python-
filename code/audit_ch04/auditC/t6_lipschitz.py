import numpy as np
from schur import *
# (i) g_s on [0,1]
print('--- g_s(t)=t/(1+s)-t/(t+s): Lip vs 1/(s(1+s))')
for s in (0.01, 0.1, 0.5, 1.0, 3.0, 10.0):
    g = lambda t,s=s: t/(1+s) - t/(t+s)
    dg = lambda t,s=s: 1/(1+s) - s/(t+s)**2
    L = lip_grid(g, dg, 0, 1, 40)
    print(s, L, 1/(s*(1+s)))
# (ii) kappa_inf on [0,1-eps]
print('--- kappa_inf on [0,1-eps]: Lip vs eps^-2-4 and max|k\'|')
for eps in (0.5,0.45,0.4,0.36,1/3,0.3,0.2,0.1,0.05):
    g = lambda t: (2*t-1)**2/(1-t)
    dg = lambda t: (2*t-1)*(3-2*t)/(1-t)**2
    L = lip_grid(g, dg, 0, 1-eps, 50)
    ts = np.linspace(0,1-eps,2001)
    print(eps, L, eps**-2-4, np.max(np.abs(dg(ts))))
