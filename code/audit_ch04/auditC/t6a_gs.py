import numpy as np
from schur import *
print('g_s(t)=t/(1+s)-t/(t+s) on [0,1]: Schur-norm of divided differences on a 24-point grid vs claimed Lip 1/(s(1+s))')
for s in (0.01, 0.1, 0.5, 1.0, 3.0, 10.0):
    g = lambda t,s=s: t/(1+s) - t/(t+s)
    dg = lambda t,s=s: 1/(1+s) - s/(t+s)**2
    print(s, lip_grid(g, dg, 0, 1, 24), 1/(s*(1+s)), flush=True)
