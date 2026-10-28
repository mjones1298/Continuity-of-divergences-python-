import mpmath as mp, numpy as np
from lib import *
mp.mp.dps=60
def mp_logm(A): return mp_funm_h(A,lambda x: mp.log(x))
def relent(r,s): return mp.re(sum((r*(mp_logm(r)-mp_logm(s)))[i,i] for i in range(r.rows)))
for p in ['0.01','0.001']:
    pm=mp.mpf(p); de=mp.mpf('0.01')*pm
    J0=mp_choi_Np(p); J1=mp_choi_Np(p,de)
    D=relent(J0/2,J1/2)   # max-entangled input output = J/2
    print('p',p,'delta',mp.nstr(de,3),'exact D(max-ent)/δ² =',mp.nstr(D/de**2,10),' *p=',mp.nstr(D/de**2*pm,6))
