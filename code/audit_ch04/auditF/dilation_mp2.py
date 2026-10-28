import sys; sys.path.insert(0,'.')
from mpdil import *
from mplib import *
import numpy as np
from lib import make_conns

rng = np.random.default_rng(5)
N, r = 4, 2
spec = ['0.3', '0.7']
d = build(N, r, spec, [0, 1], rng)
Th = d['Th']
P2 = Th * Th - Th
print('projection defect', mp.nstr(mp.norm(P2), 3))
W, T = d['W'], d['T']
JPhi = mp_herm(W * T * W.H); JPsi = mp_herm(W * (eye(r) - T) * W.H)
mpf = {'geo': f_geo, 'harm': f_harm, 'log': f_log, 'alpha0.3': f_a(mp.mpf('0.3')), 'alpha0.8': f_a(mp.mpf('0.8')), 'mix': f_mix, 'skew': f_skew, 'arith': f_arith}
conns = make_conns()
for t in [mp.mpf(10)**-4, mp.mpf(10)**-8, mp.mpf(10)**-12]:
    JPhit, JPsit, Wt = pair(d, t)
    print('t=', mp.nstr(t, 3), ' |JPhit-JPhi|_F/sqrt t =', mp.nstr(mp.norm(JPhit - JPhi) / mp.sqrt(t), 4))
    for name in ['geo', 'harm', 'log', 'alpha0.3', 'alpha0.8', 'mix', 'skew', 'arith']:
        cn = conns[name]
        L = mp_reg_mean(JPhit, JPsit, mpf[name], mp.mpf(10) ** (-60))
        aff = cn.a * JPhi + cn.b * JPsi
        print('   %-9s |L - (aJPhi+bJPsi)|_F = %s' % (name, mp.nstr(mp.norm(L - aff), 4)))
