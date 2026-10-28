import sys; sys.path.insert(0,'.')
from mpdil import *
from mplib import *
import numpy as np
from lib import make_conns
mp.mp.dps = 120
rng = np.random.default_rng(21)
conns = make_conns()
mpf = {'geo': f_geo, 'harm': f_harm, 'log': f_log, 'alpha0.3': f_a(mp.mpf('0.3')), 'alpha0.8': f_a(mp.mpf('0.8')), 'mix': f_mix, 'skew': f_skew, 'arith': f_arith}

def phi_mp(name, x):
    # phi(t) = t f((1-t)/t) in mp
    if x == 0: return mp.mpf(conns[name].b)
    if x == 1: return mp.mpf(conns[name].a)
    return x * mpf[name]((1 - x) / x)

def run(N, r, spec, Uidx, ts, names, eta_exp=90):
    d = build(N, r, spec, Uidx, rng)
    W, T, Q = d['W'], d['T'], d['Q']
    P = d['P']
    JPhi = mp_herm(W * T * W.H); JPsi = mp_herm(W * (eye(r) - T) * W.H)
    for t in ts:
        JPhit, JPsit, Wt = pair(d, t)
        print(' t=', mp.nstr(t, 2))
        for name in names:
            cn = conns[name]
            L = mp_reg_mean(JPhit, JPsit, mpf[name], mp.mpf(10) ** (-eta_exp))
            # prediction: W (phi(T) - P eta(T) P) W^*, with eta = phi - ell; U spanned by Uidx eigenvectors
            phiT = Q * mp.diag([phi_mp(name, mp.mpf(s)) for s in spec]) * Q.H
            etaT = Q * mp.diag([phi_mp(name, mp.mpf(s)) - (mp.mpf(cn.a) * mp.mpf(s) + mp.mpf(cn.b) * (1 - mp.mpf(s))) for s in spec]) * Q.H
            pred = mp_herm(W * (phiT - P * etaT * P) * W.H)
            full = mp_herm(W * phiT * W.H)
            print('   %-9s |L-pred|=%s   |L-full phi|=%s   (jump |pred-full|=%s)' % (name, mp.nstr(mp.norm(L - pred), 3), mp.nstr(mp.norm(L - full), 3), mp.nstr(mp.norm(pred - full), 3)))

print("Case N=4, r=3, c=1, U = eigvec with eigenvalue 0.4 (rank 1 <= c)")
run(4, 3, ['0.2', '0.4', '0.85'], [1], [mp.mpf(10)**-8, mp.mpf(10)**-14], ['harm', 'alpha0.8', 'mix', 'skew', 'arith', 'geo'])
print("Case N=6, r=3, c=3, U = two eigvecs (rank 2 <= 3)")
run(6, 3, ['0.2', '0.4', '0.85'], [0, 1], [mp.mpf(10)**-8, mp.mpf(10)**-14], ['harm', 'alpha0.8', 'mix', 'skew', 'arith', 'geo'])
print("Case U contains eigenvalue 0 (rank R smaller than dim U): spec 0,0.5,1")
run(5, 3, ['0', '0.5', '1'], [0, 1, 2], [mp.mpf(10)**-8, mp.mpf(10)**-14], ['harm', 'alpha0.8', 'mix', 'skew', 'geo'])
