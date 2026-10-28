import sys; sys.path.insert(0,'.')
import mpmath as mp
from mpdil2 import *
from mplib import f_geo, mp_reg_mean
from mpdil import rand_c
import numpy as np
mp.mp.dps = 100
rng = np.random.default_rng(31)
dY, dX = 2, 2; N = 4
def run(r, spec, Uidx, ts):
    c = N - r
    W = rand_c(N, r, rng)
    Qr, _ = mp.qr(rand_c(r, r, rng))
    T = Qr * mp.diag([mp.mpf(s) for s in spec]) * Qr.H
    d = dilate(W, T, Uidx)
    E, Q = d['E'], d['Q']
    h = lambda x: 1 - 2 * mp.sqrt(max(x * (1 - x), mp.mpf(0)))
    hT = Q * mp.diag([h(E[j]) for j in range(r)]) * Q.H
    P = d['Uv'] * d['Uv'].H
    # predicted: lam_max Tr_Y(W (h(T) + P (ell - h)(T) P) W^*), ell_f = 1
    ellmh = Q * mp.diag([1 - h(E[j]) for j in range(r)]) * Q.H
    pred = lam_max(ptrace_Y_mp(W * (hT + P * ellmh * P) * W.H, dY, dX))
    D0 = lam_max(ptrace_Y_mp(W * hT * W.H, dY, dX))            # D_H(Phi||Psi)
    Ef = lam_max(ptrace_Y_mp(W * W.H, dY, dX))                   # E_f = lam_max Tr_Y(J(Omega))  (f(0+)=f'(inf)=1)
    print("r=%d spec=%s U=%s k=%d :  D_H(Phi||Psi)=%s   E_f=%s   predicted limit=%s" % (r, spec, Uidx, d['k'], mp.nstr(D0, 8), mp.nstr(Ef, 8), mp.nstr(pred, 8)))
    for t in ts:
        JPt, JQt, Wt = pair(d, t)
        G = mp_reg_mean(JPt, JQt, f_geo, mp.mpf(10)**-80)
        DH = lam_max(ptrace_Y_mp(JPt + JQt - 2 * G, dY, dX))
        print("    t=%s  D_H(Phi_t||Psi_t) [via independent mean formula] = %s" % (mp.nstr(t, 2), mp.nstr(DH, 8)))
run(2, ['0.3', '0.6'], [0, 1], [mp.mpf(10)**-8, mp.mpf(10)**-12])      # c=2, full rank-2 U -> E_f
run(3, ['0.3', '0.6', '0.5'], [0], [mp.mpf(10)**-8, mp.mpf(10)**-12])   # c=1, one-vector U
run(3, ['0.3', '0.6', '0.5'], [1], [mp.mpf(10)**-8, mp.mpf(10)**-12])
run(3, ['0', '0.6', '1'], [0, 1, 2], [mp.mpf(10)**-8, mp.mpf(10)**-12])  # T near-projection except 0.6
