import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
from mplib import *
from dilation_test import halmos, setup
from scipy.linalg import null_space

rng = np.random.default_rng(11)
dY, dX = 2, 2; N = 4
r = 2; c = 2
W, T, Q, spec = setup(N, r, rng, spec=np.array([0.3, 0.7]))
Ups = null_space(W.conj().T)
JPhi = W @ T @ W.conj().T; JPsi = W @ (np.eye(r) - T) @ W.conj().T
conns = make_conns()
mpf = {'geo': f_geo, 'harm': f_harm, 'log': f_log, 'alpha0.3': f_a(mp.mpf('0.3')), 'alpha0.8': f_a(mp.mpf('0.8')), 'mix': f_mix, 'skew': f_skew, 'arith': f_arith}
# Option: U = C^r (whole), predicted limit = a JPhi + b JPsi
That, Phat = halmos(T, Q, c)
for t in [1e-3, 1e-6, 1e-9]:
    Wt = np.hstack([W, np.sqrt(t) * Ups])
    JPhit = herm(Wt @ That @ Wt.conj().T); JPsit = herm(Wt @ (np.eye(r + c) - That) @ Wt.conj().T)
    A = tomp(JPhit); B = tomp(JPsit)
    print('t=', t)
    for name in ['geo', 'harm', 'log', 'alpha0.3', 'alpha0.8', 'mix', 'skew', 'arith']:
        cn = conns[name]
        L = mp_reg_mean(A, B, mpf[name], mp.mpf(10) ** (-45))   # eta = 1e-45
        Ln = to_np(L)
        aff = cn.a * JPhi + cn.b * JPsi
        print('   %-9s |L - (a JPhi + b JPsi)| = %.3e' % (name, np.linalg.norm(Ln - aff, 2)))
