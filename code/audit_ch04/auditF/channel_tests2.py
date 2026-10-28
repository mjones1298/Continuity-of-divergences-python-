import sys; sys.path.insert(0,'.')
import mpmath as mp
from mpdil2 import *
from mplib import f_geo, mp_reg_mean
mp.mp.dps = 100

def kraus_W(Ks):
    # columns vec(K) with vec(K) = sum_j K e_j (x) e_j, K is dY x dX (mp matrices)
    cols_ = []
    for K in Ks:
        dY, dX = K.rows, K.cols
        v = mp.zeros(dY * dX, 1)
        for a in range(dY):
            for j in range(dX):
                v[a * dX + j] = K[a, j]
        cols_.append(v)
    W = mp.zeros(cols_[0].rows, len(cols_))
    for k, v in enumerate(cols_):
        for i in range(v.rows): W[i, k] = v[i]
    return W

def Fhat_mp(JP, JQ, dY, dX, eta):
    M = mp_reg_mean(JP, JQ, f_geo, eta)
    return lam_min(ptrace_Y_mp(M, dY, dX))

def report(name, Ks, Tsp, Ulist, ts, dY=2, dX=2):
    W = kraus_W(Ks)
    r = W.cols
    d = None
    JP0 = mp_herm(W * Tsp * W.H); JQ0 = mp_herm(W * (eye(r) - Tsp) * W.H)
    F0 = Fhat_mp(JP0, JQ0, dY, dX, mp.mpf(10)**-70)
    print("==", name, " r=", r, " c=", W.rows - r, "  Fhat(Phi,Psi)=", mp.nstr(F0, 8))
    for Uidx in Ulist:
        try:
            d = dilate(W, Tsp, Uidx)
        except ValueError as e:
            print("  U=", Uidx, "skip:", e); continue
        # prediction from theorem (b): lam_min Tr_Y(W (g(T) - P g(T) P) W^*)
        E, Q = d['E'], d['Q']
        g = lambda x: mp.sqrt(max(x * (1 - x), mp.mpf(0)))
        gT = Q * mp.diag([g(E[j]) for j in range(r)]) * Q.H
        P = d['Uv'] * d['Uv'].H
        pred = lam_min(ptrace_Y_mp(W * (gT - P * gT * P) * W.H, dY, dX))
        for t in ts:
            JPt, JQt, Wt = pair(d, t)
            F = Fhat_mp(JPt, JQt, dY, dX, mp.mpf(10)**-80)
            print("  U idx=%s k=%d  t=%s  Fhat(Phi_t,Psi_t)=%s   predicted limit=%s   |J(Phi_t)-J(Phi)|_F/sqrt t=%s" % (Uidx, d['k'], mp.nstr(t, 2), mp.nstr(F, 8), mp.nstr(pred, 8), mp.nstr(mp.norm(JPt - JP0) / mp.sqrt(t), 4)))

mpm = mp.mpf
I2 = mp.matrix([[1, 0], [0, 1]]); X = mp.matrix([[0, 1], [1, 0]]); Yp = mp.matrix([[0, -1j], [1j, 0]]); Zp = mp.matrix([[1, 0], [0, -1]])
# amplitude damping gamma=0.3, rank 2 of 4: T = 1/2 * 1_2
g = mpm('0.3')
K1 = mp.matrix([[1, 0], [0, mp.sqrt(1 - g)]]); K2 = mp.matrix([[0, mp.sqrt(g)], [0, 0]])
Th2 = mp.eye(2) / 2
report("amplitude damping, pair (Phi,Phi)", [K1, K2], Th2, [[0, 1], [0]], [mpm(10)**-6, mpm(10)**-10])

# Pauli channel with p3=0 (rank 3): p=(1/2,1/4,1/4,0)
p = [mpm('0.5'), mpm('0.25'), mpm('0.25')]
Ks = [mp.sqrt(p[0]) * I2, mp.sqrt(p[1]) * X, mp.sqrt(p[2]) * Yp]
T3 = mp.eye(3) / 2
report("Pauli rank 3 (c=1), pair (Phi,Phi)", Ks, T3, [[0], [1], [0, 1, 2]], [mpm(10)**-8, mpm(10)**-12])

# full-rank Pauli: continuity point (c=0): nothing to dilate.
