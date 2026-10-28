import mpmath as mp
import numpy as np, sys
sys.path.insert(0, '.')
from mplib import mp_herm, mp_fun, mp_reg_mean, to_np, tomp
from mpdil import block, cols, eye

def dilate(W, T, Uidx, tolk=mp.mpf(10)**-30):
    """W: N x r (mp), T: r x r Hermitian 0<=T<=1 (mp). U = span of eigenvectors (indices Uidx) of T.
    Returns dict with Ups (N x c orthonormal basis of ker W^*), Th ((r+c)x(r+c)), c, r, eigen-data."""
    N, r = W.rows, W.cols
    c = N - r
    E, Q = mp.eigh(mp_herm(T))
    Qf, _ = mp.qr(W, mode='full')
    Ups = cols(Qf, list(range(r, N)))
    Uv = cols(Q, Uidx); m = len(Uidx)
    Uperp_idx = [j for j in range(r) if j not in Uidx]
    TU = mp.diag([E[j] for j in Uidx])
    Rd = mp.diag([mp.sqrt(max(E[j] * (1 - E[j]), mp.mpf(0))) for j in Uidx])
    R0 = mp.zeros(c, m); k = 0
    for a, j in enumerate(Uidx):
        if E[j] * (1 - E[j]) > tolk:
            if k >= c: raise ValueError("rank R = %d > c = %d" % (k + 1, c))
            R0[k, a] = 1; k += 1
    M11 = mp.zeros(r, r)
    if Uperp_idx:
        Up = cols(Q, Uperp_idx)
        M11 += Up * mp.diag([E[j] for j in Uperp_idx]) * Up.H
    M11 += Uv * TU * Uv.H
    M12 = Uv * (Rd * R0.H)
    M22 = R0 * (eye(m) - TU) * R0.H
    Th = block([[M11, M12], [M12.H, M22]])
    return dict(W=W, T=T, Ups=Ups, Th=Th, c=c, r=r, N=N, E=E, Q=Q, Uidx=Uidx, Uv=Uv, k=k)

def pair(d, t):
    W, Ups, Th, r, c = d['W'], d['Ups'], d['Th'], d['r'], d['c']
    Wt = block([[W, mp.sqrt(t) * Ups]])
    I = eye(r + c)
    return mp_herm(Wt * Th * Wt.H), mp_herm(Wt * (I - Th) * Wt.H), Wt

def ptrace_Y_mp(J, dY, dX):
    M = mp.zeros(dX, dX)
    for a in range(dY):
        for i in range(dX):
            for j in range(dX):
                M[i, j] += J[a * dX + i, a * dX + j]
    return M

def lam_min(M):
    E = mp.eigh(mp_herm(M), eigvals_only=True)
    return min(E)
def lam_max(M):
    E = mp.eigh(mp_herm(M), eigvals_only=True)
    return max(E)
