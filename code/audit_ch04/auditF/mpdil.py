import mpmath as mp
import numpy as np, sys
sys.path.insert(0, '.')
mp.mp.dps = 90
from mplib import mp_herm, mp_fun, mp_reg_mean, to_np

def rand_c(n, m, rng):
    M = mp.matrix(n, m)
    for i in range(n):
        for j in range(m):
            M[i, j] = mp.mpc(rng.normal(), rng.normal())
    return M

def eye(n): return mp.eye(n)
def zeros(n, m): return mp.zeros(n, m)

def block(Ms):
    rows = [sum(M.rows for M in row[:1]) for row in Ms]
    R = sum(Ms[i][0].rows for i in range(len(Ms)))
    C = sum(M.cols for M in Ms[0])
    out = mp.zeros(R, C)
    ro = 0
    for row in Ms:
        co = 0
        for M in row:
            for i in range(M.rows):
                for j in range(M.cols):
                    out[ro + i, co + j] = M[i, j]
            co += M.cols
        ro += row[0].rows
    return out

def cols(M, idx):
    out = mp.zeros(M.rows, len(idx))
    for k, j in enumerate(idx):
        for i in range(M.rows): out[i, k] = M[i, j]
    return out

def build(N, r, spec, Uidx, rng):
    """returns dict with W, Q, T, Ups, That(r+c), and helper"""
    c = N - r
    W = rand_c(N, r, rng)
    Qr, _ = mp.qr(rand_c(r, r, rng))        # unitary
    D = mp.diag([mp.mpf(s) for s in spec])
    T = Qr * D * Qr.H
    # kernel of W^*
    Qf, _ = mp.qr(W, mode='full')
    Ups = cols(Qf, list(range(r, N)))
    # Halmos with U spanned by columns Uidx of Qr (eigenvectors of T)
    Uv = cols(Qr, Uidx); m = len(Uidx)
    Uperp_idx = [j for j in range(r) if j not in Uidx]
    Up = cols(Qr, Uperp_idx)
    TU = mp.diag([mp.mpf(spec[j]) for j in Uidx])         # in the eigenbasis -> diagonal
    Rd = mp.diag([mp.sqrt(mp.mpf(spec[j]) * (1 - mp.mpf(spec[j]))) for j in Uidx])
    R0 = mp.zeros(c, m)
    k = 0
    for a, j in enumerate(Uidx):
        s = mp.mpf(spec[j])
        if s * (1 - s) > mp.mpf(10) ** (-40):
            assert k < c, "rank R > c"
            R0[k, a] = 1; k += 1
    Phat = block([[TU, Rd * R0.H], [R0 * Rd, R0 * (eye(m) - TU) * R0.H]])
    # embed into C^r (+) C^c: basis ordering: [Up, Uv] on C^r (as Qr columns) and C^c
    Tperp = mp.diag([mp.mpf(spec[j]) for j in Uperp_idx]) if Uperp_idx else mp.zeros(0, 0)
    Th = mp.zeros(r + c, r + c)
    # C^r part:  Up Tperp Up^H + Uv TU Uv^H
    M11 = mp.zeros(r, r)
    if Uperp_idx: M11 += Up * Tperp * Up.H
    M11 += Uv * TU * Uv.H
    M12 = Uv * (Rd * R0.H)     # r x c
    M22 = R0 * (eye(m) - TU) * R0.H
    Th = block([[M11, M12], [M12.H, M22]])
    return dict(W=W, Q=Qr, T=T, Ups=Ups, Th=Th, c=c, r=r, N=N, spec=spec, Uidx=Uidx, Uv=Uv, Uperp=Up, P=Uv * Uv.H)

def pair(d, t):
    W, Ups, Th, r, c = d['W'], d['Ups'], d['Th'], d['r'], d['c']
    Wt = block([[W, mp.sqrt(t) * Ups]])
    I = eye(r + c)
    JPhi_t = mp_herm(Wt * Th * Wt.H)
    JPsi_t = mp_herm(Wt * (I - Th) * Wt.H)
    return JPhi_t, JPsi_t, Wt
