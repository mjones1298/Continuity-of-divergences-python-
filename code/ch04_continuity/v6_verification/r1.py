# R1: channel-valued discontinuity at (Phi,Phi) for every channel with singular Choi matrix.
import random, mlib as m, mpmath as mp
mp.mp.dps = 60
rng = random.Random(5)

def short(J, Bsub):
    """Anderson short of PSD J onto subspace spanned by columns of Bsub (orthonormal).
       J = W W^*; short = W Pi_N W^* with N = W^+ (span) i.e. {x : W x in span}."""
    W = m.frame(J); n = J.rows
    if Bsub.cols == 0: return m.zeros(n)
    P = Bsub*m.dag(Bsub)
    # N = ker((1-P) W)
    Mx = (m.eye(n)-P)*W
    E, Q = m.eigh(m.dag(Mx)*Mx)
    cols = [i for i, e in enumerate(E) if e < mp.mpf(10)**(-25)]
    if not cols: return m.zeros(n)
    Nb = mp.zeros(W.cols, len(cols))
    for c, i in enumerate(cols):
        for r in range(W.cols): Nb[r, c] = Q[r, i]
    return W*Nb*m.dag(Nb)*m.dag(W)

def orth_basis(vecs_mat, tol=mp.mpf(10)**(-25)):
    E, Q = m.eigh(vecs_mat*m.dag(vecs_mat))
    idx = [i for i, e in enumerate(E) if e > tol]
    B = mp.zeros(vecs_mat.rows, len(idx))
    for c, i in enumerate(idx):
        for r in range(vecs_mat.rows): B[r, c] = Q[r, i]
    return B

def RH_basis(J, H, dX):
    """orthonormal basis of R cap (H (x) 1)^{-1} R, R = range J."""
    W = m.frame(J); n = J.rows
    Rb = orth_basis(W)
    P = Rb*m.dag(Rb)
    HH = m.kron(H, m.eye(dX))
    Mx = (m.eye(n)-P)*HH*Rb          # x in C^{dim R}: Rb x in R_H iff Mx x = 0
    E, Q = m.eigh(m.dag(Mx)*Mx)
    cols = [i for i, e in enumerate(E) if e < mp.mpf(10)**(-25)]
    Nb = mp.zeros(Rb.cols, len(cols))
    for c, i in enumerate(cols):
        for r in range(Rb.cols): Nb[r, c] = Q[r, i]
    return Rb*Nb

def run(dX, dY, k, label):
    Ks = m.rand_channel_kraus(dX, dY, k, rng)
    J = m.choi_kraus(Ks, dX, dY)
    H = m.rand_herm(dY, rng)
    RH = RH_basis(J, H, dX)
    FN = m.lmin(m.ptrace_Y(short(J, RH), dY, dX))
    # sanity: Tr_Y J = 1
    tp = m.opnorm(m.ptrace_Y(J, dY, dX) - m.eye(dX))
    out = []
    for t in ['1e-2', '1e-4', '1e-6', '1e-8']:
        t = mp.mpf(t)
        U = m.kron(m.expm_herm(H, t), m.eye(dX))
        Jt = U*J*m.dag(U)
        out.append(m.Fhat(J, Jt, dY, dX))
    print(f'{label}: dX={dX} dY={dY} Kraus rank={k} (rank J={k} < {dX*dY}), TP err {mp.nstr(tp,3)}, dim R_H={RH.cols}')
    print('   Fhat(Phi, Ad_{U_t} Phi), t=1e-2,1e-4,1e-6,1e-8:', [mp.nstr(x, 12) for x in out])
    print('   bound F_N = lambda_min Tr_Y (short of J onto R_H) =', mp.nstr(FN, 12))
    return out[-1], FN

for (dX, dY, k) in [(2,2,1),(2,2,2),(2,2,3),(2,3,2),(2,3,4),(3,2,3),(3,2,5),(3,3,4),(2,4,7)]:
    run(dX, dY, k, 'random channel')

# Lemma SG check: exact block formula vs direct mean for one instance at finite t
dX, dY, k = 2, 3, 3
Ks = m.rand_channel_kraus(dX, dY, k, rng); J = m.choi_kraus(Ks, dX, dY)
H = m.rand_herm(dY, rng); t = mp.mpf('0.3')
U = m.kron(m.expm_herm(H, t), m.eye(dX)); Jt = U*J*m.dag(U)
G1 = m.gmean(J, Jt)
# formula: frame W of Omega=2J, T=1/2, S=2^{-1/2}; A_t=W^+ U W S, B_t = Ups^* U W S
W = m.frame(2*J); Wp = m.pinv_frame(W); n = J.rows
Rb = orth_basis(W); P = Rb*m.dag(Rb)
Ups = orth_basis(m.eye(n)-P)
S = 1/mp.sqrt(2)
A = Wp*U*W*S; B = m.dag(Ups)*U*W*S
E, Q = m.eigh(m.dag(B)*B)
cols = [i for i, e in enumerate(E) if e < mp.mpf(10)**(-25)]
Pi = m.zeros(W.cols)
for i in cols: Pi += Q[:, i]*m.dag(Q[:, i])
G2 = W*m.gmean(m.eye(W.cols)/2, A*Pi*m.dag(A))*m.dag(W)
print('Lemma SG: || J#J_t - W[T#(A Pi A^*)]W^* || =', mp.nstr(m.opnorm(G1-G2), 5), ' dim ker B_t =', len(cols))
