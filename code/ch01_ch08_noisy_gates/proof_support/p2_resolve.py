# Resolution of the "p*sup = 5.34" entry (p2_worst.out, p=1e-3, restart 0).
# High-precision (mpmath, dps=80) re-evaluation of the P2 objective at the saved point p2_restart0_x.npy.
# Route: exact-subspace / factor route. J(Phi)=W W^* with W the 4x4 matrix of vec'd Kraus operators,
#   B#A = B^{1/2} (Y Y^*)^{1/2} B^{1/2},  Y = B^{-1/2} W,  (Y Y^*)^{1/2} = U diag(s) U^*  (SVD of Y),
# so no square root of a round-off eigenvalue is ever taken and no threshold is needed.
# The derivative is taken on the 2x2 MATRIX T(e)=Tr_Y(B_e # A) (real-analytic in e since B>0) by central
# differences at several steps h, then reduced to lambda_min (first-order perturbation; degeneracy checked).
import numpy as np, mpmath as mp, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # glib.py in code/common
from glib import choi
mp.mp.dps = 80
exec(open(Path(__file__).resolve().parent / 'p2_common.py').read())   # JDp, trYcp, grad_op (double, thresholded), best_dir

def phi_kraus(x):
    Gm = (x[:16]+1j*x[16:]).reshape(8, 2); Q, _ = np.linalg.qr(Gm); return [Q[2*k:2*k+2] for k in range(4)]
def tomp(M): return mp.matrix([[mp.mpc(complex(M[i, j])) for j in range(M.shape[1])] for i in range(M.shape[0])])
def mherm(A): return (A+A.H)/2
def fnm(A, f):
    w, U = mp.eighe(mherm(A)); return U*mp.diag([f(x) for x in w])*U.H
def Wmat(Ks):            # columns vec(K_k), row-major vec as in glib
    W = mp.matrix(4, len(Ks))
    for k, K in enumerate(Ks):
        for i in range(2):
            for j in range(2): W[2*i+j, k] = mp.mpc(complex(K[i, j]))
    return W
def mtrY(M):
    T = mp.matrix(2, 2)
    for i in range(2):
        for a in range(2):
            for b in range(2): T[a, b] += M[i*2+a, i*2+b]
    return T
def Tmat(W, B):
    Bh = fnm(B, mp.sqrt); Bih = fnm(B, lambda x: 1/mp.sqrt(x))
    U, S, V = mp.svd_c(Bih*W)
    G = Bh*U*mp.diag(S)*U.H*Bh
    return mherm(mtrY(G))
def hbasis():
    n = 4; basis = []
    for i in range(n):
        for j in range(i, n):
            if i == j: E = np.zeros((n, n), complex); E[i, i] = 1; basis.append(E)
            else:
                E = np.zeros((n, n), complex); E[i, j] = E[j, i] = 1/np.sqrt(2); basis.append(E)
                E = np.zeros((n, n), complex); E[i, j] = 1j/np.sqrt(2); E[j, i] = -1j/np.sqrt(2); basis.append(E)
    return basis
def dT_all(W, B, h):
    return [(Tmat(W, B+h*tomp(E))-Tmat(W, B-h*tomp(E)))/(2*h) for E in hbasis()]
def tonp(M): return np.array([[complex(M[i, j]) for j in range(M.cols)] for i in range(M.rows)])

def analyse(x, p, label, hs=('1e-12', '1e-16', '1e-20', '1e-25', '1e-30')):
    Ks = phi_kraus(x); W = Wmat(Ks); A = W*W.H; B = tomp(JDp(p))
    eA = mp.eighe(mherm(A))[0]
    T0 = Tmat(W, B); w, V = mp.eighe(T0); F0 = w[0]; gap = w[1]-w[0]
    print(f'{label}  (p={p:g}, dps={mp.mp.dps})')
    print('  Choi eigs of Phi (exact from saved Kraus): ' + ', '.join(mp.nstr(e, 6) for e in eA))
    print(f'  Fhat = {mp.nstr(F0, 15)},  Tr_Y eigen-gap = {mp.nstr(gap, 6)}')
    Gd = grad_op(choi(Ks), JDp(p), 1e-9); vd = p*best_dir(Gd)
    print(f'  double, thresholded FD (h=1e-9, the search objective): p*sup = {vd:.6f}')
    v = V[:, 0]; Gs = {}
    for hs_ in hs:
        dT = dT_all(W, B, mp.mpf(hs_))
        g = [mp.re((v.H*D*v)[0]) for D in dT]
        Gs[hs_] = sum(complex(gi)*E for gi, E in zip(g, hbasis()))
    ref = Gs[hs[-1]]
    for hs_ in hs:
        print(f'  mp FD h={hs_:>6}: p*sup = {p*best_dir(Gs[hs_]):.9f}   max|G_h - G_ref| = {np.abs(Gs[hs_]-ref).max():.2e}')
    # independent check: one-sided slopes of lambda_min along the SDP-optimal direction for the mp gradient
    import cvxpy as cp
    n = 4; E = cp.Variable((n, n), hermitian=True); Y0 = cp.Variable((n, n), hermitian=True); Y1 = cp.Variable((n, n), hermitian=True)
    t0 = cp.Variable(); t1 = cp.Variable()
    cons = [cp.bmat([[Y0, -E], [-E, Y1]]) >> 0, trYcp(Y0) << t0*np.eye(2), trYcp(Y1) << t1*np.eye(2), (t0+t1)/2 <= 1, trYcp(E) == 0]
    for name, G in (('mp-optimal', ref), ('double-optimal (the 5.34 direction)', Gd)):
        cp.Problem(cp.Maximize(cp.real(cp.trace(E@G))), cons).solve(solver='CLARABEL'); Em = tomp(herm(E.value))
        for hh in ('1e-10', '1e-20'):
            h = mp.mpf(hh)
            Fp = mp.eighe(Tmat(W, B+h*Em))[0][0]; Fm = mp.eighe(Tmat(W, B-h*Em))[0][0]
            print(f'  one-sided slopes along {name} E, h={hh}: p*dF/de(+) = {float((Fp-F0)/h)*p:.9f}, '
                  f'p*dF/de(-) = {float((F0-Fm)/h)*p:.9f}')
    return p*best_dir(ref)

if __name__ == '__main__':
    p = float(sys.argv[2]); x = np.load(sys.argv[1]); analyse(x, p, sys.argv[1])
