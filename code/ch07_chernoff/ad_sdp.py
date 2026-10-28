"""Exact optimal n-use adaptive (and parallel) discrimination error of AD_g vs AD_h
via SDP over quantum testers (Chiribella-D'Ariano-Perinotti / Gutoski-Watrous).
Spaces ordered X1 Y1 X2 Y2 ... Xn Yn (each qubit). All data real -> real symmetric tester WLOG."""
from pathlib import Path
import numpy as np, cvxpy as cp, sys, time, json
from math import comb

def kraus(g):
    return [np.array([[1,0],[0,np.sqrt(1-g)]]), np.array([[0,np.sqrt(g)],[0,0]])]

def choi_XY(Ks):
    J = np.zeros((4,4))
    for i in range(2):
        for j in range(2):
            Eij = np.zeros((2,2)); Eij[i,j] = 1
            J += np.kron(Eij, sum(K@Eij@K.T for K in Ks))
    return J

def kronn(J, n):
    out = np.array([[1.0]])
    for _ in range(n): out = np.kron(out, J)
    return out

def ptrace_last(X, dkeep, dlast):
    return cp.partial_trace(X, [dkeep, dlast], axis=1)

def tester_constraints(T0, T1, n, parallel=False):
    """returns constraints and the 'Xi' variables"""
    cons = [T0 >> 0, T1 >> 0]
    S = T0 + T1
    if parallel:
        # inputs X1..Xn prepared jointly: S = rho_{X1..Xn} (x) 1_{Y1..Yn} in interleaved order
        # build via permutation: easier to impose with explicit structure
        rho = cp.Variable((2**n, 2**n), symmetric=True)
        cons += [rho >> 0, cp.trace(rho) == 1]
        # embed: S[(x1 y1 .. xn yn),(x1' y1' ..)] = rho[(x..),(x'..)] * delta(y,y')
        dim = 4**n
        P = np.zeros((dim, dim))  # permutation from (X1..Xn,Y1..Yn) to interleaved
        for idx in range(dim):
            bits = [(idx >> (2*n-1-k)) & 1 for k in range(2*n)]  # X1..Xn Y1..Yn
            xs, ys = bits[:n], bits[n:]
            inter = []
            for k in range(n): inter += [xs[k], ys[k]]
            j = int(''.join(map(str, inter)), 2)
            P[j, idx] = 1
        cons += [S == P @ cp.kron(rho, np.eye(2**n)) @ P.T]
        return cons
    Xi_prev = None
    # Xi_k on X1 Y1 ... X_k  (dim 2^(2k-1))
    Xis = []
    for k in range(1, n+1):
        d = 2**(2*k-1)
        Xi = cp.Variable((d, d), symmetric=True)
        Xis.append(Xi)
    cons += [S == cp.kron(Xis[-1], np.eye(2))]
    cons += [cp.trace(Xis[0]) == 1]
    for k in range(2, n+1):
        dk = 2**(2*k-2)
        cons += [ptrace_last(Xis[k-1], dk, 2) == cp.kron(Xis[k-2], np.eye(2))]
    return cons

def opt_error(g, h, n, parallel=False, verbose=False):
    JP, JQ = kronn(choi_XY(kraus(g)), n), kronn(choi_XY(kraus(h)), n)
    d = 4**n
    T0 = cp.Variable((d, d), symmetric=True); T1 = cp.Variable((d, d), symmetric=True)
    cons = tester_constraints(T0, T1, n, parallel)
    # T0: guess Phi, T1: guess Psi.  error = 1/2 [Tr T1 JP + Tr T0 JQ]
    obj = 0.5*(cp.trace(T1@JP) + cp.trace(T0@JQ))
    pr = cp.Problem(cp.Minimize(obj), cons)
    pr.solve(solver='CLARABEL', verbose=verbose, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
    return pr.value, pr.status

def bern_err(g, h, n):
    return 0.5*sum(comb(n, k)*min(g**k*(1-g)**(n-k), h**k*(1-h)**(n-k)) for k in range(n+1))

def mu_star(g, h):
    return np.sqrt(g*h)/(1-np.sqrt((1-g)*(1-h)))

def pure_lb(g, h, n):
    m = mu_star(g, h)**(2*n)
    return 0.5*(1-np.sqrt(1-m))

if __name__ == '__main__':
    g = 0.5
    Ds = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0.1, 0.03, 0.01]
    ns = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [1, 2, 3]
    res = []
    for D in Ds:
        h = g-D
        for n in ns:
            t = time.time()
            pa, st = opt_error(g, h, n)
            pp, st2 = opt_error(g, h, n, parallel=True) if n > 1 else (pa, st)
            pb = bern_err(g, h, n); lb = pure_lb(g, h, n)
            row = dict(Delta=D, n=n, adapt=pa, parallel=pp, prod1=pb, purelb=lb, status=st+'/'+st2, sec=time.time()-t)
            res.append(row)
            print("D=%.3f n=%d adapt=%.10f parallel=%.10f |1>^n=%.10f  adapt-prod=%.2e  pureprog LB=%.10f  [%s %.1fs]" %
                  (D, n, pa, pp, pb, pa-pb, lb, row['status'], row['sec']), flush=True)
    json.dump(res, open(Path(__file__).resolve().parent / ('sdp_results_%s.json' % ('_'.join(map(str, Ds)))), 'w'), indent=1)
