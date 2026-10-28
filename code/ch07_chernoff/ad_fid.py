"""Exact minimal root-fidelity of the final states over all n-use adaptive strategies:
  min_comb sqrt F(rho_Phi^(n), rho_Psi^(n)) = max_{||C||<=1} min_comb Re <<comb, J_C>>   (Uhlmann + minimax)
where J_C = sum C_{l,k} |K_k>><<L_l| (Kraus of Phi^{(x)n}, Psi^{(x)n}, interleaved X1Y1..XnYn) and the inner
min over deterministic combs is dualised:  max t  s.t. H_C >= Z_n (x) 1_{Xn},  Tr_{Y_{j}} Z_{j+1} >= Z_j (x) 1_{Xj}, Tr_{Y1} Z_2 >= t 1.
All data real -> real variables WLOG."""
from pathlib import Path
import numpy as np, cvxpy as cp, sys, time, json

def kraus(g):
    return [np.array([[1,0],[0,np.sqrt(1-g)]]), np.array([[0,np.sqrt(g)],[0,0]])]

def vecXY(K):   # |K>> = sum_x |x> (x) K|x>, ordering X(x)Y
    return np.concatenate([np.kron(np.eye(2)[x], K[:, x]) for x in range(2)]) if False else \
        sum(np.kron(np.eye(2)[x], K@np.eye(2)[x]) for x in range(2))

def kraus_mat(Ks, n):
    cols1 = [vecXY(K) for K in Ks]
    import itertools
    cols = []
    for idx in itertools.product(range(len(Ks)), repeat=n):
        v = np.array([1.0])
        for i in idx: v = np.kron(v, cols1[i])
        cols.append(v)
    return np.array(cols).T     # (4^n, r^n)

def min_root_fid(g, h, n, verbose=False):
    KM, LM = kraus_mat(kraus(g), n), kraus_mat(kraus(h), n)
    D = 4**n; r = KM.shape[1]
    # partial trace over Y_n (last factor): rows indexed (rest, y)
    Ky = [KM.reshape(D//2, 2, r)[:, y, :] for y in range(2)]
    Ly = [LM.reshape(D//2, 2, r)[:, y, :] for y in range(2)]
    C = cp.Variable((r, r))                       # C_{l,k}
    Hraw = sum(Ky[y] @ C.T @ Ly[y].T for y in range(2))   # Tr_{Yn} J_C  on X1Y1..Xn
    H = (Hraw + Hraw.T)/2
    t = cp.Variable()
    cons = [cp.sigma_max(C) <= 1]
    # Z_j on X1Y1..X_{j-1}Y_{j-1}, dim 4^(j-1); j = n..2 ; constraint chain
    Zs = {j: cp.Variable((4**(j-1), 4**(j-1)), symmetric=True) for j in range(2, n+1)}
    if n == 1:
        cons += [H - t*np.eye(2) >> 0]
    else:
        cons += [H - cp.kron(Zs[n], np.eye(2)) >> 0]
        for j in range(n, 2, -1):
            TrZ = cp.partial_trace(Zs[j], [4**(j-1)//2, 2], axis=1)   # trace Y_{j-1}
            cons += [TrZ - cp.kron(Zs[j-1], np.eye(2)) >> 0]
        TrZ2 = cp.partial_trace(Zs[2], [2, 2], axis=1)
        cons += [TrZ2 - t*np.eye(2) >> 0]
    pr = cp.Problem(cp.Maximize(t), cons)
    pr.solve(solver='CLARABEL', verbose=verbose)
    return pr.value, pr.status, C.value

if __name__ == '__main__':
    g = 0.5
    Ds = [float(x) for x in sys.argv[1].split(',')]
    ns = [int(x) for x in sys.argv[2].split(',')]
    out = []
    for D in Ds:
        h = g-D
        f = np.sqrt(g*h)+np.sqrt((1-g)*(1-h))
        mu = np.sqrt(g*h)/(1-np.sqrt((1-g)*(1-h)))
        for n in ns:
            t0 = time.time()
            v, st, C = min_root_fid(g, h, n)
            print("D=%.3f n=%d  min sqrtF=%.12f  f^n=%.12f  mu*^n=%.12f   -ln(minF)/n=%.6e  -ln f^2=%.6e  -2ln mu*=%.6e  [%s %.1fs]" %
                  (D, n, v, f**n, mu**n, -2*np.log(v)/n, -2*np.log(f), -2*np.log(mu), st, time.time()-t0), flush=True)
            out.append(dict(Delta=D, n=n, minrootF=v, f=f, mu=mu, status=st))
    json.dump(out, open(Path(__file__).resolve().parent / ('fid_results_%s_%s.json' % (sys.argv[1], sys.argv[2])), 'w'), indent=1)
