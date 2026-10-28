import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
rng = np.random.default_rng(8)
conns = make_conns()
def rand_psd(n, rank):
    G = rng.normal(size=(n, rank)) + 1j*rng.normal(size=(n, rank)); return G@G.conj().T
print("(a) a A + b B <= A sigma B  (min eigenvalue of difference, relative):")
for name, cn in conns.items():
    worst = 0
    for trial in range(300):
        n = rng.integers(2, 6)
        A = rand_psd(n, rng.integers(1, n+1)); B = rand_psd(n, rng.integers(1, n+1))
        L = ka_frame_conn(A, B, cn, tol=1e-12)
        D = L - cn.a*A - cn.b*B
        worst = min(worst, np.linalg.eigvalsh(herm(D))[0]/max(1, np.linalg.norm(L, 2)))
    print("   %-9s %.2e" % (name, worst))
print("(b) upper bound L <= A sigma B for limit points of A_n sigma B_n; sequences A_n=A+eps G (G arbitrary PSD, random directions, incl. kernel):")
for name in ['geo', 'harm', 'mix', 'skew']:
    cn = conns[name]
    res = []
    for eps in [1e-2, 1e-4, 1e-6]:
        worst = 0; lowviol = 0
        for trial in range(200):
            n = 4
            A = rand_psd(n, rng.integers(1, 3)); B = rand_psd(n, rng.integers(1, 3))
            G1 = rand_psd(n, rng.integers(1, 4)); G2 = rand_psd(n, rng.integers(1, 4))
            An = A + eps*G1/np.linalg.norm(G1, 2); Bn = B + eps*G2/np.linalg.norm(G2, 2)
            Ln = ka_frame_conn(An, Bn, cn, tol=1e-13); L0 = ka_frame_conn(A, B, cn, tol=1e-13)
            worst = min(worst, np.linalg.eigvalsh(herm(L0 - Ln))[0])      # upper bound violation (L<=L0)
            lowviol = min(lowviol, np.linalg.eigvalsh(herm(Ln - cn.a*A - cn.b*B))[0])  # lower bound violation
        res.append((eps, worst, lowviol))
    print("   %-6s" % name, ["eps=%.0e upper-viol=%.1e lower-viol=%.1e" % r for r in res])
