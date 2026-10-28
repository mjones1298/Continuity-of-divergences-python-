import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
rng = np.random.default_rng(14)
geo = make_conns()['geo']
def rand_psd(n, rank):
    G = rng.normal(size=(n, rank)) + 1j*rng.normal(size=(n, rank)); return G@G.conj().T
print("== Lemma A.3: A#B feasible (incl. singular) and dominates every feasible X (PD case) ==")
infeas = 0; viol = 0; cnt = 0
for trial in range(300):
    n = rng.integers(2, 6)
    A = rand_psd(n, rng.integers(1, n+1)); B = rand_psd(n, rng.integers(1, n+1))
    G = ka_frame_conn(A, B, geo, tol=1e-12)
    blk = np.block([[A, G], [G, B]])
    infeas = min(infeas, np.linalg.eigvalsh(herm(blk))[0]/max(1, np.linalg.norm(blk, 2)))
    if np.linalg.eigvalsh(A)[0] > 1e-3 and np.linalg.eigvalsh(B)[0] > 1e-3:
        Ah = mpow(A, .5); Aih = mpow(A, -.5); M = herm(Aih@B@Aih)
        Mih = mpow(M, -.5)
        for k in range(20):
            H = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)); H = herm(H)
            H = H/np.linalg.norm(H@Mih, 2)*rng.uniform(0.1, 1.0)
            X = herm(Ah@H@Ah)
            blkX = np.block([[A, X], [X, B]]); assert np.linalg.eigvalsh(herm(blkX))[0] > -1e-9*np.linalg.norm(blkX, 2)
            viol = min(viol, np.linalg.eigvalsh(herm(G - X))[0]/np.linalg.norm(G, 2)); cnt += 1
        # also X = A#B itself and X = -(A#B)
print("  min rel eig of block[[A,A#B],[A#B,B]] (>= -1e-12 ok):", infeas, "; min rel eig of (A#B - X) over %d feasible X:" % cnt, viol)

def sv(M): return np.linalg.svd(M, compute_uv=False)
def uis(M, n):
    s = sv(M); out = {}
    for k in range(1, n+1): out[('KF', k)] = s[:k].sum()
    for p in [1.5, 2, 3, 6]: out[('S', p)] = (s**p).sum()**(1/p)
    out[('S', np.inf)] = s.max()
    for p in [2, 3]:
        for k in range(1, n+1): out[('pk', p, k)] = (s[:k]**p).sum()**(1/p)
    return out
print("== Lemma A.6: block Cauchy-Schwarz in unitarily invariant norms ==")
worst = 0
for trial in range(2000):
    n = rng.integers(2, 7)
    A = rand_psd(n, rng.integers(1, n+1)); B = rand_psd(n, rng.integers(1, n+1))
    C = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)); C = C/np.linalg.norm(C, 2)*rng.uniform(0.2, 1)
    X = mpow(A, .5)@C@mpow(B, .5)
    nx, na, nb = uis(X, n), uis(A, n), uis(B, n)
    for key in nx: worst = max(worst, nx[key]**2/(na[key]*nb[key]))
    G = ka_frame_conn(A, B, geo); ng = uis(G, n)
    for key in ng: worst = max(worst, ng[key]**2/(na[key]*nb[key]))
print("  max ||X||^2/(||A|| ||B||) =", worst, "(<=1 expected)")
# tightness: A=B, X=A
A = rand_psd(5, 5); print("  tight case A=B=X ratio:", max(uis(A,5)[k]**2/uis(A,5)[k]**2 for k in uis(A,5)))
# Horn-Mathias CS itself for non-PSD: ||Y^*Z||^2 <= ||Y^*Y|| ||Z^*Z||
w2 = 0
for trial in range(2000):
    n = rng.integers(2, 6)
    Y = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)); Z = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    a, b, c = uis(Y.conj().T@Z, n), uis(Y.conj().T@Y, n), uis(Z.conj().T@Z, n)
    for key in a: w2 = max(w2, a[key]**2/(b[key]*c[key]))
print("  Horn-Mathias: max ||Y*Z||^2/(||Y*Y|| ||Z*Z||) =", w2)
