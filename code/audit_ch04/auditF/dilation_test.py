import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
from scipy.linalg import null_space

def halmos(T, Uidx_vecs, c, tol=1e-12):
    """T: r x r Hermitian in [0,1]. U = span of given orthonormal eigenvectors (columns of Uvecs, each an eigvec of T).
    Returns That on C^r (+) C^c, using eq:halmos with R0 a partial isometry U->C^c. Ordering of space: C^r (+) C^c."""
    r = T.shape[0]
    Uv = Uidx_vecs                   # r x m, orthonormal, T-invariant
    m = Uv.shape[1]
    TU = Uv.conj().T @ T @ Uv        # m x m
    w, V = eigh(herm(TU))
    Rw = np.sqrt(np.clip(w * (1 - w), 0, None))
    R = (V * Rw) @ V.conj().T        # m x m
    k = int(np.sum(Rw > tol))
    assert k <= c, (k, c)
    # R0: partial isometry U->C^c with R0^* R0 = range proj of R: map eigvecs of R with nonzero eigenvalue to distinct basis vectors e_1..e_k
    R0 = np.zeros((c, m), complex)
    idx = [i for i in range(m) if Rw[i] > tol]
    for j, i in enumerate(idx):
        R0[j, :] = V[:, i].conj()
    # full T-hat on C^r (+) C^c:
    # coordinates: U^perp part stays T_{U^perp}; U-part and C^c form the projection Phat
    Phat = np.block([[TU, R @ R0.conj().T], [R0 @ R, R0 @ (np.eye(m) - TU) @ R0.conj().T]])
    # embed: basis of C^r (+) C^c = [Uperp, U, C^c]
    P = Uv @ Uv.conj().T
    Uperp = null_space(Uv.conj().T)  # r x (r-m)
    Tperp = Uperp.conj().T @ T @ Uperp if Uperp.shape[1] > 0 else np.zeros((0, 0))
    # assemble in standard coords
    Th = np.zeros((r + c, r + c), complex)
    Th[:r, :r] = Uperp @ Tperp @ Uperp.conj().T if Uperp.shape[1] > 0 else 0
    Th[:r, :r] += Uv @ Phat[:m, :m] @ Uv.conj().T
    Th[:r, r:] = Uv @ Phat[:m, m:]
    Th[r:, :r] = Phat[m:, :m] @ Uv.conj().T
    Th[r:, r:] = Phat[m:, m:]
    return herm(Th), Phat

def setup(N, r, rng, spec=None):
    W = rng.normal(size=(N, r)) + 1j * rng.normal(size=(N, r))
    Q, _ = np.linalg.qr(rng.normal(size=(r, r)) + 1j * rng.normal(size=(r, r)))
    if spec is None: spec = rng.uniform(0.05, 0.95, size=r)
    T = herm(Q @ np.diag(spec) @ Q.conj().T)
    return W, T, Q, spec

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    dY, dX = 2, 2; N = dY * dX
    conns = make_conns()
    for trial in range(3):
        r = 2; c = N - r
        W, T, Q, spec = setup(N, r, rng, spec=np.array([0.3, 0.7]))
        Ups = null_space(W.conj().T)  # N x c, kernel of J = ker W^*
        # U = all of C^r (rank R = 2 <= c = 2)
        Uv = Q  # eigenvectors of T
        JPhi = W @ T @ W.conj().T; JPsi = W @ (np.eye(r) - T) @ W.conj().T
        for t in [1e-2, 1e-4, 1e-6]:
            Wt = np.hstack([W, np.sqrt(t) * Ups])
            That, Phat = halmos(T, Uv, c)
            assert np.allclose(Phat @ Phat, Phat, atol=1e-10)
            assert np.linalg.eigvalsh(That).min() > -1e-10 and np.linalg.eigvalsh(That).max() < 1 + 1e-10
            JPhit = herm(Wt @ That @ Wt.conj().T); JPsit = herm(Wt @ (np.eye(r + c) - That) @ Wt.conj().T)
            print('t=%g  |JPhit-JPhi|=%.2e  ratio/sqrt t=%.3f  minEig(JPhit)=%.1e' % (t, np.linalg.norm(JPhit - JPhi, 2), np.linalg.norm(JPhit - JPhi, 2) / np.sqrt(t), np.linalg.eigvalsh(JPhit).min()))
            for name, cn in conns.items():
                L = ka_frame_conn(JPhit, JPsit, cn)
                # predicted limit
                pred = W @ (cn.phi_mat(T) if hasattr(cn,'phi_mat') else (Q @ np.diag(cn.phi(spec)) @ Q.conj().T)) @ W.conj().T
                P_eta = Q @ np.diag(cn.eta(spec)) @ Q.conj().T  # U = whole C^r
                pred2 = pred - W @ P_eta @ W.conj().T
                # in this case: U = C^r  => limit = W ell_sigma(T) W^* = a J(Phi) + b J(Psi)
                aff = cn.a * JPhi + cn.b * JPsi
                print('   %-9s |L-pred|=%.2e  |pred - (aJPhi+bJPsi)|=%.1e' % (name, np.linalg.norm(L - pred2, 2), np.linalg.norm(pred2 - aff, 2)))
        break
