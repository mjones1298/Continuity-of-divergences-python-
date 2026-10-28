"""Referee check 3: asymptotic adaptive QFI F_inf of the depolarized qubit rotation (Zhou-Jiang).

F_SQL = 4 min_{h Hermitian, beta(h)=0} ||alpha(h)||, alpha = A^dag A, A = Kdot - i (h (x) 1) K,
beta = i K^dag A. Solved as an SDP: min t s.t. [[t 1_2, A^dag],[A, 1_8]] >= 0, K^dag A = 0.
Kraus: K0 = sqrt(1-3p/4) R, Kj = sqrt(p/4) sigma_j R, R = exp(-i delta Z/2), evaluated at delta = 0.
Compared with the claimed closed form 2(1-p)^2/(p(3-2p)).
"""
import numpy as np
import cvxpy as cp

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1]).astype(complex)

for p in [1e-1, 3e-2, 1e-2, 1e-3]:
    Ks = [np.sqrt(1 - 3 * p / 4) * I2] + [np.sqrt(p / 4) * S for S in (X, Y, Z)]
    K = np.vstack(Ks)                                   # 8x2
    Kd = np.vstack([k @ (-0.5j * Z) for k in Ks])       # derivative wrt delta
    # sanity: HNKS span check -> dimension of span{K_i^dag K_j} (real)
    span = np.array([np.concatenate([(a.conj().T @ b).real.ravel(), (a.conj().T @ b).imag.ravel()])
                     for a in Ks for b in Ks])
    h = cp.Variable((4, 4), hermitian=True)
    A = Kd - 1j * cp.kron(h, I2) @ K
    t = cp.Variable()
    M = cp.bmat([[t * np.eye(2), A.H], [A, np.eye(8)]])
    cons = [M >> 0, K.conj().T @ A == 0]
    prob = cp.Problem(cp.Minimize(t), cons)
    prob.solve(solver=cp.CLARABEL)
    F_sdp = 4 * t.value
    F_cf = 2 * (1 - p) ** 2 / (p * (3 - 2 * p))
    FR = 2 * (1 - p) ** 2 * (2 - p) / (p * (4 - 3 * p))
    print(f"p={p:g}: rank of Kraus-span={np.linalg.matrix_rank(span)} | F_inf SDP={F_sdp:.10g} closed={F_cf:.10g} "
          f"rel.diff={abs(F_sdp-F_cf)/F_cf:.1e} | p*F_inf={p*F_sdp:.6f} | F_RLD/F_inf={FR/F_sdp:.6f}")
