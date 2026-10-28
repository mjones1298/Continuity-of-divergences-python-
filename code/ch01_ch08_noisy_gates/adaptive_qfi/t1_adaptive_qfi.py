"""T1: asymptotic adaptive (error-corrected) QFI of theta -> D_p o exp(-i theta Z/2) U  (Zhou & Jiang, PRX Quantum 2, 010343 (2021),
arXiv:2003.10559). Statements verified against the arXiv text (WebFetch, 27 Oct 2026):
  H = i K^dag Kdot,  S = span_H{K_i^dag K_j};  Thm 1: F_N = Theta(N^2) iff H not in S (HNKS), else Theta(N)  (also for sequential).
  Thm 2 (H in S):  F_SQL = lim F_N/N = 4 min_{h: beta=0} ||alpha||,
     alpha = (Kdot - i h K)^dag (Kdot - i h K),  beta = i K^dag (Kdot - i h K),  h Hermitian r x r, ||.|| operator norm.
U is an input-side unitary: it does not change any quantity, so U = 1 below.

Certification: (upper) evaluate ||alpha(h)|| in mpmath at a feasible h (beta(h)=0 exactly);
(lower) minimax: min_h max_rho tr(rho alpha) >= min_{h:beta=0} tr(rho0 alpha(h)) for any density rho0 -- an equality-constrained
convex quadratic, solved exactly in mpmath (KKT via SVD null-space).
"""
import json, sys
import numpy as np, cvxpy as cp, mpmath as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # common.py, sharp_alpha.py
from common import *

mp.mp.dps = 50
PS = ['1e-1', '3e-2', '1e-2', '3e-3', '1e-3', '1e-4']


def kraus_mp(p):
    p = mp.mpf(p)
    P = [mpmat(s) for s in PAULI]
    K = [mp.sqrt(1 - 3 * p / 4) * P[0]] + [mp.sqrt(p / 4) * P[i] for i in (1, 2, 3)]
    G = mp.mpc(0, -1) * mpmat(Z) / 2      # d/dtheta exp(-i theta Z/2) at 0
    Kd = [k * G for k in K]
    return K, Kd


def herm_basis(r):
    B = []
    for i in range(r):
        E = mp.zeros(r, r); E[i, i] = 1; B.append(E)
    for i in range(r):
        for j in range(i + 1, r):
            E = mp.zeros(r, r); E[i, j] = 1; E[j, i] = 1; B.append(E)
            E = mp.zeros(r, r); E[i, j] = -1j; E[j, i] = 1j; B.append(E)
    return B


def alpha_beta(K, Kd, h):
    r = len(K)
    A = []
    for i in range(r):
        S = Kd[i].copy()
        for j in range(r):
            S = S - 1j * h[i, j] * K[j]
        A.append(S)
    alpha = mp.zeros(2, 2); beta = mp.zeros(2, 2)
    for i in range(r):
        alpha += A[i].H * A[i]
        beta += 1j * K[i].H * A[i]
    return alpha, beta


def sdp_double(p):
    K, Kd = kraus_mp(p)
    Kn = [np.array(k.tolist(), dtype=complex) for k in K]
    Kdn = [np.array(k.tolist(), dtype=complex) for k in Kd]
    r = 4
    h = cp.Variable((r, r), hermitian=True); t = cp.Variable()
    A = [Kdn[i] - 1j * sum(h[i, j] * Kn[j] for j in range(r)) for i in range(r)]
    Ast = cp.vstack(A)
    beta = 1j * sum(Kn[i].conj().T @ Kdn[i] for i in range(r)) + sum(h[i, j] * Kn[i].conj().T @ Kn[j] for i in range(r) for j in range(r))
    M = cp.bmat([[t * np.eye(2), Ast.H], [Ast, np.eye(2 * r)]])
    cons = [M >> 0, beta == 0]
    prob = cp.Problem(cp.Minimize(t), cons)
    prob.solve(solver='CLARABEL')
    # dual variable of the LMI: top-left block ~ rho (optimal state)
    Dv = cons[0].dual_value
    rho = Dv[:2, :2]; rho = (rho + rho.conj().T) / 2; rho = rho / np.trace(rho).real
    return 4 * t.value, h.value, rho, prob.status


def project_feasible(K, Kd, h0):
    """Return h = h0 + correction (least-norm) with beta(h) = 0 exactly (mp)."""
    r = len(K); B = herm_basis(r)
    x0 = [mp.re(h0[i, i]) for i in range(r)]
    # coordinates of h0 in basis B
    coords = []
    for i in range(r):
        coords.append(mp.re(h0[i, i]))
    for i in range(r):
        for j in range(i + 1, r):
            coords.append(mp.re(h0[i, j])); coords.append(-mp.im(h0[i, j]))
    # linear map x -> beta(x) real 4-vector (beta Hermitian 2x2): beta = beta0 + sum x_k beta_k
    def bvec(b):
        return [mp.re(b[0, 0]), mp.re(b[1, 1]), mp.re(b[0, 1]), mp.im(b[0, 1])]
    _, b0 = alpha_beta(K, Kd, mp.zeros(r, r))
    b0v = bvec(b0)
    cols = []
    for E in B:
        _, bE = alpha_beta(K, Kd, E)
        cols.append([bE_ - b0_ for bE_, b0_ in zip(bvec(bE), b0v)])
    N = mp.matrix(4, len(B))
    for k in range(len(B)):
        for q in range(4):
            N[q, k] = cols[k][q]
    x = mp.matrix(coords)
    res = N * x + mp.matrix(b0v)
    # least-norm correction: dx = -N^T (N N^T)^{-1} res
    dx = -(N.T * mp.inverse(N * N.T) * res)
    x = x + dx
    h = mp.zeros(r, r)
    for k, E in enumerate(B):
        h += x[k] * E
    return h, N, b0v


def lower_bound_fixed_rho(K, Kd, rho):
    """min_{h: beta=0} tr(rho alpha(h)) exactly. tr(rho alpha) = sum_i ||A_i rho^{1/2}||_F^2, A_i affine in x."""
    r = len(K); B = herm_basis(r); n = len(B)
    sr = sqrtm(rho)

    def vecA(h):
        alpha_list = []
        for i in range(r):
            S = Kd[i].copy()
            for j in range(r):
                S = S - 1j * h[i, j] * K[j]
            S = S * sr
            for a in range(2):
                for b in range(2):
                    alpha_list += [mp.re(S[a, b]), mp.im(S[a, b])]
        return mp.matrix(alpha_list)
    v0 = vecA(mp.zeros(r, r))
    M = mp.matrix(len(v0), n)
    for k, E in enumerate(B):
        vk = vecA(E) - v0
        for q in range(len(v0)):
            M[q, k] = vk[q]
    _, N, b0v = project_feasible(K, Kd, mp.zeros(r, r))
    # KKT: minimize ||v0 + M x||^2 s.t. N x = -b0
    Q = 2 * M.T * M
    c = 2 * M.T * v0
    KKT = mp.zeros(n + 4, n + 4)
    rhs = mp.zeros(n + 4, 1)
    for i in range(n):
        for j in range(n):
            KKT[i, j] = Q[i, j]
        for q in range(4):
            KKT[i, n + q] = N[q, i]; KKT[n + q, i] = N[q, i]
        rhs[i] = -c[i]
    for q in range(4):
        rhs[n + q] = -b0v[q]
    # KKT may be singular (gauge directions with M x = 0); use least squares via SVD pseudo-inverse
    U_, S_, V_ = mp.svd_r(KKT)
    smax = max(S_)
    Sinv = [1 / s if s > smax * mp.mpf(10) ** (-35) else 0 for s in S_]
    sol = V_.T * mp.diag(Sinv) * U_.T * rhs
    x = mp.matrix([sol[i] for i in range(n)])
    val = mp.norm(v0 + M * x) ** 2
    feas = mp.norm(N * x + mp.matrix(b0v))
    h = mp.zeros(r, r)
    for k, E in enumerate(B):
        h += x[k] * E
    return val, feas, h


def F_RLD(p):
    """F_RLD = lambda_max Tr_Y[(dJ) J^{-1} (dJ)], dJ by exact derivative of the Choi of D_p o Rz(theta) at theta=0."""
    p = mp.mpf(p)
    J = choi_qubit(p, 0)
    # dJ = (1-p)(G|Om><Om| + |Om><Om|G^dag), G = (-iZ/2) (x) 1 on output
    Om = choi_qubit(0, 0)
    G = kron(mp.mpc(0, -1) * mpmat(Z) / 2, mp.eye(2))
    dJ = (1 - p) * (G * Om + Om * G.H)
    # finite-difference cross-check
    h = mp.mpf(10) ** (-20)
    dJfd = (choi_qubit(p, h) - choi_qubit(p, -h)) / (2 * h)
    err = mp.norm(dJ - dJfd)
    M = herm(dJ * mp.inverse(J) * dJ)
    ev = eigvals(ptrace_Y(M, 2, 2))
    return ev[-1], ev[0], err


def span_check(p):
    K, Kd = kraus_mp(p)
    Hm = 1j * mp.fsum([0])  # placeholder
    Hm = mp.zeros(2, 2)
    for i in range(4):
        Hm += 1j * K[i].H * Kd[i]
    # real span of K_i^dag K_j (Hermitian parts) as 4-vectors in Pauli coordinates
    vecs = []
    for i in range(4):
        for j in range(4):
            M = K[i].H * K[j]
            for Mh in ((M + M.H) / 2, (M - M.H) / (2j)):
                vecs.append([mp.re(tr(mpmat(s) * Mh)) / 2 for s in PAULI])
    A = mp.matrix(vecs)
    S = mp.svd_r(A, compute_uv=False)
    rank = sum(1 for s in S if s > mp.mpf(10) ** (-30))
    return Hm, rank


def closed_forms(p):
    p = mp.mpf(p)
    w = p / 2 + p * (1 - 3 * p / 4) / (1 - p / 2)        # Zhou-Jiang Sec VII.A with px=py=pz=p/4, total 3p/4
    FZJ = (1 - w) / w
    Finf_simpl = 2 * (1 - p) ** 2 / (p * (3 - 2 * p))
    FRLD = 2 * (2 - p) * (1 - p) ** 2 / (p * (4 - 3 * p))
    return FZJ, Finf_simpl, FRLD


out = {}
for ps in PS:
    p = mp.mpf(ps)
    K, Kd = kraus_mp(p)
    Hm, rank = span_check(p)
    Fsdp, hval, rho_d, status = sdp_double(float(p))
    h0 = mpmat(hval)
    hf, _, _ = project_feasible(K, Kd, h0)
    al, be = alpha_beta(K, Kd, hf)
    Fup = 4 * eigvals(al)[-1]
    # lower bound with rho = 1/2 and with the SDP dual rho
    lb_half, feas_half, hstar = lower_bound_fixed_rho(K, Kd, mp.eye(2) / 2)
    Flow = 4 * lb_half
    lb_dual, _, _ = lower_bound_fixed_rho(K, Kd, mpmat(rho_d))
    # polish upper bound: alpha at the KKT minimiser for rho=1/2
    al2, be2 = alpha_beta(K, Kd, hstar)
    Fup2 = 4 * eigvals(al2)[-1]
    Fup_best = min(Fup, Fup2)
    FZJ, Fsimpl, FRLDc = closed_forms(p)
    FR, FRmin, fderr = F_RLD(p)
    # single-use entanglement-assisted QFI F_1 = 4 min_h ||alpha|| (no beta constraint): via SDP (double) for reference
    r = 4
    Kn = [np.array(k.tolist(), dtype=complex) for k in K]; Kdn = [np.array(k.tolist(), dtype=complex) for k in Kd]
    hh = cp.Variable((r, r), hermitian=True); tt = cp.Variable()
    A = cp.vstack([Kdn[i] - 1j * sum(hh[i, j] * Kn[j] for j in range(r)) for i in range(r)])
    cp.Problem(cp.Minimize(tt), [cp.bmat([[tt * np.eye(2), A.H], [A, np.eye(2 * r)]]) >> 0]).solve(solver='CLARABEL')
    F1 = 4 * tt.value
    rec = dict(p=float(p), HNKS_rank_of_Kraus_span=rank, H=str([[mp.nstr(Hm[i, j], 8) for j in range(2)] for i in range(2)]),
               F_SQL_sdp_double=Fsdp, sdp_status=status,
               F_SQL_upper_mp=mp.nstr(Fup_best, 30), F_SQL_lower_mp=mp.nstr(Flow, 30),
               F_SQL_lower_dualrho=mp.nstr(4 * lb_dual, 20),
               cert_gap_rel=mp.nstr((Fup_best - Flow) / Flow, 5), beta_resid=mp.nstr(mp.norm(be2), 5),
               F_SQL_closed_ZJ=mp.nstr(FZJ, 30), F_SQL_closed_simplified=mp.nstr(Fsimpl, 30),
               Finf_times_p=mp.nstr(Flow * p, 12),
               F_RLD=mp.nstr(FR, 30), F_RLD_closed=mp.nstr(FRLDc, 30), F_RLD_times_p=mp.nstr(FR * p, 12),
               TrY_is_scalar=mp.nstr(FR - FRmin, 5), dJ_fd_err=mp.nstr(fderr, 5),
               ratio_Finf_FRLD=mp.nstr(Flow / FR, 15), ratio_closed=mp.nstr((4 - 3 * p) / ((3 - 2 * p) * (2 - p)), 15),
               F1_single_use_sdp=F1,
               zeta_local_lower_over_delta2=mp.nstr(Flow / 2, 15), zeta_local_lower_times_p=mp.nstr(Flow * p / 2, 12),
               DBS_local_over_delta2=mp.nstr(FR / 2, 15),
               DBS_over_zeta_upper_bound=mp.nstr(FR / Flow, 15))
    out[ps] = rec
    print(ps, {k: rec[k] for k in ['HNKS_rank_of_Kraus_span', 'F_SQL_sdp_double', 'F_SQL_upper_mp', 'F_SQL_lower_mp', 'cert_gap_rel',
                                   'F_SQL_closed_ZJ', 'Finf_times_p', 'F_RLD_times_p', 'ratio_Finf_FRLD', 'F1_single_use_sdp',
                                   'DBS_over_zeta_upper_bound', 'beta_resid', 'TrY_is_scalar', 'dJ_fd_err']}, flush=True)

json.dump(out, open(Path(__file__).resolve().parent / 'results' / 't1.json', 'w'), indent=1)
