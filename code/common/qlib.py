"""Core numerics for ARN-derivative / geometric-channel-fidelity experiments.

Conventions (Watrous):
  * vec is row-major:  vec(|y><x|) = y (x) x, so vec(V) = V.reshape(-1).
  * J(Phi) = sum_ij Phi(E_ij) (x) E_ij  (output Y first, input X second)
           = sum_k vec(V_k) vec(V_k)^*  for Kraus operators V_k (dY x dX).
  * W = [vec(V_1) ... vec(V_r)] with linearly independent V_k obtained from the
    eigendecomposition of J(Omega): columns sqrt(lam_k) u_k, lam_k > tol.
  * ARN derivative of Psi <= Omega:  T = W^+ J(Psi) (W^+)^*,  W^+ = (W^*W)^{-1} W^*.
  * Geometric channel fidelity (unsquared):
        Fhat(Phi,Psi) = lambda_min( Tr_Y ( J(Phi) # J(Psi) ) ),
    computed three independent ways: (i) SDP, (ii) eps-regularised # with
    eps -> 0 extrapolation, (iii) ARN formula W g(T) W^*, g(t)=sqrt(t(1-t)).

Tolerances: TOL_RANK (relative eigenvalue cutoff, eigen-based frames) = 1e-12; the Fhat ARN
  route uses arn_pair (SVD of stacked Choi factors, singular-value cutoff 1e-10 relative), which
  resolves lambda_min^+(J(Omega)) down to ~1e-20.  A naive eigen-frame of J(Omega) (arn_frame)
  squares the conditioning and gave errors up to 1 for AD_gamma vs AD_{gamma-1e-6}
  (lambda_min^+ ~ 1e-13) -- recorded in E2.
"""
import numpy as np
import scipy.linalg as sla
import cvxpy as cp
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

TOL_RANK = 1e-12  # relative cutoff: eigenvalues <= TOL_RANK*max(1,lam_max) treated as 0

# ----------------------------------------------------------------------------- basics
I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def herm(A):
    return 0.5 * (A + A.conj().T)


def vec(V):
    return np.asarray(V).reshape(-1)


def unvec(w, dY, dX):
    return np.asarray(w).reshape(dY, dX)


def choi_from_kraus(Ks):
    J = 0
    for K in Ks:
        v = vec(K)
        J = J + np.outer(v, v.conj())
    return herm(np.asarray(J, dtype=complex))


def ptrace_Y(M, dY, dX):
    """Tr_Y of an operator on Y (x) X (Y first)."""
    return np.trace(M.reshape(dY, dX, dY, dX), axis1=0, axis2=2)


def ptrace_X(M, dY, dX):
    return np.trace(M.reshape(dY, dX, dY, dX), axis1=1, axis2=3)


def apply_choi(J, rho, dY, dX):
    """Phi(rho) = Tr_X[ J (1_Y (x) rho^T) ]."""
    return ptrace_X(J @ np.kron(np.eye(dY), rho.T), dY, dX)


def psd_fn(A, f, clip=True):
    w, U = np.linalg.eigh(herm(A))
    if clip:
        w = np.clip(w, 0, None)
    return herm((U * f(w)) @ U.conj().T)


def psd_sqrt(A):
    return psd_fn(A, np.sqrt)


def pinv_sqrt(A, tol=TOL_RANK):
    w, U = np.linalg.eigh(herm(A))
    cut = tol * max(1.0, np.max(np.abs(w)))
    d = np.array([1 / np.sqrt(x) if x > cut else 0.0 for x in w])
    return herm((U * d) @ U.conj().T)


def opnorm(A):
    return np.linalg.norm(A, 2)


def lmin_pos(A, tol=TOL_RANK):
    w = np.linalg.eigvalsh(herm(A))
    cut = tol * max(1.0, np.max(np.abs(w)))
    w = w[w > cut]
    return float(w.min()) if len(w) else 0.0


# ----------------------------------------------------------------------------- channels
def kraus_unitary(U):
    return [np.asarray(U, dtype=complex)]


def kraus_replacement(rho, dX):
    """Phi(X) = tr(X) rho, Kraus sqrt(l_i)|e_i><x_j|."""
    w, U = np.linalg.eigh(herm(rho))
    Ks = []
    for i, l in enumerate(w):
        if l > 1e-15:
            for j in range(dX):
                e = np.zeros(dX); e[j] = 1
                Ks.append(np.sqrt(l) * np.outer(U[:, i], e))
    return Ks


def choi_replacement(rho, dX):
    return herm(np.kron(rho, np.eye(dX)))


def choi_unitary(U):
    return choi_from_kraus([U])


def choi_depolarize_after(J, p, dY, dX):
    """J(D_p o Phi), D_p(rho)=(1-p)rho + p tr(rho) 1/dY, for TP Phi."""
    return herm((1 - p) * J + p * np.kron(np.eye(dY) / dY, ptrace_Y(J, dY, dX)))


def kraus_ampdamp(g):
    return [np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex),
            np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)]


def kraus_dephase(p):
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Z2]


def kraus_depol(p, d=2):
    """Depolarizing (1-p)rho + p 1/d via Weyl/Pauli Kraus (qubit)."""
    assert d == 2
    return [np.sqrt(1 - 3 * p / 4) * I2] + [np.sqrt(p / 4) * P for P in (X2, Y2, Z2)]


def kraus_from_choi(J, dY, dX, tol=TOL_RANK):
    W = arn_frame(J, tol)
    return [unvec(W[:, k], dY, dX) for k in range(W.shape[1])]


def random_kraus(rng, dY, dX, r):
    G = (rng.normal(size=(r * dY, dX)) + 1j * rng.normal(size=(r * dY, dX)))
    Q, _ = np.linalg.qr(G)  # isometry (r dY x dX)
    return [Q[k * dY:(k + 1) * dY, :] for k in range(r)]


def tp_renormalize(Ks):
    S = sum(K.conj().T @ K for K in Ks)
    Sm = np.linalg.inv(psd_sqrt(S))
    return [K @ Sm for K in Ks]


def random_unitary(rng, d):
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(G)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def tp_error(J, dY, dX):
    return opnorm(ptrace_Y(J, dY, dX) - np.eye(dX))


# ----------------------------------------------------------------------------- ARN machinery
def arn_frame(JOm, tol=TOL_RANK):
    """W = [sqrt(l_k) u_k] for eigenpairs of J(Omega) with l_k > tol*max(1,lmax)."""
    w, U = np.linalg.eigh(herm(JOm))
    cut = tol * max(1.0, w.max())
    idx = np.where(w > cut)[0]
    return U[:, idx] * np.sqrt(w[idx])


def arn_derivative(W, JPsi):
    Wp = np.linalg.pinv(W, rcond=1e-15)  # = (W^*W)^{-1}W^* for injective W
    return herm(Wp @ JPsi @ Wp.conj().T)


def gfun(t):
    t = np.clip(t, 0, 1)
    return np.sqrt(t * (1 - t))


def psd_factor(A, tol=1e-15):
    """R with A = R R^*, columns sqrt(l_k) u_k (relative eigenvalue cutoff tol)."""
    w, U = np.linalg.eigh(herm(A))
    idx = w > tol * max(1.0, w.max())
    return U[:, idx] * np.sqrt(w[idx])


def arn_pair(JPhi, JPsi, sv_tol=1e-10, RPhi=None, RPsi=None):
    """Numerically stable ARN data for the pair (Phi, Psi), Omega = Phi + Psi.
    Stack factors W0 = [R_Phi, R_Psi] (J(Phi)=R_Phi R_Phi^*, J(Psi)=R_Psi R_Psi^*); thin SVD
    W0 = U S V^*; keep singular values > sv_tol*s_max.  Then W = U_r S_r is a linearly
    independent Kraus frame of Omega (W W^* = J(Omega)) and the derivative of Phi is
    T = V_r^* P_Phi V_r (P_Phi = projector on the R_Phi columns) = W^+ J(Phi) (W^+)^*.
    Conditioning: singular values (~sqrt of eigenvalues of J(Omega)) are resolved, so the
    route is accurate down to lambda_min^+(J(Omega)) ~ (sv_tol*s_max)^2."""
    R1 = psd_factor(JPhi) if RPhi is None else RPhi
    R2 = psd_factor(JPsi) if RPsi is None else RPsi
    W0 = np.hstack([R1, R2])
    U, S, Vh = np.linalg.svd(W0, full_matrices=False)
    keep = S > sv_tol * S.max()
    Ur, Sr, Vr = U[:, keep], S[keep], Vh[keep, :].conj().T  # Vr: (a+b) x r
    W = Ur * Sr
    a = R1.shape[1]
    T = herm(Vr[:a, :].conj().T @ Vr[:a, :])
    return W, T


def arn_spec(JPhi, JPsi, sv_tol=1e-10):
    """Eigen-data of the ARN derivative with endpoint-accurate eigenvalues.
    With W0=[R_Phi,R_Psi]=U S V^*, V_r = [B; C] (isometry, B^*B + C^*C = 1),  T = B^*B,
    1-T = C^*C.  For eigenvectors u_k of T we return t_k=||B u_k||^2 and s_k=||C u_k||^2
    (each accurate to ~1e-32 absolute near 0), so endpoint-singular functions such as
    t^alpha (1-t)^(1-alpha) are evaluated without the ~1e-16^alpha floor of eigvalsh."""
    R1 = psd_factor(JPhi); R2 = psd_factor(JPsi)
    W0 = np.hstack([R1, R2])
    U, S, Vh = np.linalg.svd(W0, full_matrices=False)
    keep = S > sv_tol * S.max()
    W = U[:, keep] * S[keep]
    Vr = Vh[keep, :].conj().T
    a = R1.shape[1]
    B, C = Vr[:a, :], Vr[a:, :]
    T = herm(B.conj().T @ B)
    _, Ue = np.linalg.eigh(T)
    t = np.sum(np.abs(B @ Ue) ** 2, axis=0)
    sv = np.sum(np.abs(C @ Ue) ** 2, axis=0)
    nrm = t + sv
    return W, T, Ue, t / nrm, sv / nrm


def fhat_arn(JPhi, JPsi, dY, dX, return_extra=False):
    W, T, Ue, t, sv = arn_spec(JPhi, JPsi)
    gT = (Ue * np.sqrt(t * sv)) @ Ue.conj().T
    M = ptrace_Y(W @ gT @ W.conj().T, dY, dX)
    F = float(np.linalg.eigvalsh(herm(M)).min())
    if return_extra:
        G = W.conj().T @ W
        return F, dict(W=W, T=T, lmin_G=float(np.linalg.eigvalsh(G).min()),
                       spec_T=t, recon_err=float(opnorm(W @ T @ W.conj().T - JPhi)))
    return F


def mean_fun_arn(JPhi, JPsi, phi2):
    """J(Phi) sigma J(Psi) = W phi(T) W^*, T = derivative of Phi wrt Phi+Psi.
    phi2(t, s) receives t and s = 1-t separately (endpoint accuracy)."""
    W, T, Ue, t, sv = arn_spec(JPhi, JPsi)
    return herm(W @ ((Ue * phi2(t, sv)) @ Ue.conj().T) @ W.conj().T), W, T


def qhat_alpha_arn(JPhi, JPsi, alpha, dY, dX):
    """Geometric Renyi quasi-entropy Qhat_alpha(Phi||Psi) = lmin Tr_Y(J(Psi) #_alpha J(Phi)),
    J(Psi)#_alpha J(Phi) = W T^alpha (1-T)^(1-alpha) W^*  (T = derivative of Phi)."""
    M, _, _ = mean_fun_arn(JPhi, JPsi, lambda t, s: t ** alpha * s ** (1 - alpha))
    return float(np.linalg.eigvalsh(herm(ptrace_Y(M, dY, dX))).min())


def dbs_arn(JPhi, JPsi, dY, dX):
    """Belavkin-Staszewski maximal divergence (natural log): lmax Tr_Y(W h(T) W^*),
    h(t) = t log(t/(1-t)); +inf if T has eigenvalue 1 (Phi not << Psi)."""
    W, T, Ue, t, sv = arn_spec(JPhi, JPsi)
    if np.min(sv) < 1e-28:
        return np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(t > 1e-300, t * np.log(t / sv), 0.0)
    M = ptrace_Y(W @ ((Ue * h) @ Ue.conj().T) @ W.conj().T, dY, dX)
    return float(np.linalg.eigvalsh(herm(M)).max())


# ----------------------------------------------------------------------------- direct geometric mean
def gmean_pd(A, B, alpha=0.5):
    """A #_alpha B for positive definite A, B (eigh-based)."""
    wa, Ua = np.linalg.eigh(herm(A))
    Ah = (Ua * np.sqrt(wa)) @ Ua.conj().T
    Aih = (Ua / np.sqrt(wa)) @ Ua.conj().T
    Mid = herm(Aih @ B @ Aih)
    wm, Um = np.linalg.eigh(Mid)
    wm = np.clip(wm, 0, None)
    return herm(Ah @ ((Um * wm ** alpha) @ Um.conj().T) @ Ah)


def fhat_reg(JPhi, JPsi, dY, dX, eps_list=None, return_curve=False):
    """lambda_min Tr_Y((J1+eps)#(J2+eps)) on a grid and extrapolation eps->0 by least
    squares in the basis {1, eps^1/2, eps, eps^3/2}."""
    if eps_list is None:
        eps_list = np.logspace(-9, -4, 16)
    n = JPhi.shape[0]
    vals = []
    for e in eps_list:
        G = gmean_pd(JPhi + e * np.eye(n), JPsi + e * np.eye(n))
        vals.append(np.linalg.eigvalsh(herm(ptrace_Y(G, dY, dX))).min())
    vals = np.array(vals)
    s = np.sqrt(eps_list)
    Amat = np.vstack([np.ones_like(s), s, s ** 2, s ** 3]).T
    coef, *_ = np.linalg.lstsq(Amat, vals, rcond=None)
    if return_curve:
        return float(coef[0]), eps_list, vals
    return float(coef[0])


# ----------------------------------------------------------------------------- SDPs
_SDP_CACHE = {}


def _solve(prob, solver_order=("SCS", "CLARABEL")):
    last = None
    for s in solver_order:
        try:
            if s == "SCS":
                prob.solve(solver="SCS", eps_abs=1e-10, eps_rel=1e-10, max_iters=100000)
            else:
                prob.solve(solver="CLARABEL")
            if prob.status in ("optimal", "optimal_inaccurate"):
                return s, prob.status
            last = prob.status
        except Exception as ex:  # solver failure: try next
            last = repr(ex)
    return None, last


def _psd_factor(A, tol=1e-12):
    w, U = np.linalg.eigh(herm(A))
    idx = w > tol * max(1.0, w.max())
    return U[:, idx] * np.sqrt(w[idx])


def fhat_sdp(JPhi, JPsi, dY, dX, solvers=("SCS", "CLARABEL")):
    """Katariya-Wilde / Jones Rem A.5:  max t s.t. [[J1, X],[X, J2]] >= 0, Tr_Y X >= t 1, X = X^*.
    Implemented in the equivalent factored form: [[J1,X],[X^*,J2]] >= 0  <=>  X = R1 K R2^*
    with ||K||_inf <= 1, where J_i = R_i R_i^* (eigen-factors, relative cutoff 1e-12).
    This is numerically far better conditioned than the raw 2n x 2n block LMI when the
    J_i are singular (the raw form returned 'optimal_inaccurate' with errors ~1e-3)."""
    R1 = _psd_factor(JPhi)
    R2 = _psd_factor(JPsi)
    a, b = R1.shape[1], R2.shape[1]
    if a == 0 or b == 0:
        return 0.0, dict(solver="trivial", status="optimal")
    K = cp.Variable((a, b), complex=True)
    t = cp.Variable()
    X = R1 @ K @ R2.conj().T
    cons = [cp.bmat([[np.eye(a), K], [K.H, np.eye(b)]]) >> 0, X == X.H,
            cp.partial_trace((X + X.H) / 2, [dY, dX], axis=0) - t * np.eye(dX) >> 0]
    prob = cp.Problem(cp.Maximize(t), cons)
    s, st = _solve(prob, solvers)
    return (float(t.value) if s else np.nan), dict(solver=s, status=st)


def diamond_sdp(JDelta, dY, dX):
    """||Delta||_diamond for Hermitian-preserving Delta:
       max Re tr(J M) s.t. -1_Y(x)sigma <= M <= 1_Y(x)sigma, sigma >= 0, tr sigma = 1."""
    key = ("D", dY, dX)
    if key not in _SDP_CACHE:
        n = dY * dX
        Jp = cp.Parameter((n, n), hermitian=True)
        M = cp.Variable((n, n), hermitian=True)
        S = cp.Variable((dX, dX), hermitian=True)
        K = cp.kron(np.eye(dY), S)
        cons = [K - M >> 0, K + M >> 0, cp.real(cp.trace(S)) == 1]
        _SDP_CACHE[key] = (cp.Problem(cp.Maximize(cp.real(cp.trace(Jp @ M))), cons), Jp)
    prob, Jp = _SDP_CACHE[key]
    Jp.value = herm(JDelta)
    s, st = _solve(prob)
    return float(prob.value) if s else np.nan


def diamond_dual_sdp(JDelta, dY, dX):
    """Watrous dual for trace-annihilating Delta: 2 min ||Tr_Y Z||_inf, Z >= J, Z >= 0."""
    n = dY * dX
    Z = cp.Variable((n, n), hermitian=True)
    s = cp.Variable()
    cons = [Z >> 0, Z - herm(JDelta) >> 0, s * np.eye(dX) - cp.partial_trace(Z, [dY, dX], axis=0) >> 0]
    prob = cp.Problem(cp.Minimize(s), cons)
    _solve(prob)
    return 2 * float(s.value)


def hockey_sdp(JPhi, JPsi, gamma, dY, dX):
    """E_gamma(Phi||Psi) = sup_rho tr((Phi(x)id)psi - gamma (Psi(x)id)psi)_+
       = max tr(J(Phi - gamma Psi) M), 0 <= M <= 1_Y (x) sigma, tr sigma = 1."""
    key = ("E", dY, dX)
    if key not in _SDP_CACHE:
        n = dY * dX
        Jp = cp.Parameter((n, n), hermitian=True)
        M = cp.Variable((n, n), hermitian=True)
        S = cp.Variable((dX, dX), hermitian=True)
        cons = [M >> 0, cp.kron(np.eye(dY), S) - M >> 0, cp.real(cp.trace(S)) == 1]
        _SDP_CACHE[key] = (cp.Problem(cp.Maximize(cp.real(cp.trace(Jp @ M))), cons), Jp)
    prob, Jp = _SDP_CACHE[key]
    Jp.value = herm(JPhi - gamma * JPsi)
    s, st = _solve(prob)
    return float(prob.value) if s else np.nan


# ----------------------------------------------------------------------------- nu(Omega)
def nu_lower_sdp(W, dY, dX):
    """Convex relaxation:  nu^2 >= max_sigma lambda_min( W^* (1_Y (x) sigma) W ).
    (nu^2 = min_c max_rho c^* M(rho) c,  M(rho) = W^*(1 (x) rho^T) W; relax cc^* to a density.)"""
    r = W.shape[1]
    S = cp.Variable((dX, dX), hermitian=True)
    t = cp.Variable()
    M = W.conj().T @ cp.kron(np.eye(dY), S) @ W
    M = 0.5 * (M + M.H)
    prob = cp.Problem(cp.Maximize(t), [S >> 0, cp.real(cp.trace(S)) == 1, M - t * np.eye(r) >> 0])
    s, st = _solve(prob)
    return float(t.value) if s else np.nan


def nu_upper_opt(W, dY, dX, rng, n_restarts=60):
    """nu(Omega) = min_{|c|=1} || sum_k c_k V_k ||_inf by smoothed local optimisation with
    restarts (plus the eigenvector start of Lemma 8.2(2)). Returns nu^2 (an upper estimate,
    exact up to optimisation error; certified from below by nu_lower_sdp)."""
    from scipy.optimize import minimize
    r = W.shape[1]
    Vs = np.array([unvec(W[:, k], dY, dX) for k in range(r)])

    def M_of(x):
        c = x[:r] + 1j * x[r:]
        c = c / np.linalg.norm(c)
        return np.tensordot(c, Vs, axes=1), c

    def exact(x):
        M, _ = M_of(x)
        return np.linalg.norm(M, 2) ** 2

    def smooth(x, beta):
        z = x[:r] + 1j * x[r:]
        nz = np.linalg.norm(z)
        c = z / nz
        M = np.tensordot(c, Vs, axes=1)
        P = M.conj().T @ M
        w, U = np.linalg.eigh(herm(P))
        wm = w.max()
        e = np.exp(beta * (w - wm))
        f = wm + np.log(e.sum()) / beta
        # gradient of f wrt c (Wirtinger): df = tr(Q dP), Q = U diag(e/sum e) U^*
        Q = (U * (e / e.sum())) @ U.conj().T
        # dP = dM^* M + M^* dM ; dM = sum dc_k V_k ; d f/d conj(c_k) = tr(Q V_k^* M)... real grad:
        gk = np.array([np.trace(M @ Q @ V.conj().T) for V in Vs])  # = d f / d conj(c_k)
        gc = 2 * gk  # gradient wrt (Re c, Im c) packed as complex
        # chain rule through c = z/|z|: project out radial part
        gz = (gc - c * np.real(np.vdot(c, gc))) / nz
        return f, np.concatenate([gz.real, gz.imag])

    starts = []
    # eigenvector of smallest eigenvalue of G=W^*W  (gives nu^2 <= lambda_min^+(J))
    G = W.conj().T @ W
    wg, Ug = np.linalg.eigh(herm(G))
    starts.append(np.concatenate([Ug[:, 0].real, Ug[:, 0].imag]))
    for _ in range(n_restarts):
        starts.append(rng.normal(size=2 * r))
    best = np.inf
    bestx = None
    scale = max(1e-300, np.linalg.norm(W) ** 2)
    for x0 in starts:
        x = x0.copy()
        for beta in (10 / scale, 1e2 / scale, 1e3 / scale, 1e4 / scale, 1e6 / scale):
            res = minimize(smooth, x, args=(beta,), jac=True, method="L-BFGS-B",
                           options=dict(maxiter=500, gtol=1e-14, ftol=1e-15))
            x = res.x
        v = exact(x)
        if v < best:
            best, bestx = v, x
    # final derivative-free polish of the best point
    res = minimize(exact, bestx, method="Nelder-Mead",
                   options=dict(maxiter=4000 * r, xatol=1e-12, fatol=1e-16))
    best = min(best, exact(res.x))
    return float(best)


def nu_bounds(JOm, dY, dX, rng, n_restarts=60):
    W = arn_frame(JOm)
    lp = lmin_pos(JOm)
    up = nu_upper_opt(W, dY, dX, rng, n_restarts)
    lo = nu_lower_sdp(W, dY, dX)
    return dict(nu2_opt=up, nu2_sdp_lower=lo, lam_min_pos=lp,
                lemma82_lower=lp / min(dX, dY), lemma82_upper=lp)


# ----------------------------------------------------------------------------- misc fidelities
def uhlmann_fid(rho, sig):
    """Root fidelity ||sqrt(rho) sqrt(sig)||_1."""
    return float(np.sum(np.linalg.svd(psd_sqrt(rho) @ psd_sqrt(sig), compute_uv=False)))


def choi_state_fid(J1, J2, dX):
    return uhlmann_fid(J1 / dX, J2 / dX)


def fhat_all(JPhi, JPsi, dY, dX, sdp=True, reg="mp"):
    """Three independent evaluations of Fhat. reg in {"mp","double",None}."""
    out = dict(arn=fhat_arn(JPhi, JPsi, dY, dX))
    if reg == "mp":
        out["reg"], _, out["reg_extrap_gap"] = fhat_reg_mp(JPhi, JPsi, dY, dX)
    elif reg == "double":
        out["reg"] = fhat_reg(JPhi, JPsi, dY, dX)
    if sdp:
        out["sdp"], info = fhat_sdp(JPhi, JPsi, dY, dX)
        out["sdp_solver"] = info["solver"]
    vals = [out[k] for k in ("arn", "reg", "sdp") if k in out and np.isfinite(out[k])]
    out["spread"] = float(max(vals) - min(vals))
    return out


# ----------------------------------------------------------------------------- high-precision regularisation
def fhat_reg_mp(JPhi, JPsi, dY, dX, eps_exps=(-24, -32, -40, -48, -56), dps=120, alpha=0.5):
    """eps-regularised  lambda_min Tr_Y((J1+eps)#_alpha(J2+eps))  in mpmath (dps digits),
    eps = 10^k for k in eps_exps, extrapolated eps->0 by Aitken (geometric-rate) acceleration.
    Needed because double-precision regularisation cannot reach eps << lambda_min(G),
    which is the scale on which the regularised mean converges (see E1).
    Returns (extrapolated value, values on grid, |last - extrapolated|)."""
    import mpmath as mp
    mp.mp.dps = dps
    n = JPhi.shape[0]

    def tomp(A):
        return mp.matrix([[mp.mpc(complex(A[i, j])) for j in range(n)] for i in range(n)])

    def fn(A, f):
        w, U = mp.eighe(A)
        D = mp.diag([f(x) if x > 0 else mp.mpf(0) for x in w])
        return U * D * U.H

    # Compress to the support of J(Omega) (basis P from SVD of the stacked factors).  The mean
    # lives there (A#B <= (A+B)/2), and inside it the generic intersection dimension of the two
    # factor ranges equals the true one (a+b-r), so double-precision noise in the factors cannot
    # fake a trivially-intersecting pair (without this, dephasing p vs p+1e-4 gave errors ~0.5).
    R1 = psd_factor(JPhi); R2 = psd_factor(JPsi)
    Uw, Sw, _ = np.linalg.svd(np.hstack([R1, R2]), full_matrices=False)
    P = Uw[:, Sw > 1e-10 * Sw.max()]
    r = P.shape[1]
    nY = dY * dX

    def tomp_rect(A):
        return mp.matrix([[mp.mpc(complex(A[i, j])) for j in range(A.shape[1])] for i in range(A.shape[0])])

    c1, c2 = tomp_rect(P.conj().T @ R1), tomp_rect(P.conj().T @ R2)
    A0, B0 = c1 * c1.H, c2 * c2.H
    Pm = tomp_rect(P)
    n = r
    vals = []
    for k in eps_exps:
        e = mp.mpf(10) ** k
        A = A0 + e * mp.eye(n)
        B = B0 + e * mp.eye(n)
        Ah = fn(A, mp.sqrt)
        Aih = fn(A, lambda x: 1 / mp.sqrt(x))
        Mid = Aih * B * Aih
        Mid = (Mid + Mid.H) / 2
        G = Pm * (Ah * fn(Mid, lambda x: x ** alpha) * Ah) * Pm.H
        # partial trace over Y
        P = mp.matrix(dX, dX)
        for a in range(dX):
            for b in range(dX):
                P[a, b] = sum(G[y * dX + a, y * dX + b] for y in range(dY))
        P = (P + P.H) / 2
        w, _ = mp.eighe(P)
        vals.append(min(w))
    # Geometric-rate (Aitken) extrapolation on an equally log-spaced eps grid: the regularised
    # value approaches its limit like c*eps^q (q=1/2 for #, q=min(alpha,1-alpha)/... in general),
    # so successive differences decay geometrically with ratio rho = 10^(q*step).
    v = [mp.mpf(x) for x in vals]
    d1, d2 = v[-2] - v[-1], v[-3] - v[-2]
    if d2 != 0 and 0 < d1 / d2 < 1:
        rho = d1 / d2
        lim = v[-1] - d1 * rho / (1 - rho)
    else:
        lim = v[-1]
    coef = [lim]
    return float(coef[0]), [float(x) for x in v], float(abs(v[-1] - coef[0]))


def gmean_mp(A, B, eps_exps=(-30, -40, -50, -60, -70), dps=140, alpha=0.5):
    """A #_alpha B for PSD A, B as the limit eps->0 of (A+eps)#(B+eps) in mpmath, computed on
    the joint support (SVD of stacked factors), Aitken-accelerated entrywise.  Returns numpy."""
    import mpmath as mp
    mp.mp.dps = dps
    R1, R2 = psd_factor(A), psd_factor(B)
    Uw, Sw, _ = np.linalg.svd(np.hstack([R1, R2]), full_matrices=False)
    P = Uw[:, Sw > 1e-10 * Sw.max()]
    n = P.shape[1]

    def tm(M):
        return mp.matrix([[mp.mpc(complex(M[i, j])) for j in range(M.shape[1])] for i in range(M.shape[0])])

    def fn(M, f):
        w, U = mp.eighe(M)
        return U * mp.diag([f(x) if x > 0 else mp.mpf(0) for x in w]) * U.H

    c1, c2 = tm(P.conj().T @ R1), tm(P.conj().T @ R2)
    A0, B0 = c1 * c1.H, c2 * c2.H
    seq = []
    for k in eps_exps:
        e = mp.mpf(10) ** k
        Ae, Be = A0 + e * mp.eye(n), B0 + e * mp.eye(n)
        Ah, Aih = fn(Ae, mp.sqrt), fn(Ae, lambda x: 1 / mp.sqrt(x))
        Mid = Aih * Be * Aih
        Mid = (Mid + Mid.H) / 2
        seq.append(Ah * fn(Mid, lambda x: x ** alpha) * Ah)
    out = np.zeros((n, n), complex)
    for i in range(n):
        for j in range(n):
            v = [s[i, j] for s in seq]
            d1, d2 = v[-2] - v[-1], v[-3] - v[-2]
            lim = v[-1]
            if abs(d2) > 0:
                rho = d1 / d2
                if abs(rho) < 1:
                    lim = v[-1] - d1 * rho / (1 - rho)
            out[i, j] = complex(lim)
    return herm(P @ out @ P.conj().T)


# ----------------------------------------------------------------------------- more channels / divergences
def kraus_gad(g, N):
    """Generalised amplitude damping (damping g, thermal excited population N)."""
    a, b = np.sqrt(1 - N), np.sqrt(N)
    return [a * np.array([[1, 0], [0, np.sqrt(1 - g)]], complex), a * np.sqrt(g) * np.array([[0, 1], [0, 0]], complex),
            b * np.array([[np.sqrt(1 - g), 0], [0, 1]], complex), b * np.sqrt(g) * np.array([[0, 0], [1, 0]], complex)]


def kraus_pauli(px, py, pz):
    return [np.sqrt(1 - px - py - pz) * I2, np.sqrt(px) * X2, np.sqrt(py) * Y2, np.sqrt(pz) * Z2]


def compose_kraus(Ka, Kb):
    """Kraus of Ka o Kb (apply Kb first)."""
    return [A @ B for A in Ka for B in Kb]


def output_on_input(J, A, dY, dX):
    """(Phi (x) id)(vec(A)vec(A)^*) = (1_Y (x) A^T) J (1_Y (x) A^T)^*  (Jones eq. 3.1)."""
    L = np.kron(np.eye(dY), A.T)
    return herm(L @ J @ L.conj().T)


def rel_ent(r, s, tol=1e-13):
    """Umegaki D(r||s) (natural log). Supports handled exactly: +inf iff r has weight > tol*tr(r)
    outside supp(s) (relative eigenvalue cutoff tol); otherwise computed on supp(s).
    (An earlier version returned +inf whenever s was singular; fixed before the final E8 run.)"""
    wr, Ur = np.linalg.eigh(herm(r)); ws, Us = np.linalg.eigh(herm(s))
    wr = np.clip(wr, 0, None)
    cut = tol * max(1e-300, ws.max())
    keep = ws > cut
    P0 = Us[:, ~keep]
    if P0.shape[1] and np.real(np.trace(P0.conj().T @ r @ P0)) > tol * max(1e-300, np.trace(r).real):
        return np.inf
    lr = np.where(wr > 0, np.log(np.where(wr > 0, wr, 1)), 0)
    logs = (Us[:, keep] * np.log(ws[keep])) @ Us[:, keep].conj().T
    return float(np.real(np.sum(wr * lr) - np.trace(r @ logs)))


def _input_opt(fun, dX, rng, n_starts=8, maximize=True):
    from scipy.optimize import minimize
    sgn = -1 if maximize else 1

    def obj(x):
        A = (x[:dX * dX] + 1j * x[dX * dX:]).reshape(dX, dX)
        A = A / np.linalg.norm(A)
        return sgn * fun(A)
    starts = [np.concatenate([np.eye(dX).reshape(-1), np.zeros(dX * dX)])]
    starts += [rng.normal(size=2 * dX * dX) for _ in range(n_starts)]
    best = None
    for x0 in starts:
        r = minimize(obj, x0, method="Nelder-Mead", options=dict(maxiter=3000 * dX, xatol=1e-10, fatol=1e-18))
        r = minimize(obj, r.x, method="BFGS", options=dict(gtol=1e-14))
        if best is None or r.fun < best:
            best = r.fun
    return sgn * best


def rel_ent_on_input(J1, J2, A, dY, dX, sv_tol=1e-4):
    """D((Phi(x)id)psi_A || (Psi(x)id)psi_A) computed on range(1_Y (x) A^T).
    For J1, J2 > 0 (full-rank channels) both outputs have exactly this support, so the relative
    entropy is evaluated there without a numerical support test (eigenvalues of the compressed
    second argument are floored at 1e-300).  Schmidt coefficients of A below sv_tol*max are dropped
    (input restricted to a nearby, better-conditioned state)."""
    U, sv, Vh = np.linalg.svd(A.T)
    keep = sv > sv_tol * sv.max()
    Ar = (U[:, keep] * sv[keep]) @ Vh[keep, :]
    Ar = Ar / np.linalg.norm(Ar)
    Q = np.kron(np.eye(dY), U[:, keep])
    L = np.kron(np.eye(dY), Ar)
    r1 = herm(Q.conj().T @ L @ J1 @ L.conj().T @ Q)
    r2 = herm(Q.conj().T @ L @ J2 @ L.conj().T @ Q)
    w1, U1 = np.linalg.eigh(r1); w2, U2 = np.linalg.eigh(r2)
    w1 = np.clip(w1, 0, None); w2 = np.clip(w2, 1e-300, None)
    l1 = np.where(w1 > 0, np.log(np.where(w1 > 0, w1, 1)), 0)
    return float(np.sum(w1 * l1) - np.real(np.trace(r1 @ (U2 * np.log(w2)) @ U2.conj().T)))


def umegaki_channel(J1, J2, dY, dX, rng, n_starts=6):
    """sup_psi D((Phi(x)id)psi || (Psi(x)id)psi), by input optimisation (lower estimate of the sup)."""
    return _input_opt(lambda A: rel_ent_on_input(J1, J2, A, dY, dX), dX, rng, n_starts)


def uhlmann_channel_opt(J1, J2, dY, dX, rng, n_starts=6):
    """min_psi root fidelity of outputs (upper estimate of the min), by input optimisation."""
    return _input_opt(lambda A: uhlmann_fid(output_on_input(J1, A, dY, dX), output_on_input(J2, A, dY, dX)), dX, rng,
                      n_starts, maximize=False)


def uhlmann_channel_sdp(JPhi, JPsi, dY, dX):
    """Root (Uhlmann) channel fidelity, Katariya-Wilde/Yuan-Fung SDP:
    max t s.t. [[J1, Z],[Z^*, J2]] >= 0, Re Tr_Y Z >= t 1  (Z general; cf. fhat_sdp with Z = Z^*)."""
    R1, R2 = _psd_factor(JPhi), _psd_factor(JPsi)
    a, b = R1.shape[1], R2.shape[1]
    K = cp.Variable((a, b), complex=True)
    t = cp.Variable()
    Z = R1 @ K @ R2.conj().T
    cons = [cp.bmat([[np.eye(a), K], [K.H, np.eye(b)]]) >> 0,
            cp.partial_trace((Z + Z.H) / 2, [dY, dX], axis=0) - t * np.eye(dX) >> 0]
    prob = cp.Problem(cp.Maximize(t), cons)
    s, st = _solve(prob)
    return float(t.value) if s else np.nan


def rld_fisher_channel(J, D, dY, dX):
    """Channel RLD Fisher information (Katariya-Wilde): || Tr_Y( D J^{-1} D ) ||_inf, D = dJ/dtheta."""
    return float(np.linalg.eigvalsh(herm(ptrace_Y(D @ np.linalg.solve(J, D), dY, dX))).max())


def chernoff_geo(JPhi, JPsi, dY, dX):
    """Geometric Chernoff divergence (Jones, chernoff.tex Sec. 2):
    C = -ln min_{s in [0,1]} lambda_min(B_s),  B_s = Psi*_{g_s(T)}(1),  g_s(t)=t^s(1-t)^{1-s},
    lambda_min(B_s) = lmin Tr_Y(W g_s(T) W^*) = qhat_alpha_arn(J(Phi),J(Psi),s).  Returns (C, s*)."""
    from scipy.optimize import minimize_scalar
    f = lambda s: qhat_alpha_arn(JPhi, JPsi, s, dY, dX)
    grid = np.linspace(0.02, 0.98, 25)
    vals = [f(s) for s in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-10))
    v = min(r.fun, vals[i])
    # lambda_min(B_s) <= ZERO_TOL (1e-13, relative to ||B_s|| <= 2) is numerical zero -> C = +inf
    return (np.inf if v <= 1e-13 else float(-np.log(v))), float(r.x)


def hockey_state(r, s, g):
    w = np.linalg.eigvalsh(herm(r - g * s))
    return float(np.sum(w[w > 0]))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def _hs_batch(r, s, gam):
    w = np.linalg.eigvalsh(herm(r)[None] - gam[:, None, None] * herm(s)[None])
    return np.sum(np.clip(w, 0, None), axis=1)


def ht_hellinger_state(r, s, n_int=96):
    """Hirche-Tomamichel squared-Hellinger divergence, f=(sqrt x - 1)^2 (f''=gamma^{-3/2}/2, self-dual):
    D = (1/2) int_1^inf gamma^{-3/2} (E_g(r||s)+E_g(s||r)) dgamma = int_0^1 (E_{u^-2}(r||s)+E_{u^-2}(s||r)) du
    (gamma = u^-2).  Composite 16-point Gauss-Legendre on n_int log-spaced panels of [1e-8, 1]
    (+[0,1e-8]) and on 2*n_int panels log-spaced in 1-u down to 1e-12."""
    # panels log-refined both at u=0 and at u=1 (for nearby states the integrand lives in 1-u = O(dist))
    edges = np.unique(np.concatenate([[0, 1], np.logspace(-8, 0, n_int), 1 - np.logspace(-12, 0, 2 * n_int)]))
    edges = edges[(edges >= 0) & (edges <= 1)]
    a, b = edges[:-1, None], edges[1:, None]
    u = (0.5 * (b - a) * _GL_X[None] + 0.5 * (a + b)).ravel()
    w = (0.5 * (b - a) * _GL_W[None]).ravel()
    g = 1 / u ** 2
    return float(np.sum(w * (_hs_batch(r, s, g) + _hs_batch(s, r, g))))


def ht_hellinger_channel(J1, J2, dY, dX, rng, n_starts=3):
    """sup over pure inputs of the state HT squared-Hellinger divergence (local optimisation;
    a lower estimate of the channel value)."""
    return _input_opt(lambda A: ht_hellinger_state(output_on_input(J1, A, dY, dX), output_on_input(J2, A, dY, dX)),
                      dX, rng, n_starts)


def xi_qc(r, s):
    """Quantum Chernoff divergence -ln min_s Tr r^s s^{1-s}."""
    from scipy.optimize import minimize_scalar
    def q(a):
        return float(np.real(np.trace(psd_fn(r, lambda x: np.where(x > 1e-300, x, 0) ** a) @
                                      psd_fn(s, lambda x: np.where(x > 1e-300, x, 0) ** (1 - a)))))
    res = minimize_scalar(q, bounds=(1e-6, 1 - 1e-6), method="bounded", options=dict(xatol=1e-10))
    return float(-np.log(res.fun)), float(res.x)
