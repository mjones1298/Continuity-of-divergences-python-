"""A_cert core library: Pauli process tomography of qubit channels, estimators, plug-in
divergences, and certified (confidence-region) one-sided bounds.

Conventions (identical to numerics/qlib.py, Watrous): J(Phi) = sum_ij Phi(E_ij) (x) E_ij,
output Y first, input X second;  Phi(rho) = Tr_X[J (1_Y (x) rho^T)];  Tr_Y J = 1_X for TP maps.
"""
import sys, time
from pathlib import Path
import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
import cvxpy as cp

sys.path.insert(0, str(Path(__file__).resolve().parent))  # qlib.py lives next to this file
import qlib as Q  # noqa: E402  (ARN-based F-hat / BS, SDP cross-checks, mp regularisation)

I2, X2, Y2, Z2, H2 = Q.I2, Q.X2, Q.Y2, Q.Z2, Q.H2
PAULI = [X2, Y2, Z2]


def herm(A):
    return 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))


# ----------------------------------------------------------------------------- channels
def choi_gate(p, delta):
    """J(D_p o (e^{-i delta Z/2} H))  (qubit)."""
    U = sla.expm(-0.5j * delta * Z2) @ H2
    return Q.choi_depolarize_after(Q.choi_unitary(U), p, 2, 2)


# ----------------------------------------------------------------------------- tomography design
def _proj(P, s):
    return 0.5 * (I2 + s * P)


INPUTS = [_proj(P, s) for P in (Z2, X2, Y2) for s in (+1, -1)]  # 6 Pauli eigenstates
BASES = PAULI                                                  # 3 measurement bases
# settings k = (input s, basis m); outcome o in {+,-}; A_{k,o} = Pi_{m,o} (x) rho_s^T
SETTINGS = [(s, m) for s in range(6) for m in range(3)]
A_PLUS = np.array([np.kron(_proj(BASES[m], +1), INPUTS[s].T) for (s, m) in SETTINGS])   # (18,4,4)
A_MINUS = np.array([np.kron(_proj(BASES[m], -1), INPUTS[s].T) for (s, m) in SETTINGS])
K_SET = len(SETTINGS)

# real basis of 4x4 Hermitian matrices (orthonormal in HS): B_j
_PB = [I2, X2, Y2, Z2]
HBASIS = np.array([np.kron(a, b) / 2.0 for a in _PB for b in _PB])  # (16,4,4), tr(B_i B_j) = delta_ij


def hcoef(J):
    return np.real(np.einsum("jab,...ba->...j", HBASIS, J))


def hmat(c):
    return np.einsum("...j,jab->...ab", c, HBASIS)


# linear-inversion map: least squares over all 36 (setting,outcome) frequencies
_AALL = np.concatenate([A_PLUS, A_MINUS])                       # (36,4,4)
_AMAT = np.real(np.einsum("kab,jba->kj", _AALL, HBASIS))        # (36,16)
_PINV = np.linalg.pinv(_AMAT)                                   # (16,36)
# J_LI = sum_k f_k+ D_k + C   (using f_k- = 1 - f_k+)
_D = hmat((_PINV[:, :K_SET] - _PINV[:, K_SET:]).T)              # (18,4,4)
_C = hmat(_PINV[:, K_SET:].sum(axis=1))
D_TRN = np.array([np.abs(np.linalg.eigvalsh(herm(d))).sum() for d in _D])   # ||D_k||_1
D_OPN = np.array([np.abs(np.linalg.eigvalsh(herm(d))).max() for d in _D])   # ||D_k||_inf
D_FRO = np.array([np.linalg.norm(d) for d in _D])


def probs_plus(J):
    """p_{k,+} for all 18 settings (vectorised over leading axes of J)."""
    return np.real(np.einsum("kab,...ba->...k", A_PLUS, J))


def sample(J, N, rng, reps):
    """Binomial counts -> frequencies f_{k,+}, shape (reps,18)."""
    p = np.clip(probs_plus(J), 0, 1)
    return rng.binomial(int(N), np.broadcast_to(p, (reps, K_SET))) / float(N)


def lin_inv(f):
    return herm(_C + np.einsum("...k,kab->...ab", f, _D))


# ----------------------------------------------------------------------------- CPTP projection
def _ptrY(J):
    return np.einsum("...iaib->...ab", J.reshape(J.shape[:-2] + (2, 2, 2, 2)))


def _tp_proj(J):
    S = _ptrY(J) - I2
    return J - 0.5 * np.einsum("ij,...ab->...iajb", I2, S).reshape(J.shape)


def _psd_proj(J):
    w, U = np.linalg.eigh(herm(J))
    w = np.clip(w, 0, None)
    return herm(np.einsum("...ij,...j,...kj->...ik", U, w, U.conj()))


def tp_congruence(J):
    """J -> (1 (x) S^{-1/2}) J (1 (x) S^{-1/2}), S = Tr_Y J: exact TP, keeps PSD and rank."""
    S = herm(_ptrY(J))
    w, U = np.linalg.eigh(S)
    Sm = np.einsum("...ij,...j,...kj->...ik", U, 1 / np.sqrt(w), U.conj())
    K = np.einsum("ij,...ab->...iajb", I2, Sm).reshape(J.shape)
    return herm(K @ J @ K)


def cptp_project(J, iters=3000, tol=1e-13):
    """Frobenius projection onto CPTP by Dykstra (PSD cone, TP affine), batched over leading axes.
    The last PSD iterate is made exactly TP by tp_congruence (keeps exact zero eigenvalues)."""
    X = J.copy()
    P = np.zeros_like(J)
    Y = X
    for it in range(iters):
        Y = _psd_proj(X + P)
        P = X + P - Y
        Xn = _tp_proj(Y)
        if np.max(np.abs(Xn - X)) < tol:
            X = Xn
            break
        X = Xn
    return tp_congruence(_psd_proj(Y))


# ----------------------------------------------------------------------------- MLE / hedged MLE (cvxpy)
class MLE:
    """argmax sum_k f_k log p_k(J) + w log det J  over CPTP J  (w = beta/N for hedging)."""

    def __init__(self, hedged=False):
        self.f = cp.Parameter(2 * K_SET, nonneg=True)
        J = cp.Variable((4, 4), hermitian=True)
        Am = np.concatenate([A_PLUS, A_MINUS]).reshape(2 * K_SET, 16)
        # p_k = tr(A_k J) = sum_ab A_ab J_ba = (A^T flattened) . vec(J)
        At = np.concatenate([A_PLUS, A_MINUS]).transpose(0, 2, 1).reshape(2 * K_SET, 16)
        pr = cp.real(At @ cp.vec(J, order="C"))
        obj = self.f @ cp.log(pr)
        cons = [J >> 0, cp.partial_trace(J, [2, 2], axis=0) == np.eye(2)]
        self.hedged = hedged
        if hedged:
            self.w = cp.Parameter(nonneg=True)
            obj = obj + self.w * cp.log_det(J)
        self.prob = cp.Problem(cp.Maximize(obj), cons)
        self.J = J

    def fit(self, fplus, w=0.0):
        self.f.value = np.concatenate([fplus, 1 - fplus]).clip(0, None)
        if self.hedged:
            self.w.value = w
        try:
            self.prob.solve(solver="CLARABEL", tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11,
                            max_iter=400)
        except Exception:
            self.prob.solve(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=200000)
        J = herm(self.J.value)
        return J


def polish(J, thr=1e-9):
    """Remove interior-point fuzz: eigenvalues < thr (absolute) set to 0, then exact TP."""
    w, U = np.linalg.eigh(herm(J))
    w = np.where(w < thr, 0.0, w)
    return tp_congruence(herm((U * w) @ U.conj().T))


# ----------------------------------------------------------------------------- rank-r MLE (Kraus rank r)
def _iso(g, r):
    G = (g[: 4 * r] + 1j * g[4 * r:]).reshape(2 * r, 2)
    w, U = np.linalg.eigh(G.conj().T @ G)
    return G @ (U / np.sqrt(w)) @ U.conj().T


def _choi_from_iso(V, r):
    Ks = [V[2 * k:2 * k + 2, :] for k in range(r)]
    vs = np.array([K.reshape(-1) for K in Ks])
    return vs.T @ vs.conj()


def lowrank_mle(fplus, r, J0):
    """Rank-r (Kraus rank <= r) CPTP maximum-likelihood fit, initialised from the top-r
    eigenvectors of J0 (L-BFGS over an unconstrained parametrisation V = G (G*G)^{-1/2})."""
    w, U = np.linalg.eigh(herm(J0))
    vecs = U[:, ::-1][:, :r] * np.sqrt(np.clip(w[::-1][:r], 1e-12, None))
    G = np.vstack([vecs[:, k].reshape(2, 2) for k in range(r)])
    g0 = np.concatenate([G.real.ravel(), G.imag.ravel()])
    fp = np.concatenate([fplus, 1 - fplus])
    Aall = np.concatenate([A_PLUS, A_MINUS])

    def nll(g):
        J = _choi_from_iso(_iso(g, r), r)
        pr = np.real(np.einsum("kab,ba->k", Aall, J))
        return -np.sum(fp * np.log(np.clip(pr, 1e-300, None)))

    res = sopt.minimize(nll, g0, method="L-BFGS-B", options=dict(maxiter=500, ftol=1e-15, gtol=1e-12))
    return herm(_choi_from_iso(_iso(res.x, r), r))


# ----------------------------------------------------------------------------- quantities
def fhat(J1, J2):
    return Q.fhat_arn(J1, J2, 2, 2)


def dbs(J1, J2):
    """D_BS(J1 || J2) (channel, natural log); +inf if supp J1 not in supp J2."""
    return Q.dbs_arn(J1, J2, 2, 2)


def _sqrt_sigma(r):
    rn = np.linalg.norm(r)
    lp, lm = (1 + rn) / 2, max((1 - rn) / 2, 0.0)
    a, b = (np.sqrt(lp) + np.sqrt(lm)) / 2, (np.sqrt(lp) - np.sqrt(lm)) / 2
    n = r / rn if rn > 0 else np.zeros(3)
    S = a * I2 + b * (n[0] * X2 + n[1] * Y2 + n[2] * Z2)
    return np.kron(I2, S)


_GRID = None


def _grid():
    global _GRID
    if _GRID is None:
        rng = np.random.default_rng(7)
        g = rng.normal(size=(120, 3))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        pts = [np.zeros(3)] + [s * v for v in g for s in (0.5, 0.9, 1.0)]
        _GRID = np.array(pts)
    return _GRID


def _sqrt_sigma_batch(R):
    rn = np.linalg.norm(R, axis=1)
    lp, lm = (1 + rn) / 2, np.clip((1 - rn) / 2, 0, None)
    a, b = (np.sqrt(lp) + np.sqrt(lm)) / 2, (np.sqrt(lp) - np.sqrt(lm)) / 2
    n = R / np.where(rn > 0, rn, 1)[:, None]
    S = a[:, None, None] * I2 + b[:, None, None] * np.einsum("ni,iab->nab", n, np.array(PAULI))
    return np.einsum("ij,nab->niajb", I2, S).reshape(-1, 4, 4)


def _opt_ball(fun, sign, fun_batch):
    """optimise over the Bloch ball; fun concave (sign=-1, maximise) or convex (sign=+1)."""
    G = _grid()
    vals = sign * fun_batch(_sqrt_sigma_batch(G))
    best = G[np.argmin(vals)]

    def h(x):
        n = np.linalg.norm(x)
        return sign * fun(x / n if n > 1 else x)

    res = sopt.minimize(h, best, method="Nelder-Mead",
                        options=dict(xatol=1e-7, fatol=1e-13, maxiter=2000, maxfev=3000))
    x = res.x
    n = np.linalg.norm(x)
    return min(sign * res.fun, sign * vals.min()) if sign > 0 else max(sign * res.fun, sign * vals.min()), x


def diamond(Jd):
    """||Delta||_diamond for Hermitian Choi Jd of a qubit map (concave max over input state)."""
    def f(r):
        S = _sqrt_sigma(r)
        return np.abs(np.linalg.eigvalsh(herm(S @ Jd @ S))).sum()

    def fb(S):
        return np.abs(np.linalg.eigvalsh(herm(S @ Jd @ S))).sum(axis=1)
    return _opt_ball(f, -1, fb)[0]


def hockey1(J1, J2):
    return 0.5 * diamond(J1 - J2)


def _rootfid(A, B):
    wa, Ua = np.linalg.eigh(herm(A))
    sa = (Ua * np.sqrt(np.clip(wa, 0, None))) @ Ua.conj().T
    M = herm(sa @ B @ sa)
    return float(np.sqrt(np.clip(np.linalg.eigvalsh(M), 0, None)).sum())


def _rootfid_b(A, B):
    wa, Ua = np.linalg.eigh(herm(A))
    sa = np.einsum("nij,nj,nkj->nik", Ua, np.sqrt(np.clip(wa, 0, None)), Ua.conj())
    M = herm(sa @ B @ sa)
    return np.sqrt(np.clip(np.linalg.eigvalsh(M), 0, None)).sum(axis=1)


def uhlmann(J1, J2):
    """Root (Uhlmann) channel fidelity min_sigma F(...): convex in sigma (minimise)."""
    def f(r):
        S = _sqrt_sigma(r)
        return _rootfid(S @ J1 @ S, S @ J2 @ S)

    def fb(S):
        return _rootfid_b(S @ J1 @ S, S @ J2 @ S)
    return _opt_ball(f, +1, fb)[0]


def all_quantities(J1, J2, fast=False):
    out = dict(F=fhat(J1, J2), BS=dbs(J1, J2))
    if not fast:
        out["E1"] = hockey1(J1, J2)
        out["FU"] = uhlmann(J1, J2)
    out["lmin1"] = float(np.linalg.eigvalsh(herm(J1)).min())
    out["lmin2"] = float(np.linalg.eigvalsh(herm(J2)).min())
    return out


# ----------------------------------------------------------------------------- confidence region
def hoeffding_t(N, alpha, n_settings):
    """two-sided Hoeffding half-width, union bound over n_settings binomial settings."""
    return np.sqrt(np.log(2 * n_settings / alpha) / (2 * N))


def region_radii(N, alpha=0.05, n_settings=2 * K_SET):
    t = hoeffding_t(N, alpha, n_settings)
    return dict(t=t, eps_tr=t * D_TRN.sum(), eps_op=t * D_OPN.sum(), eps_fro=t * D_FRO.sum())


class CertSDP:
    """Extremal values of concave/convex channel functionals over the confidence region
    R = {(J1,J2) CPTP : |tr(A_k J_i) - f_ik| <= t for all 18 settings, i=1,2}   (kind="box")
    or  R = {(J1,J2) CPTP : ||J_i - C_i||_1 <= eps}                               (kind="ball").
    Objectives:  'F' max Fhat(J1,J2);  'Q3','Q7' max Qhat_alpha(J1||J2) (alpha=3/4, 7/8, weight alpha
    on J1, i.e. lambda_min Tr_Y (J2 #_alpha J1));  'FU' max Uhlmann;  'E1' min (1/2)||J1-J2||_diamond."""

    def __init__(self, obj, kind="box"):
        self.obj, self.kind = obj, kind
        J1 = cp.Variable((4, 4), hermitian=True)
        J2 = cp.Variable((4, 4), hermitian=True)
        cons = [J1 >> 0, J2 >> 0,
                cp.partial_trace(J1, [2, 2], axis=0) == np.eye(2),
                cp.partial_trace(J2, [2, 2], axis=0) == np.eye(2)]
        At = A_PLUS.transpose(0, 2, 1).reshape(K_SET, 16)
        if kind == "box":
            self.f1 = cp.Parameter(K_SET)
            self.f2 = cp.Parameter(K_SET)
            self.t = cp.Parameter(nonneg=True)
            for J, f in ((J1, self.f1), (J2, self.f2)):
                pr = cp.real(At @ cp.vec(J, order="C"))
                cons += [pr - f <= self.t, f - pr <= self.t]
        else:
            self.C1 = cp.Parameter((4, 4), hermitian=True)
            self.C2 = cp.Parameter((4, 4), hermitian=True)
            self.eps = cp.Parameter(nonneg=True)
            for J, C in ((J1, self.C1), (J2, self.C2)):
                P = cp.Variable((4, 4), hermitian=True)
                Qm = cp.Variable((4, 4), hermitian=True)
                cons += [P >> 0, Qm >> 0, J - C == P - Qm, cp.real(cp.trace(P + Qm)) <= self.eps]
        s = cp.Variable()
        if obj == "F":
            X = cp.Variable((4, 4), hermitian=True)
            cons += [cp.bmat([[J1, X], [X, J2]]) >> 0,
                     cp.partial_trace(X, [2, 2], axis=0) - s * np.eye(2) >> 0]
            prob = cp.Problem(cp.Maximize(s), cons)
        elif obj in ("Q3", "Q7"):
            # J2 #_{1/2} J1 >= Z1;  Z1 #_{1/2} J1 >= Z2 (=> alpha=3/4);  Z2 # J1 >= Z3 (alpha=7/8)
            levels = 2 if obj == "Q3" else 3
            Zs = [cp.Variable((4, 4), hermitian=True) for _ in range(levels)]
            prev = J2
            for Z in Zs:
                cons += [cp.bmat([[prev, Z], [Z, J1]]) >> 0]
                prev = Z
            cons += [cp.partial_trace(Zs[-1], [2, 2], axis=0) - s * np.eye(2) >> 0]
            prob = cp.Problem(cp.Maximize(s), cons)
        elif obj == "FU":
            Z = cp.Variable((4, 4), complex=True)
            TZ = cp.partial_trace(Z, [2, 2], axis=0)
            cons += [cp.bmat([[J1, Z], [Z.H, J2]]) >> 0, (TZ + TZ.H) / 2 - s * np.eye(2) >> 0]
            prob = cp.Problem(cp.Maximize(s), cons)
        elif obj == "E1":
            Z = cp.Variable((4, 4), hermitian=True)
            cons += [Z >> 0, Z - (J1 - J2) >> 0,
                     s * np.eye(2) - cp.partial_trace(Z, [2, 2], axis=0) >> 0]
            prob = cp.Problem(cp.Minimize(s), cons)   # (1/2)||J1-J2||_diamond = min s (TP difference)
        else:
            raise ValueError(obj)
        self.prob, self.s, self.J1, self.J2 = prob, s, J1, J2

    def solve(self, a1, a2, rad):
        if self.kind == "box":
            self.f1.value, self.f2.value, self.t.value = a1, a2, rad
        else:
            self.C1.value, self.C2.value, self.eps.value = herm(a1), herm(a2), rad
        try:
            self.prob.solve(solver="CLARABEL")   # default tolerances (1e-8); checked vs tight/SCS
            st = self.prob.status
        except Exception:
            st = "error"
        if st not in ("optimal", "optimal_inaccurate"):
            try:
                self.prob.solve(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=100000)
                st = self.prob.status
            except Exception:
                st = "error"
        if st not in ("optimal", "optimal_inaccurate"):
            return np.nan, st
        return float(self.s.value), st


class UhlSDP:
    """Parametric Uhlmann channel-fidelity SDP (Katariya-Wilde / Yuan-Fung form)."""

    def __init__(self):
        self.A = cp.Parameter((4, 4), hermitian=True)
        self.B = cp.Parameter((4, 4), hermitian=True)
        Z = cp.Variable((4, 4), complex=True)
        s = cp.Variable()
        TZ = cp.partial_trace(Z, [2, 2], axis=0)
        self.prob = cp.Problem(cp.Maximize(s), [cp.bmat([[self.A, Z], [Z.H, self.B]]) >> 0,
                                                (TZ + TZ.H) / 2 - s * np.eye(2) >> 0])
        self.s = s

    def __call__(self, J1, J2):
        self.A.value, self.B.value = herm(J1), herm(J2)
        try:
            self.prob.solve(solver="CLARABEL", tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
        except Exception:
            self.prob.solve(solver="SCS", eps_abs=1e-10, eps_rel=1e-10, max_iters=100000)
        return float(self.s.value)
