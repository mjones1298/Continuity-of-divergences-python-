"""S2b: compare certification pipelines (pair delta = 2 sqrt(p), true F-hat ~ 0.71):
 (a)  full Pauli tomography + trace-norm ("diamond") ball around the linear-inversion estimate (Hoeffding);
 (a') full tomography + Hoeffding box in probability space          [from s2_certified shots runs];
 (a'')full tomography + Clopper-Pearson (exact binomial, variance-adaptive) box in probability space;
 (b)  direct Pauli-error-rate estimation in the KNOWN gate frame (model assumption: N = Pauli o U, U known):
      3 settings per channel (input U^dag|P+>, measure P; flip probability = sum of the two anticommuting
      Pauli error rates), Clopper-Pearson intervals -> polytope for (q_X,q_Y,q_Z) of each channel;
      certified F-hat / D_7/8 by SDP over that polytope.
All regions hold with probability >= 0.95 (union bound).  Certified one-sided bounds as in s2_certified.
usage: python3 s2b_pipelines.py scan | coverage | contrast
  scan      N grid per pipeline, 10 runs per point, all p           -> results/s2b_raw/scan_p*.json
  refine    1/8-decade grid, 50 runs, around eta in [0.025, 0.1]   -> results/s2b_raw/refine_p*.json
  coverage  1000 runs per (p, pipeline) at the eta = 0.05 budget    -> results/s2b_raw/cov_p*_{ball,cp,pauli}.json
  contrast  E1 / Uhlmann through pipeline (a), pair delta = 1 rad    -> results/s2b_raw/contrast_p*.json
"""
import os, sys, json, time, warnings
import numpy as np
import cvxpy as cp
from scipy.stats import beta
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
sys.path.insert(0, str(Path(HERE).resolve().parents[1] / "common"))  # certlib.py, qlib.py in code/common
from certlib import *  # noqa
from certlib import _proj  # noqa

ALPHA, SAFETY = 0.05, 1e-5
OUT = os.path.join(HERE, "results", "s2b_raw"); os.makedirs(OUT, exist_ok=True)


def clopper_pearson(k, n, a):
    lo = np.where(k > 0, beta.ppf(a / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - a / 2, k + 1, n - k), 1.0)
    return lo, hi


# ----------------------------------------------------------------------------- (a'') CP box SDP
class CPBox:
    def __init__(self, obj):
        J1 = cp.Variable((4, 4), hermitian=True); J2 = cp.Variable((4, 4), hermitian=True)
        self.lo = [cp.Parameter(K_SET), cp.Parameter(K_SET)]; self.hi = [cp.Parameter(K_SET), cp.Parameter(K_SET)]
        At = A_PLUS.transpose(0, 2, 1).reshape(K_SET, 16)
        cons = [J1 >> 0, J2 >> 0, cp.partial_trace(J1, [2, 2], axis=0) == np.eye(2),
                cp.partial_trace(J2, [2, 2], axis=0) == np.eye(2)]
        for i, J in enumerate((J1, J2)):
            pr = cp.real(At @ cp.vec(J, order="C"))
            cons += [pr >= self.lo[i], pr <= self.hi[i]]
        s = cp.Variable()
        cons += _obj_cons(obj, J1, J2, s)
        self.prob, self.s = cp.Problem(cp.Maximize(s), cons), s

    def solve(self, lo1, hi1, lo2, hi2):
        for P, v in zip(self.lo + self.hi, (lo1, lo2, hi1, hi2)):
            P.value = v
        return _solve(self.prob, self.s)


def _obj_cons(obj, J1, J2, s):
    if obj == "F":
        X = cp.Variable((4, 4), hermitian=True)
        return [cp.bmat([[J1, X], [X, J2]]) >> 0, cp.partial_trace(X, [2, 2], axis=0) - s * np.eye(2) >> 0]
    Zs = [cp.Variable((4, 4), hermitian=True) for _ in range(3)]   # Q7: alpha = 7/8
    cons, prev = [], J2
    for Z in Zs:
        cons += [cp.bmat([[prev, Z], [Z, J1]]) >> 0]; prev = Z
    return cons + [cp.partial_trace(Zs[-1], [2, 2], axis=0) - s * np.eye(2) >> 0]


def _solve(prob, s):
    try:
        prob.solve(solver="CLARABEL")
    except Exception:
        prob.solve(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=100000)
    return float(s.value) if prob.status in ("optimal", "optimal_inaccurate") else np.nan


# ----------------------------------------------------------------------------- (b) known-frame Pauli
PAULIS = [X2, Y2, Z2]
ANTI = {0: (1, 2), 1: (0, 2), 2: (0, 1)}   # flip under measurement of P_j: error rates anticommuting with P_j


def frame_chois(U):
    """J of rho -> P U rho U^dag P for P in (I, X, Y, Z)."""
    return [Q.choi_unitary(P @ U) for P in (I2, X2, Y2, Z2)]


def flip_probs(J, U):
    """for each P in X,Y,Z: prob of outcome -1 when measuring P after sending U^dag |P+> through the channel."""
    out = []
    for P in PAULIS:
        rho = U.conj().T @ _proj(P, +1) @ U
        out.append(float(np.real(np.trace(_proj(P, -1) @ Q.apply_choi(J, rho, 2, 2)))))
    return np.array(out)


class PauliSDP:
    def __init__(self, obj, U1, U2):
        self.B = [frame_chois(U1), frame_chois(U2)]
        self.lo = [cp.Parameter(3), cp.Parameter(3)]; self.hi = [cp.Parameter(3), cp.Parameter(3)]
        qs = [cp.Variable(3, nonneg=True), cp.Variable(3, nonneg=True)]
        cons, Js = [], []
        for i in range(2):
            q = qs[i]
            cons += [cp.sum(q) <= 1]
            J = (1 - cp.sum(q)) * self.B[i][0] + sum(q[j] * self.B[i][j + 1] for j in range(3))
            Js.append(J)
            for j in range(3):
                a, b = ANTI[j]
                cons += [q[a] + q[b] >= self.lo[i][j], q[a] + q[b] <= self.hi[i][j]]
        s = cp.Variable()
        cons += _obj_cons(obj, Js[0], Js[1], s)
        self.prob, self.s = cp.Problem(cp.Maximize(s), cons), s

    def solve(self, lo1, hi1, lo2, hi2):
        for P, v in zip(self.lo + self.hi, (lo1, lo2, hi1, hi2)):
            P.value = v
        return _solve(self.prob, self.s)


def gates(p):
    d = 2 * np.sqrt(p)
    U1, U0 = sla.expm(-0.5j * d * Z2) @ H2, H2
    return U1, U0, choi_gate(p, d), choi_gate(p, 0.0)


_C = {}


def solvers(p):
    if p not in _C:
        U1, U0, _, _ = gates(p)
        _C[p] = dict(cpF=CPBox("F"), cpQ=CPBox("Q7"), paF=PauliSDP("F", U1, U0), paQ=PauliSDP("Q7", U1, U0),
                     ballF=CertSDP("F", "ball"), ballQ=CertSDP("Q7", "ball"))
    return _C[p]


def run_once(p, N, rng, which=("ball", "cp", "pauli")):
    U1, U0, J1, J0 = gates(p)
    S = solvers(p)
    out = {}
    if "ball" in which or "cp" in which:
        f1, f0 = sample(J1, N, rng, 1)[0], sample(J0, N, rng, 1)[0]
        if "ball" in which:
            r = region_radii(N, ALPHA, 2 * K_SET)
            out["ball_F"] = S["ballF"].solve(lin_inv(f1), lin_inv(f0), r["eps_tr"])[0]
            out["ball_Q"] = S["ballQ"].solve(lin_inv(f1), lin_inv(f0), r["eps_tr"])[0]
        if "cp" in which:
            a = ALPHA / (2 * K_SET)
            lo1, hi1 = clopper_pearson(np.round(f1 * N), N, a)
            lo0, hi0 = clopper_pearson(np.round(f0 * N), N, a)
            out["cp_F"] = S["cpF"].solve(lo1, hi1, lo0, hi0)
            out["cp_Q"] = S["cpQ"].solve(lo1, hi1, lo0, hi0)
    if "pauli" in which:
        a = ALPHA / 6
        k1 = rng.binomial(int(N), flip_probs(J1, U1)); k0 = rng.binomial(int(N), flip_probs(J0, U0))
        lo1, hi1 = clopper_pearson(k1, N, a); lo0, hi0 = clopper_pearson(k0, N, a)
        out["pauli_F"] = S["paF"].solve(lo1, hi1, lo0, hi0)
        out["pauli_Q"] = S["paQ"].solve(lo1, hi1, lo0, hi0)
    res = {}
    for k, v in out.items():
        v = min(1.0, v + SAFETY)
        res[k] = v
        res[k.replace("_F", "_D12").replace("_Q", "_D78")] = (-2 * np.log(v)) if k.endswith("_F") else np.log(v) / (0.875 - 1)
    for pre in ("ball", "cp", "pauli"):
        if pre + "_F" in res:
            res[pre + "_BSlcb"] = max(res[pre + "_D12"], res[pre + "_D78"])
    return res


PS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]


def truth(p):
    _, _, J1, J0 = gates(p)
    return dict(F=fhat(J1, J0), BS=dbs(J1, J0), D12=-2 * np.log(fhat(J1, J0)), lmin_choi_state=float(np.linalg.eigvalsh(J0 / 2).min()))


def scan():
    runs = int(os.environ.get("RUNS", 10))
    grids = dict(ball=np.arange(4, 15.01, 0.5), cp=np.arange(2, 11.01, 0.5), pauli=np.arange(1, 10.01, 0.5))
    for p in PS:
        fn = os.path.join(OUT, f"scan_p{p:.0e}.json")
        if os.path.exists(fn):
            continue
        rng = np.random.default_rng(int(1e6 * p) + 5)
        tq = truth(p)
        rec = dict(p=p, truth=tq, runs=runs)
        for w, g in grids.items():
            rows = []
            for e in g:
                N = 10 ** e
                rr = [run_once(p, N, rng, (w,)) for _ in range(runs)]
                rows.append(dict(N=N, **{k: [x[k] for x in rr] for k in rr[0]}))
                med = np.median([x[w + "_D12"] for x in rr]) / tq["D12"]
                if med > 0.95 and e > g[0] + 2:   # converged: stop scanning this pipeline
                    break
            rec[w] = rows
            print(f"scan p={p:.0e} {w} done", flush=True)
        json.dump(rec, open(fn, "w"))


def refine():
    """Denser grid (1/8 decade) and more runs (REF_RUNS, default 50) per pipeline around the shot budgets that
    certify -ln F-hat to eta in [0.025, 0.1] (window located from the coarse scan, padded by 1/2 decade).
    Same record layout as scan(); one file per p."""
    runs = int(os.environ.get("REF_RUNS", 50))
    for ip, p in enumerate(PS):
        fn = os.path.join(OUT, f"refine_p{p:.0e}.json")
        if os.path.exists(fn):
            continue
        t0 = time.time()
        rng = np.random.default_rng(777000 + ip)
        rec = dict(p=p, truth=truth(p), runs=runs)
        for w in ("pauli", "cp", "ball"):
            lo, hi = n_at_eta(p, w, 0.1), n_at_eta(p, w, 0.025)
            lo = lo if lo is not None else n_at_eta(p, w, 0.2)
            g = np.arange(np.floor(8 * (np.log10(lo) - 1.0)) / 8, np.log10(hi) + 0.51, 0.125)
            rows = []
            for e in g:
                N = 10 ** e
                rr = [run_once(p, N, rng, (w,)) for _ in range(runs)]
                rows.append(dict(N=N, **{k: [x[k] for x in rr] for k in rr[0]}))
            rec[w] = rows
            print(f"refine p={p:.0e} {w} {len(g)} pts {time.time() - t0:.0f}s", flush=True)
        json.dump(rec, open(fn, "w"))


ETA_COV = 0.05   # coverage is checked at the shot budget that certifies -ln F-hat to accuracy ETA_COV


def n_at_eta(p, w, eta=ETA_COV, q=0.5):
    """per-setting N (on the scan grid, first grid point at or above the log-interpolated N_req) at which the
    q-quantile over runs of ln(F_ucb / F_true) is <= eta, from results/s2b_raw/scan_p*.json."""
    d = json.load(open(os.path.join(OUT, f"scan_p{p:.0e}.json")))
    F = d["truth"]["F"]
    Ns = np.array([r["N"] for r in d[w]])
    acc = np.array([np.quantile(np.log(np.array(r[w + "_F"]) / F), q) for r in d[w]])
    ok = acc <= eta
    j = len(ok) - 1
    if not ok[j]:
        return None
    while j > 0 and ok[j - 1]:
        j -= 1
    return float(Ns[j])


def coverage():
    """Empirical coverage of the certified F-hat UCB and BS LCB for pipelines (a) ball, (a'') C-P box and (b)
    known-frame Pauli, at every p, at the per-setting N where the median run certifies -ln F-hat to ETA_COV.
    One file per (p, pipeline), written as each finishes."""
    runs = int(os.environ.get("COV_RUNS", 1000))
    for p in PS:
        tq = truth(p)
        for iw, w in enumerate(("pauli", "cp", "ball")):
            fn = os.path.join(OUT, f"cov_p{p:.0e}_{w}.json")
            if os.path.exists(fn):
                continue
            N = n_at_eta(p, w)
            if N is None:
                print(f"coverage p={p:.0e} {w}: no N reached eta", flush=True)
                continue
            t0 = time.time()
            rng = np.random.default_rng(4242 + 17 * PS.index(p) + 1000 * iw)
            rr = [run_once(p, N, rng, (w,)) for _ in range(runs)]
            Fv = np.array([x[w + "_F"] for x in rr]); Bv = np.array([x[w + "_BSlcb"] for x in rr])
            nan = int(np.sum(~np.isfinite(Fv)) + np.sum(~np.isfinite(Bv)))
            d = dict(p=p, pipeline=w, N_per_setting=N, n_settings=(3 if w == "pauli" else 18), eta=ETA_COV,
                     truth=tq, runs=runs, alpha=ALPHA, n_nan=nan,
                     cov_F=float(np.mean(Fv >= tq["F"])), cov_BS=float(np.mean(Bv <= tq["BS"])),
                     cov_both=float(np.mean((Fv >= tq["F"]) & (Bv <= tq["BS"]))),
                     med_F=float(np.nanmedian(Fv)), med_BS=float(np.nanmedian(Bv)),
                     F_ucb=Fv.tolist(), BS_lcb=Bv.tolist(), time=time.time() - t0)
            json.dump(d, open(fn, "w"))
            print(f"coverage p={p:.0e} {w} N={N:.2e} covF={d['cov_F']:.4f} covBS={d['cov_BS']:.4f} "
                  f"{d['time']:.0f}s", flush=True)


# ----------------------------------------------------------------------------- contrast: E1 and F_U through (a)
def contrast():
    """Pipeline (a) (Pauli tomography + Hoeffding trace-norm ball, 18 settings) applied to the hockey-stick
    E1 = (1/2)||.||_diamond (certified LCB) and the Uhlmann channel fidelity F_U (certified UCB), on the pair
    delta = 1 rad, where both are O(1) at every p.  (At delta = 2 sqrt(p) both vanish with p, so an additive
    accuracy target would be met by the trivial bound.)  One file per p."""
    runs = int(os.environ.get("CON_RUNS", 20))
    grid = np.arange(1.0, 9.01, 0.25)
    sE, sU = CertSDP("E1", "ball"), CertSDP("FU", "ball")
    for ip, p in enumerate(PS):
        fn = os.path.join(OUT, f"contrast_p{p:.0e}.json")
        if os.path.exists(fn):
            continue
        t0 = time.time()
        rng = np.random.default_rng(9090 + ip)
        J1, J0 = choi_gate(p, 1.0), choi_gate(p, 0.0)
        tq = dict(E1=hockey1(J1, J0), FU=uhlmann(J1, J0), F=fhat(J1, J0))
        rows = []
        for e in grid:
            N = 10 ** e
            f1s, f0s = sample(J1, N, rng, runs), sample(J0, N, rng, runs)
            r = region_radii(N, ALPHA, 2 * K_SET)
            E, U = [], []
            for i in range(runs):
                L1, L0 = lin_inv(f1s[i]), lin_inv(f0s[i])
                E.append(max(0.0, sE.solve(L1, L0, r["eps_tr"])[0] - SAFETY))
                U.append(min(1.0, sU.solve(L1, L0, r["eps_tr"])[0] + SAFETY))
            rows.append(dict(N=N, eps_tr=r["eps_tr"], E1_lcb=E, FU_ucb=U))
        json.dump(dict(p=p, delta=1.0, truth=tq, runs=runs, alpha=ALPHA, rows=rows), open(fn, "w"))
        print(f"contrast p={p:.0e} done {time.time() - t0:.0f}s", flush=True)


# ============================================================================= CONVERSE SIDE
# A converse (xi <= -2 ln F-hat, Stein <= D_BS, P_succ <= (1+E1)/2, error lower bounds through F_U) is certified
# by a LOWER confidence bound on F-hat / F_U and an UPPER confidence bound on D_BS / E1.  (The SDPs above give the
# opposite, convex-tractable side: max_R F-hat, min_R D, which bound distinguishability from below.)
#  (a) Theorem C (theorems.md 7.1) on the Hoeffding trace-norm ball around the linear-inversion estimate J_LI:
#      ||J_i - J_LI,i||_1 <= eps_tr on the 95% event; J_LI is exactly TP, so Lemma J gives ||.||_inf <= eps_tr/2 and
#      F-hat(true) >= (1 - kappa) F-hat(J_LI,1, J_LI,0),  kappa = eps_tr / (2 lambda_hat),  lambda_hat = min_i lambda_min(J_LI,i).
#      Vacuous (LCB = 0) when kappa >= 1.
#  (b) known-frame Pauli polytope (same data model as above): F-hat is jointly concave and D_BS jointly convex in the
#      Choi pair, and the pair is affine in (q1, q0), so min F-hat and max D_BS over the product polytope are attained
#      at pairs of vertices; both are computed exactly by vertex enumeration.
_MS = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)     # flip probabilities s_j = sum of anticommuting rates


def pauli_vertices(lo, hi):
    A = np.vstack([_MS, -_MS, -np.eye(3), np.ones((1, 3))])
    b = np.concatenate([hi, -lo, np.zeros(3), [1.0]])
    V = []
    from itertools import combinations
    for c in combinations(range(len(b)), 3):
        M = A[list(c)]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        q = np.linalg.solve(M, b[list(c)])
        if np.all(A @ q <= b + 1e-12) and not any(np.allclose(q, v, atol=1e-14, rtol=0) for v in V):
            V.append(q)
    return np.array(V)


def _chois_from_q(B, Q):
    return [(1 - q.sum()) * B[0] + sum(q[j] * B[j + 1] for j in range(3)) for q in Q]


_FR = {}


def lcb_once(p, N, rng, w):
    U1, U0, J1, J0 = gates(p)
    if w == "thmC":
        f1, f0 = sample(J1, N, rng, 1)[0], sample(J0, N, rng, 1)[0]
        L1, L0 = lin_inv(f1), lin_inv(f0)
        eps = region_radii(N, ALPHA, 2 * K_SET)["eps_tr"]
        lam = min(np.linalg.eigvalsh(L1).min(), np.linalg.eigvalsh(L0).min())
        kap = eps / (2 * lam) if lam > 0 else np.inf
        Fc = fhat(L1, L0) if lam > 0 else np.nan
        return dict(F_lcb=max(0.0, (1 - kap) * Fc - 1e-10) if kap < 1 else 0.0, kappa=float(kap), F_center=float(Fc))
    if p not in _FR:
        _FR[p] = (frame_chois(U1), frame_chois(U0))
    B1, B0 = _FR[p]
    a = ALPHA / 6
    k1 = rng.binomial(int(N), flip_probs(J1, U1)); k0 = rng.binomial(int(N), flip_probs(J0, U0))
    lo1, hi1 = clopper_pearson(k1, N, a); lo0, hi0 = clopper_pearson(k0, N, a)
    C1, C0 = _chois_from_q(B1, pauli_vertices(lo1, hi1)), _chois_from_q(B0, pauli_vertices(lo0, hi0))
    Fmin, Bmax = 1.0, 0.0
    for A_ in C1:
        for B_ in C0:
            Fmin = min(Fmin, fhat(A_, B_))
            Bmax = max(Bmax, dbs(A_, B_))
    return dict(F_lcb=max(0.0, Fmin - 1e-10), BS_ucb=float(Bmax + 1e-10), nvert=[len(C1), len(C0)])


LCB_GRIDS = dict(thmC=(np.arange(7.0, 15.01, 0.125), 100), vert=(np.arange(2.0, 9.01, 0.125), 50))


def lcb_scan():
    """converse-side certificates (a) Theorem C ball, (b) Pauli polytope; per-run records; one file per p."""
    for ip, p in enumerate(PS):
        fn = os.path.join(OUT, f"lcb_p{p:.0e}.json")
        if os.path.exists(fn):
            continue
        t0 = time.time()
        rng = np.random.default_rng(55500 + ip)
        rec = dict(p=p, truth=truth(p), alpha=ALPHA)
        for w, (g, runs) in LCB_GRIDS.items():
            rows = []
            for e in g:
                rr = [lcb_once(p, 10 ** e, rng, w) for _ in range(runs)]
                rows.append(dict(N=10 ** e, **{k: [x[k] for x in rr] for k in rr[0]}))
            rec[w] = rows
            rec[w + "_runs"] = runs
            print(f"lcb p={p:.0e} {w} done {time.time() - t0:.0f}s", flush=True)
        json.dump(rec, open(fn, "w"))


def lcb_n_at_eta(p, w, eta=ETA_COV):
    d = json.load(open(os.path.join(OUT, f"lcb_p{p:.0e}.json")))
    F = d["truth"]["F"]
    Ns = np.array([r["N"] for r in d[w]])
    acc = np.array([np.median(np.log(F / np.maximum(np.array(r["F_lcb"]), 1e-300))) for r in d[w]])
    ok = acc <= eta
    j = len(ok) - 1
    if not ok[j]:
        return None
    while j > 0 and ok[j - 1]:
        j -= 1
    return float(Ns[j])


def lcb_coverage():
    """1000 runs per (p, pipeline) at the per-setting N where the median run certifies -ln F-hat to ETA_COV."""
    runs = int(os.environ.get("COV_RUNS", 1000))
    for ip, p in enumerate(PS):
        tq = truth(p)
        for iw, w in enumerate(("vert", "thmC")):
            fn = os.path.join(OUT, f"lcbcov_p{p:.0e}_{w}.json")
            if os.path.exists(fn):
                continue
            N = lcb_n_at_eta(p, w)
            if N is None:
                continue
            t0 = time.time()
            rng = np.random.default_rng(88800 + 10 * ip + iw)
            rr = [lcb_once(p, N, rng, w) for _ in range(runs)]
            Fv = np.array([x["F_lcb"] for x in rr])
            d = dict(p=p, pipeline=w, N_per_setting=N, n_settings=(3 if w == "vert" else 18), eta=ETA_COV, truth=tq,
                     runs=runs, alpha=ALPHA, cov_F=float(np.mean(Fv <= tq["F"])), med_F=float(np.median(Fv)), F_lcb=Fv.tolist())
            if w == "vert":
                Bv = np.array([x["BS_ucb"] for x in rr])
                d.update(cov_BS=float(np.mean(Bv >= tq["BS"])), med_BS=float(np.median(Bv)), BS_ucb=Bv.tolist(),
                         cov_both=float(np.mean((Fv <= tq["F"]) & (Bv >= tq["BS"]))))
            d["time"] = time.time() - t0
            json.dump(d, open(fn, "w"))
            print(f"lcbcov p={p:.0e} {w} N={N:.2e} covF={d['cov_F']:.4f} {d['time']:.0f}s", flush=True)


def lcb_contrast():
    """Converse side of the hockey-stick / Uhlmann contrast, same tomography and Hoeffding event, pair delta = 1 rad:
       E1 UCB = (1/2)||J_LI,1 - J_LI,0||_diamond + eps_tr          (triangle inequality; theorems.md 7.3 with gamma = 1)
       F_U LCB = sqrt(1 - (P_hat + 2 sqrt(2 eps_fro))^2)            (purified distance P = sqrt(1 - F_U^2) is a metric;
         P_hat from the CPTP projections J^ of J_LI; ||J - J^||_1 <= 2||J - J^||_F <= 2||J - J_LI||_F <= 2 eps_fro;
         per input P <= sqrt(2 T) <= sqrt(||Phi - Phi^||_diamond) <= sqrt(2 eps_fro))."""
    runs = int(os.environ.get("CON_RUNS", 20))
    grid = np.arange(3.0, 12.01, 0.25)
    for ip, p in enumerate(PS):
        fn = os.path.join(OUT, f"lcbcon_p{p:.0e}.json")
        if os.path.exists(fn):
            continue
        t0 = time.time()
        rng = np.random.default_rng(66600 + ip)
        J1, J0 = choi_gate(p, 1.0), choi_gate(p, 0.0)
        tq = dict(E1=hockey1(J1, J0), FU=uhlmann(J1, J0), F=fhat(J1, J0))
        rows = []
        for e in grid:
            N = 10 ** e
            r = region_radii(N, ALPHA, 2 * K_SET)
            f1s, f0s = sample(J1, N, rng, runs), sample(J0, N, rng, runs)
            E, U = [], []
            for i in range(runs):
                L1, L0 = lin_inv(f1s[i]), lin_inv(f0s[i])
                E.append(min(1.0, 0.5 * diamond(L1 - L0) + r["eps_tr"]))
                P_hat = np.sqrt(max(0.0, 1 - uhlmann(cptp_project(L1), cptp_project(L0)) ** 2))
                Pu = P_hat + 2 * np.sqrt(2 * r["eps_fro"])
                U.append(float(np.sqrt(max(0.0, 1 - Pu ** 2))) if Pu < 1 else 0.0)
            rows.append(dict(N=N, eps_tr=r["eps_tr"], eps_fro=r["eps_fro"], E1_ucb=E, FU_lcb=U))
        json.dump(dict(p=p, delta=1.0, truth=tq, runs=runs, alpha=ALPHA, rows=rows), open(fn, "w"))
        print(f"lcbcon p={p:.0e} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"scan": scan, "refine": refine, "coverage": coverage, "contrast": contrast,
     "lcb_scan": lcb_scan, "lcb_coverage": lcb_coverage, "lcb_contrast": lcb_contrast}[sys.argv[1]]()
