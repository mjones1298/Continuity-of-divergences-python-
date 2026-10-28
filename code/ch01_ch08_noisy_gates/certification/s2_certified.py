"""S2: certified one-sided converse bounds from Pauli process-tomography data.

Confidence region (valid, conservative): two-sided Hoeffding per setting, union bound over the
36 binary settings of the two channels, alpha = 0.05:  |f_k - p_k| <= t = sqrt(ln(2*36/alpha)/(2N)).
 * box region   R_box  = {(J1,J2) CPTP : |tr(A_k J_i) - f_ik| <= t  for all k, i}
 * trace ball   R_ball = {(J1,J2) CPTP : ||J_i - J_LI,i||_1 <= eps_tr = t * sum_k ||D_k||_1}
   (J_LI - J_true = sum_k (f_k - p_k) D_k exactly, so the ball holds on the same event.)
Both contain the true pair with probability >= 0.95.
Certified one-sided bounds (direct convex optimisation over the region, qubit SDPs):
   F-hat  : UCB = max_R F-hat                     (F-hat jointly concave)
   BS     : LCB = max_{alpha in {1/2,7/8}} min_R D-hat_alpha  <= BS   (geometric Renyi monotone in alpha)
   E_1    : LCB = min_R (1/2)||J1-J2||_diamond
   Uhlmann: UCB = max_R F_U                        (jointly concave)
Validity direction: a converse is invalidated by an F too HIGH or a divergence too LOW.
usage: python3 s2_certified.py coverage <worker> <nworkers> | shots <worker> <nworkers>
"""
import os, sys, time, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
sys.path.insert(0, str(Path(HERE).resolve().parents[1] / "common"))  # certlib.py, qlib.py in code/common
from certlib import *  # noqa

ALPHA = 0.05
SAFETY = 1e-5     # added to UCBs / subtracted from LCBs to absorb SDP solver tolerance
OUT = os.path.join(HERE, "results", "s2_raw")
os.makedirs(OUT, exist_ok=True)

_S = {}


def sdp(obj, kind="box"):
    if (obj, kind) not in _S:
        _S[(obj, kind)] = CertSDP(obj, kind)
    return _S[(obj, kind)]


def truth(p, delta):
    J1, J0 = choi_gate(p, delta), choi_gate(p, 0.0)
    q = dict(F=fhat(J1, J0), BS=dbs(J1, J0), E1=hockey1(J1, J0), FU=uhlmann(J1, J0),
             D12=-2 * np.log(fhat(J1, J0)),
             D78=np.log(Q.qhat_alpha_arn(J1, J0, 0.875, 2, 2)) / (0.875 - 1),
             lmin=float(np.linalg.eigvalsh(J0).min()),
             lmin_sum=float(np.linalg.eigvalsh(J1 + J0).min()))
    return J1, J0, q


def certify(f1, f2, N, objs=("F", "Q7", "FU", "E1"), ball_F=False):
    r = region_radii(N, ALPHA, 2 * K_SET)
    out = {}
    for o in objs:
        v, st = sdp(o).solve(f1, f2, r["t"])
        out[o] = v
        out[o + "_st"] = st
    if ball_F:
        v, st = sdp("F", "ball").solve(lin_inv(f1), lin_inv(f2), r["eps_tr"])
        out["F_ball"] = v
    # certified quantities (one-sided)
    res = {}
    if "F" in out:
        res["F_ucb"] = min(1.0, out["F"] + SAFETY)
    if "Q7" in out:
        q7 = min(1.0, out["Q7"] + SAFETY)
        d78 = np.log(q7) / (0.875 - 1)
        d12 = -2 * np.log(res["F_ucb"]) if "F_ucb" in res else 0.0
        res["BS_lcb"] = max(d78, d12, 0.0)
    if "FU" in out:
        res["FU_ucb"] = min(1.0, out["FU"] + SAFETY)
    if "E1" in out:
        res["E1_lcb"] = max(0.0, out["E1"] - SAFETY)
    if ball_F:
        res["F_ucb_ball"] = min(1.0, out["F_ball"] + SAFETY)
    res["status"] = {o: out[o + "_st"] for o in objs}
    return res, r


def theorem_route(F_plug, lmin_sum_hat, r):
    """v5 Cor 10.7 + Cor 10.8 + Lemma 8.2 + Weyl, applied to the CPTP-projected LI pair (Jhat1, Jhat2):
    |F(true) - F(hat)| <= (11/2) sqrt(delta_rel);  delta_rel <= max_i ||J_i - Jhat_i||_diamond / nu(Omega')^2;
    nu(Omega')^2 >= lambda_min J(Omega') / 2 (Lemma 8.2, d=2);  Omega' = sum of all four channels.
    Frobenius projection onto CPTP is non-expansive: ||Jhat_i - J_i||_F <= eps_fro, hence
    diamond <= ||.||_1 <= 2 eps_fro and lambda_min J(Omega') >= 2 lambda_min(Jhat1+Jhat2) - 2 eps_fro.
    Returns the UCB on F-hat (1.0 when vacuous: floor not certified or delta_rel >= 1/2)."""
    lam = 2 * lmin_sum_hat - 2 * r["eps_fro"]
    if lam <= 0:
        return 1.0
    drel = 2 * r["eps_fro"] / (lam / 2)
    if drel >= 0.5:
        return 1.0
    return min(1.0, F_plug + 5.5 * np.sqrt(drel))


def theorem_N_required(tq, target_ucb, alpha=ALPHA):
    """deterministic version (plug-in = truth): smallest N with theorem-route UCB <= target."""
    for e in np.arange(2, 30.01, 0.05):
        r = region_radii(10 ** e, alpha, 2 * K_SET)
        if theorem_route(tq["F"], tq["lmin_sum"], r) <= target_ucb:
            return 10 ** e
    return np.inf


# ----------------------------------------------------------------------------- coverage
COV_CFGS = [(1e-2, 1e6), (1e-3, 1e8), (1e-4, 1e10), (1e-3, 1e10)]
COV_RUNS = int(os.environ.get("COV_RUNS", 1000))


def coverage(ic):
    p, N = COV_CFGS[ic]
    fn = os.path.join(OUT, f"cov_p{p:.0e}_N{N:.0e}.json")
    if os.path.exists(fn):
        return
    rng = np.random.default_rng(777 + ic)
    J1, J0, tq = truth(p, 2 * np.sqrt(p))
    f1s, f0s = sample(J1, N, rng, COV_RUNS), sample(J0, N, rng, COV_RUNS)
    recs = []
    t0 = time.time()
    for i in range(COV_RUNS):
        f1, f0 = f1s[i], f0s[i]
        res, r = certify(f1, f0, N, ball_F=True)
        A, B = cptp_project(lin_inv(f1)), cptp_project(lin_inv(f0))
        L1, L0 = lin_inv(f1), lin_inv(f0)
        res["F_plug"], res["BS_plug"] = fhat(A, B), dbs(A, B)
        lmin_hat = float(np.linalg.eigvalsh(L0).min())
        res["floor_lcb"] = lmin_hat - r["eps_op"]            # Weyl: lambda_min(J_true) >= this
        res["F_ucb_thm"] = theorem_route(res["F_plug"], float(np.linalg.eigvalsh(A + B).min()), r)
        # was the true pair inside the Hoeffding event?
        res["in_event"] = bool(np.all(np.abs(f1 - probs_plus(J1)) <= r["t"]) and
                               np.all(np.abs(f0 - probs_plus(J0)) <= r["t"]))
        recs.append(res)
    keys = ["F_ucb", "BS_lcb", "FU_ucb", "E1_lcb", "F_ucb_ball", "F_plug", "BS_plug", "floor_lcb", "F_ucb_thm"]
    arr = {k: [float(x[k]) for x in recs] for k in keys}
    arr["in_event"] = [x["in_event"] for x in recs]
    arr["status_bad"] = sum(any(s not in ("optimal", "optimal_inaccurate") for s in x["status"].values())
                            for x in recs)
    arr["status_inacc"] = sum(any(s == "optimal_inaccurate" for s in x["status"].values()) for x in recs)
    json.dump(dict(p=p, N=N, truth=tq, radii=r, runs=COV_RUNS, time=time.time() - t0, data=arr),
              open(fn, "w"))
    print(f"coverage p={p:.0e} N={N:.0e} done {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- shots vs p
SH_PS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
SH_NS = [10 ** e for e in np.arange(3.0, 12.01, 0.5)]
SH_RUNS = int(os.environ.get("SH_RUNS", 10))
SH_DELTAS = {"scaled": lambda p: 2 * np.sqrt(p), "fixed": lambda p: 1.0}


def shots(job):
    ip, fam = job
    p = SH_PS[ip]
    fn = os.path.join(OUT, f"shots_{fam}_p{p:.0e}.json")
    if os.path.exists(fn):
        return
    rng = np.random.default_rng(31 + 7 * ip + (0 if fam == "scaled" else 100))
    J1, J0, tq = truth(p, SH_DELTAS[fam](p))
    rows = []
    t0 = time.time()
    for N in SH_NS:
        f1s, f0s = sample(J1, N, rng, SH_RUNS), sample(J0, N, rng, SH_RUNS)
        rr = []
        for i in range(SH_RUNS):
            res, r = certify(f1s[i], f0s[i], N)
            A, B = cptp_project(lin_inv(f1s[i])), cptp_project(lin_inv(f0s[i]))
            res["F_plug"] = fhat(A, B)
            res["F_ucb_thm"] = theorem_route(res["F_plug"],
                                             float(np.linalg.eigvalsh(A + B).min()), r)
            res.pop("status")
            rr.append(res)
        rows.append(dict(N=N, radii=r, **{k: [float(x[k]) for x in rr] for k in rr[0]}))
        print(f"shots {fam} p={p:.0e} N={N:.1e} {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(p=p, fam=fam, delta=SH_DELTAS[fam](p), truth=tq, rows=rows), open(fn, "w"))


if __name__ == "__main__":
    mode, w, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if mode == "coverage":
        jobs = list(range(len(COV_CFGS)))
        fn = coverage
    else:
        jobs = [(ip, fam) for fam in ("scaled", "fixed") for ip in range(len(SH_PS))]
        fn = shots
    for j, job in enumerate(jobs):
        if j % nw == w:
            fn(job)
