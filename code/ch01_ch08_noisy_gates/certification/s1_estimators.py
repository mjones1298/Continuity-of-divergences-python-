"""S1: plug-in estimates of F-hat, BS, E_1, Uhlmann from simulated Pauli process tomography.
usage: python3 s1_estimators.py <worker> <nworkers>      -> results/s1_raw/p{p}_N{N}.npz
Pairs per repetition:
  'delta': (est N_{p,delta}, est N_p)   delta = 2 sqrt(p)  [e^{-i delta Z/2} H]   (true F-hat ~ 0.71)
  'self' : (est N_p, independent est N_p)                                       (true F-hat = 1)
"""
import os, sys, time, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # certlib.py, qlib.py in code/common
from certlib import *  # noqa
from certlib import _psd_proj

PS = [1e-2, 1e-3, 1e-4]
NS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
REPS = int(os.environ.get("REPS", 200))
REPS_SLOW = min(20, REPS)          # E_1 and Uhlmann (input optimisation) on the first 50 repetitions
POLISH_THR = 1e-7       # MLE interior-point fuzz threshold (sensitivity reported separately)
ESTS = ["LI", "MLE", "HMLE0.5", "HMLE0.05", "LR1", "LR2"]
SLOW_ESTS = ["LI", "MLE", "HMLE0.5", "LR2"]   # E_1 / Uhlmann evaluated for these only
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "s1_raw"))
os.makedirs(OUT, exist_ok=True)


def run_config(ip, iN):
    p, N = PS[ip], NS[iN]
    fn = os.path.join(OUT, f"p{p:.0e}_N{N:.0e}.npz")
    if os.path.exists(fn):
        return
    rng = np.random.default_rng(1000 * ip + iN + 12345)
    d = 2 * np.sqrt(p)
    Jt = {"d": choi_gate(p, d), "a": choi_gate(p, 0.0), "b": choi_gate(p, 0.0)}
    freqs = {k: sample(J, N, rng, REPS) for k, J in Jt.items()}
    mle, hmle = MLE(), MLE(hedged=True)
    est = {e: {} for e in ESTS}
    raw_mle = {}
    t0 = time.time()
    for k in Jt:
        f = freqs[k]
        LI = cptp_project(lin_inv(f))
        est["LI"][k] = LI
        M = np.array([mle.fit(fi) for fi in f])
        raw_mle[k] = M
        est["MLE"][k] = np.array([polish(m, POLISH_THR) for m in M])
        est["HMLE0.5"][k] = np.array([tp_congruence(_psd_proj(hmle.fit(fi, 0.5 / N))) for fi in f])
        est["HMLE0.05"][k] = np.array([tp_congruence(_psd_proj(hmle.fit(fi, 0.05 / N))) for fi in f])
        est["LR1"][k] = np.array([lowrank_mle(f[i], 1, LI[i]) for i in range(REPS)])
        est["LR2"][k] = np.array([lowrank_mle(f[i], 2, LI[i]) for i in range(REPS)])
    t1 = time.time()
    res = {}
    for e in ESTS:
        for pair, (k1, k2) in (("delta", ("d", "a")), ("self", ("b", "a"))):
            A, B = est[e][k1], est[e][k2]
            F = np.array([fhat(A[i], B[i]) for i in range(REPS)])
            BS = np.array([dbs(A[i], B[i]) for i in range(REPS)])
            E1 = np.full(REPS, np.nan); FU = np.full(REPS, np.nan)
            for i in range(REPS_SLOW if e in SLOW_ESTS else 0):
                E1[i] = hockey1(A[i], B[i]); FU[i] = uhlmann(A[i], B[i])
            l1 = np.linalg.eigvalsh(A).min(axis=1); l2 = np.linalg.eigvalsh(B).min(axis=1)
            res[f"{e}|{pair}|F"] = F; res[f"{e}|{pair}|BS"] = BS
            res[f"{e}|{pair}|E1"] = E1; res[f"{e}|{pair}|FU"] = FU
            res[f"{e}|{pair}|lmin1"] = l1; res[f"{e}|{pair}|lmin2"] = l2
    # MLE polish sensitivity (F and BS only)
    for thr in (0.0, 1e-9, 1e-6):
        for pair, (k1, k2) in (("delta", ("d", "a")), ("self", ("b", "a"))):
            A = raw_mle[k1]; B = raw_mle[k2]
            if thr > 0:
                A = np.array([polish(m, thr) for m in A]); B = np.array([polish(m, thr) for m in B])
            res[f"MLEthr{thr:g}|{pair}|F"] = np.array([fhat(A[i], B[i]) for i in range(REPS)])
            res[f"MLEthr{thr:g}|{pair}|BS"] = np.array([dbs(A[i], B[i]) for i in range(REPS)])
    np.savez_compressed(fn, p=p, N=N, t_est=t1 - t0, t_q=time.time() - t1, **res)
    print(f"p={p:.0e} N={N:.0e} est {t1 - t0:.0f}s quant {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    w, nw = int(sys.argv[1]), int(sys.argv[2])
    cfgs = [(ip, iN) for iN in range(len(NS)) for ip in range(len(PS))]
    for j, (ip, iN) in enumerate(cfgs):
        if j % nw == w:
            run_config(ip, iN)
