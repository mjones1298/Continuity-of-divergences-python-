"""Summarise S1 raw results -> results/s1_summary.json (+ printed tables)."""
import os, sys, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
sys.path.insert(0, str(Path(HERE).resolve().parents[1] / "common"))  # certlib.py, qlib.py in code/common
from certlib import choi_gate, fhat, dbs, hockey1, uhlmann  # noqa

PS = [1e-2, 1e-3, 1e-4]
NS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
ESTS = ["LI", "MLE", "HMLE0.5", "HMLE0.05", "LR1", "LR2", "MLEthr0", "MLEthr1e-09", "MLEthr1e-06"]
QS = ["F", "BS", "E1", "FU"]
TOL = dict(F=1e-9, BS=1e-9, E1=1e-7, FU=1e-7)    # |error| below TOL counts as "exact"
# Validity direction: fidelities (F, FU) overclaim when TOO HIGH; divergences (BS, E1) when TOO LOW.
OVER_SIGN = dict(F=+1, FU=+1, BS=-1, E1=-1)


def truths():
    T = {}
    for p in PS:
        J1, J0 = choi_gate(p, 2 * np.sqrt(p)), choi_gate(p, 0.0)
        T[p] = {"delta": dict(F=fhat(J1, J0), BS=dbs(J1, J0), E1=hockey1(J1, J0), FU=uhlmann(J1, J0)),
                "self": dict(F=1.0, BS=0.0, E1=0.0, FU=1.0)}
    return T


def stats(x, t, q):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return None
    err = x - t
    s = OVER_SIGN[q]
    over = np.mean(s * err > TOL[q])
    loose = np.mean(-s * err > TOL[q])
    fin = np.isfinite(x)
    d = dict(n=int(len(x)), median=float(np.median(x)), q10=float(np.quantile(x, 0.1)),
             q90=float(np.quantile(x, 0.9)),
             mean=float(np.mean(x)) if fin.all() else float("inf"),
             bias=float(np.mean(err)) if fin.all() else float("inf"),
             median_err=float(np.median(err)), err_q10=float(np.quantile(err, 0.1)),
             err_q90=float(np.quantile(err, 0.9)),
             overclaim=float(over), loose=float(loose), frac_inf=float(np.mean(~fin)))
    if q == "F":
        d["frac_zero"] = float(np.mean(x < 1e-8))
    return d


if __name__ == "__main__":
    T = truths()
    out = dict(truth={str(p): T[p] for p in PS}, table=[])
    for p in PS:
        for N in NS:
            fn = os.path.join(HERE, "results", "s1_raw", f"p{p:.0e}_N{N:.0e}.npz")
            if not os.path.exists(fn):
                continue
            D = np.load(fn)
            for e in ESTS:
                for pair in ("delta", "self"):
                    for q in QS:
                        k = f"{e}|{pair}|{q}"
                        if k not in D.files:
                            continue
                        s = stats(D[k], T[p][pair][q], q)
                        if s is None:
                            continue
                        row = dict(p=p, N=N, est=e, pair=pair, q=q, truth=T[p][pair][q], **s)
                        k1 = f"{e}|{pair}|lmin2"
                        if k1 in D.files:
                            row["frac_rankdef_2nd"] = float(np.mean(D[k1] < 1e-12))
                            row["frac_rankdef_any"] = float(np.mean(np.minimum(D[k1], D[f"{e}|{pair}|lmin1"]) < 1e-12))
                        out["table"].append(row)
    json.dump(out, open(os.path.join(HERE, "results", "s1_summary.json"), "w"), indent=1)
    # compact printout
    for pair in ("delta", "self"):
        for q in QS:
            print(f"\n=== pair={pair} q={q} : median [q10,q90] | over / loose | inf")
            for r in out["table"]:
                if r["pair"] == pair and r["q"] == q and r["est"] in ("LI", "MLE", "HMLE0.5", "HMLE0.05", "LR1", "LR2"):
                    print(f"p={r['p']:.0e} N={r['N']:.0e} {r['est']:9s} {r['median']:.4g} [{r['q10']:.4g},{r['q90']:.4g}] "
                          f"| {r['overclaim']:.2f}/{r['loose']:.2f} | inf {r['frac_inf']:.2f} "
                          f"rankdef {r.get('frac_rankdef_any', float('nan')):.2f}")
