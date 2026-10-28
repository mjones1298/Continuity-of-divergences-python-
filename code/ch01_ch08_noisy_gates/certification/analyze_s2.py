"""Summarise S2 (coverage, shots-vs-p) -> results/s2_summary.json."""
import os, sys, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
sys.path.insert(0, str(Path(HERE).resolve().parents[1] / "common"))  # certlib.py, qlib.py in code/common
from certlib import region_radii, K_SET  # noqa

R = os.path.join(HERE, "results", "s2_raw")


def cov_summary():
    rows = []
    for fn in sorted(glob.glob(os.path.join(R, "cov_*.json"))):
        d = json.load(open(fn))
        t, D = d["truth"], {k: np.array(v) for k, v in d["data"].items() if isinstance(v, list)}
        row = dict(p=d["p"], N=d["N"], runs=d["runs"], t=d["radii"]["t"], eps_tr=d["radii"]["eps_tr"],
                   eps_op=d["radii"]["eps_op"], truth=t, status_bad=d["data"]["status_bad"],
                   status_inaccurate=d["data"]["status_inacc"])
        cov = dict(F=D["F_ucb"] >= t["F"], BS=D["BS_lcb"] <= t["BS"], FU=D["FU_ucb"] >= t["FU"],
                   E1=D["E1_lcb"] <= t["E1"], F_ball=D["F_ucb_ball"] >= t["F"], F_thm=D["F_ucb_thm"] >= t["F"],
                   floor=D["floor_lcb"] <= t["lmin"])
        cov["all4"] = cov["F"] & cov["BS"] & cov["FU"] & cov["E1"]
        row["coverage"] = {k: float(np.mean(v)) for k, v in cov.items()}
        row["hoeffding_event"] = float(np.mean(D["in_event"]))
        n = len(D["F_ucb"])
        row["coverage_ci95_halfwidth_at_0.95"] = float(1.96 * np.sqrt(0.95 * 0.05 / n))
        med = lambda x: float(np.median(x))
        row["median"] = dict(F_ucb=med(D["F_ucb"]), F_ucb_ball=med(D["F_ucb_ball"]), F_ucb_thm=med(D["F_ucb_thm"]),
                             BS_lcb=med(D["BS_lcb"]), FU_ucb=med(D["FU_ucb"]), E1_lcb=med(D["E1_lcb"]),
                             F_plug=med(D["F_plug"]), BS_plug=med(D["BS_plug"]), floor_lcb=med(D["floor_lcb"]))
        row["nontrivial"] = dict(F=float(np.mean(D["F_ucb"] < 1 - 1e-6)), BS=float(np.mean(D["BS_lcb"] > 1e-6)),
                                 F_thm=float(np.mean(D["F_ucb_thm"] < 1)), floor=float(np.mean(D["floor_lcb"] > 0)))
        # naive plug-in used as if it were a one-sided bound
        row["plugin_as_bound_coverage"] = dict(F=float(np.mean(D["F_plug"] >= t["F"])),
                                               BS=float(np.mean(D["BS_plug"] <= t["BS"])))
        rows.append(row)
    return rows


def divform(name, v):
    v = np.asarray(v, float)
    if name in ("F", "FU", "F_thm"):
        return -2 * np.log(np.clip(v, 1e-300, 1))
    return v


def nreq(Ns, ratios, target=0.5):
    """smallest N (log-interp) such that median ratio >= target here and for all larger N."""
    ok = np.array([r >= target for r in ratios])
    if not ok[-1]:
        return float("inf")
    j = len(ok) - 1
    while j > 0 and ok[j - 1]:
        j -= 1
    if j == 0:
        return float(Ns[0])
    x0, x1 = np.log10(Ns[j - 1]), np.log10(Ns[j])
    r0, r1 = ratios[j - 1], ratios[j]
    if not np.isfinite(r0) or r1 == r0:
        return float(Ns[j])
    return float(10 ** (x0 + (target - r0) / (r1 - r0) * (x1 - x0)))


def shots_summary():
    out = {}
    keymap = dict(F=("F_ucb", "D12"), BS=("BS_lcb", "BS"), E1=("E1_lcb", "E1"), FU=("FU_ucb", "FU"),
                  F_thm=("F_ucb_thm", "D12"))
    for fn in sorted(glob.glob(os.path.join(R, "shots_*.json"))):
        d = json.load(open(fn))
        t = d["truth"]
        tv = dict(D12=t["D12"], BS=t["BS"], E1=t["E1"], FU=-2 * np.log(t["FU"]))
        Ns = [r["N"] for r in d["rows"]]
        res = dict(p=d["p"], fam=d["fam"], delta=d["delta"], truth=t, N=Ns, ratio={}, q10={}, q90={}, Nreq={})
        for q, (k, tk) in keymap.items():
            rat = [divform(q, r[k]) / tv[tk] for r in d["rows"]]
            res["ratio"][q] = [float(np.median(x)) for x in rat]
            res["q10"][q] = [float(np.quantile(x, 0.1)) for x in rat]
            res["q90"][q] = [float(np.quantile(x, 0.9)) for x in rat]
            res["Nreq"][q] = nreq(Ns, res["ratio"][q])
        # plug-in F (CPTP-projected LI): N with median within 10% of truth (two-sided accuracy, not a bound)
        pl = [np.median(np.abs(np.array(r["F_plug"]) - t["F"]) / t["F"]) for r in d["rows"]]
        res["plugin_F_relerr_median"] = [float(x) for x in pl]
        res["Nreq"]["F_plugin_10pc"] = nreq(Ns, [0.1 / max(x, 1e-12) for x in pl], 1.0)
        # analytic: Weyl floor certified to lambda_min/2, i.e. eps_op <= lambda_min/2
        e = np.arange(1, 30, 0.01)
        eo = np.array([region_radii(10 ** x, 0.05, 2 * K_SET)["eps_op"] for x in e])
        res["Nreq"]["weyl_floor_half"] = float(10 ** e[np.argmax(eo <= t["lmin"] / 2)])
        out[f"{d['fam']}|{d['p']:.0e}"] = res
    # exponent fits  N_req ~ p^{-a}
    fits = {}
    for fam in ("scaled", "fixed"):
        rs = [v for k, v in out.items() if v["fam"] == fam]
        if len(rs) < 2:
            continue
        ps = np.array([r["p"] for r in rs])
        for q in rs[0]["Nreq"]:
            n = np.array([r["Nreq"][q] for r in rs])
            m = np.isfinite(n)
            if m.sum() >= 2:
                a, b = np.polyfit(np.log10(ps[m]), np.log10(n[m]), 1)
                fits[f"{fam}|{q}"] = dict(exponent=float(-a), prefactor_log10=float(b), npts=int(m.sum()),
                                          Nreq={f"{p:.0e}": float(x) for p, x in zip(ps, n)})
            else:
                fits[f"{fam}|{q}"] = dict(exponent=None, npts=int(m.sum()),
                                          Nreq={f"{p:.0e}": float(x) for p, x in zip(ps, n)})
    return out, fits


if __name__ == "__main__":
    cov = cov_summary()
    sh, fits = shots_summary()
    json.dump(dict(coverage=cov, shots=sh, fits=fits), open(os.path.join(HERE, "results", "s2_summary.json"), "w"),
              indent=1)
    for r in cov:
        print(f"p={r['p']:.0e} N={r['N']:.0e} cov={ {k: round(v, 3) for k, v in r['coverage'].items()} } "
              f"event={r['hoeffding_event']:.3f} med={ {k: round(v, 4) for k, v in r['median'].items()} } "
              f"nontriv={r['nontrivial']} plugin={r['plugin_as_bound_coverage']}")
    for k, v in fits.items():
        print(k, v)


def s2b_summary():
    """pipelines: N_req (median certified/true >= 1/2) for D12 = -2 ln F_ucb and for the BS lower bound."""
    rows = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "s2b_raw", "scan_*.json"))):
        d = json.load(open(fn))
        t = d["truth"]
        r = dict(p=d["p"], truth=t)
        for w, nset in (("ball", 18), ("cp", 18), ("pauli", 3)):
            if w not in d:
                continue
            Ns = [x["N"] for x in d[w]]
            for q, tv in (("D12", t["D12"]), ("BSlcb", t["BS"])):
                rat = [float(np.median(np.array(x[f"{w}_{q}"]) / tv)) for x in d[w]]
                n = nreq(Ns, rat)
                r[f"{w}|{q}|Nreq_per_setting"] = n
                r[f"{w}|{q}|Nreq_total_per_channel"] = n * nset
                r[f"{w}|{q}|curve"] = [Ns, rat]
        rows[f"{d['p']:.0e}"] = r
    # Hoeffding box (a') from the s2 shots runs
    sh = os.path.join(HERE, "results", "s2_summary.json")
    if os.path.exists(sh):
        S = json.load(open(sh))["shots"]
        for k, v in S.items():
            if v["fam"] == "scaled" and f"{v['p']:.0e}" in rows:
                rr = rows[f"{v['p']:.0e}"]
                rr["hbox|D12|Nreq_per_setting"] = v["Nreq"]["F"]; rr["hbox|D12|Nreq_total_per_channel"] = 18 * v["Nreq"]["F"]
                rr["hbox|BSlcb|Nreq_per_setting"] = v["Nreq"]["BS"]; rr["hbox|BSlcb|Nreq_total_per_channel"] = 18 * v["Nreq"]["BS"]
    fits = {}
    ps = sorted(rows, key=float)
    for w in ("ball", "hbox", "cp", "pauli"):
        for q in ("D12", "BSlcb"):
            key = f"{w}|{q}|Nreq_total_per_channel"
            P = np.array([float(p) for p in ps if key in rows[p] and np.isfinite(rows[p][key])])
            N = np.array([rows[p][key] for p in ps if key in rows[p] and np.isfinite(rows[p][key])])
            if len(P) >= 2:
                a, b = np.polyfit(np.log10(P), np.log10(N), 1)
                fits[f"{w}|{q}"] = dict(exponent=float(-a), log10_prefactor=float(b),
                                        Nreq_total={f"{x:.0e}": float(y) for x, y in zip(P, N)})
    return rows, fits


# ----------------------------------------------------------------------------- accuracy-eta analysis of S2b
# Criterion: the certified UCB on F-hat reaches additive accuracy eta in -ln F-hat, i.e.
#   ln(F_ucb / F_true) <= eta,   in the q-quantile run (q = 0.5: median run; q = 0.75: three runs in four).
# For E1 (hockey-stick) the criterion is additive in E1:  E1_true - E1_lcb <= eta.
# For the Uhlmann fidelity it is the same as for F-hat:  ln(FU_ucb / FU_true) <= eta.
ETAS, QS, ETA0, Q0 = (0.1, 0.05, 0.025), (0.5, 0.75), 0.05, 0.5
C0, DELTA_F, C_N2 = 0.0246, 0.05, 0.0589          # theorems.md Prop N3 c0; N4 failure prob; Prop N2 constant
NSET = dict(ball=18, hbox=18, cp=18, pauli=3, E1=18, FU=18, thmC=18, vert=3, vertBS=3, E1u=18, FUl=18)
# converse side (certifies the converse): thmC, vert, vertBS, E1u, FUl.  Opposite side (SDP max F / min D): the rest.
SERIES = ("thmC", "vert", "vertBS", "E1u", "FUl", "ball", "hbox", "cp", "pauli", "E1", "FU")
B_BOOT = 2000


def _nreq_acc(Ns, acc, eta):
    """smallest N (log-log interpolation of accuracy vs N) with acc <= eta here and at all larger grid N."""
    acc = np.asarray(acc, float)
    ok = acc <= eta
    if not ok[-1]:
        return float("inf")
    j = len(ok) - 1
    while j > 0 and ok[j - 1]:
        j -= 1
    if j == 0:
        return float("nan")     # already reached at the first grid point: unresolved
    x0, x1 = np.log10(Ns[j - 1]), np.log10(Ns[j])
    a0, a1 = acc[j - 1], acc[j]
    if a1 <= 0 or a0 <= 0 or a0 == a1:
        return float(Ns[j])
    y0, y1, yt = np.log(a0), np.log(a1), np.log(eta)
    return float(10 ** (x0 + (yt - y0) / (y1 - y0) * (x1 - x0)))


def _err_arrays():
    """per (p, pipeline): (Ns, list of per-run error arrays) for every pipeline and the contrast quantities."""
    E = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "s2b_raw", "scan_*.json"))):
        d = json.load(open(fn))
        F = d["truth"]["F"]
        for w in ("ball", "cp", "pauli"):
            if w in d:
                E[(d["p"], w)] = ([x["N"] for x in d[w]], [np.log(np.array(x[f"{w}_F"]) / F) for x in d[w]])
    if os.environ.get("USE_REFINE", "0") == "1":    # refined scan (1/8 decade, 50 runs); off by default: only p = 1e-2 completed
        for fn in sorted(glob.glob(os.path.join(HERE, "results", "s2b_raw", "refine_*.json"))):
            d = json.load(open(fn))
            F = d["truth"]["F"]
            for w in ("ball", "cp", "pauli"):
                if w in d:
                    E[(d["p"], w)] = ([x["N"] for x in d[w]], [np.log(np.array(x[f"{w}_F"]) / F) for x in d[w]])
    for fn in sorted(glob.glob(os.path.join(R, "shots_scaled_*.json"))):     # (a') Hoeffding box, from S2
        d = json.load(open(fn))
        F = d["truth"]["F"]
        E[(d["p"], "hbox")] = ([x["N"] for x in d["rows"]], [np.log(np.array(x["F_ucb"]) / F) for x in d["rows"]])
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "s2b_raw", "contrast_*.json"))):
        d = json.load(open(fn))
        t = d["truth"]
        E[(d["p"], "E1")] = ([x["N"] for x in d["rows"]], [t["E1"] - np.array(x["E1_lcb"]) for x in d["rows"]])
        E[(d["p"], "FU")] = ([x["N"] for x in d["rows"]], [np.log(np.array(x["FU_ucb"]) / t["FU"]) for x in d["rows"]])
    # converse side: lower confidence bounds on F-hat / F_U, upper confidence bounds on D_BS / E1
    big = 1e3          # stands for a vacuous certificate (F_lcb = 0 or D_BS ucb = inf)
    lnr = lambda a, b: np.where(np.asarray(b) > 0, np.log(a / np.maximum(np.asarray(b, float), 1e-300)), big)
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "s2b_raw", "lcb_p*.json"))):
        d = json.load(open(fn))
        t = d["truth"]
        for w in ("thmC", "vert"):
            E[(d["p"], w)] = ([x["N"] for x in d[w]], [lnr(t["F"], x["F_lcb"]) for x in d[w]])
        E[(d["p"], "vertBS")] = ([x["N"] for x in d["vert"]],
                                 [np.minimum(np.array(x["BS_ucb"], float) - t["BS"], big) for x in d["vert"]])
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "s2b_raw", "lcbcon_p*.json"))):
        d = json.load(open(fn))
        t = d["truth"]
        E[(d["p"], "E1u")] = ([x["N"] for x in d["rows"]], [np.array(x["E1_ucb"]) - t["E1"] for x in d["rows"]])
        E[(d["p"], "FUl")] = ([x["N"] for x in d["rows"]], [lnr(t["FU"], x["FU_lcb"]) for x in d["rows"]])
    return E


def _fit(P, N):
    """OLS of log10 N on log10 p; returns slope, intercept, slope standard error, 95% t half-width."""
    from scipy.stats import t as tdist
    x, y = np.log10(P), np.log10(N)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    n = len(x)
    if n <= 2:
        return float(coef[0]), float(coef[1]), float("nan"), float("nan")
    s2 = np.sum((y - A @ coef) ** 2) / (n - 2)
    se = float(np.sqrt(s2 / np.sum((x - x.mean()) ** 2)))
    return float(coef[0]), float(coef[1]), se, float(tdist.ppf(0.975, n - 2) * se)


def s2b_eta():
    E = _err_arrays()
    rng = np.random.default_rng(20261027)
    ps_all = sorted({k[0] for k in E}, reverse=True)
    out = dict(criterion="q-quantile over runs of ln(F_ucb/F_true) <= eta (E1: E1_true - E1_lcb <= eta)",
               etas=list(ETAS), quantiles=list(QS), primary=dict(eta=ETA0, q=Q0), n_settings=NSET,
               bootstrap_reps=B_BOOT, nreq={}, fits={}, eta_exponent={}, bounds={}, ratio_a_over_b={})
    for w in SERIES:
        for q in QS:
            accB = {}      # bootstrap accuracy curves per p, shape (B, n_rows): runs resampled within each N
            for p in ps_all:
                if (p, w) in E:
                    Ns, errs = E[(p, w)]
                    accB[p] = np.array([np.quantile(e[rng.integers(0, len(e), (B_BOOT, len(e)))], q, axis=1)
                                        for e in errs]).T
            for eta in ETAS:
                key = f"{w}|q{q}|eta{eta}"
                P, Nt, boots = [], [], []
                for p in ps_all:
                    if (p, w) not in E:
                        continue
                    Ns, errs = E[(p, w)]
                    acc = [np.quantile(e, q) for e in errs]
                    n = _nreq_acc(Ns, acc, eta)
                    out["nreq"].setdefault(key, {})[f"{p:.0e}"] = dict(
                        per_setting=n, total_per_channel=n * NSET[w], runs=int(len(errs[0])))
                    if np.isfinite(n):
                        bs = [_nreq_acc(Ns, a, eta) * NSET[w] for a in accB[p]]
                        P.append(p); Nt.append(n * NSET[w]); boots.append(np.array(bs))
                        out["nreq"][key][f"{p:.0e}"]["total_boot95"] = [float(np.nanquantile(bs, .025)),
                                                                        float(np.nanquantile(bs, .975))]
                if len(P) >= 2:
                    P, Nt = np.array(P), np.array(Nt)
                    sl, ic, se, hw = _fit(P, Nt)
                    B = np.array(boots)                    # (np, B)
                    good = np.all(np.isfinite(B), axis=0)
                    bsl = np.array([np.polyfit(np.log10(P), np.log10(B[:, b]), 1)[0] for b in np.where(good)[0]])
                    out["fits"][key] = dict(slope=sl, intercept_log10=ic, slope_se=se, slope_ci95_ols=[sl - hw, sl + hw],
                                            slope_ci95_boot=[float(np.quantile(bsl, .025)), float(np.quantile(bsl, .975))],
                                            boot_valid=int(good.sum()), npts=int(len(P)),
                                            ps=[float(x) for x in P], N_total=[float(x) for x in Nt])
            # eta exponent (N ~ eta^-b) per p
            for p in ps_all:
                ks = [(eta, out["nreq"].get(f"{w}|q{q}|eta{eta}", {}).get(f"{p:.0e}", {}).get("per_setting", np.nan)) for eta in ETAS]
                ks = [(e, n) for e, n in ks if np.isfinite(n)]
                if len(ks) >= 2:
                    b = -np.polyfit(np.log10([k[0] for k in ks]), np.log10([k[1] for k in ks]), 1)[0]
                    out["eta_exponent"].setdefault(f"{w}|q{q}", {})[f"{p:.0e}"] = float(b)
    for p in ps_all:
        lt = p / 4
        out["bounds"][f"{p:.0e}"] = {
            f"eta{eta}": dict(N3_c0_over_eta2_lt=C0 / (eta ** 2 * lt), N3_certifier_reduction=4 * C0 / (eta ** 2 * lt),
                              N4_3ln2overdf_over_eta2_lt=3 * np.log(2 / DELTA_F) / (eta ** 2 * lt),
                              N2_incoherent_ball_eps_4etalt=C_N2 * (1 - p) ** 2 / (4 * eta * lt) ** 2)
            for eta in ETAS}
        for q in QS:
            for eta in ETAS:
                for side, (wa, wb) in (("converse", ("thmC", "vert")), ("opposite", ("ball", "pauli"))):
                    a = out["nreq"].get(f"{wa}|q{q}|eta{eta}", {}).get(f"{p:.0e}", {}).get("total_per_channel", np.nan)
                    b = out["nreq"].get(f"{wb}|q{q}|eta{eta}", {}).get(f"{p:.0e}", {}).get("total_per_channel", np.nan)
                    out["ratio_a_over_b"].setdefault(f"{side}|q{q}|eta{eta}", {})[f"{p:.0e}"] = dict(
                        ratio=float(a / b), ratio_times_lt=float(a / b * lt))
    # coverage (1000-run files)
    from scipy.stats import beta as _beta
    cov = []
    files = [(f, "opposite") for f in glob.glob(os.path.join(HERE, "results", "s2b_raw", "cov_p*_*.json"))] + \
            [(f, "converse") for f in glob.glob(os.path.join(HERE, "results", "s2b_raw", "lcbcov_p*_*.json"))]
    for fn, side in sorted(files):
        d = json.load(open(fn))
        n = d["runs"]
        row = {k: d[k] for k in ("p", "pipeline", "N_per_setting", "n_settings", "eta", "runs", "alpha", "n_nan",
                                 "cov_F", "cov_BS", "cov_both", "med_F", "med_BS") if k in d}
        row["side"] = side
        row["truth"] = d["truth"]
        for k in ("cov_F", "cov_BS", "cov_both"):
            if k not in d:
                continue
            x = round(d[k] * n)
            row[k + "_cp95"] = [float(_beta.ppf(0.025, x, n - x + 1)) if x > 0 else 0.0,
                                float(_beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0]
        row["med_acc_F"] = float(abs(np.log(d["med_F"] / d["truth"]["F"]))) if d["med_F"] > 0 else float("inf")
        cov.append(row)
    out["coverage"] = sorted(cov, key=lambda r: (r["side"], r["pipeline"], -r["p"]))
    con = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "s2b_raw", "lcbcon_p*.json"))):
        d = json.load(open(fn))
        con[f"{d['p']:.0e}"] = dict(truth=d["truth"], runs=d["runs"], delta=d["delta"])
    out["contrast_truth"] = con
    return out


if __name__ == "__main__":
    b_rows, b_fits = s2b_summary()
    json.dump(dict(rows=b_rows, fits=b_fits, eta=s2b_eta()),
              open(os.path.join(HERE, "results", "s2b_summary.json"), "w"), indent=1)
    for k, v in b_fits.items():
        print("PIPELINE", k, v)
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "s2b_raw", "cov_*.json"))):
        print("PIPELINE-COV", json.load(open(fn)))
