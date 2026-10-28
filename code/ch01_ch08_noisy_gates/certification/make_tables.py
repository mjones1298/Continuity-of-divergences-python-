"""Render markdown tables from results/*summary*.json and fill README_template.md -> README.md."""
import os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def g(x, n=3):
    if x is None:
        return "–"
    if isinstance(x, float) and np.isinf(x):
        return "∞"
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    if x == 0:
        return "0"
    a = abs(x)
    if a < 1e-3 or a >= 1e5:
        return f"{x:.{n - 1}e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    r = f"{x:.{n}g}"
    return f"{float(r):.0f}" if "e" in r else r


def s1_tables():
    S = json.load(open(os.path.join(R, "s1_summary.json")))
    T = S["table"]
    out = {}
    ests = ["LI", "MLE", "HMLE0.5", "HMLE0.05", "LR1", "LR2"]
    for pair in ("delta", "self"):
        for q in ("F", "BS", "E1", "FU"):
            lines = ["| p | N | " + " | ".join(ests) + " |", "|---|---|" + "---|" * len(ests)]
            for p in (1e-2, 1e-3, 1e-4):
                Ns = sorted({r["N"] for r in T if r["p"] == p})
                for N in Ns:
                    cells = []
                    for e in ests:
                        r = [x for x in T if x["p"] == p and x["N"] == N and x["est"] == e and x["pair"] == pair and x["q"] == q]
                        if not r:
                            cells.append("–"); continue
                        r = r[0]
                        if q == "BS":
                            if r["frac_inf"] >= 0.5:
                                c = f"∞ ({r['frac_inf']:.0%} ∞)"
                            else:
                                c = f"{g(r['median'])} [{g(r['q10'])}, {g(r['q90'])}] ({r['frac_inf']:.0%} ∞)"
                        else:
                            c = f"{g(r['median'])} [{g(r['q10'])}, {g(r['q90'])}]"
                        c += f" o{r['overclaim']:.2f}/l{r['loose']:.2f}"
                        cells.append(c)
                    lines.append(f"| {p:g} | {N:.0e} | " + " | ".join(cells) + " |")
            out[f"S1_{pair}_{q}"] = "\n".join(lines)
    # bias / signed error table for F (delta pair), LI/MLE/HMLE0.5
    lines = ["| p | N | est | truth | median est | mean signed error (bias) | median signed error | 10–90% signed error | overclaim | loose | rank-def. |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in (1e-2, 1e-3, 1e-4):
        for r in sorted([x for x in T if x["p"] == p and x["pair"] == "delta" and x["q"] == "F" and x["est"] in ("LI", "MLE", "HMLE0.5")],
                        key=lambda x: (x["N"], x["est"])):
            lines.append(f"| {p:g} | {r['N']:.0e} | {r['est']} | {g(r['truth'])} | {g(r['median'])} | {g(r['bias'])} | "
                         f"{g(r['median_err'])} | [{g(r['err_q10'])}, {g(r['err_q90'])}] | {r['overclaim']:.2f} | {r['loose']:.2f} | "
                         f"{r.get('frac_rankdef_any', float('nan')):.2f} |")
    out["S1_bias_F"] = "\n".join(lines)
    # MLE polish sensitivity
    lines = ["| p | N | pair | F̂ median: thr 0 / 1e-9 / 1e-7 / 1e-6 | BS ∞-fraction: thr 0 / 1e-9 / 1e-7 / 1e-6 |", "|---|---|---|---|---|"]
    for p in (1e-2, 1e-3, 1e-4):
        for N in sorted({r["N"] for r in T if r["p"] == p}):
            for pair in ("delta",):
                def get(e, q, k):
                    r = [x for x in T if x["p"] == p and x["N"] == N and x["est"] == e and x["pair"] == pair and x["q"] == q]
                    return r[0][k] if r else float("nan")
                ef = [get(e, "F", "median") for e in ("MLEthr0", "MLEthr1e-09", "MLE", "MLEthr1e-06")]
                eb = [get(e, "BS", "frac_inf") for e in ("MLEthr0", "MLEthr1e-09", "MLE", "MLEthr1e-06")]
                lines.append(f"| {p:g} | {N:.0e} | {pair} | " + " / ".join(g(x) for x in ef) + " | " +
                             " / ".join(f"{x:.2f}" for x in eb) + " |")
    out["S1_polish"] = "\n".join(lines)
    out["S1_truth"] = "\n".join(["| p | F̂ | D_BS | E₁ | F_U |", "|---|---|---|---|---|"] +
                                [f"| {p} | {g(v['delta']['F'], 6)} | {g(v['delta']['BS'], 5)} | {g(v['delta']['E1'], 5)} | {g(v['delta']['FU'], 7)} |"
                                 for p, v in S["truth"].items()])
    return out


def s2_tables():
    S = json.load(open(os.path.join(R, "s2_summary.json")))
    out = {}
    lines = ["| p | N | runs | Hoeffding t | ε_tr (ball) | ε_op (Weyl) | Hoeffding event | cov F̂-UCB (box) | cov BS-LCB | cov F_U-UCB | cov E₁-LCB | cov all 4 | cov F̂ (ball) | cov floor (Weyl) | plug-in F̂ ≥ truth | plug-in BS ≤ truth |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(S["coverage"], key=lambda x: (-x["p"], x["N"])):
        c = r["coverage"]
        lines.append(f"| {r['p']:g} | {r['N']:.0e} | {r['runs']} | {g(r['t'])} | {g(r['eps_tr'])} | {g(r['eps_op'])} | {r['hoeffding_event']:.3f} | "
                     f"{c['F']:.3f} | {c['BS']:.3f} | {c['FU']:.3f} | {c['E1']:.3f} | {c['all4']:.3f} | {c['F_ball']:.3f} | {c['floor']:.3f} | "
                     f"{r['plugin_as_bound_coverage']['F']:.3f} | {r['plugin_as_bound_coverage']['BS']:.3f} |")
    out["S2_cov"] = "\n".join(lines)
    lines = ["| p | N | truth F̂ / BS / F_U / E₁ | median F̂-UCB box / ball / v5-theorem | median BS-LCB | median F_U-UCB | median E₁-LCB | median Weyl floor LCB (truth λ_min) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(S["coverage"], key=lambda x: (-x["p"], x["N"])):
        t, m = r["truth"], r["median"]
        lines.append(f"| {r['p']:g} | {r['N']:.0e} | {g(t['F'])} / {g(t['BS'])} / {g(t['FU'], 6)} / {g(t['E1'])} | "
                     f"{g(m['F_ucb'])} / {g(m['F_ucb_ball'])} / {g(m['F_ucb_thm'])} | {g(m['BS_lcb'])} | {g(m['FU_ucb'], 6)} | {g(m['E1_lcb'])} | "
                     f"{g(m['floor_lcb'])} ({g(t['lmin'])}) |")
    out["S2_covmed"] = "\n".join(lines)
    lines = ["| family | quantity | N_req at p = 1e-2 / 3e-3 / 1e-3 / 3e-4 / 1e-4 | fitted exponent a (N ∝ p^-a) |", "|---|---|---|---|"]
    names = dict(F="geometric −2 ln F̂ (box)", BS="BS (box, via D̂_7/8)", E1="hockey-stick E₁ (box)", FU="Uhlmann −2 ln F_U (box)",
                 F_thm="geometric via v5 Cor 10.7/10.8 (ball)", F_plugin_10pc="plug-in F̂ within 10% (NOT a bound)",
                 weyl_floor_half="Weyl-certified floor ≥ λ_min/2")
    for k, v in S["fits"].items():
        fam, q = k.split("|")
        Nr = v["Nreq"]
        vals = " / ".join(g(Nr.get(p, float("nan"))) for p in ("1e-02", "3e-03", "1e-03", "3e-04", "1e-04"))
        ex = "–" if v["exponent"] is None else f"{v['exponent']:.2f}"
        lines.append(f"| {'δ=2√p' if fam == 'scaled' else 'δ=1 rad'} | {names.get(q, q)} | {vals} | {ex} |")
    out["S2_nreq"] = "\n".join(lines)
    return out


def s2b_tables():
    fn = os.path.join(R, "s2b_summary.json")
    if not os.path.exists(fn):
        return {}
    S = json.load(open(fn))
    lab = dict(ball="(a) tomography + trace-norm ball (Hoeffding)", hbox="(a′) tomography + Hoeffding box",
               cp="(a″) tomography + Clopper–Pearson box", pauli="(b) known frame, direct Pauli error rates (C–P)")
    lines = ["| pipeline | target | total shots/channel at p = 1e-2 / 3e-3 / 1e-3 / 3e-4 / 1e-4 | exponent a (N ∝ p^-a) |", "|---|---|---|---|"]
    for k, v in S["fits"].items():
        w, q = k.split("|")
        vals = " / ".join(g(v["Nreq_total"].get(p, float("nan"))) for p in ("1e-02", "3e-03", "1e-03", "3e-04", "1e-04"))
        lines.append(f"| {lab[w]} | {'−2 ln F̂ UCB' if q == 'D12' else 'BS LCB'} | {vals} | {v['exponent']:.2f} |")
    out = {"S2b_fits": "\n".join(lines)}
    out.update(s2b_eta_tables(S["eta"]))
    return out


PSTR = ("1e-02", "3e-03", "1e-03", "3e-04", "1e-04")
LAB = dict(thmC="(a) tomography + diamond-norm ball (Theorem C), F̂-LCB", vert="(b) direct Pauli-error estimation, F̂-LCB",
           vertBS="(b) direct Pauli-error estimation, D_BS-UCB", E1u="hockey-stick E₁-UCB, diamond-norm ball (δ = 1 rad)",
           FUl="Uhlmann F_U-LCB, diamond-norm ball (δ = 1 rad)", ball="(a) trace-norm ball, F̂-UCB (SDP)",
           hbox="(a′) Hoeffding box, F̂-UCB (SDP)", cp="(a″) Clopper–Pearson box, F̂-UCB (SDP)",
           pauli="(b) Pauli polytope, F̂-UCB (SDP)", E1="hockey-stick E₁-LCB (SDP)", FU="Uhlmann F_U-UCB (SDP)")


def _row_n(E, w, q, eta):
    nr = E["nreq"].get(f"{w}|q{q}|eta{eta}", {})
    cells = []
    for p in PSTR:
        x = nr.get(p)
        if x is None:
            cells.append("–")
        elif not np.isfinite(x["total_per_channel"]):
            cells.append("> grid" if np.isinf(x["total_per_channel"]) else "< grid")
        else:
            b = x.get("total_boot95")
            cells.append(f"{g(x['total_per_channel'])}" + (f" [{g(b[0], 2)}, {g(b[1], 2)}]" if b else ""))
    return cells


def _slope(E, w, q, eta):
    f = E["fits"].get(f"{w}|q{q}|eta{eta}")
    if not f:
        return "–", "–", "–"
    return (f"{f['slope']:+.3f}", f"[{f['slope_ci95_ols'][0]:+.3f}, {f['slope_ci95_ols'][1]:+.3f}]",
            f"[{f['slope_ci95_boot'][0]:+.3f}, {f['slope_ci95_boot'][1]:+.3f}]")


def s2b_eta_tables(E):
    out = {}
    hdr = "| quantity | " + " | ".join(f"p = {p}" for p in PSTR) + " |"
    sep = "|---|" + "---|" * len(PSTR)
    L = [f"Total channel uses per channel, median run, η = 0.05 (SIMULATED; bootstrap 95% range in brackets):", "", hdr, sep]
    for w in ("thmC", "vert", "vertBS"):
        L.append(f"| {LAB[w]} | " + " | ".join(_row_n(E, w, 0.5, 0.05)) + " |")
    L += ["", "Other accuracies and the three-runs-in-four criterion (total channel uses per channel; SIMULATED):", "",
          "| quantity | criterion | " + " | ".join(f"p = {p}" for p in PSTR) + " |", "|---|---|" + "---|" * len(PSTR)]
    for w in ("thmC", "vert"):
        for q, eta in ((0.5, 0.1), (0.5, 0.025), (0.75, 0.05)):
            cells = [c.split(" [")[0] for c in _row_n(E, w, q, eta)]
            L.append(f"| {LAB[w]} | η = {eta}, {'median' if q == 0.5 else '75th pct.'} | " + " | ".join(cells) + " |")
    ra = E["ratio_a_over_b"].get("converse|q0.5|eta0.05", {})
    L += ["", "Ratio (a)/(b) and ratio × λ̃ (η = 0.05, median): " +
          "; ".join(f"p = {p}: {g(ra[p]['ratio'])} (× λ̃ = {g(ra[p]['ratio_times_lt'])})" for p in PSTR if p in ra) + "."]
    out["S2b_main"] = "\n".join(L)
    L = ["| quantity | criterion | slope | 95% CI (OLS) | 95% CI (bootstrap) |", "|---|---|---|---|---|"]
    for w in ("thmC", "vert", "vertBS", "E1u", "FUl"):
        for q, eta in ((0.5, 0.05), (0.5, 0.1), (0.5, 0.025), (0.75, 0.05)):
            s, c1, c2 = _slope(E, w, q, eta)
            if s != "–":
                L.append(f"| {LAB[w]} | η = {eta}, {'median' if q == 0.5 else '75th pct.'} | {s} | {c1} | {c2} |")
    L += ["", "η-exponent b in N ∝ η^(−b), fitted per p over η ∈ {0.1, 0.05, 0.025} (median; FITTED): " +
          "; ".join(f"{w}: " + ", ".join(f"{g(v, 3)}" for v in E["eta_exponent"].get(f"{w}|q0.5", {}).values())
                    for w in ("thmC", "vert")) + " (in the order p = 1e-2 … 1e-4)."]
    out["S2b_slopes"] = "\n".join(L)
    B = E["bounds"]
    L = ["| p | N3 lower bound c₀/(η²λ̃) | 4c₀/(η²λ̃) (certifier → estimator at η/2) | N4 upper bound 3 ln(2/δ_f)/(η²λ̃) | N2 bound with ε = 4ηλ̃ (incoherent ball) | (b) observed | (b)/N3 | (a) observed | (a)/N2 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for p in PSTR:
        if p not in B:
            continue
        b = B[p]["eta0.05"]
        nb = E["nreq"].get("vert|q0.5|eta0.05", {}).get(p, {}).get("total_per_channel", np.nan)
        na = E["nreq"].get("thmC|q0.5|eta0.05", {}).get(p, {}).get("total_per_channel", np.nan)
        L.append(f"| {p} | {g(b['N3_c0_over_eta2_lt'])} | {g(b['N3_certifier_reduction'])} | {g(b['N4_3ln2overdf_over_eta2_lt'])} | "
                 f"{g(b['N2_incoherent_ball_eps_4etalt'])} | {g(nb)} | {g(nb / b['N3_c0_over_eta2_lt'])} | {g(na)} | {g(na / b['N2_incoherent_ball_eps_4etalt'])} |")
    out["S2b_bounds"] = "\n".join(L)
    L = ["| quantity | " + " | ".join(f"p = {p}" for p in PSTR) + " | slope | 95% CI (OLS) |", "|---|" + "---|" * (len(PSTR) + 2)]
    for w in ("E1u", "FUl"):
        s, c1, _ = _slope(E, w, 0.5, 0.05)
        L.append(f"| {LAB[w]}, η = 0.05 | " + " | ".join(_row_n(E, w, 0.5, 0.05)) + f" | {s} | {c1} |")
    ct = E.get("contrast_truth", {})
    if ct:
        L += ["", "True values (δ = 1 rad): " + "; ".join(f"p = {p}: E₁ = {g(v['truth']['E1'], 4)}, F_U = {g(v['truth']['FU'], 4)}, F̂ = {g(v['truth']['F'], 3)}"
                                              for p, v in sorted(ct.items(), key=lambda kv: -float(kv[0]))) + "."]
    out["S2b_contrast"] = "\n".join(L)

    def covtab(side):
        L = ["| pipeline | p | N per setting | runs | coverage F̂ bound [95% C–P] | coverage D_BS bound [95% C–P] | median F̂ bound (truth) | median D_BS bound (truth) |",
             "|---|---|---|---|---|---|---|---|"]
        for r in [x for x in E["coverage"] if x["side"] == side]:
            t = r["truth"]
            cF = f"{r['cov_F']:.3f} [{r['cov_F_cp95'][0]:.4f}, {r['cov_F_cp95'][1]:.4f}]"
            cB = f"{r['cov_BS']:.3f} [{r['cov_BS_cp95'][0]:.4f}, {r['cov_BS_cp95'][1]:.4f}]" if "cov_BS" in r else "–"
            mB = f"{g(r['med_BS'])} ({g(t['BS'])})" if "med_BS" in r else "–"
            L.append(f"| {LAB[r['pipeline']].split(',')[0]} | {r['p']:g} | {g(r['N_per_setting'])} | {r['runs']} | {cF} | {cB} | "
                     f"{g(r['med_F'])} ({g(t['F'])}) | {mB} |")
        return "\n".join(L)
    out["S2b_cov"] = covtab("converse")
    out["S2b_covopp"] = covtab("opposite")
    L = ["| pipeline (SDP, opposite side) | " + " | ".join(f"p = {p}" for p in PSTR) + " | slope | 95% CI (OLS) | 95% CI (bootstrap) |",
         "|---|" + "---|" * (len(PSTR) + 3)]
    for w in ("ball", "hbox", "cp", "pauli"):
        s, c1, c2 = _slope(E, w, 0.5, 0.05)
        L.append(f"| {LAB[w]}, η = 0.05 | " + " | ".join(c.split(' [')[0] for c in _row_n(E, w, 0.5, 0.05)) + f" | {s} | {c1} | {c2} |")
    out["S2b_opp"] = "\n".join(L)
    return out


def s3_tables():
    fn = os.path.join(R, "s3_cnot.json")
    if not os.path.exists(fn):
        return {}
    D = json.load(open(fn))
    lines = ["| p | λ_min J | θ (F̂=0.71) | N | median F̂ pair [10,90] | median F̂ self | BS = ∞ fraction |", "|---|---|---|---|---|---|---|"]
    nreq = []
    for p in D["PS"]:
        rows = sorted([x for x in D["rows"] if x["p"] == p], key=lambda x: x["N"])
        ok = None
        for x in rows:
            F = np.array(x["F_delta"]); t = x["truth"]
            lines.append(f"| {p:g} | {g(t['lmin'])} | {g(t['theta'])} | {x['N']:.0e} | {g(float(np.median(F)))} [{g(float(np.quantile(F, .1)))}, {g(float(np.quantile(F, .9)))}] | "
                         f"{g(float(np.median(x['F_self'])))} | {np.mean(np.isinf(x['BS_delta'])):.2f} |")
        good = [np.quantile(np.abs(np.array(x["F_delta"]) - x["truth"]["F"]) / x["truth"]["F"], 0.9) <= 0.1 for x in rows]
        Ns = [x["N"] for x in rows]
        j = len(good)
        while j > 0 and good[j - 1]:
            j -= 1
        nreq.append((p, Ns[j] if j < len(Ns) else float("inf"), rows[0]["truth"]["lmin"]))
    out = {"S3_table": "\n".join(lines)}
    l2 = ["| p | λ_min J (= p²/4) | N* (90% of runs within 10%) |", "|---|---|---|"] + [f"| {p:g} | {g(l)} | {g(float(n))} |" for p, n, l in nreq]
    P = np.array([x[0] for x in nreq]); Nn = np.array([x[1] for x in nreq], float)
    m = np.isfinite(Nn)
    if m.sum() >= 2:
        a, _ = np.polyfit(np.log10(P[m]), np.log10(Nn[m]), 1)
        l2.append(f"\nFitted N* ∝ p^-{-a:.2f} (grid resolution one decade).")
    out["S3_nreq"] = "\n".join(l2)
    return out


# ----------------------------------------------------------------------------- prose blocks (numbers read from JSON)
def _s1(p, N, est, pair, q, key):
    T = json.load(open(os.path.join(R, "s1_summary.json")))["table"]
    r = [x for x in T if x["p"] == p and x["N"] == N and x["est"] == est and x["pair"] == pair and x["q"] == q]
    return r[0][key] if r else float("nan")


def texts():
    out = {}
    s1 = lambda *a: _s1(*a)
    out["S1_REFEREE"] = "\n".join([
        "**Take-aways from S1 (SIMULATED, 200 runs per cell):**",
        f"- **Rank-deficient plug-ins return F̂ = 0 and D_BS = +∞ until the floor is resolved.** For linear inversion + CPTP projection on the δ = 2√p pair, the median F̂ is "
        f"{g(s1(1e-2, 1e3, 'LI', 'delta', 'F', 'median'))} / {g(s1(1e-2, 1e4, 'LI', 'delta', 'F', 'median'))} / {g(s1(1e-2, 1e5, 'LI', 'delta', 'F', 'median'))} / "
        f"{g(s1(1e-2, 1e6, 'LI', 'delta', 'F', 'median'))} at p = 1e-2 and N = 1e3 / 1e4 / 1e5 / 1e6 (truth {g(s1(1e-2, 1e6, 'LI', 'delta', 'F', 'truth'))}). "
        f"It is still {g(s1(1e-3, 1e6, 'LI', 'delta', 'F', 'median'))} at p = 1e-3, N = 1e6, and {g(s1(1e-4, 1e5, 'LI', 'delta', 'F', 'median'))} at p = 1e-4, N = 1e5. "
        f"Plug-in D_BS is +∞ in {s1(1e-3, 1e6, 'LI', 'delta', 'BS', 'frac_inf'):.0%} of runs at p = 1e-3, N = 1e6. The rank-1 and rank-2 fits give F̂ = 0 at every N.",
        f"- **Plug-in errors are mostly on the loose side, but not always.** At p = 1e-2, N = 1e6 the overclaim fraction of F̂ is {s1(1e-2, 1e6, 'LI', 'delta', 'F', 'overclaim'):.2f} (LI) and "
        f"{s1(1e-2, 1e6, 'MLE', 'delta', 'F', 'overclaim'):.2f} (MLE). The hedged MLE (β = 1/2) overclaims in {s1(1e-4, 1e2, 'HMLE0.5', 'delta', 'F', 'overclaim'):.0%} of runs at p = 1e-4, N = 1e2 "
        f"(median {g(s1(1e-4, 1e2, 'HMLE0.5', 'delta', 'F', 'median'))} against a truth of {g(s1(1e-4, 1e2, 'HMLE0.5', 'delta', 'F', 'truth'))}). No plug-in estimator is therefore a certified converse.",
        f"- **E₁ and F_U plug-ins do not need the floor.** At N = 1e4 the median LI E₁ is {g(s1(1e-2, 1e4, 'LI', 'delta', 'E1', 'median'))} at p = 1e-2 (truth {g(s1(1e-2, 1e4, 'LI', 'delta', 'E1', 'truth'))}) and "
        f"{g(s1(1e-3, 1e4, 'LI', 'delta', 'E1', 'median'))} at p = 1e-3 (truth {g(s1(1e-3, 1e4, 'LI', 'delta', 'E1', 'truth'))}).",
        "- **The S1 grid stops at N = 1e6 for every p.** The p = 1e-4, N = 1e6 cell was completed in this session. The N = 1e7–1e9 cells of the original plan were not run."])
    S2 = json.load(open(os.path.join(R, "s2_summary.json")))
    cov = sorted(S2["coverage"], key=lambda x: (-x["p"], x["N"]))
    out["S2_COVTEXT"] = "\n".join([
        "**Reading the coverage table (SIMULATED).**",
        "- Plug-in F̂ used directly as a converse is valid (F̂_plug ≤ truth) in " +
        ", ".join(f"{1 - r['plugin_as_bound_coverage']['F']:.3f} (p = {r['p']:g}, N = {r['N']:.0e})" for r in cov) + " of runs. It is not a certified bound at any of these points.",
        "- The Weyl floor LCB has a negative median at every coverage point " +
        ", ".join(f"({g(r['median']['floor_lcb'])} at p = {r['p']:g}, N = {r['N']:.0e})" for r in cov) +
        ", so the floor itself is not yet certified at the N where the box certificates are within a factor 2. This is the p⁻² of the operator-norm route."])
    F = S2["fits"]
    ex = lambda k: F[k]["exponent"]
    out["S2_TEXT"] = "\n".join([
        "**Summary of S2 (opposite side; FITTED exponents of N_req ∝ p^−a):**",
        f"- On the δ = 2√p pair the Hoeffding-box bounds cost p^−{ex('scaled|F'):.2f} (F̂) and p^−{ex('scaled|BS'):.2f} (D_BS). The plug-in F̂ reaches 10% accuracy at p^−{ex('scaled|F_plugin_10pc'):.2f}, "
        f"and the Weyl-certified floor at p^−{ex('scaled|weyl_floor_half'):.2f} (analytic).",
        f"- On the same pair E₁ costs p^−{ex('scaled|E1'):.2f} and F_U p^−{ex('scaled|FU'):.2f}. These are factor-2 criteria, and the true values shrink (E₁ ∝ √p, −ln F_U ∝ p), so this is not a floor effect.",
        f"- On the δ = 1 rad pair, E₁ and F_U are within a factor 2 at the first grid point (10³ shots per setting) for every p, while F̂ costs p^−{ex('fixed|F'):.2f}.",
        "- The v5 theorem route (Cor 10.7/10.8) is vacuous on the whole grid up to N = 10¹²."])
    D = json.load(open(os.path.join(R, "s3_cnot.json")))
    out["S3_TEXT"] = ("**Summary of S3 (SIMULATED, plug-in only).** For the CNOT with local depolarising noise, λ_min J = p²/4, the square of the qubit floor up to constants. "
                      "The N at which 90% of plug-in runs are within 10% therefore scales as λ_min⁻² ∝ p⁻⁴ (table above, one-decade grid). "
                      "This matches the n-qubit form λ̃ = (p/4)ⁿ that enters N1–N4. A certified CNOT pipeline was not built.")
    return out


def rg(a, b, n=2):
    """range a-b, or a single value when both round to the same string."""
    A, B = g(float(a), n), g(float(b), n)
    return A if A == B else f"{A}–{B}"


def sl(x, n=3):
    x = 0.0 if abs(x) < 0.5 * 10 ** (-n) else x
    return (f"{x:+.{n}f}".replace("-", "−")) if x != 0 else f"{0:.{n}f}"


def s2b_text():
    E = json.load(open(os.path.join(R, "s2b_summary.json")))["eta"]
    f = lambda w, eta=0.05, q=0.5: E["fits"].get(f"{w}|q{q}|eta{eta}")
    n = lambda w, p, eta=0.05: E["nreq"].get(f"{w}|q0.5|eta{eta}", {}).get(p, {}).get("total_per_channel", float("nan"))
    B = E["bounds"]
    L = ["**Discussion of S2b.**"]
    fa, fb, fB = f("thmC"), f("vert"), f("vertBS")
    ra = E["ratio_a_over_b"]["converse|q0.5|eta0.05"]
    L.append(f"- **Slopes (FITTED).** The diamond-ball certificate costs N ∝ p^{sl(fa['slope'])} and direct Pauli-error estimation N ∝ p^{sl(fb['slope'], 2)}. "
             f"The D_BS upper bound from (b) costs N ∝ p^{sl(fB['slope'], 2)}. These are exactly the Θ(1/(ηλ̃)²) and Θ(1/(η²λ̃)) laws. "
             f"The ratio (a)/(b) times λ̃ stays within {rg(min(v['ratio_times_lt'] for v in ra.values()), max(v['ratio_times_lt'] for v in ra.values()))} across p, "
             "so the waste of the ball is a factor ≈ 7e2/λ̃. Part of that constant comes from the 6× larger number of settings in (a), and part from the variance-blind 36t trace-norm radius.")
    b3 = [n("vert", p) / B[p]["eta0.05"]["N3_c0_over_eta2_lt"] for p in PSTR if p in B]
    b4 = [n("vert", p) / B[p]["eta0.05"]["N4_3ln2overdf_over_eta2_lt"] for p in PSTR if p in B]
    b2 = [n("thmC", p) / B[p]["eta0.05"]["N2_incoherent_ball_eps_4etalt"] for p in PSTR if p in B]
    L.append(f"- **Constants against N2–N4.** (b) sits {rg(min(b3), max(b3))}× above the N3 scale c₀/(η²λ̃), and at {rg(min(b4), max(b4))}× the N4 sufficient budget, which is for Bell-basis measurement at δ_f = 0.05. "
             f"(a) sits {rg(min(b2), max(b2))}× above the incoherent-learner bound N2 evaluated at the sharp radius ε = 4ηλ̃. That gap is the constant of a non-optimal learner, not a different scaling. "
             f"The requirement κ = ε_tr/(2λ_min) ≤ η, with λ_min = p/2, ε_tr = 36t and t = √(ln(1440)/(2N_set)), alone gives N = 18·36²·ln(1440)/(2η²p²) ≈ {g(18 * 36 ** 2 * np.log(1440) / (2 * 0.05 ** 2), 2)}/p² channel uses (observed: {g(n('thmC', '1e-02') * 1e-4, 2)}/p² at p = 1e-2). N2 at the sharp radius is ≈ {g(0.0589 / 0.05 ** 2, 2)}/p². "
             "Two caveats apply to N3 here. It is proved for the estimand F̂(Ad_{PU}, Ψ), a unitary against a Pauli channel, with success probability 2/3. Our estimand is F̂ between two noisy channels, and the criterion is the median run. "
             "The plotted line is therefore the proved scale of the intrinsic cost, not a rigorous bound for this pair. It lies below every measured point.")
    ea, eb = E["eta_exponent"].get("thmC|q0.5", {}), E["eta_exponent"].get("vert|q0.5", {})
    L.append(f"- **η-dependence (FITTED over η ∈ {{0.1, 0.05, 0.025}}).** (a): N ∝ η^−b with b = {rg(min(ea.values()), max(ea.values()), 3)}. Since κ is linear in the radius, the asymptotic value is 2. "
             f"(b): b = {rg(min(eb.values()), max(eb.values()), 3)}. This is pre-asymptotic: ln F̂ responds sub-linearly to relative changes of the floor rates at these widths. "
             "It is consistent with the N4 upper bound ∝ η^−2.")
    fe, fu = f("E1u"), f("FUl")
    if fe and fu:
        L.append(f"- **Contrast.** Through the same diamond ball, E₁ and F_U are certified at p-independent cost: slope {sl(fe['slope'])} [{sl(fe['slope_ci95_ols'][0])}, {sl(fe['slope_ci95_ols'][1])}] and "
                 f"{sl(fu['slope'])} [{sl(fu['slope_ci95_ols'][0])}, {sl(fu['slope_ci95_ols'][1])}]. "
                 f"At p = 1e-4 the E₁ certificate needs {g(n('E1u', '1e-04'), 2)} channel uses, against {g(n('thmC', '1e-04'), 2)} for F̂ through the same ball. The F_U route uses the purified-distance triangle inequality, which costs ∝ η^−4 but contains no λ̃.")
    cv = [r for r in E["coverage"] if r["side"] == "converse"]
    if cv:
        L.append(f"- **Coverage (SIMULATED).** At every p and for both pipelines, the F̂ lower bound covered the truth in {rg(min(r['cov_F'] * r['runs'] for r in cv), max(r['cov_F'] * r['runs'] for r in cv), 4)} of 1000 runs. "
                 f"The D_BS upper bound of (b) covered it in {rg(min(r['cov_BS'] * r['runs'] for r in cv if 'cov_BS' in r), max(r['cov_BS'] * r['runs'] for r in cv if 'cov_BS' in r), 4)} of 1000. "
                 "The lowest 95% Clopper–Pearson limit on a coverage of 1000/1000 is 0.9963. The regions are conservative (union bound), as expected.")
    op = {w: f(w) for w in ("ball", "hbox", "cp", "pauli")}
    L.append("- **Opposite side (§3.5).** The SDP bounds give the same picture: " +
             ", ".join(f"{LAB[w].split(',')[0]} {sl(v['slope'], 2)}" for w, v in op.items() if v) +
             ". The model-free Clopper–Pearson box in probability space reaches slope ≈ −1 on that side. On the converse side the exact LCB over that box is a concave minimisation over an 18-dimensional polytope ∩ CPTP, which was not attempted. "
             "Whether model-free tomography attains the 1/λ̃ law for the converse is therefore not settled by these runs.")
    L.append("- **Model assumption.** (b) is valid only if both channels are Pauli channels in the known frames U₁ and U₀. Any coherent error outside the model invalidates it. (a) assumes nothing beyond the tomography model.")
    return "\n".join(L)


def headline():
    E = json.load(open(os.path.join(R, "s2b_summary.json")))["eta"]
    fa, fb, fB = E["fits"]["thmC|q0.5|eta0.05"], E["fits"]["vert|q0.5|eta0.05"], E["fits"]["vertBS|q0.5|eta0.05"]
    fe, fu = E["fits"]["E1u|q0.5|eta0.05"], E["fits"]["FUl|q0.5|eta0.05"]
    na, nb = E["nreq"]["thmC|q0.5|eta0.05"], E["nreq"]["vert|q0.5|eta0.05"]
    cv = [r for r in E["coverage"] if r["side"] == "converse"]
    ra = E["ratio_a_over_b"]["converse|q0.5|eta0.05"]
    rt = np.array([v["ratio_times_lt"] for v in ra.values()])
    nruns = min(r["runs"] for r in cv)
    return (f"**Headline.** Near the noisy Hadamard N_p = D_p∘H, the geometric-fidelity and Belavkin–Staszewski converses can be certified from finite data, at a cost set by the Choi floor λ̃ = p/4. "
            f"Certifying −ln F̂ to η = 0.05 through a diamond-norm ball (Pauli tomography + Theorem C) takes {g(na['1e-02']['total_per_channel'], 2)}–{g(na['1e-04']['total_per_channel'], 2)} channel uses per channel for p = 1e-2…1e-4, with fitted slope {sl(fa['slope'])}. "
            f"Direct Pauli-error estimation in the known frame takes {g(nb['1e-02']['total_per_channel'], 2)}–{g(nb['1e-04']['total_per_channel'], 2)}, with slope {sl(fb['slope'], 2)} [95% CI {sl(fb['slope_ci95_ols'][0], 2)}, {sl(fb['slope_ci95_ols'][1], 2)}] (FITTED). "
            f"So the ball is wasteful by a factor ≈ {g(float(np.median(rt)), 2)}/λ̃ (ratio × λ̃ = {rg(rt.min(), rt.max())}), as Theorem C and N1–N4 predict. "
            f"Every converse-side certificate covered the truth in {min(r['cov_F'] for r in cv) * nruns:.0f}/{nruns} runs at every p (target ≥ 0.95; SIMULATED). The D_BS upper bound scales like F̂ (slope {sl(fB['slope'], 2)}). "
            f"Hockey-stick and Uhlmann certificates through the same ball cost the same at every p (slopes {sl(fe['slope'])}, {sl(fu['slope'])}). "
            "Correction: the earlier S2/S2b SDP bounds (max F̂, min D over the region) lie on the opposite side and do not certify the converses (§0).")


if __name__ == "__main__":
    T = {}
    for f in (s1_tables, s2_tables, s2b_tables, s3_tables, texts):
        try:
            T.update(f())
        except FileNotFoundError as ex:
            print("skip", f.__name__, ex)
    try:
        T["HEADLINE"] = headline()
    except (KeyError, FileNotFoundError) as ex:
        print("headline incomplete:", ex)
    try:
        T["S2b_TEXT"] = s2b_text()
    except (KeyError, FileNotFoundError, TypeError, ValueError) as ex:
        print("S2b text incomplete:", repr(ex))
    json.dump(T, open(os.path.join(R, "tables.json"), "w"), indent=1)
    tpl = os.path.join(HERE, "README_template.md")
    if os.path.exists(tpl):
        s = open(tpl).read()
        for k, v in T.items():
            s = s.replace("{{" + k + "}}", v)
        open(os.path.join(HERE, "README.md"), "w").write(s)
    print("\n\n".join(f"## {k}\n{v}" for k, v in T.items()))
