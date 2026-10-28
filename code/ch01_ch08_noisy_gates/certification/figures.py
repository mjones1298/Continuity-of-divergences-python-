"""Figures for A_cert (reads results/*.json, results/s1_summary.json, results/s2_summary.json)."""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(HERE, "results")
# validated categorical palette (dataviz reference instance, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})
EST = ["LI", "MLE", "HMLE0.5", "HMLE0.05", "LR1", "LR2"]
ELAB = {"LI": "lin. inv. + CPTP proj.", "MLE": "MLE", "HMLE0.5": "hedged MLE β=1/2", "HMLE0.05": "hedged MLE β=0.05",
        "LR1": "rank-1 MLE", "LR2": "rank-2 MLE"}
PS = [1e-2, 1e-3, 1e-4]


def s1_figs():
    S = json.load(open(os.path.join(RES, "s1_summary.json")))
    T = S["table"]

    def sel(p, e, pair, q):
        r = sorted([x for x in T if x["p"] == p and x["est"] == e and x["pair"] == pair and x["q"] == q],
                   key=lambda x: x["N"])
        return r

    # Fig 1: plug-in F-hat
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    for j, p in enumerate(PS):
        for i, pair in enumerate(("delta", "self")):
            a = ax[i, j]
            for k, e in enumerate(EST):
                r = sel(p, e, pair, "F")
                if not r:
                    continue
                N = [x["N"] for x in r]
                a.plot(N, [x["median"] for x in r], color=C[k], marker="o", ms=3.5, label=ELAB[e])
                a.fill_between(N, [x["q10"] for x in r], [x["q90"] for x in r], color=C[k], alpha=0.12, lw=0)
            tv = S["truth"][str(p)][pair]["F"]
            a.axhline(tv, color=INK, ls="--", lw=1)
            a.set_xscale("log"); a.set_ylim(-0.03, 1.03)
            a.set_title(f"{'pair (N_p,δ , N_p), δ=2√p' if pair == 'delta' else 'gate vs itself'}   p={p:g}",
                        fontsize=9, color=INK)
            if i == 1:
                a.set_xlabel("shots per setting N")
            if j == 0:
                a.set_ylabel("plug-in geometric fidelity F̂\n(median, 10–90% band)")
    ax[0, 0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("S1 — plug-in F̂ from Pauli process tomography (dashed: truth). Above the truth = overclaim.",
                 fontsize=10, color=INK)
    fig.savefig(os.path.join(FIG, "s1_plugin_fhat.png")); plt.close(fig)

    # Fig 2: BS infinite fraction + overclaim fraction of F and BS (delta pair)
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    for j, p in enumerate(PS):
        for k, e in enumerate(EST):
            r = sel(p, e, "delta", "BS")
            if r:
                ax[0, j].plot([x["N"] for x in r], [x["frac_inf"] for x in r], color=C[k], marker="o", ms=3.5,
                              label=ELAB[e])
            r = sel(p, e, "delta", "F")
            if r:
                ax[1, j].plot([x["N"] for x in r], [x["overclaim"] for x in r], color=C[k], marker="o", ms=3.5)
        ax[0, j].set_title(f"p={p:g}", fontsize=9, color=INK)
        ax[1, j].set_xlabel("shots per setting N")
        for a in ax[:, j]:
            a.set_xscale("log"); a.set_ylim(-0.03, 1.03)
    ax[0, 0].set_ylabel("fraction of runs with\nplug-in D̂_BS = +∞")
    ax[1, 0].set_ylabel("overclaim fraction of F̂\n(estimate > truth)")
    ax[0, 0].legend(loc="center left", fontsize=7.5)
    fig.suptitle("S1 — support failures of BS and the invalid (overclaiming) direction of F̂, pair δ=2√p",
                 fontsize=10, color=INK)
    fig.savefig(os.path.join(FIG, "s1_bs_inf_overclaim.png")); plt.close(fig)


def s2_figs():
    S = json.load(open(os.path.join(RES, "s2_summary.json")))
    sh = S["shots"]
    Q = [("F", "geometric  −2 ln F̂"), ("BS", "Belavkin–Staszewski"), ("E1", "hockey-stick E₁"),
         ("FU", "Uhlmann  −2 ln F_U"), ("F_thm", "geometric via v5 Cor 10.7/10.8")]
    for fam in ("scaled", "fixed"):
        keys = sorted([k for k in sh if sh[k]["fam"] == fam], key=lambda k: -sh[k]["p"])
        if not keys:
            continue
        fig, ax = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.4), sharey=True)
        ax = np.atleast_1d(ax)
        for j, k in enumerate(keys):
            r = sh[k]
            for i, (q, lab) in enumerate(Q):
                y = np.array(r["ratio"][q])
                ax[j].plot(r["N"], y, color=C[i], marker="o", ms=3, label=lab, ls=":" if q == "F_thm" else "-")
            ax[j].axhline(0.5, color=INK2, lw=1, ls="--")
            ax[j].set_xscale("log"); ax[j].set_ylim(-0.03, 1.1)
            ax[j].set_title(f"p={r['p']:g}, δ={r['delta']:.3g}", fontsize=9, color=INK)
            ax[j].set_xlabel("shots per setting N")
        ax[0].set_ylabel("certified bound / true value\n(median over runs; divergence form)")
        ax[0].legend(loc="upper left", fontsize=7)
        fig.suptitle(f"S2 — certified one-sided converses (95% Hoeffding box region), "
                     f"{'δ = 2√p (geometric scale)' if fam == 'scaled' else 'δ = 1 rad (fixed)'}; dashed: factor 2",
                     fontsize=10, color=INK)
        fig.savefig(os.path.join(FIG, f"s2_ratio_{fam}.png")); plt.close(fig)

    # N_req vs p
    fits = S["fits"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for j, fam in enumerate(("scaled", "fixed")):
        a = ax[j]
        allq = Q + [("weyl_floor_half", "Weyl floor ≥ λ_min/2 (analytic)"),
                    ("F_plugin_10pc", "plug-in F̂ within 10% (not a bound)")]
        for i, (q, lab) in enumerate(allq):
            f = fits.get(f"{fam}|{q}")
            if not f:
                continue
            ps = np.array([float(x) for x in f["Nreq"]]); n = np.array(list(f["Nreq"].values()))
            m = np.isfinite(n)
            if not m.any():
                continue
            ex = f["exponent"]
            a.plot(ps[m], n[m], color=C[i], marker="o", ms=4, ls=":" if q in ("F_thm", "weyl_floor_half") else "-",
                   label=f"{lab}" + (f"  (∝ p^-{ex:.2f})" if ex is not None else ""))
        a.set_xscale("log"); a.set_yscale("log"); a.invert_xaxis()
        a.set_xlabel("depolarizing floor p")
        a.set_title("δ = 2√p" if fam == "scaled" else "δ = 1 rad", fontsize=9, color=INK)
        a.legend(fontsize=6.8, loc="upper left")
    ax[0].set_ylabel("shots per setting for certified\nbound within factor 2 of truth")
    fig.suptitle("S2 — certification cost vs noise floor (median run; ∞ = not reached on grid, omitted)",
                 fontsize=10, color=INK)
    fig.savefig(os.path.join(FIG, "s2_nreq_vs_p.png")); plt.close(fig)


def s3_figs():
    fn = os.path.join(RES, "s3_cnot.json")
    if not os.path.exists(fn):
        return
    D = json.load(open(fn))
    q1 = json.load(open(os.path.join(RES, "s1_summary.json")))["table"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for i, p in enumerate(D["PS"]):
        r = sorted([x for x in D["rows"] if x["p"] == p], key=lambda x: x["N"])
        if not r:
            continue
        N = [x["N"] for x in r]
        for a, key in ((ax[0], "F_delta"), (ax[1], "F_self")):
            med = [np.median(x[key]) for x in r]
            a.plot(N, med, color=C[i], marker="o", ms=3.5, label=f"CNOT, local depol. p={p:g} (λ_min={r[0]['truth']['lmin']:.1e})")
            a.fill_between(N, [np.quantile(x[key], .1) for x in r], [np.quantile(x[key], .9) for x in r],
                           color=C[i], alpha=0.12, lw=0)
    for k, p in enumerate([1e-2]):
        for a, pair in ((ax[0], "delta"), (ax[1], "self")):
            r = sorted([x for x in q1 if x["p"] == p and x["est"] == "LI" and x["pair"] == pair and x["q"] == "F"],
                       key=lambda x: x["N"])
            a.plot([x["N"] for x in r], [x["median"] for x in r], color=INK2, ls="--", lw=1.5,
                   label="qubit H, p=0.01 (λ_min=5e-3)")
    ax[0].axhline(0.7071, color=INK, lw=1, ls=":")
    ax[0].set_title("pair with true F̂ = 0.71", fontsize=9); ax[1].set_title("gate vs itself (true F̂ = 1)", fontsize=9)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("shots per setting N")
    ax[0].set_ylabel("plug-in F̂ (LI + CPTP proj.)\nmedian, 10–90%")
    ax[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("S3 — two-qubit CNOT: the floor λ_min = p²/4 sets the shot cost", fontsize=10)
    fig.savefig(os.path.join(FIG, "s3_cnot.png")); plt.close(fig)


if __name__ == "__main__":
    for f in (s1_figs, s2_figs, s3_figs):
        try:
            f()
        except FileNotFoundError as ex:
            print("skip", f.__name__, ex)


def s2b_fig():
    fn = os.path.join(RES, "s2b_summary.json")
    if not os.path.exists(fn):
        return
    S = json.load(open(fn))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    lab = {"ball": "(a) tomography + trace-norm (diamond) ball, Hoeffding",
           "hbox": "(a′) tomography + Hoeffding box",
           "cp": "(a″) tomography + Clopper–Pearson box",
           "pauli": "(b) known frame: direct Pauli error rates, C–P"}
    for j, (q, tt) in enumerate((("D12", "geometric −2 ln F̂ (UCB on F̂)"), ("BSlcb", "Belavkin–Staszewski (LCB)"))):
        a = ax[j]
        for i, w in enumerate(("ball", "hbox", "cp", "pauli")):
            f = S["fits"].get(f"{w}|{q}")
            if not f:
                continue
            P = np.array([float(x) for x in f["Nreq_total"]]); N = np.array(list(f["Nreq_total"].values()))
            o = np.argsort(P)
            a.plot(P[o], N[o], color=C[i], marker="o", ms=4, label=f"{lab[w]}  ∝ p^-{f['exponent']:.2f}")
        a.set_xscale("log"); a.set_yscale("log"); a.invert_xaxis()
        a.set_xlabel("depolarizing floor p   (Choi-state floor λ̃ = p/4)")
        a.set_title(tt, fontsize=9)
        a.legend(fontsize=6.8, loc="upper left")
    ax[0].set_ylabel("total shots per channel for certified\nbound within factor 2 (median run)")
    fig.suptitle("S2b — certification pipelines, pair δ = 2√p: isotropic regions cost ∝ p⁻², variance-adaptive / "
                 "direct estimation ∝ p⁻¹", fontsize=10)
    fig.savefig(os.path.join(FIG, "s2b_pipelines_slopes.png")); plt.close(fig)


if __name__ == "__main__":
    s2b_fig()


# ----------------------------------------------------------------------------- main letter figure
def _fmt(x):
    """slope formatting: proper minus sign, no negative zero."""
    x = 0.0 if abs(x) < 0.005 else x
    return f"{x:+.2f}".replace("-", "\u2212") if x != 0 else "0.00"

def cert_pipelines_fig(eta=0.05, q=0.5):
    """figures/cert_pipelines.{pdf,png}: one column (3.4 in).  (a) total channel uses N per channel needed for a
    certified upper confidence bound on F-hat with ln(F_ucb/F) <= eta (median run), against the depolarising
    floor p, for three certification pipelines, the proved lower-bound scale c0/(eta^2 lt) (Prop. N3), and the
    hockey-stick / Uhlmann contrast through the same diamond ball.  (b) empirical coverage of the F-hat bound."""
    S = json.load(open(os.path.join(RES, "s2b_summary.json")))["eta"]
    BLUE, ORANGE, AQUA = C[0], C[1], C[2]
    rc = {"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
          "legend.fontsize": 6, "lines.linewidth": 1.2, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
          "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(3.4, 3.55))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 0.75], hspace=0.07)
        a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        pgrid = np.logspace(-4.15, -1.85, 50)
        series = [("thmC", BLUE, "s", "full", "Tomography + diamond-norm ball"),
                  ("vert", ORANGE, "o", "full", "Direct Pauli-error estimation"),
                  ("E1u", AQUA, "^", "none", r"Hockey-stick $E_1$, diamond-norm ball"),
                  ("FUl", AQUA, "D", "none", r"Uhlmann $F_U$, diamond-norm ball")]
        for w, col, mk, fill, lab in series:
            f = S["fits"].get(f"{w}|q{q}|eta{eta}")
            if not f:
                continue
            nr = S["nreq"][f"{w}|q{q}|eta{eta}"]
            P = np.array(f["ps"]); N = np.array(f["N_total"])
            lo = np.array([nr[f"{p:.0e}"]["total_boot95"][0] for p in P])
            hi = np.array([nr[f"{p:.0e}"]["total_boot95"][1] for p in P])
            sl = f["slope"]; ci = f["slope_ci95_ols"]
            a.plot(pgrid, 10 ** (f["intercept_log10"] + sl * np.log10(pgrid)), color=col, lw=0.9,
                   ls="-" if fill == "full" else (0, (4, 1.5)), zorder=2)
            a.errorbar(P, N, yerr=[N - lo, hi - N], fmt=mk, ms=4.2 if mk != "D" else 3.6, mfc=col if fill == "full" else "white",
                       mec=col, mew=1.0, ecolor=col, elinewidth=0.7, capsize=0, zorder=3,
                       label=f"{lab}; slope {_fmt(sl)} [{_fmt(ci[0])}, {_fmt(ci[1])}]")
        bd = S["bounds"]
        c0 = 0.0246
        nb = c0 / (eta ** 2 * pgrid / 4)
        a.plot(pgrid, nb, color=INK, lw=0.9, ls=(0, (1.2, 1.4)), zorder=2,
               label=r"Intrinsic lower bound $c_0/(\eta^2\tilde\lambda)$")
        a.fill_between(pgrid, 1, nb, color="#d9d8d3", alpha=0.45, lw=0, zorder=1)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(pgrid[0], pgrid[-1]); a.set_ylim(1e3, 1e17)
        a.set_ylabel(r"channel uses per channel, $N$")
        a.tick_params(labelbottom=False)
        a.set_yticks([1e3, 1e5, 1e7, 1e9, 1e11, 1e13, 1e15, 1e17])
        a.text(0.03, 0.97, "(a)", transform=a.transAxes, va="top", fontsize=7.5, color=INK)
        a.text(0.97, 0.97, rf"accuracy $\eta = {eta}$ in $-\ln\hat F$, $\delta_f = 0.05$", transform=a.transAxes, va="top", ha="right", fontsize=6, color=INK2)
        h, l = a.get_legend_handles_labels()
        order = [1, 2, 3, 4, 0] if len(h) == 5 else list(range(len(h)))
        fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", bbox_to_anchor=(0.0, 0.0),
                   ncol=1, handlelength=1.8, labelspacing=0.3, borderaxespad=0.0, frameon=False, fontsize=6)
        # (b) coverage
        cov = [r for r in S["coverage"] if r["side"] == "converse"]
        off = {"thmC": 0.95, "vert": 1.05}
        for w, col, mk, fill, lab in series[:2]:
            rows = sorted([r for r in cov if r["pipeline"] == w], key=lambda r: r["p"])
            if not rows:
                continue
            P = np.array([r["p"] for r in rows]) * off[w]
            y = np.array([r["cov_F"] for r in rows])
            lo = np.array([r["cov_F_cp95"][0] for r in rows]); hi = np.array([r["cov_F_cp95"][1] for r in rows])
            b.errorbar(P, y, yerr=[y - lo, hi - y], fmt=mk, ms=3.6, mfc=col if fill == "full" else "white", mec=col,
                       mew=1.0, ecolor=col, elinewidth=0.7, capsize=1.2, zorder=3)
        b.axhline(0.95, color=INK, lw=0.8, ls=(0, (4, 2)))
        b.text(2.0e-3, 0.953, r"target $1-\delta_f = 0.95$", fontsize=6, color=INK2, va="bottom")
        b.set_xscale("log"); b.set_xlim(pgrid[0], pgrid[-1]); b.set_ylim(0.935, 1.006)
        b.set_yticks([0.95, 1.0])
        b.set_ylabel("coverage", labelpad=6)
        b.set_xlabel(r"depolarising strength $p$  (Choi-state floor $\tilde\lambda = p/4$)")
        b.text(0.03, 0.5, "(b)", transform=b.transAxes, va="center", fontsize=7.5, color=INK)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG, f"cert_pipelines.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)


if __name__ == "__main__":
    cert_pipelines_fig()
