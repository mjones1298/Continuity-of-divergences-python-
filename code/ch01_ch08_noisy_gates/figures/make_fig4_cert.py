"""Fig. 4 of the letter (revision 1): cost of certifying converses from simulated single-qubit Pauli tomography.

Regenerated from the A_cert data (A_cert/results/s2b_summary.json), adapted from A_cert/figures.py:
cert_pipelines_fig() (copied here, A_cert is not edited). Changes requested by the referee (M4(c), minor 32):
 - the dotted line is relabelled as a reference scale c0/(eta^2 lt), proved for the estimand F_hat(Ad_PU, Psi),
   and the shaded "forbidden" region below it is removed;
 - Computer Modern fonts shared with Figs. 1-3, US spelling ("depolarizing").
Output: fig4_cert.pdf (one column, vector).
"""
import json, os
import numpy as np
from figstyle import plt, BLUE, ORANGE, AQUA, INK, INK2, W1, panel_tag

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "certification", "results", "s2b_summary.json")


def _fmt(x):
    x = 0.0 if abs(x) < 0.005 else x
    return f"{x:+.2f}".replace("+", "") if x != 0 else "0.00"


def main(eta=0.05, q=0.5):
    S = json.load(open(SRC))["eta"]
    fig = plt.figure(figsize=(W1, 3.75))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 0.8], hspace=0.08)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    pgrid = np.logspace(-4.15, -1.85, 50)
    series = [("thmC", BLUE, "s", "full", "diamond-norm ball and Theorem 2"),
              ("vert", ORANGE, "o", "full", "Pauli error rates in the known frame"),
              ("E1u", AQUA, "^", "none", r"hockey-stick $E_1$, same ball ($\delta=1$)"),
              ("FUl", AQUA, "D", "none", r"Uhlmann $F_{\rm U}$, same ball ($\delta=1$)")]
    slopes = {}
    for w, col, mk, fill, lab in series:
        key = f"{w}|q{q}|eta{eta}"
        f = S["fits"][key]; nr = S["nreq"][key]
        P = np.array(f["ps"]); N = np.array(f["N_total"])
        lo = np.array([nr[f"{p:.0e}"]["total_boot95"][0] for p in P])
        hi = np.array([nr[f"{p:.0e}"]["total_boot95"][1] for p in P])
        sl = f["slope"]; slopes[w] = (sl, f["slope_ci95_ols"])
        a.plot(pgrid, 10 ** (f["intercept_log10"] + sl * np.log10(pgrid)), color=col, lw=0.9,
               ls="-" if fill == "full" else (0, (4, 1.5)), zorder=2)
        a.errorbar(P, N, yerr=[N - lo, hi - N], fmt=mk, ms=4.2 if mk != "D" else 3.6,
                   mfc=col if fill == "full" else "white", mec=col, mew=1.0, ecolor=col, elinewidth=0.7,
                   capsize=0, zorder=3, label=f"{lab}; slope {_fmt(sl)}")
    c0 = 0.0246
    a.plot(pgrid, c0 / (eta ** 2 * pgrid / 4), color=INK, lw=0.9, ls=(0, (1.2, 1.4)), zorder=2,
           label=r"reference scale $c_0/(\eta^2\tilde\lambda)$")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(pgrid[0], pgrid[-1]); a.set_ylim(1e3, 1e17)
    a.set_ylabel(r"channel uses per channel, $N$")
    a.tick_params(labelbottom=False)
    a.set_yticks([1e3, 1e5, 1e7, 1e9, 1e11, 1e13, 1e15, 1e17])
    panel_tag(a, "(a)", x=0.025)
    a.text(0.97, 0.97, rf"$\eta={eta}$ in $\ln\hat F$, $1-\alpha=0.95$", transform=a.transAxes, va="top",
           ha="right", fontsize=7, color=INK2)
    h, l = a.get_legend_handles_labels()
    order = [1, 2, 3, 4, 0] if len(h) == 5 else list(range(len(h)))
    h, l = [h[i] for i in order], [l[i] for i in order]
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=1, handlelength=1.8, labelspacing=0.3,
               borderaxespad=0.0, frameon=False, fontsize=7)
    cov = [r for r in S["coverage"] if r["side"] == "converse"]
    off = {"thmC": 0.94, "vert": 1.06}
    for w, col, mk, fill, lab in series[:2]:
        rows = sorted([r for r in cov if r["pipeline"] == w], key=lambda r: r["p"])
        P = np.array([r["p"] for r in rows]) * off[w]
        y = np.array([r["cov_F"] for r in rows])
        lo = np.array([r["cov_F_cp95"][0] for r in rows]); hi = np.array([r["cov_F_cp95"][1] for r in rows])
        b.errorbar(P, y, yerr=[y - lo, hi - y], fmt=mk, ms=3.6, mfc=col, mec=col, mew=1.0, ecolor=col,
                   elinewidth=0.7, capsize=1.2, zorder=3)
    b.axhline(0.95, color=INK, lw=0.8, ls=(0, (4, 2)))
    b.text(2.0e-3, 0.953, r"target $1-\alpha=0.95$", fontsize=7, color=INK2, va="bottom")
    b.set_xscale("log"); b.set_xlim(pgrid[0], pgrid[-1]); b.set_ylim(0.935, 1.008)
    b.set_yticks([0.95, 1.0])
    b.set_ylabel("coverage", labelpad=6)
    b.set_xlabel(r"depolarizing probability $p$ (Choi-state floor $\tilde\lambda=p/4$)")
    panel_tag(b, "(b)", x=0.025, y=0.6)
    fig.savefig(os.path.join(HERE, "fig4_cert.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    for k, v in slopes.items():
        print(k, "slope %.3f CI [%.3f, %.3f]" % (v[0], v[1][0], v[1][1]))


if __name__ == "__main__":
    main()
