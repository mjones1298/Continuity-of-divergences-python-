"""Figs. 2 and 3 of the letter, regenerated as one-column vector PDFs from the
A_numerics data (A_numerics/results/t3.json and t2b_merged.json), with the
letter's notation (Fisher informations I_RLD, I_inf; Chernoff exponent xi is not
used as an axis variable).

fig2_window.pdf   : (a) window delta_0.1 vs p for weights w = 1, 2, 3 (local) and
                        global noise; (b) collapse of Fhat against u = delta*sqrt(kappa)/4.
fig3_crossover.pdf: (a) D_BS / zeta bracket vs delta/(p ln(1/p)); (b) zeta*p/delta^2 bracket.
"""
import json, os
import numpy as np
from figstyle import plt, INK2
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
R_T3 = os.path.join(HERE, "..", "multiqubit_window", "results", "")   # t3.json
R_T2 = os.path.join(HERE, "..", "crossover", "results", "")           # t2b_merged.json
C3 = ["#2a78d6", "#eb6834", "#1baf7a"]
GREY, INK = "#7a7974", "#222222"


# ------------------------------------------------------------------ Fig. 2
def fig2():
    T3 = json.load(open(R_T3 + "t3.json"))
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.9))
    ax = axs[0]
    mk = {1: "o", 2: "s", 3: "D"}
    for (name, w, c) in [("n1_local_Z", 1, C3[0]), ("n2_local_ZZ", 2, C3[1]), ("n3_local_ZZZ", 3, C3[2])]:
        rows = T3["window"][name]["rows"]
        p = np.array([r["p"] for r in rows]); ds = np.array([r["delta_star"] for r in rows])
        ax.loglog(p, ds, mk[w], color=c, ms=3.5, mfc="none", mew=1.0)
        pp = np.logspace(-5, -1, 50)
        ax.loglog(pp, 1.9372884 * (pp / 4) ** (w / 2), "-", color=c, lw=1.0)
    rows = T3["window"]["n2_local_ZI"]["rows"]   # weight-one error on two qubits (referee minor 21)
    p = np.array([r["p"] for r in rows]); ds = np.array([r["delta_star"] for r in rows])
    ax.loglog(p, ds, "x", color=INK2, ms=4.0, mew=0.9)
    for (name, n, ls) in [("n2_global_ZZ", 2, "--"), ("n3_global_ZZZ", 3, ":")]:
        rows = T3["window"][name]["rows"]
        p = np.array([r["p"] for r in rows]); ds = np.array([r["delta_star"] for r in rows])
        ax.loglog(p, ds, "^", color=GREY, ms=3.2, mfc="none", mew=0.9)
        pp = np.logspace(-5, -1, 50)
        ax.loglog(pp, 1.9372884 * np.sqrt(pp) / 2 ** n, ls, color=GREY, lw=1.0)
    h = [Line2D([], [], color=C3[0], marker="o", mfc="none", lw=1.0, ms=3.5),
         Line2D([], [], color=C3[1], marker="s", mfc="none", lw=1.0, ms=3.5),
         Line2D([], [], color=C3[2], marker="D", mfc="none", lw=1.0, ms=3.5),
         Line2D([], [], color=INK2, marker="x", ls="none", ms=4.0, mew=0.9),
         Line2D([], [], color=GREY, marker="^", mfc="none", ls="--", lw=1.0, ms=3.2),
         Line2D([], [], color=GREY, marker="^", mfc="none", ls=":", lw=1.0, ms=3.2)]
    ax.legend(h, [r"local, $Z$ ($w=1$)", r"local, $Z\otimes Z$ ($w=2$)", r"local, $Z^{\otimes3}$ ($w=3$)", r"local, $Z\otimes\mathbb{1}$ ($n=2$, $w=1$)",
                  r"global, $n=2$", r"global, $n=3$"], loc="upper left", ncol=2, fontsize=6.6,
              columnspacing=0.8, handlelength=1.8, labelspacing=0.25)
    ax.set_ylim(top=2e3)
    ax.set_xlabel(r"depolarizing probability $p$")
    ax.set_ylabel(r"window $\delta_{0.1}$ (rad)")
    ax.set_title(r"(a) window where $\hat F$ drops to $0.9$", fontsize=8, loc="left")

    ax = axs[1]
    styles = {"1e-2": "-", "1e-3": "--", "1e-4": ":"}
    for i, (name, n) in enumerate([("n1_local_Z", 1), ("n2_local_ZZ", 2), ("n3_local_ZZZ", 3)]):
        for pk, ls in styles.items():
            cv = T3["curves"][f"{name}_p{pk}"]
            u = np.array(cv["u"], float); F = np.array(cv["Fhat"], float)
            p = float(cv["p"]); delta = u * cv["scale"]
            m = delta <= np.pi
            kappa = ((4 - 3 * p) / p) ** n          # ratio of coupled Choi eigenvalues (w = n)
            ax.semilogx(delta[m] * np.sqrt(kappa) / 4, F[m], ls, color=C3[i], lw=1.2)
    uu = np.logspace(-2, 2, 200)
    ax.semilogx(uu, (1 + uu ** 2) ** -0.5, "-", color=INK, lw=0.7)
    ax.set_xlim(1e-2, 1e2)
    ax.set_xlabel(r"$u=\delta\sqrt{\kappa_n}/4$")
    ax.set_ylabel(r"$\hat F(\mathcal{N}_{p,\delta},\mathcal{N}_p)$")
    h = [Line2D([], [], color=C3[i]) for i in range(3)] + \
        [Line2D([], [], color=GREY, ls=ls) for ls in styles.values()] + [Line2D([], [], color=INK, lw=0.7)]
    ax.legend(h, [r"$n=w=1$", r"$n=w=2$", r"$n=w=3$", r"$p=10^{-2}$", r"$p=10^{-3}$", r"$p=10^{-4}$",
                  r"$(1+u^2)^{-1/2}$"], loc="lower left", ncol=1, fontsize=6.6, labelspacing=0.25)
    ax.set_title(r"(b) collapse of the exact curves", fontsize=8, loc="left")
    fig.tight_layout(h_pad=0.8)
    fig.savefig(os.path.join(HERE, "fig2_window.pdf"))
    print("wrote fig2_window.pdf")
    for name in ["n1_local_Z", "n2_local_ZZ", "n3_local_ZZZ", "n2_global_ZZ", "n3_global_ZZZ"]:
        print(name, T3["window"][name]["slope_p_le_1em3"])


# ------------------------------------------------------------------ Fig. 3
def fig3():
    T2 = json.load(open(R_T2 + "t2b_merged.json"))["results"]  # final bracket incl. two-copy D^#
    fig, axs = plt.subplots(2, 1, figsize=(3.37, 4.8), sharex=True)
    labs = {"0.01": r"$p=10^{-2}$", "0.001": r"$p=10^{-3}$", "0.0001": r"$p=10^{-4}$"}
    peaks = {}
    for i, pk in enumerate(["0.01", "0.001", "0.0001"]):
        rows = T2[pk]; p = float(pk); c = C3[i]
        x = np.array([r["x"] for r in rows]); d = np.array([r["delta"] for r in rows])
        lo = np.array([r["DBS_over_zu"] for r in rows]); hi = np.array([r["DBS_over_zl"] for r in rows])
        zl = np.array([r["zeta_lower"] for r in rows]); zu = np.array([r["zeta_upper"] for r in rows])
        dbs = np.array([r["DBS"] for r in rows])
        axs[0].fill_between(x, lo, hi, color=c, alpha=0.13, lw=0)
        axs[0].plot(x, lo, "-", color=c)
        axs[0].plot(x, hi, "--", color=c, lw=1.0)
        peaks[pk] = (lo.max(), x[lo.argmax()], d[lo.argmax()])
        axs[1].fill_between(x, zl * p / d ** 2, zu * p / d ** 2, color=c, alpha=0.13, lw=0)
        axs[1].plot(x, zu * p / d ** 2, "-", color=c)
        axs[1].plot(x, zl * p / d ** 2, "--", color=c, lw=1.0)
        axs[1].plot(x, dbs * p / d ** 2, ":", color=c, lw=1.2)
    ax = axs[0]
    ax.axhline(1, color=GREY, lw=0.6)
    ax.errorbar([0.0085], [np.sqrt(1.5)], yerr=[[np.sqrt(1.5) - 1], [1.5 - np.sqrt(1.5)]], fmt="none",
                ecolor=INK, capsize=2.5, lw=1.1)
    ax.text(0.0105, 1.62, r"$\delta\to0$: $[1,3/2]$", fontsize=6.5, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$D_{\rm BS}/\zeta$")
    h = [Line2D([], [], color=C3[i]) for i in range(3)] + [Line2D([], [], color=GREY, ls="-"),
                                                          Line2D([], [], color=GREY, ls="--")]
    ax.legend(h, [labs[k] for k in labs] + [r"divided by upper bound on $\zeta$", r"divided by lower bound on $\zeta$"],
              loc="upper left")
    ax.set_title(r"(a) converse relative to the Stein exponent", fontsize=8, loc="left")
    ax = axs[1]
    ax.axhline(1 / 3, color=GREY, lw=0.6, ls="-.")
    ax.text(0.0105, 0.375, r"$F_\infty/2\to1/3$", fontsize=6.5, color=GREY)
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax.text(0.0105, 0.6, r"$F_{\rm RLD}/2\to1/2$", fontsize=6.5, color=GREY)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\delta/(p\ln(1/p))$")
    ax.set_ylabel(r"$p\,\zeta/\delta^2$")
    ax.set_ylim(top=0.9)
    h = [Line2D([], [], color=GREY, ls="-"), Line2D([], [], color=GREY, ls="--"), Line2D([], [], color=GREY, ls=":")]
    ax.legend(h, [r"upper bound on $\zeta$", r"lower bound on $\zeta$", r"$D_{\rm BS}$"], loc="lower left")
    ax.set_title(r"(b) bracket on $\zeta$ in units of $\delta^2/p$", fontsize=8, loc="left")
    fig.tight_layout(h_pad=0.8)
    fig.savefig(os.path.join(HERE, "fig3_crossover.pdf"))
    print("wrote fig3_crossover.pdf", peaks)


if __name__ == "__main__":
    fig2()   # Fig. 3 is now produced by make_fig3_crossover.py
