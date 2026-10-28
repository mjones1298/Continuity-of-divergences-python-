"""Fig. 3 of the letter and Fig. S2 of the Supplemental Material (revision 1).

Adapted from A_numerics/make_figures.py:crossover() (copied, not edited in place), same data:
A_numerics/results/t2b_merged.json (zeta bracket incl. the two-copy Fawzi-Fawzi bound).

fig3_crossover.pdf  (one column): (a) D_BS/zeta bracket, (b) p*zeta/delta^2 bracket with the local limits.
figS2_fidelity_ratio.pdf: (-2 ln F_hat)/zeta (former panel (b) of Fig. 3; not a Stein converse, moved to the SM).

Changes requested by the referee (M5, minor 27): two panels instead of three, height 4.5 in instead of 6.1 in,
labels >= 7 pt, no overlapping annotations; the numerical small-delta bracket [1, 3.16] and the proved
delta -> 0 bracket [1, F_RLD/F_inf] are both labelled; arrows mark delta = 2 sqrt(p), where the certified overshoot peaks.
"""
import json, os
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from figstyle import plt, BLUE, ORANGE, AQUA, INK, INK2, W1, panel_tag

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "crossover", "results", "")
PSET = [("0.01", r"$p=10^{-2}$", BLUE, "o"), ("0.001", r"$p=10^{-3}$", ORANGE, "s"), ("0.0001", r"$p=10^{-4}$", AQUA, "^")]
T2 = json.load(open(R + "t2b_merged.json"))["results"]
plt.rcParams.update({"figure.constrained_layout.use": True,
                     "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.02})


def arrays(pk):
    rows = T2[pk]
    g = lambda k: np.array([r[k] for r in rows], float)
    return g("x"), g("delta"), g("zeta_lower"), g("zeta_upper"), g("DBS"), g("m2lnFhat")


def fig3():
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.5), sharex=True, height_ratios=[1.15, 1.0])
    me = (0, 5)
    for pk, lab, c, mk in PSET:
        p = float(pk)
        x, d, zl, zu, dbs, fh = arrays(pk)
        ax[0].fill_between(x, dbs / zu, dbs / zl, color=c, alpha=0.15, lw=0)
        ax[0].plot(x, dbs / zu, color=c, ls="-", marker=mk, markevery=me, mfc=c, mec="white", mew=0.5)
        ax[0].plot(x, dbs / zl, color=c, ls="--", lw=0.9, marker=mk, markevery=me, mfc="white", mec=c, mew=0.7)
        s = p / d ** 2
        ax[1].fill_between(x, zl * s, zu * s, color=c, alpha=0.15, lw=0)
        ax[1].plot(x, zu * s, color=c, ls="-", marker=mk, markevery=me, mfc=c, mec="white", mew=0.5)
        ax[1].plot(x, zl * s, color=c, ls="--", lw=0.9, marker=mk, markevery=me, mfc="white", mec=c, mew=0.7)
        ax[1].plot(x, dbs * s, color=c, ls=":", lw=1.1)
        # arrow at delta = 2 sqrt(p), near the peak of the certified overshoot
        xpk = 2 * np.sqrt(p) / (p * np.log(1 / p))
        tr = blended_transform_factory(ax[0].transData, ax[0].transAxes)
        ax[0].plot([xpk], [0.965], marker="v", ms=5, color=c, mec="white", mew=0.4, transform=tr, clip_on=False, zorder=6)
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log")
    ax[0].axhline(1, color=INK2, lw=0.6)
    tr = blended_transform_factory(ax[0].transData, ax[0].transAxes)
    ax[0].text(3.2, 0.965, r"$\delta=2\sqrt{p}$", transform=tr, ha="right", va="center", fontsize=7, color=INK)
    # proved delta -> 0 bracket [1, F_RLD/F_inf] subset [1, 3/2]
    xb = 0.0075
    ax[0].plot([xb, xb], [1, 1.5], color=INK, lw=1.1, solid_capstyle="butt", zorder=5)
    for yb in (1.0, 1.5):
        ax[0].plot([xb * 0.88, xb * 1.13], [yb, yb], color=INK, lw=1.1, zorder=5)
    ax[0].text(0.0092, 1.22, r"proved, $\delta\to0$: $[1,\,3/2]$", fontsize=7, color=INK, va="center")
    ax[0].text(0.0092, 3.6, r"numerical: $[1,\,3.16]$", fontsize=7, color=INK, va="bottom")
    ax[0].set_ylabel(r"$D_{\rm BS}/\zeta$")
    ax[0].set_ylim(0.75, 60)
    ax[0].set_xlim(0.006, 110)
    hp = [Line2D([], [], color=c, marker=mk, mfc=c, mec="white", mew=0.5, label=lab) for _, lab, c, mk in PSET]
    ax[0].legend(handles=hp, loc="upper left", bbox_to_anchor=(0.06, 0.86), fontsize=7, labelspacing=0.25)
    # (b)
    ax[1].axhline(1 / 3, color=INK2, lw=0.7, ls="-.")
    ax[1].axhline(0.5, color=INK2, lw=0.7, ls=(0, (1, 1.5)))
    ax[1].set_ylabel(r"$p\,\zeta/\delta^{2}$")
    ax[1].set_xlabel(r"$\bar\delta=\delta\,/\,[p\,\ln(1/p)]$")
    ax[1].set_ylim(0.004, 1.2)
    hz = [Line2D([], [], color=INK2, ls="-", label=r"upper bound $\zeta_+$"),
          Line2D([], [], color=INK2, ls="--", lw=0.9, label=r"lower bound $\zeta_-$"),
          Line2D([], [], color=INK2, ls=":", lw=1.1, label=r"converse $D_{\rm BS}$"),
          Line2D([], [], color=INK2, ls=(0, (1, 1.5)), lw=0.7, label=r"$pF_{\rm RLD}/2\to1/2$"),
          Line2D([], [], color=INK2, ls="-.", lw=0.7, label=r"$pF_\infty/2\to1/3$ (proved, $\delta\to0$)")]
    ax[1].legend(handles=hz, loc="lower left", fontsize=6.8, labelspacing=0.2, handlelength=2.0)
    panel_tag(ax[0], "(a)"); panel_tag(ax[1], "(b)", x=0.94)
    fig.align_ylabels(ax)
    fig.savefig(os.path.join(HERE, "fig3_crossover.pdf"))
    plt.close(fig)


def figS2():
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    me = (0, 5)
    for pk, lab, c, mk in PSET:
        x, d, zl, zu, dbs, fh = arrays(pk)
        ax.fill_between(x, fh / zu, fh / zl, color=c, alpha=0.15, lw=0)
        ax.plot(x, fh / zu, color=c, ls="-", marker=mk, markevery=me, mfc=c, mec="white", mew=0.5, label=lab)
        ax.plot(x, fh / zl, color=c, ls="--", lw=0.9, marker=mk, markevery=me, mfc="white", mec=c, mew=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_ylabel(r"$-2\ln\hat F/\zeta$")
    ax.set_xlabel(r"$\bar\delta=\delta\,/\,[p\,\ln(1/p)]$")
    ax.set_xlim(0.006, 110); ax.set_ylim(0.3, 80)
    h = [Line2D([], [], color=c, marker=mk, mfc=c, mec="white", mew=0.5, label=lab) for _, lab, c, mk in PSET]
    h += [Line2D([], [], color=INK2, ls="-", label=r"divided by $\zeta_+$"), Line2D([], [], color=INK2, ls="--", lw=0.9, label=r"divided by $\zeta_-$")]
    ax.legend(handles=h, loc="upper left", fontsize=6.8, ncol=2, labelspacing=0.2, columnspacing=0.8)
    fig.savefig(os.path.join(HERE, "figS2_fidelity_ratio.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig3(); figS2()
    for pk, *_ in PSET:
        x, d, zl, zu, dbs, fh = arrays(pk)
        i = int(np.argmax(dbs / zu))
        print(pk, "peak D_BS/zeta_+ = %.3f at delta = %.4f = %.2f sqrt(p), x = %.2f; small-x D_BS/zeta_- = %.3f"
              % ((dbs / zu)[i], d[i], d[i] / np.sqrt(float(pk)), x[i], (dbs / zl)[0]))
