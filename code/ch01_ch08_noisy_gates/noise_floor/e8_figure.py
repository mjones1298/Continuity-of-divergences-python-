"""Figure for E8 from results/e8.json (after e8b_fix_d2.py)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "results/e8.json")))["gap_rows"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
cols = dict(zip([1e-2, 1e-3, 1e-4, 1e-5], plt.cm.viridis(np.linspace(0.1, 0.85, 4))))
for p in cols:
    sub = [r for r in rows if r["p"] == p]
    x = [r["x"] for r in sub]
    ax[0].loglog(x, [r["BS_over_D1"] for r in sub], "o-", color=cols[p], ms=3, label=f"p={p:g}")
    ax[0].loglog(x, [r["m2lnF_over_D1"] for r in sub], "s--", color=cols[p], ms=3)
    ax[0].loglog(x, [r["C_over_D1"] for r in sub], ":", color=cols[p])
    ax[1].loglog(x, [r["BS_over_zeta_upper"] for r in sub], "o-", color=cols[p], ms=3, label=f"p={p:g}: lower bound")
    s2 = [r for r in sub if "D2_half_opt" in r]
    if s2:
        ax[1].loglog([r["x"] for r in s2], [r["BS"] / r["D2_half_opt"] for r in s2], "v:", color=cols[p], ms=5,
                     label=f"p={p:g}: upper bound ($/D^{{(2)}}/2$)")
    ax[2].loglog(np.array(x) / np.log(1 / p), [r["BS_over_paper3"] for r in sub], "o-", color=cols[p], ms=3, label=f"p={p:g}")
ax[0].set_xlabel(r"$\delta/p$"); ax[0].set_ylabel(r"converse / $D^{(1)}(N_p\|N_{p,\delta})$")
ax[0].set_title("Ratio to single-letter Umegaki $D^{(1)}$\n" + r"solid: $\hat D_{BS}$, dashed: $-2\ln\hat F$, dotted: $\hat C$", fontsize=9)
ax[0].legend(fontsize=7)
ax[1].axhline(1, color="gray", lw=0.8)
ax[1].set_xlabel(r"$\delta/p$"); ax[1].set_ylabel(r"bracket on $\hat D_{BS}/\zeta$ (Stein)")
ax[1].set_title(r"$\hat D_{BS}/\zeta\ \in$ [solid, triangles]", fontsize=9); ax[1].legend(fontsize=6)
xx = np.logspace(-3, 3, 50)
ax[2].loglog(xx, xx, "k--", lw=1, label=r"$\propto\delta/(p\ln(1/p))$")
ax[2].axhline(1, color="gray", lw=0.8)
ax[2].set_xlabel(r"$\delta/(p\,\ln(1/p))$"); ax[2].set_ylabel(r"$\hat D_{BS}\,/\,U_3$  ($U_3\geq\zeta$, paper 3)")
ax[2].set_title("Certified overshoot where > 1: collapse in $\\delta/(p\\ln(1/p))$", fontsize=9); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e8_gap.png"), dpi=160)
print("ok")
