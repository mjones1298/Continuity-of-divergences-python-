"""Schematic figure (revision 1, referee Sec. 6 and fix item 9): the three regimes and the delta-scales.

Top: classification by Choi supports. Bottom: at a noisy qubit gate (p = 1e-4) the exact BS converse
D_BS = (2-p) phi(x) (closed form, SM) against a schematic exponent zeta_s = min(delta^2/(3p), delta ln(1/p)),
which only illustrates the two growth laws of Sec. VI (capped by D_BS and ln(4/p)); the certified bracket on zeta is in Fig. 3.
Output: fig0_schematic.pdf (one column).
"""
import os
import numpy as np
from figstyle import plt, BLUE, ORANGE, AQUA, RED, INK, INK2, W1
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
fig = plt.figure(figsize=(W1, 2.9))
top = fig.add_axes([0.0, 0.76, 1.0, 0.24]); top.axis("off")
boxes = [(0.01, r"$J(\Psi)$ singular", "genuine:\nexponent jumps", RED),
         (0.345, r"only $J(\Phi+\Psi)$ singular", "spurious:\nconverse jumps", ORANGE),
         (0.68, r"$J(\Phi+\Psi)$ invertible", "stable: scales\nset by the floor", BLUE)]
for x0, head, body, col in boxes:
    top.add_patch(FancyBboxPatch((x0, 0.08), 0.31, 0.84, boxstyle="round,pad=0.01,rounding_size=0.03",
                                 fc="white", ec=col, lw=1.0, transform=top.transAxes))
    top.text(x0 + 0.155, 0.72, head, ha="center", va="center", fontsize=7, color=INK, transform=top.transAxes)
    top.text(x0 + 0.155, 0.35, body, ha="center", va="center", fontsize=7, color=col, transform=top.transAxes)

ax = fig.add_axes([0.14, 0.13, 0.83, 0.58])
p = 1e-4
d = np.logspace(np.log10(p * np.log(1 / p)) - 2, np.log10(np.pi) - 0.2, 400)
x = 4 * (1 - p) ** 2 * np.sin(d / 2) ** 2 / (p * (4 - 3 * p))
phi = np.sqrt(x / (1 + x)) * np.arcsinh(np.sqrt(x))
dbs = (2 - p) * phi
zs = np.minimum.reduce([d ** 2 / (3 * p), d * np.log(1 / p), 0.999 * dbs, np.full_like(d, np.log(4 / p))])  # zeta <= D_BS and zeta <= ln(4/p)
ax.loglog(d, dbs, color=BLUE, lw=1.4, label=r"converse $D_{\rm BS}$ (exact)")
ax.loglog(d, zs, color=ORANGE, lw=1.4, ls="--", label=r"exponent $\zeta$ (schematic)")
d1, d2 = p * np.log(1 / p), 2 * np.sqrt(p)
ylo, yhi = 1e-7, 1e2
ax.axvspan(d[0], d1, color=AQUA, alpha=0.10, lw=0)
ax.axvspan(d1, d2, color=ORANGE, alpha=0.10, lw=0)
for xv in (d1, d2):
    ax.axvline(xv, color=INK2, lw=0.6, ls=":")
ax.text(np.sqrt(d[0] * d1), 3e0, "genuine", ha="center", fontsize=7, color=INK)
ax.text(np.sqrt(d1 * d2), 3e0, "spurious", ha="center", fontsize=7, color=INK)
ax.text(np.sqrt(d2 * d[-1]) * 1.2, 1e-3, "saturation", ha="center", fontsize=7, color=INK)
ax.set_xticks([d1, d2]); ax.set_xticklabels([r"$p\ln(1/p)$", r"$2\sqrt{p}$"])
ax.minorticks_off()
ax.set_xlim(d[0], d[-1]); ax.set_ylim(ylo, yhi)
ax.set_xlabel(r"miscalibration $\delta$ (log scale)")
ax.set_ylabel("nats per use")
ax.legend(loc="lower right", fontsize=7)
fig.savefig(os.path.join(HERE, "fig0_schematic.pdf"))
print("wrote fig0_schematic.pdf")
