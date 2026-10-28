"""Fig. 1 of the letter (revision 1): spurious fragility of single-letter Chernoff converses.

(a) amplitude damping A_gamma vs A_gamma' at gamma = 1/2 (closed forms of the companion Chernoff paper:
    Bernoulli Chernoff information = product-input lower bound xi_prod; -2 ln mu_* = proved adaptive upper bound;
    -2 ln f = conjectured bound);
(b) replacement channels (data: numerics/results/e7.json, 'fragile_rows').
Changes from v1: panels swapped (amplitude damping, the physical example, first); legend placed outside the
shaded band; the +infinity line is labelled as schematic; Computer Modern fonts shared with Figs. 2-4.
Output: fig1_regimes.pdf (one column, vector).
"""
import json, os
import numpy as np
from scipy.optimize import minimize_scalar
from figstyle import plt, BLUE, ORANGE, AQUA, RED, GREY, INK2, W1, panel_tag

HERE = os.path.dirname(os.path.abspath(__file__))
E7 = json.load(open(os.path.join(HERE, "..", "noise_floor", "results", "e7.json")))
TOP = 100.0  # schematic plotting height that represents +infinity

fig, axs = plt.subplots(2, 1, figsize=(W1, 4.9))


def bern_chernoff(a, b):
    f = lambda s: np.log(a ** s * b ** (1 - s) + (1 - a) ** s * (1 - b) ** (1 - s))
    r = minimize_scalar(f, bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
    return -r.fun


# ---------------- (a) amplitude damping ----------------
ax = axs[0]
g = 0.5
Delta = np.logspace(-3, np.log10(0.3), 60)
gp = g - Delta
lower = np.array([bern_chernoff(g, b) for b in gp])
a = np.sqrt((1 - g) * (1 - gp))
mu = np.sqrt(g * gp) / (1 - a)
upper = -2 * np.log(mu)
fb = np.sqrt(g * gp) + a
conj = -2 * np.log(fb)
ax.loglog(Delta, np.full_like(Delta, TOP), color=RED, lw=2.4, solid_capstyle="butt")
ax.text(Delta[0] * 1.15, TOP / 2.2, va="top", s= r"$\hat C=-2\ln\hat F=C_{\rm bar}=+\infty$ for every $\gamma'\neq\gamma$ (schematic height)",
        color=RED, fontsize=6.6)
ax.fill_between(Delta, lower, upper, color=BLUE, alpha=0.15, lw=0, label=r"range allowed for $\xi$")
ax.loglog(Delta, upper, "-", color=BLUE, label=r"$-2\ln\mu_*$ (proved)")
ax.loglog(Delta, conj, ":", color=AQUA, lw=1.5, label=r"$-2\ln f$ (conjectured)")
ax.loglog(Delta, lower, "--", color=ORANGE, label=r"$\xi_{\rm prod}$ (lower bound)")
ax.set_ylim(1e-7, 1e4)
ax.set_xlim(Delta[0], Delta[-1])
ax.set_xlabel(r"difference of damping rates $\Delta=\gamma-\gamma'$ ($\gamma=1/2$)")
ax.set_ylabel("exponent (nats per use)")
# legend in the empty upper-left region, between the infinite line and the curves
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.74), fontsize=6.6, labelspacing=0.3)
panel_tag(ax, "(a) amplitude damping", x=0.0, y=1.09)

# ---------------- (b) replacement channels ----------------
ax = axs[1]
fr = E7["fragile_rows"]
dia = np.array([r["diamond_Phi_t_vs_Phi"] for r in fr])
xi = np.array([r["xi_QC"] for r in fr])
e1 = np.array([r["hockey1"] for r in fr])
assert all(r["C_geo"] == float("inf") for r in fr)
ax.loglog(dia, np.full_like(dia, TOP), color=RED, lw=2.4, solid_capstyle="butt")
ax.text(dia[0] * 1.15, TOP / 2.2, va="top", s= r"$\hat C=-2\ln\hat F=C_{\rm bar}=+\infty$ for every $t>0$ (schematic height)",
        color=RED, fontsize=6.6)
ax.loglog(dia, xi, "o-", color=BLUE, ms=3.2, label=r"true exponent $\xi\approx4t$")
ax.loglog(dia, e1, "s--", color=GREY, ms=2.8, lw=1.0, label=r"$\frac{1}{2}\|\Phi_t-\Psi_t\|_\diamond$")
ax.set_ylim(1e-6, 1e4)
ax.set_xlabel(r"distance to the limiting pair $\|\Phi_t-\Phi\|_\diamond\approx2\sqrt{t}$")
ax.set_ylabel("exponent (nats per use)")
ax.legend(loc="lower right")
panel_tag(ax, "(b) replacement channels", x=0.0, y=1.09)

fig.tight_layout(h_pad=1.6)
out = os.path.join(HERE, "fig1_regimes.pdf")
fig.savefig(out)
print("wrote", out)
for D in (0.1, 0.03, 0.01):
    b = g - D
    aa = np.sqrt((1 - g) * (1 - b))
    print(D, bern_chernoff(g, b), -2 * np.log(np.sqrt(g * b) / (1 - aa)), -2 * np.log(np.sqrt(g * b) + aa))
