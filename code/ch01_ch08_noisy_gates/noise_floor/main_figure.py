"""Letter main figure (4 panels), from results/e1.json, e6.json, e7.json.
(a) replacement-channel jump: Fhat and C-hat vs xi_QC along paper-2 'fragile' sequence + Cor 3.3
(b) noise-floor sensitivity per divergence family vs 1/p
(c) data collapse Fhat(N_p, N_{p,delta}) vs delta/sqrt(p)
(d) tomography cost N* vs p
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
E6 = json.load(open(os.path.join(HERE, "results/e6.json")))
E7 = json.load(open(os.path.join(HERE, "results/e7.json")))

fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))

# (a)
ax = axs[0, 0]
fr = E7["fragile_rows"]
t = np.array([r["t"] for r in fr]); dia = np.array([r["diamond_Phi_t_vs_Phi"] for r in fr])
ax.loglog(dia, [r["xi_QC"] for r in fr], "o-", color="tab:green", label=r"true adaptive exponent $\xi_{QC}(\phi_t,\psi_t)\approx4t$")
ax.loglog(dia, [r["hockey1"] for r in fr], "s-", color="tab:gray", label=r"hockey-stick $E_1(\phi_t\|\psi_t)$")
ax.loglog(dia, [1 - r["uhlmann_root_fid"] for r in fr], "^-", color="tab:purple", label=r"$1-F_{Uhl}(\phi_t,\psi_t)$")
top = 30
ax.loglog(dia, np.full_like(dia, top), "r-", lw=3, label=r"$\hat C=-2\ln\hat F=C_{bar}=+\infty$ (all t)")
ax.annotate(r"$\hat F(\Phi_t,\Psi_t)=0$ for every $t$", (dia[2], top * 0.45), color="r", fontsize=8)
ax.set_xlabel(r"$\|\Phi_t-\Phi\|_\diamond$  ($\Phi=\Psi=\mathrm{tr}(\cdot)\phi$)")
ax.set_ylabel("value")
ax.set_title("(a) spurious jump at replacement channels", fontsize=10)
ax.legend(fontsize=7, loc="lower right"); ax.set_ylim(1e-7, 100)

# (b)
ax = axs[0, 1]
S = E7["sensitivity_rows"]; sl = E7["loglog_slopes_vs_1_over_p"]
ip = np.array([1 / r["p"] for r in S])
series = [("geo_1mF", r"geometric $1-\hat F$", "o-", "tab:red"),
          ("chernoff", r"geometric Chernoff $\hat C$", "s--", "darkred"),
          ("BS", r"Belavkin–Staszewski", "D-", "tab:orange"),
          ("umegaki", r"Umegaki $D(\Phi\|\Psi)$", "^-", "tab:blue"),
          ("ht_hellinger", r"HT squared Hellinger", "v-", "tab:cyan"),
          ("uhlmann_1mF", r"Uhlmann $1-F$", "x-", "tab:purple")]
for k, lab, mk, c in series:
    ax.loglog(ip, [r[k] for r in S], mk, color=c, ms=4, label=f"{lab}  [slope {sl[k]:.2f}]")
ax.loglog(ip, [r["hockey"]["1.0"] for r in S], "k:",
          label=f"hockey-stick $E_1/\\delta$  [slope {sl['hockey_1']:.2f}]")
ax.set_xlabel(r"$1/p$"); ax.set_ylabel(r"divergence $/\delta^2$  (hockey: $/\delta$)")
ax.set_title(r"(b) local sensitivity at $\delta=10^{-2}\sqrt{p}$ (depolarized H)", fontsize=10)
ax.legend(fontsize=6.5, loc="upper left")

# (c)
ax = axs[1, 0]
R = E6["rows_depol"]
cols = plt.cm.viridis(np.linspace(0.1, 0.85, 4))
for c, p in zip(cols, [1e-2, 1e-3, 1e-4, 1e-5]):
    rows = R[f"H_p{p:g}"]
    th = np.array([r["theta"] for r in rows])
    ax.semilogx(th / np.sqrt(p), [r["F"] for r in rows], "-", color=c, lw=2.2, label=f"p = {p:g}")
    ax.semilogx(th / np.sqrt(p), [r["diamond"] / 2 for r in rows], ":", color=c, lw=1)
rows0 = R["H_p0"]
x = np.logspace(-3, 3, 200)
ax.semilogx(x, 1 / np.sqrt(1 + x ** 2), "k--", lw=1, label=r"$(1+x^2)^{-1/2}$")
ax.set_xlim(1e-3, 1e3); ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"$\delta/\sqrt{p}$"); ax.set_ylabel(r"$\hat F(N_p,N_{p,\delta})$  (dotted: $\frac{1}{2}\|\cdot\|_\diamond$)")
ax.set_title(r"(c) data collapse; p = 0: $\hat F$ = 0 for all $\delta>0$", fontsize=10)
ax.legend(fontsize=7, loc="center left")

# (d)
ax = axs[1, 1]
T = E7["tomography"]
pp = sorted(float(p) for p in T)
for key, lab, mk, c in (("Nstar_F", r"geometric $\hat F$", "o-", "tab:red"),
                        ("Nstar_uhl", r"Choi Uhlmann", "^-", "tab:purple"),
                        ("Nstar_hockey", r"Choi hockey-stick $E_1$", "s-", "k")):
    ys = [T[[k for k in T if float(k) == p][0]][key] for p in pp]
    xs = [p for p, y in zip(pp, ys) if y is not None]
    ys2 = [y for y in ys if y is not None]
    ax.loglog(xs, ys2, mk, color=c, label=lab)
    miss = [p for p, y in zip(pp, ys) if y is None]
    if miss:
        ax.loglog(miss, [3e10] * len(miss), "v", color=c, mfc="none", ms=9)
ax.loglog(pp, [1e2 * (1e-2 / p) ** 2 for p in pp], "r:", lw=1, label=r"$\propto p^{-2}$")
ax.set_xlabel(r"noise floor $p$"); ax.set_ylabel(r"shots/setting $N^*$ (90% within 10%)")
ax.set_title(r"(d) plug-in tomography cost ($\delta=\sqrt{p}$; open ▽: $>10^{10}$)", fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures/main_figure.png"), dpi=170)
print("main figure written")
