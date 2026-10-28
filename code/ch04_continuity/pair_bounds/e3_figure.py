"""Figure for E3 (reads results/e3.json, e3b.json)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "results/e3.json")))
rows = D["rows"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
bins = np.logspace(-16, 0.5, 60)
for key, lab in (("rhs_candidate", "candidate bound"), ("rhs_thm104b", "v5 Thm 10.4(b)"), ("rhs_cor107", "v5 Cor 10.7")):
    r = [float(x["lhs"]) / float(x[key]) for x in rows if str(x[key]) not in ("inf", "Infinity") and float(x[key]) > 0]
    ax[0].hist(np.clip(r, 1e-16, None), bins=bins, histtype="step", lw=1.5, label=f"{lab} (max {max(r):.3f})")
ax[0].axvline(1, color="r", lw=1); ax[0].set_xscale("log")
ax[0].set_xlabel("LHS / RHS"); ax[0].set_ylabel("instances"); ax[0].set_title(f"E3: {len(rows)} quadruples, 0 violations"); ax[0].legend(fontsize=8)
fams = sorted(set(x["family"] for x in rows))
for f in fams:
    s = [x for x in rows if x["family"] == f and x["same_support"] and float(x["rhs_thm104b"]) > 0]
    if s:
        ax[1].loglog([float(x["rhs_thm104b"]) for x in s], [max(1e-16, float(x["lhs"])) for x in s], ".", ms=3, label=f)
xx = np.logspace(-8, 1, 10); ax[1].loglog(xx, xx, "k-", lw=1)
ax[1].set_ylim(1e-14, 10)
ax[1].set_xlabel(r"RHS of Thm 10.4(b): $\|\bar\Omega^*(1)\|(\tau^{1/2}+\sqrt{2}\tanh(d_T/2))$"); ax[1].set_ylabel(r"$|\hat F(\Phi_1,\Psi_1)-\hat F(\Phi_2,\Psi_2)|$")
ax[1].legend(fontsize=6); ax[1].set_title("Thm 10.4(b) by family (same-support quadruples)")
L = [x for x in rows if "lem105_delta" in x]
ax[2].semilogx([float(x["delta"]) for x in L], [float(x["lem105_gamma"]) for x in L], ".", ms=3, alpha=.5, label="random families")
ex = [x for x in L if x["family"] == "jones_ex10.6_CPmaps"]
ax[2].semilogx([float(x["delta"]) for x in ex], [float(x["lem105_gamma"]) for x in ex], "r-", lw=2, label="v5 Example 10.6")
B = json.load(open(os.path.join(HERE, "results/e3b.json")))["rows"]
B = [b for b in B if b["delta"] < 0.5]
ax[2].semilogx([b["delta"] for b in B], [b["ratio_gamma_bound"] for b in B], "k^", label="local-search maximiser (E3b)")
ax[2].axhline(1, color="gray", lw=0.8)
ax[2].set_xlabel(r"$\delta=\max(\|A_1-A_2\|,\|B_1-B_2\|)$"); ax[2].set_ylabel(r"$\tau\,/\,[2\sqrt{2}\delta/\sqrt{1-4\gamma^2}]$")
ax[2].set_title("Lemma 10.5 tightness"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e3_bounds.png"), dpi=150)
print("ok")
