"""Figures for E6 (reads results/e6.json)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "results/e6.json")))
S = D["summary"]
R = D["rows_depol"]
ps = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
cols = dict(zip(ps, plt.cm.viridis(np.linspace(0, 0.9, len(ps)))))


def col(rows, k, sub=None):
    return np.array([(r[k] if sub is None else r[k][sub]) for r in rows], float)


# ---- Fig 1: noise floor plot
fig, ax = plt.subplots(figsize=(7, 4.6))
for p in ps:
    rows = R[f"H_p{p:g}"]
    th = col(rows, "theta")
    ax.semilogx(th, col(rows, "F"), "-", color=cols[p], lw=2, label=f"p = {p:g}")
    sd = [(r["theta"], r["F_sdp"]) for r in rows if "F_sdp" in r]
    ax.semilogx(*zip(*sd), "x", color=cols[p], ms=5)
rows0 = R["H_p0"]
ax.semilogx(col(rows0, "theta"), col(rows0, "diamond") / 2, "k--", lw=1.2, label=r"$\frac{1}{2}\|N_0-N_{0,\theta}\|_\diamond$ (p=0)")
ax.semilogx(col(rows0, "theta"), col(rows0, "uhlmann_channel"), "k:", lw=1.5, label="Uhlmann channel fidelity (p=0)")
ax.set_xlabel(r"miscalibration angle $\theta$ (rad),  $U_\theta = e^{-i\theta Z}H$")
ax.set_ylabel(r"$\hat F(N_p, N_{p,\theta})$")
ax.set_title("Noise floor: geometric fidelity of a depolarized Hadamard\n(lines: ARN formula, x: SDP)", fontsize=10)
ax.set_ylim(-0.03, 1.05)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e6_noise_floor.png"), dpi=160)

# ---- Fig 2: data collapse  F vs theta/sqrt(p), and 1-F vs theta
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for p in ps[1:]:
    rows = R[f"H_p{p:g}"]
    th = col(rows, "theta")
    ax[0].semilogx(th / np.sqrt(p), col(rows, "F"), "-", color=cols[p], lw=2, label=f"p = {p:g}")
xs = np.logspace(-4, 3, 200)
ax[0].semilogx(xs, 1 / np.sqrt(1 + xs ** 2), "k--", lw=1, label=r"$(1+x^2)^{-1/2}$ (empirical law, p$\to$0)")
ax[0].set_xlabel(r"$\theta/\sqrt{p}$"); ax[0].set_ylabel(r"$\hat F(N_p,N_{p,\theta})$")
ax[0].set_title("Data collapse (depolarized Hadamard, $H_{err}=Z$)"); ax[0].legend(fontsize=8)
ax[0].set_xlim(1e-3, 3e2)
for p in ps[1:]:
    rows = R[f"H_p{p:g}"]
    ax[1].loglog(col(rows, "diamond"), np.maximum(1e-16, 1 - col(rows, "F")), "-", color=cols[p], lw=2, label=f"1−F̂, p={p:g}")
    ax[1].loglog(col(rows, "diamond"), [r["thm124a_rhs_nu_base"] for r in rows], ":", color=cols[p], lw=1.2)
ax[1].axhline(1, color="gray", lw=0.8)
ax[1].set_xlabel(r"$d_{cb}=\|N_p-N_{p,\theta}\|_\diamond$"); ax[1].set_ylabel(r"$|\Delta\hat F|$")
ax[1].set_title(r"measured $1-\hat F$ (solid) vs v5 Thm 12.4(a) bound (dotted)", fontsize=10)
ax[1].set_ylim(1e-14, 1e4); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e6_collapse_and_bound.png"), dpi=160)

# ---- Fig 3: local sensitivities vs 1/p
sens = S["sensitivity"]["rows"]
sl = S["sensitivity"]["loglog_slopes_vs_1_over_p"]
ip = np.array([1 / s["p"] for s in sens])
fig, ax = plt.subplots(figsize=(7, 4.6))
for key, lab, mk in [("geo", r"$(1-\hat F)/\theta^2$ geometric", "o-"), ("BS", r"$\hat D_{BS}/\theta^2$", "s-"),
                     ("I_RLD", r"channel RLD Fisher info $I_{RLD}$", "k--"),
                     ("umegaki", r"Umegaki channel $D/\theta^2$", "^-"), ("uhlmann", r"$(1-F_{Uhl})/\theta^2$ channel", "v-"),
                     ("hockey105_over_theta", r"$E_{1.05}/\theta$ hockey-stick", "d-"),
                     ("hockey1_over_theta", r"$E_{1}/\theta$ (= $\frac{1}{2}\|\cdot\|_\diamond/\theta$)", "x-")]:
    y = np.array([s[key] for s in sens])
    ax.loglog(ip, np.maximum(y, 1e-12), mk, ms=4, label=f"{lab}  (slope {sl[key]:.2f})")
ax.set_xlabel(r"$1/p$"); ax.set_ylabel(r"local sensitivity (per $\theta^2$; hockey per $\theta$)")
ax.set_title(r"Local sensitivity at $\theta = 10^{-2}\sqrt{p}$ (depolarized Hadamard)", fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e6_sensitivity_vs_p.png"), dpi=160)

# ---- Fig 5: Qhat_alpha
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for k, a in enumerate(["0.25", "0.5", "0.75"]):
    for p in ps:
        rows = R[f"H_p{p:g}"]
        ax[k].semilogx(col(rows, "theta"), [r["Q"][a] for r in rows], color=cols[p], label=f"p={p:g}")
    ax[k].set_xlabel(r"$\theta$ (rad)"); ax[k].set_ylabel(rf"$\hat Q_{{{a}}}(N_{{p,\theta}}\|N_p)$")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e6_qalpha.png"), dpi=150)
# ---- Fig 4: noise models and CNOT
M = D.get("rows_models")
if M is None or D.get("rows_cnot") is None:
    print("models/cnot not yet merged; skipping fig 4")
    raise SystemExit
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
mcols = dict(gad="tab:blue", gad_lowT="tab:cyan", t1t2="tab:red", t1="tab:orange", pauli="tab:green")
for model, c in mcols.items():
    for p, ls in zip([1e-4, 1e-2], ["-", "--"]):
        rows = M[f"{model}_p{p:g}"]
        ax[0].semilogx(col(rows, "theta"), col(rows, "F"), ls, color=c, label=f"{model}, p={p:g}")
ax[0].set_xlabel(r"$\theta$ (rad),  $U_\theta=e^{-i\theta\, \hat n\cdot\vec\sigma}H$"); ax[0].set_ylabel(r"$\hat F$")
ax[0].set_title("Noise models (gad: γ=p,N=0.05; gad_lowT: γ=0.01,N=p;\n t1: AD γ=p; t1t2: AD γ=p + dephasing p; pauli: p(.1,.2,.7))", fontsize=8)
ax[0].legend(fontsize=6, ncol=2)
C = D["rows_cnot"]
for model, ls in (("depol", "-"), ("depol_local2", "--")):
    for p in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
        rows = C[f"{model}_p{p:g}"]
        ax[1].semilogx(col(rows, "theta"), col(rows, "F"), ls, color=cols.get(p, "k"), label=f"{model}, p={p:g}")
ax[1].set_xlabel(r"$\theta$ (rad),  $U_\theta = e^{-i\theta Z\otimes Z}$CNOT"); ax[1].set_ylabel(r"$\hat F$")
ax[1].set_title("Two-qubit CNOT (global vs local depolarizing)", fontsize=9); ax[1].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e6_models_cnot.png"), dpi=160)

print("figures written")
