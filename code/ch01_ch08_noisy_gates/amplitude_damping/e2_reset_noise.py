"""E2: reset channels R_0 vs R_theta, with depolarising noise; AD near gamma=1; controls;
geometric Renyi Qhat_alpha, Belavkin-Staszewski and Hellinger maximal divergences."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
from qlib import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = 2
ps = [0.0, 1e-3, 1e-2, 5e-2, 0.2]
thetas = np.logspace(-5, np.log10(np.pi / 2), 31)
alphas = [0.25, 0.5, 0.75]


def ket(th):
    return np.array([np.cos(th), np.sin(th)], complex)


def rho_reset(th, p):
    k = ket(th)
    return (1 - p) * np.outer(k, k.conj()) + p * np.eye(2) / 2


def closed_form_F(th, p):
    """Matsumoto fidelity tr(rho_0 # rho_theta) for the depolarised reset outputs
    (derived from the 2x2 formula A#B = sqrt(ab)(A/a+B/b)/sqrt(det(A/a+B/b)), a=sqrt(detA)):
       F = [1 + (1-p)^2 sin^2(theta) / (2p - p^2)]^{-1/2}   (p>0)."""
    if p == 0:
        return 1.0 if th == 0 else 0.0
    return 1 / np.sqrt(1 + (1 - p) ** 2 * np.sin(th) ** 2 / (2 * p - p ** 2))


rows = []
for p in ps:
    J0 = choi_replacement(rho_reset(0, p), d)
    for th in thetas:
        Jt = choi_replacement(rho_reset(th, p), d)
        F = fhat_all(J0, Jt, d, d)
        r = dict(p=p, theta=th, F=F, F_closed=closed_form_F(th, p),
                 uhlmann_out=uhlmann_fid(rho_reset(0, p), rho_reset(th, p)),
                 choi_state_fid=choi_state_fid(J0, Jt, d),
                 half_diamond=0.5 * diamond_sdp(J0 - Jt, d, d),
                 half_diamond_exact=(1 - p) * np.sin(th),
                 lam_min_JOm=float(np.linalg.eigvalsh(J0 + Jt).min()),
                 Q={a: qhat_alpha_arn(J0, Jt, a, d, d) for a in alphas},
                 Q_reg={a: fhat_reg_mp(Jt, J0, d, d, alpha=a)[0] for a in alphas},  # J(Psi)#_a J(Phi)
                 D_BS=dbs_arn(J0, Jt, d, d),
                 D_H=2 - 2 * F["arn"])
        rows.append(r)

# Hellinger identity check: D_H(Phi||Psi) = lmax Tr_Y(J1 + J2 - 2 J1#J2) = 2 - 2 Fhat for channels
# (verify with the generic h_f route for f=(1-sqrt t)^2: h(t) = 1 - 2 sqrt(t(1-t)))
hell_check = []
for r in rows[::7]:
    J0 = choi_replacement(rho_reset(0, r["p"]), d); Jt = choi_replacement(rho_reset(r["theta"], r["p"]), d)
    M, W, T = mean_fun_arn(J0, Jt, lambda t, s: 1 - 2 * np.sqrt(t * s))
    hell_check.append(abs(np.linalg.eigvalsh(ptrace_Y(M, d, d)).max() - r["D_H"]))

# empirical modulus 1 - F <= L theta^beta  (fit on 1e-10 < 1-F < 1e-2)
fits = {}
for p in ps:
    sub = [r for r in rows if r["p"] == p]
    th = np.array([r["theta"] for r in sub]); omf = np.array([1 - r["F"]["arn"] for r in sub])
    m = (omf > 1e-10) & (omf < 1e-2)
    if m.sum() >= 3:
        beta, logL = np.polyfit(np.log(th[m]), np.log(omf[m]), 1)
        L = float(np.max(omf[m] / th[m] ** beta))  # smallest L making it an upper bound on the fit window
        fits[p] = dict(beta=float(beta), L=L, L_fit=float(np.exp(logL)), L_times_p=L * p, n=int(m.sum()))
    else:
        fits[p] = dict(beta=0.0, L=float(omf.max()), note="1-F = 1 for every theta>0: jump (no modulus)")
    # theta where F = 1/2
    Fv = np.array([r["F"]["arn"] for r in sub])
    idx = np.where(Fv < 0.5)[0]
    if len(idx) and idx[0] > 0:
        i = idx[0]
        x = np.interp(0.5, [Fv[i], Fv[i - 1]], [np.log(th[i]), np.log(th[i - 1])])
        fits[p]["theta_half"] = float(np.exp(x))
        fits[p]["theta_half_pred_sqrt(6p)"] = float(np.arcsin(min(1, np.sqrt(3 * (2 * p - p * p)) / (1 - p))))

# amplitude damping near gamma = 1 and general gamma vs gamma'
ad = []
for g0 in [1.0, 0.9, 0.5]:
    for dg in np.logspace(-6, -1, 11):
        g1 = g0 - dg
        J0 = choi_from_kraus(kraus_ampdamp(g0)); J1 = choi_from_kraus(kraus_ampdamp(g1))
        F = fhat_all(J0, J1, d, d)
        ad.append(dict(gamma=g0, gamma2=g1, F=F, diamond=diamond_sdp(J0 - J1, d, d),
                       choi_state_fid=choi_state_fid(J0, J1, d),
                       lam_min_JOm=float(np.linalg.eigvalsh(J0 + J1).min()),
                       Q={a: qhat_alpha_arn(J0, J1, a, d, d) for a in alphas},
                       range_intersection_dim=int(np.linalg.matrix_rank(J0, 1e-10) + np.linalg.matrix_rank(J1, 1e-10)
                                                  - np.linalg.matrix_rank(np.hstack([J0, J1]), 1e-10))))
# controls: dephasing p vs p+dp, depolarising p vs p+dp
ctrl = []
for name, kf in [("dephasing", kraus_dephase), ("depolarizing", kraus_depol)]:
    for p0 in [0.05, 0.3]:
        for dp in np.logspace(-6, -1, 11):
            J0 = choi_from_kraus(kf(p0)); J1 = choi_from_kraus(kf(p0 + dp))
            F = fhat_all(J0, J1, d, d)
            ctrl.append(dict(channel=name, p=p0, dp=dp, F=F, one_minus_F=1 - F["arn"],
                             diamond=diamond_sdp(J0 - J1, d, d),
                             lam_min_JOm=float(np.linalg.eigvalsh(J0 + J1).min())))
ctrl_fit = {}
for name in ("dephasing", "depolarizing"):
    for p0 in [0.05, 0.3]:
        s = [c for c in ctrl if c["channel"] == name and c["p"] == p0]
        x = np.array([c["dp"] for c in s]); y = np.array([c["one_minus_F"] for c in s])
        m = y > 1e-12
        ctrl_fit[f"{name}_p{p0}"] = dict(beta=float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]),
                                         max_1mF=float(y.max()))

summary = dict(
    max_spread_three_methods=max(r["F"]["spread"] for r in rows + ad + ctrl),
    max_abs_arn_minus_closed_form=max(abs(r["F"]["arn"] - r["F_closed"]) for r in rows),
    max_abs_halfdiamond_minus_exact=max(abs(r["half_diamond"] - r["half_diamond_exact"]) for r in rows),
    max_abs_uhlmann_minus_choistate=max(abs(r["uhlmann_out"] - r["choi_state_fid"]) for r in rows),
    p0_uhlmann_minus_cos=max(abs(r["uhlmann_out"] - np.cos(r["theta"])) for r in rows if r["p"] == 0),
    Q_half_equals_F=max(abs(r["Q"][0.5] - r["F"]["arn"]) for r in rows),
    Q_arn_vs_mpreg=max(abs(r["Q"][a] - r["Q_reg"][a]) for r in rows for a in alphas),
    hellinger_identity_err=max(hell_check),
    modulus_fits=fits,
    Q_at_theta_near_1em3={str(p): (lambda rr: dict(theta=rr["theta"], **{str(a): rr["Q"][a] for a in alphas}))(
        min([r for r in rows if r["p"] == p], key=lambda r: abs(np.log(r["theta"] / 1e-3)))) for p in ps},
    D_BS_at_smallest_theta={p: [r["D_BS"] for r in rows if r["p"] == p][1] for p in ps},
    ampdamp=dict(max_F_over_gamma_ne_gamma2=max(a["F"]["arn"] for a in ad),
                 max_spread=max(a["F"]["spread"] for a in ad),
                 min_diamond=min(a["diamond"] for a in ad),
                 range_intersection_dims=sorted(set(a["range_intersection_dim"] for a in ad))),
    controls=ctrl_fit,
)
json.dump(dict(summary=summary, rows=rows, ampdamp=ad, controls=ctrl), open(os.path.join(HERE, "results/e2.json"), "w"),
          indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
print(json.dumps(summary, indent=1, default=float))

# ---------------- figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cols = plt.cm.viridis(np.linspace(0, 0.85, len(ps)))
for c, p in zip(cols, ps):
    s = [r for r in rows if r["p"] == p]
    th = [r["theta"] for r in s]
    ax[0].semilogx(th, [r["F"]["arn"] for r in s], "-", color=c, label=f"p={p:g}")
    ax[0].semilogx(th, [r["F"]["sdp"] for r in s], "x", color=c, ms=4)
s = [r for r in rows if r["p"] == 0]
ax[0].semilogx([r["theta"] for r in s], [r["uhlmann_out"] for r in s], "k--", label=r"Uhlmann $F(\rho_0,\rho_\theta)=\cos\theta$")
ax[0].semilogx([r["theta"] for r in s], [r["half_diamond"] for r in s], "k:", label=r"$\frac{1}{2}\|R_0-R_\theta\|_\diamond=\sin\theta$")
ax[0].set_xlabel(r"$\theta$ (rad)"); ax[0].set_ylabel(r"$\hat F(R^p_0,R^p_\theta)$ (lines ARN, x SDP)")
ax[0].set_title("E2: noisy reset channels"); ax[0].legend(fontsize=7)
for c, p in zip(cols, ps[1:]):
    s = [r for r in rows if r["p"] == p]
    ax[1].loglog([r["theta"] for r in s], [max(1e-16, 1 - r["F"]["arn"]) for r in s], "-", color=c, label=f"p={p:g}")
    ax[1].loglog(thetas, thetas ** 2 / (4 * p), ":", color=c)
ax[1].set_ylim(1e-12, 2)
ax[1].set_xlabel(r"$\theta$ (rad)"); ax[1].set_ylabel(r"$1-\hat F$")
ax[1].set_title(r"$1-\hat F$ (dotted: $\theta^2/4p$)"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e2_reset_noise.png"), dpi=150)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for k, a in enumerate(alphas):
    for c, p in zip(cols, ps):
        s = [r for r in rows if r["p"] == p]
        ax[k].semilogx([r["theta"] for r in s], [r["Q"][a] for r in s], "-", color=c, label=f"p={p:g}")
    ax[k].set_xlabel(r"$\theta$ (rad)"); ax[k].set_ylabel(r"$\hat Q_\alpha(R^p_0\|R^p_\theta)$")
    ax[k].set_title(rf"geometric Rényi quasi-entropy, $\alpha={a}$")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e2_qalpha.png"), dpi=150)

fig, ax = plt.subplots(figsize=(5.5, 4))
for g0, mk in zip([1.0, 0.9, 0.5], "os^"):
    s = [a for a in ad if a["gamma"] == g0]
    ax.semilogx([a["diamond"] for a in s], [a["F"]["arn"] for a in s], mk + "-", label=rf"$\hat F(AD_{{{g0}}},AD_{{\gamma'}})$")
    ax.semilogx([a["diamond"] for a in s], [a["choi_state_fid"] for a in s], mk + "--", mfc="none", label=f"Choi-state fid., γ={g0}")
ax.set_xlabel(r"$\|AD_\gamma-AD_{\gamma'}\|_\diamond$"); ax.set_ylabel("fidelity")
ax.set_title("Amplitude damping: channel-valued jump"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e2_ampdamp.png"), dpi=150)
