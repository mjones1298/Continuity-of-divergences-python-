"""E1: reproduce Jones Cor 3.3 (replacement channels, same sum)."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
from qlib import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = 2
thetas = np.logspace(-4, np.log10(np.pi / 4), 25)
rows = []
for th in thetas:
    e0 = np.array([1, 0], complex)
    et = np.array([np.cos(th), np.sin(th)], complex)
    J1 = choi_replacement(np.outer(e0, e0.conj()), d)
    K1 = choi_replacement(np.outer(et, et.conj()), d)
    JOm = J1 + K1
    J2 = K2 = JOm / 2
    W = arn_frame(JOm)
    T1 = arn_derivative(W, J1)
    T2 = arn_derivative(W, J2)
    F1 = fhat_all(J1, K1, d, d)
    F2 = fhat_all(J2, K2, d, d)
    dn = diamond_sdp(J1 - J2, d, d)
    G = W.conj().T @ W
    nu = nu_bounds(JOm, d, d, np.random.default_rng(0), 20)
    rows.append(dict(theta=th, diamond_Phi1_Phi2=dn, diamond_exact_sin=np.sin(th),
                     diamond_Psi1_Psi2=diamond_sdp(K1 - K2, d, d),
                     F1=F1, F2=F2, T1_minus_T2=opnorm(T1 - T2),
                     lmin_G=float(np.linalg.eigvalsh(G).min()), rank_Omega=W.shape[1],
                     specT1=np.linalg.eigvalsh(T1).tolist(), nu2=nu["nu2_opt"], nu2_lower=nu["nu2_sdp_lower"],
                     lam_min_pos_J=nu["lam_min_pos"]))

def maxdev(key, target):
    return max(abs(r[key][m] - target) for r in rows for m in ("arn", "reg", "sdp"))

summary = dict(
    reg_note="'reg' = mpmath (90 digits) eps-regularised mean, eps=1e-24..1e-40, extrapolated eps->0",
    double_precision_reg_failure=dict(theta=1e-4, value=fhat_reg(choi_replacement(np.diag([1,0]).astype(complex),2),
        choi_replacement(np.outer([np.cos(1e-4),np.sin(1e-4)],[np.cos(1e-4),np.sin(1e-4)]).astype(complex),2),2,2),
        note="double-precision eps in [1e-9,1e-4] cannot reach eps << lambda_min(G)=theta^2/2; gives a spurious value"),
    max_spread_methods=max(max(r["F1"]["spread"], r["F2"]["spread"]) for r in rows),
    n_theta=len(thetas),
    max_abs_F1_minus_0_over_methods=maxdev("F1", 0.0),
    max_abs_F2_minus_1_over_methods=maxdev("F2", 1.0),
    max_abs_T1T2_minus_half=max(abs(r["T1_minus_T2"] - 0.5) for r in rows),
    max_abs_diamond_minus_sin=max(abs(r["diamond_Phi1_Phi2"] - r["diamond_exact_sin"]) for r in rows),
    lmin_G_fit_at_small_theta=dict(theta=[r["theta"] for r in rows[:3]], lmin_G=[r["lmin_G"] for r in rows[:3]]),
    lmin_G_over_theta2_small=rows[0]["lmin_G"] / rows[0]["theta"] ** 2,
    nu2_over_theta2_small=rows[0]["nu2"] / rows[0]["theta"] ** 2,
)
json.dump(dict(summary=summary, rows=rows), open(os.path.join(HERE, "results/e1.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
th = np.array([r["theta"] for r in rows])
ax[0].semilogx(th, [r["F1"]["arn"] for r in rows], "o-", label=r"$\hat F(\Phi_1,\Psi_1)$ (ARN)")
ax[0].semilogx(th, [r["F1"]["sdp"] for r in rows], "x", label=r"$\hat F(\Phi_1,\Psi_1)$ (SDP)")
ax[0].semilogx(th, [r["F2"]["arn"] for r in rows], "s-", label=r"$\hat F(\Phi_2,\Psi_2)$ (ARN)")
ax[0].semilogx(th, [r["F2"]["sdp"] for r in rows], "+", label=r"$\hat F(\Phi_2,\Psi_2)$ (SDP)")
ax[0].semilogx(th, [r["diamond_Phi1_Phi2"] for r in rows], "k--", label=r"$\|\Phi_1-\Phi_2\|_\diamond$")
ax[0].set_xlabel(r"$\theta$ (rad)"); ax[0].set_ylabel("value (dimensionless)")
ax[0].set_title("E1: Jones Cor. 3.3"); ax[0].legend(fontsize=8)
ax[1].loglog(th, [r["lmin_G"] for r in rows], "o-", label=r"$\lambda_{\min}(G)$")
ax[1].loglog(th, [r["nu2"] for r in rows], "s-", label=r"$\nu(\Omega)^2$")
ax[1].loglog(th, [r["T1_minus_T2"] for r in rows], "^-", label=r"$\|T_1-T_2\|_\infty$")
ax[1].loglog(th, th ** 2, "k:", label=r"$\theta^2$")
ax[1].set_xlabel(r"$\theta$ (rad)"); ax[1].set_ylabel("value (dimensionless)")
ax[1].legend(fontsize=8); ax[1].set_title("ARN-side quantities")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e1_cor33.png"), dpi=150)
