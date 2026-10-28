"""E5: channel hockey-stick divergence E_gamma(Phi||Psi) = sup_psi Tr((Phi(x)id)psi - gamma(Psi(x)id)psi)_+
(SDP) is diamond-Lipschitz:  |E_g(Phi1||Psi1) - E_g(Phi2||Psi2)| <= (||dPhi||_dia + g ||dPsi||_dia)/2
on the E1 (Cor 3.3) and E2 (noisy reset, amplitude damping) families, plus random pairs."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ch04_continuity" / "pair_bounds"))  # e3_pair_bounds.py
from qlib import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(55)
gammas = [1.0, 1.05, 2.0, 10.0]


def rep(th, p=0.0):
    k = np.array([np.cos(th), np.sin(th)], complex)
    return choi_replacement((1 - p) * np.outer(k, k.conj()) + p * np.eye(2) / 2, 2)


quads = []
# E1 family: (Phi1,Psi1) = (R_0, R_th), (Phi2,Psi2) = ((R_0+R_th)/2, same)
for th in np.logspace(-4, np.log10(np.pi / 4), 12):
    J1, K1 = rep(0), rep(th)
    quads.append(("E1_cor33", th, J1, K1, (J1 + K1) / 2, (J1 + K1) / 2))
# E2 family: (R0^p, R0^p) vs (R0^p, Rth^p) and (R0^p, R_a^p) vs (R0^p, R_{a+th}^p)
for p in (0.0, 1e-3, 1e-2, 0.2):
    for th in np.logspace(-5, 0, 10):
        quads.append(("E2_reset_vs_self", (p, th), rep(0, p), rep(0, p), rep(0, p), rep(th, p)))
        quads.append(("E2_reset_interior", (p, th), rep(0, p), rep(0.3, p), rep(0, p), rep(0.3 + th, p)))
# AD near gamma=1
for g in (1.0, 0.9):
    for dg in np.logspace(-6, -1, 8):
        J0, J1 = choi_from_kraus(kraus_ampdamp(g)), choi_from_kraus(kraus_ampdamp(g - dg))
        quads.append(("E2_ampdamp", (g, dg), J0, J0, J0, J1))
        quads.append(("E2_ampdamp_vs_id", (g, dg), J0, choi_unitary(I2), J1, choi_unitary(I2)))
# random qubit/qutrit perturbation quadruples
from e3_pair_bounds import perturb_kraus, rand_channel
for k in range(60):
    d = 2 if k % 2 == 0 else 3
    KP, KQ = rand_channel(rng, d), rand_channel(rng, d)
    dl = 10 ** rng.uniform(-5, 0)
    quads.append(("random", dl, choi_from_kraus(KP), choi_from_kraus(KQ),
                  choi_from_kraus(perturb_kraus(KP, dl, rng)), choi_from_kraus(perturb_kraus(KQ, dl, rng))))

rows = []
for fam, par, P1, Q1, P2, Q2 in quads:
    d = int(round(np.sqrt(P1.shape[0])))
    dP, dQ = diamond_sdp(P1 - P2, d, d), diamond_sdp(Q1 - Q2, d, d)
    for g in gammas:
        E1 = hockey_sdp(P1, Q1, g, d, d); E2 = hockey_sdp(P2, Q2, g, d, d)
        rhs = (dP + g * dQ) / 2
        rows.append(dict(family=fam, par=par, gamma=g, E1=E1, E2=E2, lhs=abs(E1 - E2), rhs=rhs, dP=dP, dQ=dQ,
                         ratio=abs(E1 - E2) / rhs if rhs > 1e-12 else (0.0 if abs(E1 - E2) < 1e-8 else np.inf),
                         F1=fhat_arn(P1, Q1, d, d), F2=fhat_arn(P2, Q2, d, d)))
SOLVER_TOL = 5e-8
viol = [r for r in rows if r["lhs"] - r["rhs"] > SOLVER_TOL]
fam = {}
for f in sorted(set(r["family"] for r in rows)):
    rr = [r for r in rows if r["family"] == f]
    fam[f] = dict(n=len(rr), max_ratio=float(max(r["ratio"] for r in rr if np.isfinite(r["ratio"]))),
                  max_lhs_minus_rhs=float(max(r["lhs"] - r["rhs"] for r in rr)),
                  max_abs_dF=float(max(abs(r["F1"] - r["F2"]) for r in rr)),
                  max_dF_over_dia=float(max(abs(r["F1"] - r["F2"]) / max(1e-300, r["dP"] + r["dQ"]) for r in rr)))
e1 = [r for r in rows if r["family"] == "E1_cor33" and r["gamma"] == 1.0]
summary = dict(n_rows=len(rows), n_quads=len(quads), solver_tol=SOLVER_TOL, n_violations=len(viol),
               max_ratio=float(max(r["ratio"] for r in rows if np.isfinite(r["ratio"]))),
               per_family=fam,
               E1_smallest_theta=dict(theta=e1[0]["par"], E_gamma1_pair1=e1[0]["E1"], E_gamma1_pair2=e1[0]["E2"],
                                      dia=e1[0]["dP"], F1=e1[0]["F1"], F2=e1[0]["F2"]),
               violations=viol[:10])
json.dump(dict(summary=summary, rows=rows), open(os.path.join(HERE, "results/e5.json"), "w"), indent=1,
          default=lambda o: float(o) if np.isscalar(o) else str(o))
print(json.dumps(summary, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o)))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for g, mk in zip(gammas, "os^d"):
    rr = [r for r in rows if r["gamma"] == g and r["rhs"] > 0]
    ax[0].loglog([r["rhs"] for r in rr], [max(1e-14, r["lhs"]) for r in rr], mk, ms=3, alpha=.6, label=f"γ={g:g}")
xx = np.logspace(-7, 1, 10)
ax[0].loglog(xx, xx, "k-", lw=1, label="LHS = RHS")
ax[0].set_xlabel(r"$(\|\Delta\Phi\|_\diamond+\gamma\|\Delta\Psi\|_\diamond)/2$")
ax[0].set_ylabel(r"$|\Delta E_\gamma|$"); ax[0].set_title("E5: hockey-stick is diamond-Lipschitz"); ax[0].legend(fontsize=7)
ax[0].set_ylim(1e-14, 10)
rr = [r for r in rows if r["gamma"] == 1.0 and r["dP"] + r["dQ"] > 0]
ax[1].loglog([r["dP"] + r["dQ"] for r in rr], [max(1e-14, r["lhs"]) for r in rr], "o", ms=3, label=r"$|\Delta E_1|$")
ax[1].loglog([r["dP"] + r["dQ"] for r in rr], [max(1e-14, abs(r["F1"] - r["F2"])) for r in rr], "x", ms=4, label=r"$|\Delta\hat F|$")
ax[1].loglog(xx, xx, "k-", lw=1)
ax[1].set_ylim(1e-14, 10)
ax[1].set_xlabel(r"$\|\Delta\Phi\|_\diamond+\|\Delta\Psi\|_\diamond$"); ax[1].set_ylabel("change")
ax[1].set_title(r"same quadruples: $E_1$ vs $\hat F$"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e5_hockey.png"), dpi=150)
