"""E8: (1) gap plot: single-letter converses vs the operational exponents near a depolarized gate,
    as a function of delta/p; (2) worst-case additive moduli per family (cross-check of the referee's
    moduli.py / moduli2.py table).

Pair: Phi = N_p = D_p o U, Psi = N_{p,delta} = D_p o (e^{-i delta Z} U), U = Hadamard.

Stein (type-II exponent, null Phi):  zeta = D^A = D^reg  (Wang-Wilde 2019; FFRS 2020).
  Brackets computed here:   D^(1) <= D^(2)/2 <= zeta <= min{ D_BS-hat ,  U_paper3 }
  where D^(n) = sup_input D((Phi^{(x)n} (x) id) psi || (Psi^{(x)n} (x) id) psi) (local optimisation -> lower
  estimates), and U_paper3 = 2 eps log dY + (1+eps) h(eps/(1+eps)) + eps log kappa(Psi), eps = ||Phi-Psi||_dia/2
  (paper 3 Cor stein(b) applied at (Psi,Psi), zeta(Psi,Psi)=0).  NOTE: the single-letter Umegaki channel
  divergence D^(1) is a LOWER bound on zeta, not an upper bound.
Chernoff: xi_adapt >= xi_prod >= xi_QC(Choi input);  converses C-hat <= -2 ln Fhat (paper 2).
"""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
from qlib import *
from qlib import _input_opt
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 8
rng = np.random.default_rng(SEED)


def pair(p, dl, U=H2, Hm=Z2):
    J0 = choi_depolarize_after(choi_unitary(U), p, 2, 2)
    J1 = choi_depolarize_after(choi_unitary(sla.expm(-1j * dl * Hm) @ U), p, 2, 2)
    return J0, J1


def choi_tensor2(J, d=2):
    K = kraus_from_choi(J, d, d)
    return choi_from_kraus([np.kron(a, b) for a in K for b in K])


def h2(x):
    return 0.0 if x <= 0 or x >= 1 else float(-x * np.log(x) - (1 - x) * np.log(1 - x))


def paper3_ub(J0, J1, dia):
    eps = dia / 2
    kap = np.linalg.eigvalsh(J1).max() / np.linalg.eigvalsh(J1).min()
    return 2 * eps * np.log(2) + (1 + eps) * h2(eps / (1 + eps)) + eps * np.log(kap)


def xi_prod(J0, J1, d=2, n_starts=3):
    return _input_opt(lambda A: xi_qc(output_on_input(J0, A, d, d), output_on_input(J1, A, d, d))[0], d, rng, n_starts)


def gap_experiment():
    rows = []
    for p in [1e-2, 1e-3, 1e-4, 1e-5]:
        for x in np.logspace(-2, 4, 19):  # x = delta / p
            dl = x * p
            if dl > 1.2:
                continue
            J0, J1 = pair(p, dl)
            dia = diamond_sdp(J0 - J1, 2, 2)
            F = fhat_arn(J0, J1, 2, 2)
            C, s = chernoff_geo(J0, J1, 2, 2)
            r = dict(p=p, delta=dl, x=x, diamond=dia,
                     BS=dbs_arn(J0, J1, 2, 2),
                     m2lnF=-2 * np.log(F) if F > 0 else np.inf, C_geo=C,
                     D1=umegaki_channel(J0, J1, 2, 2, rng, n_starts=2),
                     D1_choi=rel_ent(J0 / 2, J1 / 2),
                     U_paper3=paper3_ub(J0, J1, dia),
                     Dmax_bound=float(np.log(2 / np.linalg.eigvalsh(J1).min())),
                     xi_QC_choi=xi_qc(J0 / 2, J1 / 2)[0])
            rows.append(r)
        print("gap p", p, flush=True)
    # n = 2 lower bound on D^reg on a sub-grid (input on X1X2 with 4-dim reference)
    for r in rows:
        if r["p"] in (1e-2, 1e-3) and (abs(np.log10(r["x"]) - np.round(np.log10(r["x"]))) < 1e-9):
            J0, J1 = pair(r["p"], r["delta"])
            J02, J12 = choi_tensor2(J0), choi_tensor2(J1)
            r["D2_half_choi"] = rel_ent(J02 / 4, J12 / 4) / 2
            r["D2_half_opt"] = _input_opt(lambda A: rel_ent(output_on_input(J02, A, 4, 4), output_on_input(J12, A, 4, 4)),
                                          4, rng, n_starts=2) / 2
    print("D2 done", flush=True)
    for r in rows:
        r["zeta_lower"] = max(r["D1"], r.get("D2_half_opt", 0.0))
        r["zeta_upper"] = min(r["BS"], r["U_paper3"])
        r["BS_over_D1"] = r["BS"] / r["D1"]
        r["m2lnF_over_D1"] = r["m2lnF"] / r["D1"]
        r["C_over_D1"] = r["C_geo"] / r["D1"]
        r["BS_over_zeta_upper"] = r["BS"] / r["zeta_upper"]       # lower bound on BS / zeta
        r["BS_over_zeta_lower"] = r["BS"] / r["zeta_lower"]       # upper bound on BS / zeta
        r["C_over_xiQC"] = r["C_geo"] / r["xi_QC_choi"]           # upper bound on C / xi_adapt (xi_adapt>=xi_QC(choi))
        r["BS_over_paper3"] = r["BS"] / r["U_paper3"]
    return rows


# ----------------------------------------------------------------------------- moduli (referee moduli.py / moduli2.py)
def rx(t):
    return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])


PA = [I2, X2, Y2, Z2]


def Kd(p, dl):
    return [np.sqrt(1 - 3 * p / 4) * rx(np.pi / 2 + dl)] + [np.sqrt(p / 4) * s @ rx(np.pi / 2 + dl) for s in PA[1:]]


def pert(K, x, eps):
    X = (x[:16] + 1j * x[16:]).reshape(4, 2, 2)
    X = X / np.linalg.norm(X) * eps
    Kp = [k + x_ for k, x_ in zip(K, X)]
    return choi_from_kraus(tp_renormalize(Kp))


def tn(A):
    return float(np.sum(np.abs(np.linalg.eigvalsh(herm(A)))))


def bs_state(A, B):
    """Choi-state BS: Tr A log(A^1/2 B^-1 A^1/2)."""
    Ah = psd_sqrt(A)
    M = herm(Ah @ np.linalg.inv(B) @ Ah)
    w, V = np.linalg.eigh(M)
    return float(np.real(np.trace(Ah @ ((V * np.log(np.clip(w, 1e-300, None))) @ V.conj().T) @ Ah)))


def moduli_experiment(eps=1e-4, dl0=0.05, n_restarts=2):
    out = {}
    for p in [1e-2, 1e-3, 1e-4]:
        res = {}
        for arg in ("first", "second"):
            Kpert = Kd(p, dl0) if arg == "first" else Kd(p, 0)
            Jfix = choi_from_kraus(Kd(p, 0)) if arg == "first" else choi_from_kraus(Kd(p, dl0))
            Jb = choi_from_kraus(Kpert)

            def pr(J):  # (Phi, Psi) ordering: Phi = gate+dl0, Psi = gate
                return (J, Jfix) if arg == "first" else (Jfix, J)
            fs = {"m2lnF_geo": lambda J: -2 * np.log(fhat_arn(*pr(J), 2, 2)),
                  "BS_channel": lambda J: dbs_arn(*pr(J), 2, 2),
                  "BS_choi_state": lambda J: bs_state(pr(J)[0] / 2, pr(J)[1] / 2),
                  "umegaki_choi_state": lambda J: rel_ent(pr(J)[0] / 2, pr(J)[1] / 2)}
            for name, f in fs.items():
                v0 = f(Jb)
                best = 0.0
                for _ in range(n_restarts):
                    for sg in (1, -1):
                        r = minimize(lambda x: -sg * (f(pert(Kpert, x, eps)) - v0), rng.normal(size=32), method="Powell",
                                     options=dict(maxiter=2000))
                        J = pert(Kpert, r.x, eps)
                        best = max(best, abs(f(J) - v0) / tn(J - Jb))
                res[f"{name}|{arg}"] = dict(modulus=best, base_value=v0)
        out[f"{p:g}"] = res
        print("moduli p", p, {k: round(v["modulus"], 3) for k, v in res.items()}, flush=True)
    return out


def main():
    t0 = time.time()
    rows = gap_experiment()
    mod = moduli_experiment()
    # collapse diagnostics: spread across p of BS/zeta_upper at fixed x, and vs x/log(1/p)
    coll = {}
    for x in sorted(set(round(r["x"], 8) for r in rows)):
        sub = [r for r in rows if abs(r["x"] - x) < 1e-8 * x]
        if len(sub) >= 3:
            coll[f"{x:.3g}"] = dict(BS_over_D1=[r["BS_over_D1"] for r in sub], BS_over_zeta_upper=[r["BS_over_zeta_upper"] for r in sub],
                                    C_over_xiQC=[r["C_over_xiQC"] for r in sub], ps=[r["p"] for r in sub])
    summary = dict(seed=SEED, gap_rows=rows, moduli=mod, collapse_table=coll,
                   referee_table=dict(m2lnF_first={"1e-2": 1.2, "1e-3": 12.5, "1e-4": 342},
                                      BS_second={"1e-2": 2.6, "1e-3": 30, "1e-4": 803},
                                      umegaki={"1e-2": "0.07-0.08", "1e-3": "0.10-0.11", "1e-4": "0.13-0.16"}),
                   runtime_s=time.time() - t0)
    json.dump(summary, open(os.path.join(HERE, "results/e8.json"), "w"), indent=1,
              default=lambda o: float(o) if np.isscalar(o) else str(o))

    # ---------- figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cols = dict(zip([1e-2, 1e-3, 1e-4, 1e-5], plt.cm.viridis(np.linspace(0.1, 0.85, 4))))
    for p in cols:
        sub = [r for r in rows if r["p"] == p]
        x = [r["x"] for r in sub]
        ax[0].loglog(x, [r["BS_over_D1"] for r in sub], "o-", color=cols[p], ms=3, label=f"BS, p={p:g}")
        ax[0].loglog(x, [r["m2lnF_over_D1"] for r in sub], "s--", color=cols[p], ms=3)
        ax[0].loglog(x, [r["C_over_D1"] for r in sub], ":", color=cols[p])
        ax[1].loglog(x, [r["BS_over_zeta_upper"] for r in sub], "o-", color=cols[p], ms=3, label=f"p={p:g}")
        ax[1].loglog(x, [r["BS_over_zeta_lower"] for r in sub], "v:", color=cols[p], ms=3)
        ax[2].loglog(np.array(x) / np.log(1 / p), [r["BS_over_paper3"] for r in sub], "o-", color=cols[p], ms=3, label=f"p={p:g}")
    ax[0].set_xlabel(r"$\delta/p$"); ax[0].set_ylabel(r"converse / $D^{(1)}(N_p\|N_{p,\delta})$")
    ax[0].set_title("Ratio to single-letter Umegaki (requested normalisation)\n"
                    r"solid: $\hat D_{BS}$, dashed: $-2\ln\hat F$, dotted: $\hat C$", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].axhline(1, color="gray", lw=0.8)
    ax[1].set_xlabel(r"$\delta/p$"); ax[1].set_ylabel(r"$\hat D_{BS}/\zeta$ bracket")
    ax[1].set_title(r"Stein: $\hat D_{BS}/\zeta$ lies between solid ($/\zeta_{up}$) and dotted ($/\zeta_{low}$)", fontsize=9)
    ax[1].legend(fontsize=7)
    xx = np.logspace(-3, 3, 50)
    ax[2].loglog(xx, xx, "k--", lw=1, label=r"$\propto\delta/(p\ln(1/p))$")
    ax[2].axhline(1, color="gray", lw=0.8)
    ax[2].set_xlabel(r"$\delta/(p\,\ln(1/p))$"); ax[2].set_ylabel(r"$\hat D_{BS}$ / paper-3 upper bound on $\zeta$")
    ax[2].set_title("Collapse of the certified overshoot", fontsize=9); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e8_gap.png"), dpi=160)
    print(json.dumps(dict(moduli=mod), indent=1, default=float))
    print("runtime", time.time() - t0)


if __name__ == "__main__":
    main()
