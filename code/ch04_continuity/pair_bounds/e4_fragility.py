"""E4: what predicts fragility?  For random perturbation pairs, the fragility ratio
    R = |dFhat| / (||Phi1-Phi2||_dia + ||Psi1-Psi2||_dia)^{1/2}
is compared against 1/lambda_min(G) (G = W^*W for Omega = Phi1+Psi1, which equals lambda_min^+(J(Omega))),
1/lambda_min^+(C_i) (relative-derivative sums in the Omega' frame), 1/nu(Omega)^2,
1/lambda_min^+(J(Phi1)), 1/lambda_min^+(J(Psi1)).  Spearman and Pearson (log-log) correlations."""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
from qlib import *
from e3_pair_bounds import frame4, perturb_kraus, rand_channel
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 44
rng = np.random.default_rng(SEED)


def make_pair(rng):
    """Base pair with a controlled conditioning: mix random low-rank channels with a weight-w full-rank one."""
    d = int(rng.choice([2, 3]))
    kind = rng.integers(0, 4)
    rP, rQ = int(rng.integers(1, d * d + 1)), int(rng.integers(1, d * d + 1))
    KP, KQ = rand_channel(rng, d, rP), rand_channel(rng, d, rQ)
    JP, JQ = choi_from_kraus(KP), choi_from_kraus(KQ)
    w = 10 ** rng.uniform(-6, -0.5)
    if kind == 0:   # both slightly full-rank
        JP = (1 - w) * JP + w * np.eye(d * d) / d
        JQ = (1 - w) * JQ + w * np.eye(d * d) / d
    elif kind == 1:  # near-identical with floor (gate-like)
        U = random_unitary(rng, d)
        JP = choi_depolarize_after(choi_unitary(U), w, d, d)
        JQ = choi_depolarize_after(choi_unitary(sla.expm(-1j * 10 ** rng.uniform(-4, 0) * herm(rng.normal(size=(d, d)))) @ U), w, d, d)
    elif kind == 2:  # replacement channels with floor
        a = random_kraus(rng, d, 1, 1)[0][:, 0]; b = random_kraus(rng, d, 1, 1)[0][:, 0]
        JP = choi_replacement((1 - w) * np.outer(a, a.conj()) + w * np.eye(d) / d, d)
        JQ = choi_replacement((1 - w) * np.outer(b, b.conj()) + w * np.eye(d) / d, d)
    # kind 3: raw random (possibly singular)
    return d, JP, JQ, kind, w


def main():
    t0 = time.time()
    rows = []
    for k in range(1500):
        d, JP, JQ, kind, w = make_pair(rng)
        KP = kraus_from_choi(JP, d, d); KQ = kraus_from_choi(JQ, d, d)
        dl = 10 ** rng.uniform(-6, -1)
        JP2 = choi_from_kraus(perturb_kraus(KP, dl, rng)); JQ2 = choi_from_kraus(perturb_kraus(KQ, dl, rng))
        F1, F2 = fhat_arn(JP, JQ, d, d), fhat_arn(JP2, JQ2, d, d)
        dP, dQ = diamond_sdp(JP - JP2, d, d), diamond_sdp(JQ - JQ2, d, d)
        JOm = JP + JQ
        W, (A1, B1, A2, B2) = frame4([JP, JQ, JP2, JQ2])
        lC = min(lmin_pos(A1 + B1), lmin_pos(A2 + B2))
        lamG = float(np.linalg.eigvalsh(arn_frame(JOm).conj().T @ arn_frame(JOm)).min())
        nu2 = nu_upper_opt(arn_frame(JOm), d, d, rng, n_restarts=3) if k % 3 == 0 else np.nan
        rows.append(dict(d=d, kind=int(kind), w=w, pert=dl, dF=abs(F1 - F2), F1=F1, dP=dP, dQ=dQ,
                         R=abs(F1 - F2) / np.sqrt(dP + dQ) if dP + dQ > 0 else 0.0,
                         R1=abs(F1 - F2) / (dP + dQ) if dP + dQ > 0 else 0.0,
                         lam_min_G=lamG, lam_min_pos_JOm=lmin_pos(JOm), lam_min_C=lC, nu2=nu2,
                         lam_min_JP=lmin_pos(JP), lam_min_JQ=lmin_pos(JQ),
                         lam_min_full_JOm=float(np.linalg.eigvalsh(JOm).min()),
                         full_rank=bool(np.linalg.eigvalsh(JOm).min() > 1e-12)))
    print("pairs", len(rows), time.time() - t0)

    preds = {"1/lam_min(G)": lambda r: 1 / r["lam_min_G"],
             "1/lam_min+(C_i)": lambda r: 1 / r["lam_min_C"],
             "1/nu^2": lambda r: 1 / r["nu2"],
             "1/lam_min+(J(Phi))": lambda r: 1 / r["lam_min_JP"],
             "1/lam_min+(J(Psi))": lambda r: 1 / r["lam_min_JQ"],
             "1/min(lam+(J(Phi)),lam+(J(Psi)))": lambda r: 1 / min(r["lam_min_JP"], r["lam_min_JQ"]),
             "1/lam_min(J(Omega)) [full spectrum, 0->1e-16]": lambda r: 1 / max(1e-16, r["lam_min_full_JOm"])}
    corr = {}
    for target in ("R", "R1"):
        for subset_name, sub in (("all", rows), ("full_rank_sum", [r for r in rows if r["full_rank"]]),
                                 ("nontrivial_dF>1e-9", [r for r in rows if r["dF"] > 1e-9])):
            for name, f in preds.items():
                xs, ys = [], []
                for r in sub:
                    x = f(r)
                    if np.isfinite(x) and r[target] > 1e-12:
                        xs.append(np.log(x)); ys.append(np.log(r[target]))
                if len(xs) > 10:
                    corr[f"{target}|{subset_name}|{name}"] = dict(n=len(xs), spearman=float(spearmanr(xs, ys)[0]),
                                                               pearson_loglog=float(pearsonr(xs, ys)[0]),
                                                               slope_loglog=float(np.polyfit(xs, ys, 1)[0]))
    # upper-envelope test: is R <= c / nu? (Cor 10.8 predicts R*nu <= (11/4)*2*sqrt(2)... order-1)
    env = {}
    for name in ("1/lam_min(G)", "1/nu^2", "1/lam_min+(C_i)"):
        vals = [r["R"] / np.sqrt(preds[name](r)) for r in rows if np.isfinite(preds[name](r)) and r["R"] > 0]
        env[name] = dict(max_R_over_sqrt_pred=float(np.max(vals)), q99=float(np.quantile(vals, 0.99)))
    summary = dict(seed=SEED, n=len(rows), n_full_rank_sum=int(sum(r["full_rank"] for r in rows)),
                   correlations=corr, envelope_R_times_sqrt_lambda=env, runtime_s=time.time() - t0)
    json.dump(dict(summary=summary, rows=rows), open(os.path.join(HERE, "results/e4.json"), "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1))

    fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    kinds = ["floor-mixed", "noisy gate pair", "noisy replacement", "raw random"]
    for ax, name in zip(axs, ["1/lam_min(G)", "1/lam_min+(C_i)", "1/nu^2", "1/min(lam+(J(Phi)),lam+(J(Psi)))"]):
        for kd, mk in zip(range(4), "os^x"):
            xs = [preds[name](r) for r in rows if r["kind"] == kd and r["R"] > 1e-12 and np.isfinite(preds[name](r))]
            ys = [r["R"] for r in rows if r["kind"] == kd and r["R"] > 1e-12 and np.isfinite(preds[name](r))]
            ax.loglog(xs, ys, mk, ms=3, alpha=0.5, label=kinds[kd])
        xx = np.logspace(0, 13, 10)
        ax.loglog(xx, np.sqrt(xx), "k--", lw=1, label=r"$\propto x^{1/2}$")
        key = f"R|nontrivial_dF>1e-9|{name}"
        rho = corr.get(key, {}).get("spearman", np.nan)
        ax.set_title(f"{name}\nSpearman ρ = {rho:.2f}", fontsize=9)
        ax.set_xlabel(name); ax.set_ylabel(r"$|\Delta\hat F| / (\Delta_\diamond^\Phi+\Delta_\diamond^\Psi)^{1/2}$")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures/e4_fragility.png"), dpi=150)


if __name__ == "__main__":
    main()
