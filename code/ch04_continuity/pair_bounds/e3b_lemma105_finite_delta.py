"""E3b: v5 open problem (1): is (1-4 delta^2)^{-1/2} in Lemma 10.5 optimal at finite delta?
Derivative-level search: positive A1,B1,A2,B2 on C^r with A1+B1+A2+B2 = 1 (any such quadruple is
realised by CP maps for any linearly independent frame, v5 Thm 2.2), C_i = A_i+B_i invertible.
Maximise  tau / delta  at fixed delta-bucket, compare to the bound 2 sqrt2 /sqrt(1-4 gamma^2) and
2 sqrt2/sqrt(1-4 delta^2).  Parametrisation: 4 PSD matrices X_j = G_j G_j^*, normalised by
S^{-1/2} X_j S^{-1/2}, S = sum X_j."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
from qlib import *
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(105)


def quad(x, r):
    Gs = (x[:4 * r * r] + 1j * x[4 * r * r:]).reshape(4, r, r)
    Xs = [G @ G.conj().T + 1e-9 * np.eye(r) for G in Gs]
    S = sum(Xs)
    Sih = pinv_sqrt(S)
    return [herm(Sih @ X @ Sih) for X in Xs]


def stats(x, r):
    A1, B1, A2, B2 = quad(x, r)
    C1, C2 = A1 + B1, A2 + B2
    T1 = pinv_sqrt(C1) @ A1 @ pinv_sqrt(C1); T2 = pinv_sqrt(C2) @ A2 @ pinv_sqrt(C2)
    tau = opnorm(T1 - T2); delta = max(opnorm(A1 - A2), opnorm(B1 - B2)); gamma = opnorm(C1 - C2) / 2
    return tau, delta, gamma


rows = []
for r in (2, 3):
    for dtarget in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45):
        best = None
        for st in range(12 if r == 2 else 6):
            def obj(x):
                tau, delta, gamma = stats(x, r)
                if delta >= 0.5 or delta <= 0:
                    return 10.0
                pen = 200 * (delta - dtarget) ** 2 / dtarget ** 2
                return -(tau / (2 * np.sqrt(2) * delta / np.sqrt(1 - 4 * delta ** 2))) + pen
            # start near a commuting-ish point with small differences
            x0 = rng.normal(size=8 * r * r)
            res = minimize(obj, x0, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-9, fatol=1e-12))
            res = minimize(obj, res.x, method="Powell", options=dict(maxiter=6000))
            tau, delta, gamma = stats(res.x, r)
            val = dict(r=r, delta_target=dtarget, delta=delta, gamma=gamma, tau=tau, tau_over_delta=tau / delta,
                       ratio_delta_bound=tau / (2 * np.sqrt(2) * delta / np.sqrt(1 - 4 * delta ** 2)),
                       ratio_gamma_bound=tau / (2 * np.sqrt(2) * delta / np.sqrt(max(1e-300, 1 - 4 * gamma ** 2))),
                       commuting_bound_ratio=tau / (2 * delta / (1 - 2 * delta)))
            if best is None or val["ratio_delta_bound"] > best["ratio_delta_bound"]:
                best = val
        rows.append(best)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()}, flush=True)
viol = [b for b in rows if b["ratio_gamma_bound"] > 1 + 1e-6]
json.dump(dict(rows=rows, n_violations=len(viol),
               note="local search (Nelder-Mead + Powell, 12/6 restarts); values are lower bounds on the sup"),
          open(os.path.join(HERE, "results/e3b.json"), "w"), indent=1, default=float)
print("violations of Lemma 10.5:", len(viol))
