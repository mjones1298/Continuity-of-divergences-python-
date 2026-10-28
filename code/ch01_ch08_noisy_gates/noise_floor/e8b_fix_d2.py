"""E8b: recompute the n=2 Umegaki lower bound D^(2)/2 in results/e8.json with the support-safe
evaluation (rel_ent_on_input), since rel_ent's support test misfired at numerically singular inputs
chosen by the optimiser (the first E8 pass gave +inf there).  Updates derived ratios and the figure."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
from qlib import *
from qlib import _input_opt
from e8_gap_moduli import pair, choi_tensor2
import e8_gap_moduli as E8
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(81)
D = json.load(open(os.path.join(HERE, "results/e8.json")))
for r in D["gap_rows"]:
    if "D2_half_opt" in r:
        J0, J1 = pair(r["p"], r["delta"])
        J02, J12 = choi_tensor2(J0), choi_tensor2(J1)
        r["D2_half_choi"] = rel_ent(J02 / 4, J12 / 4) / 2
        r["D2_half_opt"] = _input_opt(lambda A: rel_ent_on_input(J02, J12, A, 4, 4), 4, rng, n_starts=1) / 2
        # sanity: D^(n)/n <= D_max(Phi||Psi) <= ln(lambda_max J0 / lambda_min J1)
        r["Dmax_cap"] = float(np.log(np.linalg.eigvalsh(J0).max() / np.linalg.eigvalsh(J1).min()))
        assert r["D2_half_opt"] <= r["Dmax_cap"] + 1e-9, (r["D2_half_opt"], r["Dmax_cap"])
        print(r["p"], r["x"], r["D1"], r["D2_half_choi"], r["D2_half_opt"], flush=True)
for r in D["gap_rows"]:
    r["zeta_lower"] = max(r["D1"], r.get("D2_half_opt", 0.0))
    r["zeta_upper"] = min(r["BS"], r["U_paper3"])
    r["BS_over_zeta_upper"] = r["BS"] / r["zeta_upper"]
    r["BS_over_zeta_lower"] = r["BS"] / r["zeta_lower"]
json.dump(D, open(os.path.join(HERE, "results/e8.json"), "w"), indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
print("fixed")
