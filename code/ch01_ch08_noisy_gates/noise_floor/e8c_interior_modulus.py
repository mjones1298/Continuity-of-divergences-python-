"""E8c: independent check of the referee's interior-pair modulus |dFhat| ~ 0.22 d/sqrt(p) (worst4.py)
and diagonal 1-Fhat ~ 0.03-0.07 d^2/p (worst3.py).
Gate Rx(pi/2) with depolarising p (referee convention, Pauli Kraus).  Interior pair (Phi,Psi_0) with
Psi_0 = gate over-rotated by dl0 chosen so Fhat ~ 0.5.  Perturb Psi by TP-preserving Kraus perturbations
of Frobenius size eps (Powell maximisation of |dFhat|), report |dF|/d for d = Choi trace-norm distance
(referee's metric) and d = diamond distance (SDP).  Also the analytic coherent direction."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
from qlib import *
from scipy.optimize import minimize
from e8_gap_moduli import Kd, pert, tn

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(83)
out = {}
for p in [1e-2, 1e-3, 1e-4]:
    J1 = choi_from_kraus(Kd(p, 0))
    dls = np.logspace(-4, 0, 400)
    dl0 = [d for d in dls if fhat_arn(J1, choi_from_kraus(Kd(p, d)), 2, 2) < 0.5][0]
    K = Kd(p, dl0); J2 = choi_from_kraus(K); F0 = fhat_arn(J1, J2, 2, 2)
    res = {}
    for scale in (0.01, 0.03, 0.1):
        eps = scale * np.sqrt(p)
        best = (0, 0, 0)
        for sg in (1, -1):
            for _ in range(2):
                r = minimize(lambda x: -sg * (fhat_arn(J1, pert(K, x, eps), 2, 2) - F0), rng.normal(size=32), method="Powell",
                             options=dict(maxiter=2500))
                J = pert(K, r.x, eps)
                v = abs(fhat_arn(J1, J, 2, 2) - F0)
                if v > best[0]:
                    best = (v, tn(J - J2), diamond_sdp(J - J2, 2, 2))
        v, dch, ddia = best
        res[f"{scale}"] = dict(dF=v, d_choi=dch, d_dia=ddia, dF_sqrtp_over_dchoi=v * np.sqrt(p) / dch,
                               dF_sqrtp_over_ddia=v * np.sqrt(p) / ddia)
    # coherent direction: numerical derivative along the over-rotation
    h = 1e-6 * np.sqrt(p)
    Jp, Jm = choi_from_kraus(Kd(p, dl0 + h)), choi_from_kraus(Kd(p, dl0 - h))
    dFd = (fhat_arn(J1, Jp, 2, 2) - fhat_arn(J1, Jm, 2, 2)) / (2 * h)
    ddia = diamond_sdp(Jp - Jm, 2, 2) / (2 * h); dch = tn(Jp - Jm) / (2 * h)
    out[f"{p:g}"] = dict(dl0=dl0, F0=F0, worst=res, coherent=dict(dF_ddelta=dFd, dF_sqrtp_over_ddia=abs(dFd) * np.sqrt(p) / ddia,
                                                                 dF_sqrtp_over_dchoi=abs(dFd) * np.sqrt(p) / dch))
    print(p, json.dumps(out[f"{p:g}"], default=float), flush=True)
json.dump(out, open(os.path.join(HERE, "results/e8c.json"), "w"), indent=1, default=float)
