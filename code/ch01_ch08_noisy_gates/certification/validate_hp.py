"""High-precision validation of F-hat on singular / near-singular ESTIMATED Choi pairs:
ARN (SVD-stacked frame, double) vs 120-digit eps-regularised geometric mean (mpmath, Aitken eps->0)
vs SDP (SCS).  -> results/validate_hp.json"""
import os, sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
sys.path.insert(0, str(Path(HERE).resolve().parents[1] / "common"))  # certlib.py, qlib.py in code/common
from certlib import *  # noqa

rng = np.random.default_rng(99)
rows = []
for p, N in ((1e-3, 1e3), (1e-3, 1e5), (1e-4, 1e6), (1e-2, 1e4)):
    J1, J0 = choi_gate(p, 2 * np.sqrt(p)), choi_gate(p, 0.0)
    f1, f0, fb = sample(J1, N, rng, 3), sample(J0, N, rng, 3), sample(J0, N, rng, 3)
    A, B, Bb = cptp_project(lin_inv(f1)), cptp_project(lin_inv(f0)), cptp_project(lin_inv(fb))
    for i in range(3):
        for lab, (X, Y) in (("LI delta", (A[i], B[i])), ("LI self", (Bb[i], B[i])),
                            ("LR2 self", (lowrank_mle(fb[i], 2, Bb[i]), lowrank_mle(f0[i], 2, B[i])))):
            arn = fhat(X, Y)
            mp_val, _, gap = Q.fhat_reg_mp(X, Y, 2, 2)
            sdp_val, _ = Q.fhat_sdp(X, Y, 2, 2)
            r = dict(p=p, N=N, case=lab, arn=arn, mp=float(mp_val), mp_extrap_gap=float(gap), sdp=sdp_val,
                     rank1=int(np.sum(np.linalg.eigvalsh(X) > 1e-12)), rank2=int(np.sum(np.linalg.eigvalsh(Y) > 1e-12)))
            rows.append(r)
            print(r, flush=True)
d = np.array([abs(r["arn"] - r["mp"]) for r in rows])
s = np.array([abs(r["arn"] - r["sdp"]) for r in rows])
json.dump(dict(rows=rows, max_arn_mp=float(d.max()), max_arn_sdp=float(np.nanmax(s))),
          open(os.path.join(HERE, "results", "validate_hp.json"), "w"), indent=1)
print("max |ARN-mp| =", d.max(), " max |ARN-SDP| =", np.nanmax(s))
