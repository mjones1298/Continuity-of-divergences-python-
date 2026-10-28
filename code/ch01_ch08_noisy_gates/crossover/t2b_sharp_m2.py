"""T2b: two-copy Fawzi-Fawzi bound zeta <= (1/2) D^#_alpha(N^{(x)2} || M^{(x)2}) (Thm 5.1, m = 2) at the grid points of T2
where the one-copy D^# was the best upper bound (every other point). alpha: the best one-copy t, and its dyadic neighbours.
Each SDP solution is certified in mpmath (certify(): exact rescaling to feasibility). Usage: python3 t2b_sharp_m2.py <p>"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # common.py, sharp_alpha.py
import numpy as np, mpmath as mp
from fractions import Fraction
from sharp_alpha import Dsharp_channel, certify, T_GRID
from common import choi_qubit


def J(p, d):
    mp.mp.dps = 30
    return np.array(choi_qubit(p, d).tolist(), dtype=complex)


def two(Ja):
    T = np.einsum('abcd,efgh->aebfcgdh', Ja.reshape(2, 2, 2, 2), Ja.reshape(2, 2, 2, 2))
    return T.reshape(16, 16)


pk = sys.argv[1]
rows = json.load(open(Path(__file__).resolve().parent / 'results' / 't2.json'))['results'][pk]
sel = [r for r in rows if r['upper_from'] == 'sharp'][::2]
out = []
for r in sel:
    p, d = float(pk), r['delta']
    t0 = Fraction(r['t_best']); i = T_GRID.index(t0)
    best = None
    for t in [t0]:
        D2, Q2, st, B2 = Dsharp_channel(two(J(p, d)), two(J(p, 0)), 4, 4, t, return_B=True)
        if B2 is None:
            continue
        c, sc = certify(two(J(p, d)), two(J(p, 0)), 4, 4, t, B2, dps=25)
        val = float(c) / 2
        if best is None or val < best[0]:
            best = (val, str(t), st, float(sc))
    rec = dict(x=r['x'], delta=d, zeta_upper_m1=r['zeta_upper'], Dsharp_m1=r['Dsharp_cert'],
               Dsharp_m2_per_copy_cert=best[0] if best else None, t=best[1] if best else None, status=best[2] if best else 'fail',
               scale=best[3] if best else None, zeta_lower=r['zeta_lower'], DBS=r['DBS'])
    out.append(rec)
    print(json.dumps(rec), flush=True)
json.dump(out, open(Path(__file__).resolve().parent / 'results' / f't2b_m2_{pk}.json', 'w'), indent=1)
