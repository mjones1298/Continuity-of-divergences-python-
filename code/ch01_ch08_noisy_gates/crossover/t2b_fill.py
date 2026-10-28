"""Fill the rows of t2b (two-copy D^#, m = 2) whose SDP returned no point ('fail') in t2b_sharp_m2.py / t2b_resume.py.
Same construction and same certify() as t2b_sharp_m2.py. Order of attempts per row: the one-copy best t with Clarabel
(as before), then its dyadic neighbours t0 -/+ 1/16 with Clarabel, then t0 with SCS. Any alpha = 1/t > 1 gives a valid
upper bound zeta <= (1/2) D^#_alpha(N^(x)2 || M^(x)2) (Fawzi-Fawzi Thm 5.1), so the first certified attempt is kept and
the smallest certified value over all successful attempts is reported.
Appends one JSON line per row to results/t2b_fill_<p>.log.  Usage: python3 t2b_fill.py <p> [<p> ...]"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # common.py, sharp_alpha.py
from fractions import Fraction
from sharp_alpha import Dsharp_channel, certify, T_GRID
R = str(Path(__file__).resolve().parent / 'results')
src = open(Path(__file__).resolve().parent / 't2b_sharp_m2.py').read()
ns = {'__file__': __file__}   # t2b_sharp_m2.py locates code/common from its own path
exec(src.split('pk = sys.argv[1]')[0], ns)
J, two = ns['J'], ns['two']

for pk in sys.argv[1:]:
    rows = json.load(open(f'{R}/t2b_m2_{pk}.json'))
    p = float(pk)
    t2rows = {round(r['delta'], 15): r for r in json.load(open(f'{R}/t2.json'))['results'][pk]}
    for rec in rows:
        if rec['Dsharp_m2_per_copy_cert'] is not None:
            continue
        d = rec['delta']
        t0 = Fraction(t2rows[round(d, 15)]['t_best'])
        i = T_GRID.index(t0)
        tries = [(t0, 'CLARABEL')] + [(T_GRID[j], 'CLARABEL') for j in (i - 1, i + 1) if 0 <= j < len(T_GRID)] + [(t0, 'SCS')]
        attempts, best = [], None
        JN, JM = two(J(p, d)), two(J(p, 0))
        for t, solver in tries:
            t1 = time.time()
            D2, Q2, st, B2 = Dsharp_channel(JN, JM, 4, 4, t, solver=solver, return_B=True)
            a = dict(t=str(t), solver=solver, status=st, sdp_per_copy=None if D2 != D2 else float(D2) / 2, wall_s=round(time.time() - t1, 1))
            if B2 is not None:
                c, sc = certify(JN, JM, 4, 4, t, B2, dps=25)
                a.update(cert_per_copy=float(c) / 2, scale=float(sc))
                if best is None or a['cert_per_copy'] < best['cert_per_copy']:
                    best = a
            attempts.append(a)
            print(pk, round(rec['x'], 3), json.dumps(a), flush=True)
            if best is not None and t == t0:
                break          # original t succeeded: same computation as the main run
        out = dict(rec, Dsharp_m2_per_copy_cert=best['cert_per_copy'] if best else None, t=best['t'] if best else None,
                   status=best['status'] if best else 'fail', scale=best['scale'] if best else None,
                   solver=best['solver'] if best else None, filled=True, attempts=attempts)
        with open(f'{R}/t2b_fill_{pk}.log', 'a') as fh:
            fh.write(json.dumps(out) + '\n')
        print(pk, 'row done', json.dumps({k: out[k] for k in ('x', 'Dsharp_m2_per_copy_cert', 't', 'status')}), flush=True)
    print(pk, 'fill complete', flush=True)
