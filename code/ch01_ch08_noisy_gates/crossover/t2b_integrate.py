"""Merge the two-copy D^# results (t2b_m2_<p>.json + t2b_fill_<p>.log) into the T2 bracket and update results/summary.json.
zeta_upper_best = min(zeta_upper (T2, one copy), (1/2) D^#_alpha(N^(x)2||M^(x)2) certified) at every grid point where the
two-copy value exists (every other point of the range where one-copy D^# was the best upper bound). Writes
results/t2b_merged.json (per-grid-point table used by make_figures.py) and the 'T2b' block / sharpened T2 numbers in summary.json."""
import json, os, math
from pathlib import Path
R = str(Path(__file__).resolve().parent / 'results') + '/'
T2 = json.load(open(R + 't2.json'))['results']
S = json.load(open(R + 'summary.json'))
PK = {'0.01': '1e-2', '0.001': '1e-3', '0.0001': '1e-4'}
merged, t2b = {}, {}
for pk, lab in PK.items():
    m2 = {round(r['delta'], 15): r for r in json.load(open(R + f't2b_m2_{pk}.json'))}
    if os.path.exists(R + f't2b_fill_{pk}.log'):
        for line in open(R + f't2b_fill_{pk}.log'):
            r = json.loads(line); m2[round(r['delta'], 15)] = r
    rows, m2rows = [], []
    for r in T2[pk]:
        k = round(r['delta'], 15); zu1 = r['zeta_upper']; zl = r['zeta_lower']; m = m2.get(k)
        v2 = m['Dsharp_m2_per_copy_cert'] if m else None
        zu = min(zu1, v2) if v2 is not None else zu1
        src = 'sharp_m2' if (v2 is not None and v2 < zu1) else r['upper_from']
        rows.append(dict(x=r['x'], delta=r['delta'], zeta_lower=zl, zeta_upper_m1=zu1, zeta_upper_m2=v2, zeta_upper=zu, upper_from=src,
                         DBS=r['DBS'], m2lnFhat=r['m2lnFhat'], DBS_over_zu=r['DBS'] / zu, DBS_over_zl=r['DBS'] / zl,
                         F_over_zu=r['m2lnFhat'] / zu, F_over_zl=r['m2lnFhat'] / zl, zu_over_zl=zu / zl))
        if m:
            m2rows.append(dict(x=round(r['x'], 3), delta=r['delta'], zeta_upper_m1=zu1, Dsharp_m2_per_copy=v2, t=m['t'], status=m['status'],
                               improvement_pct=None if v2 is None else round(100 * (1 - v2 / zu1), 2), binding=bool(v2 is not None and v2 < zu1),
                               filled=bool(m.get('filled', False))))
    merged[pk] = rows
    p = float(pk)
    i = max(range(len(rows)), key=lambda j: rows[j]['DBS_over_zu'])
    iF = max(range(len(rows)), key=lambda j: rows[j]['F_over_zu'])
    onset = next(r['x'] for r in rows if r['DBS_over_zu'] > 1.01)
    t2b[lab] = dict(rows=m2rows, n_rows=len(m2rows), n_missing=sum(r['Dsharp_m2_per_copy'] is None for r in m2rows),
                    n_binding=sum(r['binding'] for r in m2rows),
                    max_improvement_pct=max([r['improvement_pct'] for r in m2rows if r['binding']] or [0.0]),
                    max_DBS_over_zeta_upper=round(rows[i]['DBS_over_zu'], 3), at_x=round(rows[i]['x'], 2), at_delta=round(rows[i]['delta'], 4),
                    peak_times_sqrtp_ln1p=round(rows[i]['DBS_over_zu'] * math.sqrt(p) * math.log(1 / p), 3),
                    max_m2lnFhat_over_zeta_upper=round(rows[iF]['F_over_zu'], 3), at_x_F=round(rows[iF]['x'], 2),
                    onset_x=round(onset, 2),
                    bracket_width_range=[round(min(r['zu_over_zl'] for r in rows), 2), round(max(r['zu_over_zl'] for r in rows), 2)])
json.dump(dict(note='T2 grid with the two-copy (m=2) Fawzi-Fawzi bound merged into zeta_upper; ratios recomputed', results=merged),
          open(R + 't2b_merged.json', 'w'), indent=1)
S['T2b'] = dict(description='two-copy bound zeta <= (1/2) D^#_alpha(N^(x)2||M^(x)2) (Fawzi-Fawzi Thm 5.1), SDP-CERTIFIED (mpmath re-check, '
                'exact rescaling); alpha = 1/t with t the best one-copy t (neighbouring dyadic t where that SDP failed). '
                'zeta_upper := min(one-copy, two-copy) at every point where both exist.', per_p=t2b)
S['T2']['max_certified_DBS_over_zeta_upper_m1_only'] = S['T2'].get('max_certified_DBS_over_zeta_upper_m1_only', S['T2']['max_certified_DBS_over_zeta_upper'])
S['T2']['max_certified_DBS_over_zeta_upper'] = {lab: t2b[lab]['max_DBS_over_zeta_upper'] for lab in t2b}
S['T2']['peak_times_sqrtp_ln1p'] = {lab: t2b[lab]['peak_times_sqrtp_ln1p'] for lab in t2b}
S['T2']['max_certified_m2lnFhat_over_zeta_upper'] = {lab: t2b[lab]['max_m2lnFhat_over_zeta_upper'] for lab in t2b}
S['T2']['bracket_width_zu_over_zl'] = {lab: t2b[lab]['bracket_width_range'] for lab in t2b}
S['T2']['onset_x'] = {lab: t2b[lab]['onset_x'] for lab in t2b}
json.dump(S, open(R + 'summary.json', 'w'), indent=1)
for lab, v in t2b.items():
    print(lab, {k: v[k] for k in v if k != 'rows'})
    for r in v['rows']:
        print('   ', r)
