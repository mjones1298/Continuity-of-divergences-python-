"""B3: two-qubit local Stein bracket. N_theta = (noise)^{(x)2} o Ad_{exp(-i theta P/2)}, P = ZZ (weight 2) or Z1 (weight 1).
Per point: HNKS check; double SDP for F_inf on the 4-dim input (16 Kraus operators, 68x68 LMI); mpmath (40 digits) upper
certificate: within-pair h* from the restricted KKT, beta(h*) = 0 and ||alpha(h*)|| evaluated exactly; double-precision
lower bound min_{beta=0} tr(alpha)/4 over the FULL 16x16 Hermitian h; closed form (1-w)/w; F_RLD = lambda_max Tr_Y
[dJ J^-1 dJ] on 16x16 Choi matrices (40 digits) vs the pair formula.  Then the closed forms for n = 3 (ZZZ).
Output: results/b3.json (checkpointed).
"""
import json, sys, os, time
import mpmath as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # pauli_biased.py, zj_generic.py
from zj_generic import *
import pauli_biased as pb

mp.mp.dps = 40
OUT = str(Path(__file__).resolve().parent / 'results' / 'b3.json')
done = json.load(open(OUT)) if os.path.exists(OUT) else {}


def dep_single(p):
    p = mp.mpf(p)
    return {'1': 1 - 3 * p / 4, 'X': p / 4, 'Y': p / 4, 'Z': p / 4}


def biased_single(pz, b):
    q = pb.qubit_rates(pz, b)
    return dict(zip(('1', 'X', 'Y', 'Z'), q))


cases = []
for p in ('1e-1', '3e-2', '1e-2', '1e-3', '1e-4'):
    cases.append((f'dep|{p}|ZZ', dep_single(p), 'ZZ'))
for p in ('1e-2', '1e-3'):
    cases.append((f'dep|{p}|Z1', dep_single(p), 'Z1'))
for pz, b in (('1e-3', '100'), ('1e-3', '0.5')):
    cases.append((f'biasZ|{pz}|b={b}|ZZ', biased_single(pz, b), 'ZZ'))
    cases.append((f'biasZ|{pz}|b={b}|XX', biased_single(pz, b), 'XX'))

for key, single, gen in cases:
    if key in done:
        continue
    t0 = time.time()
    rates = local_rates(single, 2)
    K, Kd, labels = kraus_mp(rates, gen)
    r1, r2 = hnks_rank(K, gen)
    pairs = pairs_from_rates(rates, gen)
    w = mp.fsum(4 * a * c / (a + c) for a, c in pairs if a * c > 0)
    Fcl = (1 - w) / w
    FRc = pb.F_RLD(pairs)
    rec = dict(case=key, gen=gen, HNKS_holds=(r2 > r1), n_pairs=len(pairs), w=mp.nstr(w, 20), F_inf_closed=mp.nstr(Fcl, 25),
               F_RLD_closed=mp.nstr(FRc, 25), ratio_closed=mp.nstr(FRc / Fcl, 20),
               F_SLD_closed=mp.nstr(pb.F_SLD(pairs), 20), F_BKM_closed=mp.nstr(pb.F_BKM(pairs), 20),
               lambda_tilde=mp.nstr(min(rates.values()), 10))
    Fsdp, st = sdp_double(K, Kd)
    ent = pair_entries(labels, gen)
    ub_tr, hstar, res = trace_min_mp(K, Kd, ent)
    al, be = alpha_beta(K, Kd, hstar)
    ev = sorted([mp.re(e) for e in mp.eigh((al + al.H) / 2)[0]])
    Fup = 4 * ev[-1]
    Flow_d, st2 = trace_min_double_full(K, Kd)
    FRd, FRmin = F_RLD_direct(rates, gen)
    rec.update(F_inf_sdp_double=Fsdp, sdp_status=st, F_inf_upper_mp=mp.nstr(Fup, 25), alpha_spread=mp.nstr(ev[-1] - ev[0], 3),
               beta_resid=mp.nstr(mp.norm(be), 3), upper_vs_closed=mp.nstr((Fup - Fcl) / Fcl, 3),
               F_inf_lower_double_fullh=Flow_d, lower_status=st2, lower_vs_closed=float((Flow_d - Fcl) / Fcl),
               sdp_vs_closed=float((Fsdp - Fcl) / Fcl),
               F_RLD_direct=mp.nstr(FRd, 25), F_RLD_dev=mp.nstr(abs(FRd - FRc) / FRc, 3), TrY_scalar_dev=mp.nstr((FRd - FRmin) / FRd, 3),
               sec=round(time.time() - t0, 1))
    done[key] = rec; json.dump(done, open(OUT, 'w'), indent=1)
    print(key, {k: rec[k] for k in ('HNKS_holds', 'F_inf_closed', 'F_RLD_closed', 'ratio_closed', 'upper_vs_closed', 'beta_resid', 'lower_vs_closed', 'sdp_vs_closed', 'F_RLD_dev', 'sec')}, flush=True)

# ---- closed forms only: weight scaling for local depolarizing, n = 1..4, P = Z^{(x)w} (independent of n >= w)
scal = {}
for wgt in (1, 2, 3, 4):
    gen = 'Z' * wgt
    rows = []
    for p in ('1e-1', '1e-2', '1e-3', '1e-4', '1e-5'):
        rates = local_rates(dep_single(p), wgt)
        pairs = pairs_from_rates(rates, gen)
        w = mp.fsum(4 * a * c / (a + c) for a, c in pairs if a * c > 0)
        Fi = (1 - w) / w; FR = pb.F_RLD(pairs)
        pp = mp.mpf(p)
        rows.append(dict(p=float(pp), F_inf=mp.nstr(Fi, 15), F_RLD=mp.nstr(FR, 15), ratio=mp.nstr(FR / Fi, 12),
                         F_inf_scaled=mp.nstr(Fi * pp ** ((wgt + 1) // 2), 10), F_RLD_scaled=mp.nstr(FR * pp ** wgt, 10),
                         ratio_scaled=mp.nstr(FR / Fi * pp ** (wgt // 2), 10)))
    scal[wgt] = rows
    print('weight', wgt, [(r['p'], r['F_inf_scaled'], r['F_RLD_scaled'], r['ratio_scaled']) for r in rows], flush=True)
done['weight_scaling_local_dep'] = scal
# n-independence check: ZZ on 3 qubits (ZZ1) equals ZZ on 2
r3 = local_rates(dep_single('1e-2'), 3); pr3 = pairs_from_rates(r3, 'ZZ1')
r2 = local_rates(dep_single('1e-2'), 2); pr2 = pairs_from_rates(r2, 'ZZ')
done['n_independence_ZZ1_vs_ZZ'] = dict(F_inf=[mp.nstr(pb.F_inf(pr3), 20), mp.nstr(pb.F_inf(pr2), 20)],
                                       F_RLD=[mp.nstr(pb.F_RLD(pr3), 20), mp.nstr(pb.F_RLD(pr2), 20)])
json.dump(done, open(OUT, 'w'), indent=1)
print('done')
