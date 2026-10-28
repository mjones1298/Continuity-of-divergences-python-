"""B2: asymptotic adaptive QFI F_inf (Zhou-Jiang) and F_RLD for Z-biased qubit Pauli noise, Z- and X-miscalibrations.
Per point: HNKS check; double SDP (Clarabel); mpmath (50 digits) lower bound min_{beta=0} tr(alpha)/2 over the FULL h
(KKT); upper bound ||alpha(h*)|| at that minimiser (beta(h*)=0 exactly); closed form (1-w)/w (ZJ Eq. (50));
F_RLD = lambda_max Tr_Y[dJ J^-1 dJ] (50 digits) vs the pair formula; ratio F_RLD/F_inf vs its closed form.
Output: results/b2.json (checkpointed after every point).
"""
import json, sys, os, time
import mpmath as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # pauli_biased.py, zj_generic.py
from zj_generic import *
import pauli_biased as pb

mp.mp.dps = 50
OUT = str(Path(__file__).resolve().parent / 'results' / 'b2.json')
done = json.load(open(OUT)) if os.path.exists(OUT) else {}

grid = []
for pz in ('1e-2', '1e-3'):
    for b in ('0.5', '1', '3', '10', '100', '1000'):
        grid.append((pz, b, (0.5, 0.5)))
grid += [('1e-3', '10', (0.9, 0.1)), ('1e-3', '100', (0.99, 0.01))]
grid += [('1e-3', 'inf', (0.5, 0.5))]          # pure dephasing


def rates_of(pz, b, split):
    if b == 'inf':
        q = (1 - mp.mpf(pz), mp.mpf(0), mp.mpf(0), mp.mpf(pz))
    else:
        q = pb.qubit_rates(pz, b, split)
    return dict(zip(('1', 'X', 'Y', 'Z'), q)), q


for pz, b, split in grid:
    for axis in ('Z', 'X'):
        key = f'{pz}|{b}|{split[0]}|{axis}'
        if key in done:
            continue
        t0 = time.time()
        rates, q = rates_of(pz, b, split)
        K, Kd, labels = kraus_mp(rates, axis)
        r1, r2 = hnks_rank(K, axis)
        rec = dict(pz=float(pz), b=b, split=split, axis=axis, rates=[mp.nstr(x, 12) for x in q],
                   HNKS_holds=(r2 > r1))
        pairs = pb.qubit_pairs(q, axis)
        if r2 > r1:
            rec['note'] = 'H not in Kraus span: Heisenberg scaling, F_N = Theta(N^2), F_inf = infinity'
            done[key] = rec; json.dump(done, open(OUT, 'w'), indent=1)
            print(key, rec['note'], flush=True); continue
        Fsdp, st = sdp_double(K, Kd)
        ent = [(i, j) for i in range(len(K)) for j in range(i, len(K))]
        lb, hstar, res = trace_min_mp(K, Kd, ent)
        al, be = alpha_beta(K, Kd, hstar)
        ev = sorted([mp.re(e) for e in mp.eigh((al + al.H) / 2)[0]])
        Fup = 4 * ev[-1]; Flow = 4 * lb
        w = sum(4 * a * c / (a + c) for a, c in pairs if a * c > 0)
        Fcl = (1 - w) / w
        rec.update(F_inf_sdp_double=Fsdp, sdp_status=st, F_inf_upper=mp.nstr(Fup, 30), F_inf_lower=mp.nstr(Flow, 30),
                   cert_gap_rel=mp.nstr((Fup - Flow) / Flow, 3), beta_resid=mp.nstr(mp.norm(be), 3),
                   F_inf_closed=mp.nstr(Fcl, 30), closed_dev_rel=mp.nstr(abs(Fcl - Flow) / Fcl, 3),
                   sdp_dev_rel=float(abs(Fsdp - Fcl) / Fcl), w=mp.nstr(w, 20))
        if all(a * c > 0 for a, c in pairs):
            FRd, FRmin = F_RLD_direct(rates, axis)
            FRc = pb.F_RLD(pairs)
            rec.update(F_RLD_direct=mp.nstr(FRd, 30), F_RLD_closed=mp.nstr(FRc, 30), F_RLD_dev_rel=mp.nstr(abs(FRd - FRc) / FRc, 3),
                       TrY_scalar_dev=mp.nstr(FRd - FRmin, 3), ratio=mp.nstr(FRc / Fcl, 20))
        else:
            # pure dephasing, Z axis: J has rank 2; D_BS computed on the common support -> pair formula with the (1,Z) pair only
            FRc = pb.F_RLD([pp for pp in pairs if pp[0] * pp[1] > 0])
            rec.update(F_RLD_closed_on_support=mp.nstr(FRc, 30), ratio=mp.nstr(FRc / Fcl, 20))
        # asymptotic closed-form ratio (p -> 0 at fixed bias): 1 + h_spec / h_main
        (a1, c1), (a2, c2) = pairs
        hm = 4 * a1 * c1 / (a1 + c1); hs = 4 * a2 * c2 / (a2 + c2) if a2 * c2 > 0 else 0
        rec['ratio_leading'] = mp.nstr(1 + hs / hm, 12)
        rec['sec'] = round(time.time() - t0, 1)
        done[key] = rec; json.dump(done, open(OUT, 'w'), indent=1)
        print(key, {k: rec.get(k) for k in ('F_inf_closed', 'cert_gap_rel', 'closed_dev_rel', 'sdp_dev_rel', 'F_RLD_dev_rel', 'ratio', 'ratio_leading', 'sec')}, flush=True)
print('done')
