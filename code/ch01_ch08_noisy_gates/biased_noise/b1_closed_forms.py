"""B1: biased Pauli noise on one qubit, Z- and X-miscalibrations.
 (a) Theorem B closed forms vs direct 40-digit matrix evaluation (Fhat, D_BS, Umegaki and Uhlmann at max-ent input),
     including near-singular Choi matrices (p_perp down to 5e-8), and a random-input check of max-ent optimality.
 (b) Second-order coefficients (sympy) for the symmetric-transverse family px = py = pperp.
 (c) Windows delta_eta (eta = 0.1) for Fhat and D_BS by 40-digit bisection; two-sided bounds; asymptotic law
     delta_eta ~ 4 sqrt(x_eta qP/q1).
Output: results/b1.json
"""
import json, sys, random
import mpmath as mp
import sympy as sp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # common.py
sys.path.insert(0, str(Path(__file__).resolve().parent))  # pauli_biased.py
from common import mpmat, PAULI, choi_from_map_mp, gmean, ptrace_Y, eigvals, herm, sqrtm, logm, tr, DBS_channel, relent
from pauli_biased import *

mp.mp.dps = 40
OUT = str(Path(__file__).resolve().parent / 'results') + '/'
res = {}


def R_mp(axis, delta):
    P = mpmat(PAULI[IDX[axis]])
    return mp.cos(mp.mpf(delta) / 2) * mp.eye(2) - 1j * mp.sin(mp.mpf(delta) / 2) * P


def choi(q, axis, delta):
    R = R_mp(axis, delta)
    Ps = [mpmat(s) for s in PAULI]

    def f(E):
        rho = R * E * R.H
        out = mp.zeros(2, 2)
        for k in range(4):
            out += q[k] * Ps[k] * rho * Ps[k]
        return out
    return choi_from_map_mp(f, 2)


def direct(q, axis, delta):
    J1, J0 = choi(q, axis, delta), choi(q, axis, 0)
    Fh = eigvals(ptrace_Y(gmean(J1, J0), 2, 2))[0]
    Db = DBS_channel(J1, J0, 2, 2)
    w1, w0 = J1 / 2, J0 / 2
    Um = relent(w1, w0)
    s1 = sqrtm(w1)
    Uh = mp.fsum(mp.sqrt(max(e, 0)) for e in eigvals(herm(s1 * w0 * s1)))
    return Fh, Db, Um, Uh, J0


# ---------------- (a) closed forms vs direct ----------------
checks = []
pts = [(1e-2, 0.5, 0.05), (1e-3, 10, 0.01), (1e-3, 100, 3e-3), (1e-3, 1e4, 1e-4), (1e-2, 1e5, 2e-4), (5e-3, 30, 0.3)]
for pz, b, d in pts:
    for split in ((0.5, 0.5), (0.8, 0.2)):
        q = qubit_rates(pz, b, split)
        for axis in ('Z', 'X'):
            pr = qubit_pairs(q, axis)
            Fh, Db, Um, Uh, J0 = direct(q, axis, d)
            cf = (Fhat(pr, d), DBS(pr, d), Umegaki(pr, d), Uhlmann(pr, d))
            rel = [abs(a - c) / abs(c) for a, c in zip((Fh, Db, Um, Uh), cf)]
            lam = eigvals(J0)[0]
            checks.append(dict(pz=pz, b=b, split=split, axis=axis, delta=d, lambda_min_J=mp.nstr(lam, 6),
                               Fhat=mp.nstr(cf[0], 15), DBS=mp.nstr(cf[1], 15), D=mp.nstr(cf[2], 15), F=mp.nstr(cf[3], 15),
                               max_rel_dev=mp.nstr(max(rel), 3)))
            print(axis, pz, b, split, d, 'lam', mp.nstr(lam, 4), 'max rel dev', mp.nstr(max(rel), 3), flush=True)
res['closed_vs_direct'] = checks

# random-input check of max-ent optimality (Umegaki sup / Uhlmann min), one biased point
random.seed(1)
q = qubit_rates(1e-2, 20, (0.5, 0.5)); d = 0.2
inp = []
for axis in ('Z', 'X'):
    pr = qubit_pairs(q, axis)
    Dme, Fme = Umegaki(pr, d), Uhlmann(pr, d)
    worstD, worstF = mp.mpf(0), mp.mpf(1)
    J1, J0 = choi(q, axis, d), choi(q, axis, 0)
    for _ in range(40):
        v = mp.matrix([mp.mpc(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(4)]); v = v / mp.norm(v)
        # input |v> on R(x)X with R first index: output (Phi(x)id) applied; use Choi: omega = (1 (x) A^T) J (1 (x) conj A) style
        B = mp.matrix([[v[0], v[1]], [v[2], v[3]]])          # |v> = sum_{r,x} B_{rx} |r>|x>
        # output on Y(x)R: (Phi (x) id)(|v><v|) = (1_Y (x) B) J (1_Y (x) B^dag), with J = sum Phi(|x><x'|) (x) |x><x'|
        L = mp.zeros(4, 4)
        for y in range(2):
            for r in range(2):
                for x in range(2):
                    L[y * 2 + r, y * 2 + x] = B[r, x]
        o1 = herm(L * J1 * L.H); o0 = herm(L * J0 * L.H)
        Dv = relent(o1, o0)
        s1 = sqrtm(o1)
        Fv = mp.fsum(mp.sqrt(max(e, 0)) for e in eigvals(herm(s1 * o0 * s1)))
        worstD = max(worstD, Dv); worstF = min(worstF, Fv)
    inp.append(dict(axis=axis, D_maxent=mp.nstr(Dme, 12), D_random_max=mp.nstr(worstD, 12),
                    F_maxent=mp.nstr(Fme, 12), F_random_min=mp.nstr(worstF, 12)))
    print('input check', inp[-1], flush=True)
res['input_optimality_random'] = inp

# ---------------- (b) second-order coefficients (sympy) ----------------
pz, pp, dl = sp.symbols('p_z p_perp u', positive=True)   # u = sin(delta/2); delta^2-coefficient = (u^2-coefficient)/4
q1 = 1 - pz - 2 * pp
s2 = dl ** 2
sym = {}
for axis, pairs in (('Z', [(q1, pz), (pp, pp)]), ('X', [(q1, pp), (pp, pz)])):
    xs = [(a - c) ** 2 * s2 / (4 * a * c) for a, c in pairs]
    Fh = sum((a + c) / sp.sqrt(1 + x) for (a, c), x in zip(pairs, xs))
    Db = 2 * sum((a + c) * sp.sqrt(x / (1 + x)) * sp.asinh(sp.sqrt(x)) for (a, c), x in zip(pairs, xs))
    Um = s2 * sum((a - c) * sp.log(a / c) if a != c else 0 for a, c in pairs)
    Uh = sum(sp.sqrt((a + c) ** 2 - (a - c) ** 2 * s2) for a, c in pairs)
    FR = sum((a + c) * (a - c) ** 2 / (4 * a * c) for a, c in pairs)
    FB = sum((a - c) * sp.log(a / c) if a != c else 0 for a, c in pairs) / 2
    FS = sum((a - c) ** 2 / (a + c) for a, c in pairs)
    c_lnF = sp.simplify(sp.series(-2 * sp.log(Fh), dl, 0, 3).removeO().coeff(dl, 2) / 4 - FR / 4)
    c_DBS = sp.simplify(sp.series(Db, dl, 0, 3).removeO().coeff(dl, 2) / 4 - FR / 2)
    c_D = sp.simplify(sp.series(Um, dl, 0, 3).removeO().coeff(dl, 2) / 4 - FB / 2)
    c_F = sp.simplify(sp.series(1 - Uh, dl, 0, 3).removeO().coeff(dl, 2) / 4 - FS / 8)
    sym[axis] = dict(F_RLD=str(sp.factor(FR)), F_BKM=str(FB), F_SLD=str(sp.factor(FS)),
                     check_m2lnF_minus_FRLD_over_4=str(c_lnF), check_DBS_minus_FRLD_over_2=str(c_DBS),
                     check_D_minus_FBKM_over_2=str(c_D), check_1mF_minus_FSLD_over_8=str(c_F))
    print('sympy', axis, sym[axis], flush=True)
res['second_order_sympy'] = sym

# ---------------- (c) windows ----------------
eta = mp.mpf('0.1')
x_eta = ginv(eta)
win = []
for pzv in ('1e-2', '1e-3', '1e-4'):
    for b in ('0.5', '1', '3', '10', '30', '100', '300', '1000', '3000', '10000'):
        q = qubit_rates(pzv, b)
        row = dict(pz=float(pzv), b=float(b), pperp=float(q[1]))
        for axis in ('Z', 'X'):
            pr = qubit_pairs(q, axis)
            dF = window_Fhat(pr, eta); dB = window_DBS(pr, eta)
            lo, up = window_bounds(pr[0], pr[1][0] + pr[1][1], eta)
            qP = pr[0][1]; q1 = pr[0][0]
            asym = 4 * mp.sqrt(x_eta * qP / q1)
            xB = mp.findroot(lambda xx: 2 * phi(xx) - eta, mp.mpf('0.05'))
            asymB = 4 * mp.sqrt(xB * qP / q1)
            row[axis] = dict(delta_Fhat=float(dF), lower=float(lo), upper=float(up), inside=bool(lo <= dF <= up),
                             asym=float(asym), ratio_to_asym=float(dF / asym), delta_DBS=float(dB),
                             DBS_ratio_to_asym=float(dB / asymB))
        win.append(row)
        print('window', pzv, b, {a: (round(row[a]['delta_Fhat'], 8), round(row[a]['ratio_to_asym'], 5), row[a]['inside']) for a in 'ZX'}, flush=True)
res['windows'] = win
res['x_eta'] = float(x_eta); res['4sqrt_x_eta'] = float(4 * mp.sqrt(x_eta))
res['x_eta_DBS'] = float(mp.findroot(lambda xx: 2 * phi(xx) - eta, mp.mpf('0.05')))
json.dump(res, open(OUT + 'b1.json', 'w'), indent=1)
print('done')
