"""M1, part 2: the Chernoff side at the noisy-gate pair N_p = D_p o U, N_{p,delta} = D_p o (Rz(delta) U).

Sections
  A. Chat = -ln Fhat: closed form vs direct 4x4 matrix evaluation; local coefficient F_RLD/8.
  B. explicit strategies (lower bounds on xi), closed forms re-verified by explicit simulation (mpmath, 40 digits):
       (P)  product, maximally entangled input, Bell measurement (classical Chernoff);
       (A)  k sequential uses on half of a Bell pair, U^dag between uses, collective measurement on i.i.d. blocks:
            xi >= -(1/k) ln[1 - sin^2(k delta/2) (sqrt(1-3q/4) - sqrt(q/4))^2],  q = 1-(1-p)^k   (k = 1: product input);
       (B)  heralded code {|00>,|11>} (probe + noiseless ancilla), abort on flag:
            xi >= -(1/k) ln[1 - (1-p/2)^k (1 - sqrt(1-r^2)) sin^2(k delta/2)],  r = (1-2a)^k, a = (p/4)/(1-p/2).
  C. local constants p*xi/delta^2: strategies (finite p and p -> 0), Zhou-Jiang F_inf/8, Chat F_RLD/8.
  D. program-simulation converse mu_* (SDP, cvxpy/Clarabel) vs Chat.
  E. finite-delta bracket for p = 1e-2, 1e-3, 1e-4 at delta = 2*delta_j (delta_j: grid of A_numerics T2):
       upper: Chat(delta);  midpoint lemma xi(delta) <= B(N_{delta/2}||N_0), B in {D_BS, D^#_alpha (1 and 2 copies,
       certified in A_numerics), D_max, ln(4/p)} (strong-converse Stein bounds);  and, separately, xi <= zeta <= U3(delta).
Output: m1_chernoff.out, results_chernoff.json
"""
import sys, json
import numpy as np
import mpmath as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # m1lib.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # sharp_alpha.py, common.py
from m1lib import *

mp.mp.dps = 40
OUT = open(Path(__file__).resolve().parent / 'm1_chernoff.out', 'w')
RES = {}


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + '\n'); OUT.flush()


def xpar(p, d):
    p, d = mp.mpf(p), mp.mpf(d)
    return 4 * (1 - p) ** 2 * mp.sin(d / 2) ** 2 / (p * (4 - 3 * p))


def Chat_closed(p, d):
    p = mp.mpf(p)
    return -mp.log(p / 2 + (1 - p / 2) / mp.sqrt(1 + xpar(p, d)))


def DBS_closed(p, d):
    p = mp.mpf(p); x = xpar(p, d)
    return (2 - p) * mp.sqrt(x / (1 + x)) * mp.asinh(mp.sqrt(x))


def Dmax_closed(p, d):
    return 2 * mp.asinh(mp.sqrt(xpar(p, d)))


def FRLD(p):
    p = mp.mpf(p); return 2 * (1 - p) ** 2 * (2 - p) / (p * (4 - 3 * p))


def Finf(p):
    p = mp.mpf(p); return 2 * (1 - p) ** 2 / (p * (3 - 2 * p))


def choi_N(p, d):
    p = mp.mpf(p)
    q = depol_rates(p, 1)
    return choi_pauli(q, Rz(d))


# ---------------- strategy closed forms (mpmath; k loop) ----------------
def xiA_k(p, d, k):
    p, d = mp.mpf(p), mp.mpf(d)
    q = -mp.expm1(k * mp.log1p(-p))
    y = mp.sin(k * d / 2) ** 2 * (mp.sqrt(1 - 3 * q / 4) - mp.sqrt(q / 4)) ** 2
    return -mp.log1p(-y) / k


def xiB_k(p, d, k):
    p, d = mp.mpf(p), mp.mpf(d)
    a = (p / 4) / (1 - p / 2)
    r = mp.exp(k * mp.log1p(-2 * a))
    y = mp.exp(k * mp.log1p(-p / 2)) * (1 - mp.sqrt(1 - r ** 2)) * mp.sin(k * d / 2) ** 2
    return -mp.log1p(-y) / k


def xiP(p, d):
    p, d = mp.mpf(p), mp.mpf(d)
    qI, qZ = 1 - 3 * p / 4, p / 4
    s2 = mp.sin(d / 2) ** 2; c2 = 1 - s2
    P0 = [qI, qZ, p / 2]
    P1 = [qI * c2 + qZ * s2, qZ * c2 + qI * s2, p / 2]
    return classical_chernoff(P1, P0)[0]


def best_k(fun, p, d, kmax):
    """maximise fun(p,d,k) over integer k <= kmax: coarse float scan, then exact mp evaluation around the argmax."""
    pf, df = float(p), float(d)
    ks = np.unique(np.concatenate([np.arange(1, min(kmax, 400) + 1), np.unique(np.geomspace(1, kmax, 400).astype(int))]))
    if fun is xiA_k:
        q = -np.expm1(ks * np.log1p(-pf))
        y = np.sin(ks * df / 2) ** 2 * (np.sqrt(1 - 3 * q / 4) - np.sqrt(q / 4)) ** 2
    else:
        a = (pf / 4) / (1 - pf / 2); r = np.exp(ks * np.log1p(-2 * a))
        y = np.exp(ks * np.log1p(-pf / 2)) * (1 - np.sqrt(1 - r ** 2)) * np.sin(ks * df / 2) ** 2
    v = -np.log1p(-y) / ks
    k0 = int(ks[int(np.argmax(v))])
    cands = range(max(1, k0 - 30), min(kmax, k0 + 30) + 1)
    best = max(((fun(p, d, k), k) for k in cands), key=lambda t: t[0])
    return best


# ================================================================ A
log('=== A. Chat = -ln Fhat at the noisy-gate pair: closed form vs 4x4 matrices ===')
RES['A'] = []
for p, d in (('1e-2', '0.05'), ('1e-3', '0.02'), ('1e-4', '0.3'), ('1e-3', '1e-5')):
    J1, J0 = choi_N(p, d), choi_N(p, 0)
    cm = Chat(J1, J0, 2, 2)
    f12 = -mp.log(Qhat_s(J1, J0, mp.mpf(1) / 2, 2, 2))
    cc = Chat_closed(p, d)
    log('   p=%s delta=%s: Chat(matrix, min over s)=%s  -ln Qhat_1/2=%s  closed=%s' % (p, d, mp.nstr(cm, 20), mp.nstr(f12, 20), mp.nstr(cc, 20)))
    RES['A'].append(dict(p=p, delta=d, dev=float(abs(cm - cc)), dev12=float(abs(f12 - cc))))
for p in ('1e-2', '1e-3', '1e-4'):
    d = mp.mpf('1e-9') * mp.sqrt(mp.mpf(p))
    log('   p=%s: Chat/delta^2 / (F_RLD/8) at delta=1e-9 sqrt(p): %s' % (p, mp.nstr(Chat_closed(p, d) / d ** 2 / (FRLD(p) / 8), 15)))

# ================================================================ B
log('\n=== B. Strategy formulas vs explicit simulation (mpmath) ===')
RES['B'] = []
p = mp.mpf('0.03'); dd = mp.mpf('0.2')
wts = {'I': 1 - 3 * p / 4, 'X': p / 4, 'Y': p / 4, 'Z': p / 4}


def depol_on(rho, sysmats):
    """apply D_p to the subsystem represented by the list of embedded Paulis sysmats[label]."""
    out = mp.zeros(rho.rows, rho.cols)
    for lab, w in wts.items():
        P = sysmats[lab]
        out += w * P * rho * P.H
    return out


# (A): probe (x) reference
emb = {lab: kron(PAULI[lab], I2) for lab in 'IXYZ'}
for k in (1, 2, 3):
    def runA(delta):
        psi = mp.matrix([1, 0, 0, 1]) / mp.sqrt(2)
        rho = outer(psi, psi)
        R = kron(Rz(delta), I2)
        for _ in range(k):
            rho = R * rho * R.H
            rho = depol_on(rho, emb)
        return rho
    sim, sopt = chernoff_states(runA(dd), runA(0))
    log('   (A) k=%d: simulated quantum Chernoff=%s (s*=%s), closed form=%s' % (k, mp.nstr(sim, 20), mp.nstr(sopt, 6), mp.nstr(xiA_k(p, dd, k) * k, 20)))
    RES['B'].append(dict(strategy='A', k=k, dev=float(abs(sim - xiA_k(p, dd, k) * k))))

# (B): probe (x) ancilla, code |0_L> = |00>, |1_L> = |11>, input |+_L>; per use: Rz on probe, D_p on probe,
# parity (Z(x)Z) measurement; abort on odd parity.
embB = {lab: kron(PAULI[lab], I2) for lab in 'IXYZ'}
Peven = mp.zeros(4, 4); Peven[0, 0] = 1; Peven[3, 3] = 1
for k in (1, 2, 3):
    def runB(delta):
        psi = mp.matrix([1, 0, 0, 1]) / mp.sqrt(2)
        rho = outer(psi, psi)
        R = kron(Rz(delta), I2)
        for _ in range(k):
            rho = R * rho * R.H
            rho = depol_on(rho, embB)
            rho = Peven * rho * Peven  # unflagged branch (unnormalised)
        return rho
    r1, r0 = runB(dd), runB(0)
    Pnf = mp.re(tr(r1)); Pnf0 = mp.re(tr(r0))
    # cq output: flag (prob 1-Pnf, identical under both hypotheses) + unflagged state
    def Qs(s):
        return (1 - Pnf) + mp.re(tr(powm(r1, s) * powm(r0, 1 - s)))
    smin = mp.findroot(lambda s: mp.diff(Qs, s), mp.mpf('0.5'))
    sim = -mp.log(Qs(smin))
    log('   (B) k=%d: P_noflag=%s (formula %s), simulated Chernoff=%s (s*=%s), closed form=%s' % (
        k, mp.nstr(Pnf, 15), mp.nstr((1 - p / 2) ** k, 15), mp.nstr(sim, 20), mp.nstr(smin, 6), mp.nstr(xiB_k(p, dd, k) * k, 20)))
    RES['B'].append(dict(strategy='B', k=k, dev=float(abs(sim - xiB_k(p, dd, k) * k))))

# (P): Bell measurement after one use = classical Chernoff; check against simulated Bell-basis probabilities
psi = mp.matrix([1, 0, 0, 1]) / mp.sqrt(2)
def bell_probs(delta):
    rho = outer(psi, psi); R = kron(Rz(delta), I2)
    rho = depol_on(R * rho * R.H, emb)
    out = []
    for lab in 'IXYZ':
        b = kron(PAULI[lab], I2) * psi
        out.append(mp.re(inner(b, rho * b)))
    return out
P1, P0 = bell_probs(dd), bell_probs(0)
cP = classical_chernoff(P1, P0)[0]
log('   (P) Bell-measurement classical Chernoff: simulated=%s closed=%s' % (mp.nstr(cP, 20), mp.nstr(xiP(p, dd), 20)))
RES['B'].append(dict(strategy='P', dev=float(abs(cP - xiP(p, dd)))))

# ================================================================ C
log('\n=== C. Local constants p*xi/delta^2 (delta -> 0) ===')


def cA_lim():
    f = lambda u: u * (mp.sqrt(4 - 3 * (1 - mp.exp(-u))) - mp.sqrt(1 - mp.exp(-u))) ** 2 / 16
    u = mp.findroot(lambda u: mp.diff(f, u), mp.mpf('0.5'))
    return f(u), u


def cB_lim():
    f = lambda u: u * mp.exp(-u / 2) * (1 - mp.sqrt(1 - mp.exp(-u))) / 4
    u = mp.findroot(lambda u: mp.diff(f, u), mp.mpf('0.6'))
    return f(u), u


def cA_SLD_lim():
    f = lambda u: u * 2 * mp.exp(-2 * u) / (1 + mp.exp(-u)) / 8
    u = mp.findroot(lambda u: mp.diff(f, u), mp.mpf('0.6'))
    return f(u), u


cA, uA = cA_lim(); cB, uB = cB_lim(); cAS, uAS = cA_SLD_lim()
log('   p->0 limits: c_A = %s (u*=%s);  c_B = %s (u*=%s);  [SLD/8 rate of strategy A: %s (u*=%s), ratio c_A/that = %s]' % (
    mp.nstr(cA, 15), mp.nstr(uA, 10), mp.nstr(cB, 15), mp.nstr(uB, 10), mp.nstr(cAS, 15), mp.nstr(uAS, 10), mp.nstr(cA / cAS, 8)))
log('   Zhou-Jiang: p F_inf/8 -> 1/12 = %s ;  Chat: p F_RLD/8 -> 1/8' % mp.nstr(mp.mpf(1) / 12, 15))
RES['C'] = dict(cA=float(cA), uA=float(uA), cB=float(cB), uB=float(uB), cA_SLD=float(cAS), rows=[])
log('   %-8s %-14s %-14s %-14s %-14s %-14s %-12s' % ('p', 'p*xiA/d^2', 'p*xiB/d^2', 'p*Finf/8', 'p*FRLD/8', 'product k=1', 'Bell-product'))
for p in ('1e-1', '1e-2', '1e-3', '1e-4'):
    pm = mp.mpf(p)
    kmax = int(20 / pm)
    # local coefficient of each k: A: k (sqrt(mu)-sqrt(nu))^2/4 ; B: k (1-p/2)^k (1-sqrt(1-r^2))/4
    bestA = max((k * (mp.sqrt(1 - 3 * (-mp.expm1(k * mp.log1p(-pm))) / 4) - mp.sqrt(-mp.expm1(k * mp.log1p(-pm)) / 4)) ** 2 / 4, k)
                for k in range(1, kmax + 1, max(1, kmax // 4000)))
    kk = bestA[1]
    bestA = max((k * (mp.sqrt(1 - 3 * (-mp.expm1(k * mp.log1p(-pm))) / 4) - mp.sqrt(-mp.expm1(k * mp.log1p(-pm)) / 4)) ** 2 / 4, k)
                for k in range(max(1, kk - 20), kk + 21))
    a = (pm / 4) / (1 - pm / 2)
    fB = lambda k: k * mp.exp(k * mp.log1p(-pm / 2)) * (1 - mp.sqrt(1 - mp.exp(2 * k * mp.log1p(-2 * a)))) / 4
    bB = max((fB(k), k) for k in range(1, kmax + 1, max(1, kmax // 4000)))
    kk = bB[1]; bB = max((fB(k), k) for k in range(max(1, kk - 20), kk + 21))
    prod1 = (mp.sqrt(1 - 3 * pm / 4) - mp.sqrt(pm / 4)) ** 2 / 4
    # numerical check of the delta->0 limit of the explicit (finite-delta) formulas
    dsm = mp.mpf('1e-8') * pm
    chkA = xiA_k(pm, dsm, bestA[1]) / dsm ** 2
    chkB = xiB_k(pm, dsm, bB[1]) / dsm ** 2
    bell = xiP(pm, mp.mpf('1e-3') * mp.sqrt(pm))
    log('   %-8s %-14s %-14s %-14s %-14s %-14s Bell(d=1e-3 sqrt p): xi/d^4*p=%s' % (
        p, mp.nstr(pm * bestA[0], 8) + ' k=%d' % bestA[1], mp.nstr(pm * bB[0], 8) + ' k=%d' % bB[1],
        mp.nstr(pm * Finf(pm) / 8, 8), mp.nstr(pm * FRLD(pm) / 8, 8), mp.nstr(pm * prod1, 6),
        mp.nstr(bell / (mp.mpf('1e-3') * mp.sqrt(pm)) ** 4 * pm, 6)))
    log('            check: finite-delta formulas at delta=1e-8 p give p*xi/d^2 = %s (A), %s (B)' % (
        mp.nstr(pm * chkA, 10), mp.nstr(pm * chkB, 10)))
    RES['C']['rows'].append(dict(p=float(pm), cA=float(pm * bestA[0]), kA=bestA[1], cB=float(pm * bB[0]), kB=bB[1],
                                 cZJ=float(pm * Finf(pm) / 8), cRLD=float(pm * FRLD(pm) / 8), cprod=float(pm * prod1),
                                 ratio_RLD_inf=float(FRLD(pm) / Finf(pm))))

# ================================================================ D
log('\n=== D. Program-simulation converse mu_* (SDP) ===')
import cvxpy as cp
RES['D'] = []
labs = ['I', 'X', 'Y', 'Z']
Pn = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]), 'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.diag([1, -1])}


def mu_star(p, d):
    q = [1 - 3 * p / 4, p / 4, p / 4, p / 4]
    R = np.diag([np.exp(-1j * d / 2), np.exp(1j * d / 2)])
    K = [np.sqrt(q[i]) * Pn[l] @ R for i, l in enumerate(labs)]
    L = [np.sqrt(q[i]) * Pn[l] for i, l in enumerate(labs)]
    C = cp.Variable((4, 4), complex=True)
    c = cp.Variable()
    S = sum(C[l, k] * (L[l].conj().T @ K[k]) for l in range(4) for k in range(4))
    cons = [S == c * np.eye(2), cp.bmat([[np.eye(4), C], [C.H, np.eye(4)]]) >> 0]
    prob = cp.Problem(cp.Maximize(c), cons)
    prob.solve(solver='CLARABEL')
    return c.value, prob.status


for p in (1e-2, 1e-3, 1e-4):
    for xx in (0.01, 0.1, 1.0, 10.0):
        d = xx * p * np.log(1 / p)
        if d > np.pi:
            continue
        mu, st = mu_star(p, d)
        conv = -2 * np.log(mu)
        ch = float(Chat_closed(p, d))
        log('   p=%g x=%g delta=%.4g: -2 ln mu_* = %.6g (%s)  Chat = %.6g  ratio = %.4g' % (p, xx, d, conv, st, ch, conv / ch))
        RES['D'].append(dict(p=p, x=xx, delta=d, m2lnmu=conv, Chat=ch, status=st))

# ================================================================ E
log('\n=== E. Finite-delta bracket on xi at delta = 2 delta_j ===')
T2 = json.load(open(Path(__file__).resolve().parents[1] / 'crossover' / 'results' / 't2.json'))['results']
M2 = json.load(open(Path(__file__).resolve().parents[1] / 'crossover' / 'results' / 't2b_merged.json'))['results']


def U3(p, d):
    p, d = mp.mpf(p), mp.mpf(d)
    eps = (1 - p) * abs(mp.sin(d / 2))
    x = eps / (1 + eps)
    h = -x * mp.log(x) - (1 - x) * mp.log(1 - x)
    return 2 * eps * mp.log(2) + (1 + eps) * h + eps * mp.log((4 - 3 * p) / p)


RES['E'] = {}
for pk in ('0.01', '0.001', '0.0001'):
    p = mp.mpf(pk)
    rows = []
    log('-- p = %s' % pk)
    log('   %-7s %-9s %-11s %-11s %-6s %-11s %-11s %-11s %-9s %-9s %-7s' % ('x', 'delta', 'Chat', 'B(d/2)', 'from', 'U3(d)', 'xi_-', 'which', 'C/xi+', 'C/xi+Sh', 'C/xi_-'))
    for r1, r2 in zip(T2[pk], M2[pk]):
        dj = mp.mpf(r1['delta'])
        d = 2 * dj
        if d > mp.pi:
            continue
        # strong-converse Stein bounds valid against ADAPTIVE strategies: D_BS (geometric Renyi chain rule, alpha->1),
        # one-copy D^#_alpha (Fawzi-Fawzi chain rule), D_max and ln(4/p) (CP domination).  The two-copy D^# is NOT used:
        # it bounds zeta only through zeta = D^reg (weak converse).
        cands = {'BS': r1['DBS'], 'sharp': r1['Dsharp_cert'], 'ln4p': r1['ln4p'], 'Dmax': float(Dmax_closed(p, dj))}
        Bfrom, Bmid = min(((k, v) for k, v in cands.items() if v is not None and np.isfinite(v)), key=lambda t: t[1])
        ch = Chat_closed(p, d)
        u3 = U3(p, d)
        kmax = int(max(10 / float(p), 50 / float(d)))
        a = best_k(xiA_k, p, d, kmax); b = best_k(xiB_k, p, d, kmax)
        pr = xiP(p, d)
        xl, which = max((a[0], 'A k=%d' % a[1]), (b[0], 'B k=%d' % b[1]), (pr, 'Bell'), key=lambda t: t[0])
        xu = min(ch, mp.mpf(Bmid))
        xuS = min(xu, u3)
        rec = dict(x=2 * r1['x'], delta=float(d), Chat=float(ch), Bmid=float(Bmid), Bfrom=Bfrom, U3=float(u3), xi_lower=float(xl), which=which,
                   xiA=float(a[0]), kA=a[1], xiB=float(b[0]), kB=b[1], xiBell=float(pr),
                   C_over_xu=float(ch / xu), C_over_xuS=float(ch / xuS), C_over_xl=float(ch / xl), xu_over_xl=float(xuS / xl),
                   pxi_l_over_d2=float(p * xl / d ** 2), pC_over_d2=float(p * ch / d ** 2))
        rows.append(rec)
        log('   %-7.3g %-9.3g %-11.4g %-11.4g %-6s %-11.4g %-11.4g %-11s %-9.4g %-9.4g %-7.4g' % (
            rec['x'], rec['delta'], rec['Chat'], rec['Bmid'], Bfrom, rec['U3'], rec['xi_lower'], which, rec['C_over_xu'], rec['C_over_xuS'], rec['C_over_xl']))
    RES['E'][pk] = rows
    mx = max(rows, key=lambda r: r['C_over_xu']); mxS = max(rows, key=lambda r: r['C_over_xuS'])
    on = next((r for r in rows if r['C_over_xu'] > 1.01), None)
    onS = next((r for r in rows if r['C_over_xuS'] > 1.01), None)
    log('   max certified Chat/xi (refereed+Chernoff-paper bounds) = %.4g at x=%.3g (delta=%.3g); onset >1.01 at x=%s' % (
        mx['C_over_xu'], mx['x'], mx['delta'], ('%.3g' % on['x']) if on else 'none'))
    log('   max certified Chat/xi (incl. U3 [Shadows]) = %.4g at x=%.3g (delta=%.3g); onset >1.01 at x=%s' % (
        mxS['C_over_xuS'], mxS['x'], mxS['delta'], ('%.3g' % onS['x']) if onS else 'none'))
    log('   max Chat/xi_- = %.4g; bracket width xi_+/xi_- in [%.3g, %.3g]; small-x p*xi_-/d^2 = %.5g, p*Chat/d^2 = %.5g' % (
        max(r['C_over_xl'] for r in rows), min(r['xu_over_xl'] for r in rows), max(r['xu_over_xl'] for r in rows),
        rows[0]['pxi_l_over_d2'], rows[0]['pC_over_d2']))
    log('   peak certified overshoot x sqrt(p) ln(1/p): %.4g (refereed), %.4g (with U3)' % (
        mx['C_over_xu'] * float(mp.sqrt(p) * mp.log(1 / p)), mxS['C_over_xuS'] * float(mp.sqrt(p) * mp.log(1 / p))))

json.dump(RES, open(Path(__file__).resolve().parent / 'results_chernoff.json', 'w'), indent=1)
log('\ndone')
