"""Independent numerical cross-checks of A_proofs/theorems.md against this directory's results (27 Oct 2026).
  (1) Theorem B (Pauli-block closed forms for F_hat and D_BS, all n, any Pauli string, any input-side U) against direct
      40-digit matrix computations from the full Choi matrices (n = 1, 2; random U for n = 2), incl. the worked example
      P = Z(x)X, p = 0.01, delta = 0.02 (theorems.md Sec. 5: F_hat = 0.458087, sandwich [0.44992, 0.45814]).
  (2) Theorem W(3)/(5): two-sided window bounds and the global exact window versus the T3 bisection values (results/t3.json).
  (3) Theorem W(4)/L1: log-modulus d/(4 lambda) = d^2/(4 p^n) by a 40-digit central finite difference of ln F_hat(Ad_W, Psi_e),
      Psi_e = Psi + (e/2)(Ad_W - Ad_W'), W = Z^{(x)n} (full weight), W' = X^{(x)n}; n = 1, 2, 3.
  (4) Corollary H-q / Prop H: sympy delta-expansion of the theorems.md closed forms against the T4 coefficients.
  (5) The R7 / Theorem S constant f*/4 = 0.14958713 (u* = 0.466968), the finite-p values 0.148832 / 0.149512 / 0.149580,
      and the heralded-strategy limit 0.158119 of T2.
Uses only common.py (does NOT import t3_multiqubit.py, whose import truncates results/t3.log).
Writes results/crosscheck_theorems.json."""
import sys, json, itertools, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import numpy as np, mpmath as mp, sympy as sp
from common import mpmat, herm, sqrtm, gmean, ptrace_Y, eigvals, DBS_channel, PAULI, kron

mp.mp.dps = 40
HERE = Path(__file__).resolve().parent
R = str(HERE / 'results') + '/'                                   # output: results/crosscheck_theorems.json
R_T3 = str(HERE.parent / 'multiqubit_window' / 'results') + '/'   # input: t3.json
R_SUM = str(HERE.parent / 'crossover' / 'results') + '/'          # input: summary.json
PM = [mpmat(s) for s in PAULI]
OUT = {}


def pstr(idx):
    M = PM[idx[0]]
    for i in idx[1:]:
        M = kron(M, PM[i])
    return M


def vec(M):
    d = M.rows
    return mp.matrix([M[a, i] for a in range(d) for i in range(d)])   # (M (x) 1)|Omega>, output index first


def choi_pauli(q, V):
    d = V.rows; J = mp.zeros(d * d, d * d)
    for Q, w in q.items():
        v = vec(pstr(Q) * V); J += w * (v * v.H)
    return herm(J)


def q_local(p, n):
    p = mp.mpf(p); w1 = [1 - 3 * p / 4, p / 4, p / 4, p / 4]
    return {idx: mp.fprod([w1[i] for i in idx]) for idx in itertools.product(range(4), repeat=n)}


def q_global(p, n):
    p = mp.mpf(p); d = 2 ** n
    return {idx: (1 - p + p / d ** 2) if not any(idx) else p / d ** 2 for idx in itertools.product(range(4), repeat=n)}


MUL = {(0, b): b for b in range(4)}
MUL.update({(a, 0): a for a in range(4)}); MUL.update({(a, a): 0 for a in range(4)})
MUL.update({(1, 2): 3, (2, 1): 3, (2, 3): 1, (3, 2): 1, (1, 3): 2, (3, 1): 2})


def thmB(q, P, delta):
    """Theorem B: F_hat = sum_pairs (q+q')(1+x)^(-1/2); D_BS = 2 sum (q+q') phi(x)."""
    s2 = mp.sin(mp.mpf(delta) / 2) ** 2; seen = set(); F = 0; DBS = 0
    for Q in q:
        if Q in seen:
            continue
        QP = tuple(MUL[(a, b)] for a, b in zip(Q, P)); seen |= {Q, QP}
        a, b = q[Q], q[QP]; x = (a - b) ** 2 * s2 / (4 * a * b)
        F += (a + b) / mp.sqrt(1 + x)
        DBS += 2 * (a + b) * (mp.sqrt(x / (1 + x)) * mp.asinh(mp.sqrt(x)) if x > 0 else 0)
    return F, DBS


def Rp(P, delta):
    M = pstr(P); return mp.cos(mp.mpf(delta) / 2) * mp.eye(M.rows) - 1j * mp.sin(mp.mpf(delta) / 2) * M


def rand_unitary(d, seed):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); H = (H + H.conj().T) / 2
    return mp.expm(1j * herm(mpmat(H)))


# ---------------- (1) Theorem B ----------------
t0 = time.time(); rowsB = []
cases = [(1, 'local', (3,), '0.01', '0.05', None), (1, 'local', (1,), '0.001', '0.3', None),
         (2, 'local', (3, 1), '0.01', '0.02', None), (2, 'local', (3, 1), '0.01', '0.02', 7),
         (2, 'local', (3, 3), '0.001', '0.004', 11), (2, 'local', (2, 0), '0.003', '0.1', 13),
         (2, 'global', (3, 3), '0.01', '0.05', 17)]
for n, model, P, p, dl, seed in cases:
    q = q_local(p, n) if model == 'local' else q_global(p, n)
    d = 2 ** n
    U = rand_unitary(d, seed) if seed else mp.eye(d)
    J0 = choi_pauli(q, U); Jd = choi_pauli(q, Rp(P, dl) * U)
    TY = ptrace_Y(gmean(Jd, J0), d, d); ev = eigvals(herm(TY))
    F_dir = ev[0]; off = ev[-1] - ev[0]
    D_dir = DBS_channel(Jd, J0, d, d); D_rev = DBS_channel(J0, Jd, d, d)
    F_cf, D_cf = thmB(q, P, dl)
    rowsB.append(dict(n=n, model=model, P=P, p=p, delta=dl, random_U=bool(seed), Fhat_direct=mp.nstr(F_dir, 20), Fhat_thmB=mp.nstr(F_cf, 20),
                      dF=mp.nstr(abs(F_dir - F_cf), 3), TrY_spread=mp.nstr(off, 3), DBS_direct=mp.nstr(D_dir, 20), DBS_thmB=mp.nstr(D_cf, 20),
                      dDBS=mp.nstr(abs(D_dir - D_cf), 3), DBS_rev_minus_fwd=mp.nstr(abs(D_rev - D_dir), 3)))
    print('B', rowsB[-1], flush=True)
# worked example sandwich (Thm W(2)) for P = Z(x)X, p = 0.01, delta = 0.02
p_, n_, w_ = mp.mpf('0.01'), 2, 2
r_w = (p_ / (4 - 3 * p_)) ** w_; xP = (1 - r_w) ** 2 * mp.sin(mp.mpf('0.01')) ** 2 / (4 * r_w)
g = lambda x: 1 - 1 / mp.sqrt(1 + x)
OUT['theorem_B'] = dict(rows=rowsB, worked_example=dict(x_P=mp.nstr(xP, 6), Fhat_sandwich=[mp.nstr(1 - g(xP), 6), mp.nstr(1 - (1 - 3 * p_ / 4) ** n_ * g(xP), 6)],
                                                       theorems_md='x_P = 3.940, F_hat in [0.44992, 0.45814], direct 0.458087'))
print('B done', time.time() - t0, flush=True)

# ---------------- (2) Theorem W window bounds vs T3 ----------------
T3 = json.load(open(R_T3 + 't3.json'))['window']
eta = mp.mpf('0.1'); x_eta = (1 - eta) ** -2 - 1
W = {}
for key, n, w, model in [('n1_local_Z', 1, 1, 'local'), ('n2_local_ZZ', 2, 2, 'local'), ('n2_local_ZI', 2, 1, 'local'),
                         ('n3_local_ZZZ', 3, 3, 'local'), ('n2_global_ZZ', 2, 2, 'global'), ('n3_global_ZZZ', 3, 3, 'global')]:
    rows = []
    for r in T3[key]['rows']:
        p = mp.mpf(r['p']); ds = mp.mpf(r['delta_star'])
        if model == 'local':
            rw = (p / (4 - 3 * p)) ** w
            xp = (1 / (1 - eta / (1 - 3 * p / 4) ** n) ** 2 - 1)
            lo = 2 * mp.asin(2 * mp.sqrt(rw * x_eta) / (1 - rw)); hi = 2 * mp.asin(min(1, 2 * mp.sqrt(rw * xp) / (1 - rw)))
            asym = 4 * mp.sqrt(x_eta) * (p / 4) ** (mp.mpf(w) / 2)
            ok = lo * (1 - mp.mpf(10) ** -12) <= ds <= hi * (1 + mp.mpf(10) ** -12)
            rows.append(dict(p=float(p), delta_star_T3=float(ds), lower=float(lo), upper=float(hi), inside=bool(ok), ratio_to_asymptotic=float(ds / asym)))
        else:
            d = 2 ** n; q1 = 1 - p + p / d ** 2; qP = p / d ** 2
            x = 1 / (1 - eta / (q1 + qP)) ** 2 - 1
            exact = 2 * mp.asin(2 * mp.sqrt(x * q1 * qP) / (q1 - qP))
            asym = 4 * mp.sqrt(x_eta * p) / d
            rows.append(dict(p=float(p), delta_star_T3=float(ds), exact_thmB=float(exact), rel_diff=float(abs(ds / exact - 1)), ratio_to_asymptotic=float(ds / asym)))
    W[key] = rows
    print('W', key, rows[-1], flush=True)
OUT['theorem_W_window'] = dict(x_eta=mp.nstr(x_eta, 10), four_sqrt_x_eta=mp.nstr(4 * mp.sqrt(x_eta), 10), rows=W)

# ---------------- (3) modulus d^2/(4 p^n) ----------------
mod = []
for n, p in [(1, '0.001'), (2, '0.001'), (2, '0.01'), (3, '0.01')]:
    t1 = time.time(); d = 2 ** n; mp.mp.dps = 60   # 40 digits lose ~1e-14 in the rank-deficient mean
    q = q_local(p, n); J = choi_pauli(q, mp.eye(d))
    w1 = vec(pstr((3,) * n)); w2 = vec(pstr((1,) * n))
    JW = herm(w1 * w1.H); JDel = herm((w1 * w1.H - w2 * w2.H) / 2)
    h = mp.mpf(10) ** -12
    lnF = lambda e: mp.log(eigvals(herm(ptrace_Y(gmean(J + e * JDel, JW), d, d)))[0])
    slope = (lnF(h) - lnF(-h)) / (2 * h)
    lnF1 = lambda e: -mp.log(mp.re((w1.H * mp.inverse(J + e * JDel) * w1)[0])) / 2     # rank-one formula (A.2 step 4)
    slope1 = (lnF1(h) - lnF1(-h)) / (2 * h)
    pred = mp.mpf(d) ** 2 / (4 * mp.mpf(p) ** n)
    # ||Delta||_diamond = 1: >= trace norm on the max-ent input (orthogonal pure outputs), <= 1 by the triangle inequality
    tn = sum(abs(e) for e in eigvals(herm(JDel / d)))
    mod.append(dict(n=n, p=p, slope_fd=mp.nstr(slope, 12), predicted_d2_over_4pn=mp.nstr(pred, 12), ratio=mp.nstr(slope / pred, 15), ratio_rank_one_formula=mp.nstr(slope1 / pred, 15),
                    maxent_trace_norm_of_Delta=mp.nstr(tn, 10), secs=round(time.time() - t1, 1)))
    print('mod', mod[-1], flush=True)
OUT['theorem_W_modulus'] = mod
mp.mp.dps = 40

# ---------------- (4) Corollary H-q vs T4 ----------------
p, dl = sp.symbols('p s', positive=True)   # dl stands for s = sin(delta/2)
x = 4 * (1 - p) ** 2 * dl ** 2 / (p * (4 - 3 * p))
forms = {'D_BS': (2 - p) * sp.sqrt(x / (1 + x)) * sp.asinh(sp.sqrt(x)),
         '-2lnFhat': -2 * sp.log(p / 2 + (1 - p / 2) / sp.sqrt(1 + x)),
         'D_Umegaki_maxent': (1 - p) * dl ** 2 * sp.log((4 - 3 * p) / p),
         '1-F_Uhlmann_maxent': 1 - (p / 2 + sp.sqrt((1 - p / 2) ** 2 - (1 - p) ** 2 * dl ** 2))}
T4 = json.load(open(R_SUM + 'summary.json'))['T4']
hq = {}
for k, e in forms.items():
    c2 = sp.simplify(sp.series(e, dl, 0, 3).removeO().coeff(dl, 2) / 4)
    t4 = sp.sympify(T4[k].replace('Abs(p - 2)', '(2 - p)'), locals={'p': p})
    hq[k] = dict(thm_coeff=str(sp.factor(c2)), t4_coeff=T4[k], difference=str(sp.simplify(c2 - t4)))
    print('Hq', k, hq[k], flush=True)
OUT['corollary_Hq_vs_T4'] = hq

# ---------------- (5) Stein constants ----------------
u = mp.mpf
fA = lambda u: u * mp.exp(-u) * mp.log((1 + 3 * mp.exp(-u)) / (1 - mp.exp(-u))) / 4
fB = lambda u: u * mp.exp(-u) * mp.atanh(mp.exp(-u / 2)) / 2
uA = mp.findroot(lambda t: mp.diff(fA, t), 0.47); uB = mp.findroot(lambda t: mp.diff(fB, t), 0.5)
fin = {}
for pk in ('0.01', '0.001', '0.0001'):
    pp = mp.mpf(pk); kmax = int(5 / pp)
    best = max((pp * k * (1 - pp) ** k * mp.log(1 + 4 * (1 - pp) ** k / (1 - (1 - pp) ** k)) / 4, k) for k in range(1, kmax))
    fin[pk] = dict(value=mp.nstr(best[0], 8), k=best[1])
OUT['stein_constants'] = dict(seq_maxent_limit=mp.nstr(fA(uA), 12), u_star=mp.nstr(uA, 8), heralded_limit=mp.nstr(fB(uB), 12), u_star_herald=mp.nstr(uB, 8),
                              seq_maxent_finite_p=fin, theorems_md='f*/4 = 0.14958713, u* = 0.466968; finite p 0.148832 / 0.149512 / 0.149580')
print('S', OUT['stein_constants'], flush=True)
json.dump(OUT, open(R + 'crosscheck_theorems.json', 'w'), indent=1, default=str)
print('done', time.time() - t0)
