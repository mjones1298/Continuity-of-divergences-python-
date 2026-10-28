"""M1, part 1: the first-order (Choi-floor) sense of fragility.

Checks, in 40-digit arithmetic (mpmath), of Propositions M1.1-M1.5 of RESULTS.md:
  A. worst case of Theorem 1: Phi = Ad_W with vec(W) an eigenvector of J(Psi):
       zeta = xi = D_BS = -2 ln Fhat = Chat = D_max = Umegaki(max-ent) = -ln(Bell-test type-II) = ln(1/mu~)
     (direct matrix evaluations of every quantity, no closed forms used).
  B. directional derivatives at the worst case in random directions and in the Theorem-1 direction:
     upper bound D_max(Ad_W||Psi_e) and lower bounds Umegaki / -ln<w|sigma_e|w> have equal derivatives.
  C. Thompson-Lipschitz property |Delta q| <= d_T(Psi,Psi') for q in {D_BS, -2 ln Fhat, Chat} at random pairs.
  D. Pauli-diagonal pairs: D_BS = KL(a||b), Qhat_s = sum a^s b^(1-s), Fhat = sum sqrt(ab) (direct matrices).
  E. first-order spurious gap: preparation channels (d_X = 1) phi pure vs sigma = diag(1-lam, lam).
  F. value gap at the absolute-slope maximiser of Prop. L2: Phi = Ad_{Rz(theta)}, sin^2(theta/2) = p/2.
Output: m1_first_order.out and results/first_order.json
"""
import sys, json, time
import numpy as np
import mpmath as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # m1lib.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # sharp_alpha.py, common.py
from m1lib import *

mp.mp.dps = 40
OUT = open(Path(__file__).resolve().parent / 'm1_first_order.out', 'w')
RES = {}


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + '\n'); OUT.flush()


rng = np.random.default_rng(20261027)

# ------------------------------------------------------------------ A
log('=== A. Worst case of Theorem 1: Phi = Ad_W, vec W eigenvector of J(Psi) ===')
CNOT = mp.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
cases = []
for p in ('1e-2', '1e-3', '1e-4'):
    cases.append(('depol n=1 p=%s, W=Z U (floor)' % p, depol_rates(p, 1), 'Z', rand_unitary(2, rng)))
qb = {'I': 1 - mp.mpf('1e-3') - 2 * mp.mpf('1e-5'), 'X': mp.mpf('1e-5'), 'Y': mp.mpf('1e-5'), 'Z': mp.mpf('1e-3')}
cases.append(('biased n=1 (qZ=1e-3,qX=qY=1e-5), W=X U (floor)', qb, 'X', rand_unitary(2, rng)))
cases.append(('biased n=1 (qZ=1e-3,qX=qY=1e-5), W=Z U (non-floor eigvec)', qb, 'Z', rand_unitary(2, rng)))
cases.append(('depol n=2 p=1e-3, U=CNOT, W=(Z(x)X) U (floor)', depol_rates('1e-3', 2), 'ZX', CNOT))

RES['A'] = []
for name, q, Wl, U in cases:
    n = len(Wl); d = 2 ** n
    JPsi = choi_pauli(q, U)
    W = pauli_string(Wl) * U
    w = vec(W)
    JPhi = outer(w, w)
    qW = q[Wl]
    target = mp.log(1 / qW)
    sig = JPsi / d
    wh = w / mp.sqrt(d)
    vals = {}
    vals['-2lnFhat'] = -2 * mp.log(Fhat(JPhi, JPsi, d, d))
    vals['D_BS'] = DBS(JPhi, JPsi, d, d)
    vals['D_max'] = Dmax(JPhi, JPsi)
    vals['Umegaki(maxent)'] = relent(outer(wh, wh), sig)
    beta = mp.re(inner(wh, sig * wh))
    vals['-ln beta(Bell test)'] = -mp.log(beta)
    vals['-2ln F_Uhlmann(maxent)'] = -mp.log(beta)
    for s in ('0.25', '0.5', '0.75'):
        s = mp.mpf(s)
        Qs = Qhat_s(JPhi, JPsi, s, d, d)
        vals['Qhat_s/qW^(1-s) at s=%s' % mp.nstr(s, 2)] = Qs / qW ** (1 - s)
    if n == 1:
        vals['Chat'] = Chat(JPhi, JPsi, d, d)
    dev = max(abs(v - target) for k, v in vals.items() if not k.startswith('Qhat'))
    devQ = max(abs(v - 1) for k, v in vals.items() if k.startswith('Qhat'))
    log('-- %s : ln(1/q_W) = %s' % (name, mp.nstr(target, 25)))
    for k, v in vals.items():
        log('   %-34s %s' % (k, mp.nstr(v, 25)))
    log('   max |q - ln(1/q_W)| = %s ; max |Qhat_s/q_W^(1-s) - 1| = %s' % (mp.nstr(dev, 3), mp.nstr(devQ, 3)))
    RES['A'].append(dict(case=name, target=float(target), maxdev=float(dev), maxdevQ=float(devQ),
                         Chat_note='Chat evaluated at s ~ 1e-12 (inf at s->0)' if n == 1 else 'not evaluated'))

# ------------------------------------------------------------------ B
log('\n=== B. Directional derivatives at the worst case (all quantities have derivative -<w^,X w^>/mu) ===')
RES['B'] = []
h = mp.mpf('1e-15')
for name, q, Wl, U in [cases[1], cases[3], cases[5]]:
    n = len(Wl); d = 2 ** n
    JPsi = choi_pauli(q, U)
    W = pauli_string(Wl) * U
    w = vec(W); wh = w / mp.sqrt(d)
    JPhi = outer(w, w)
    mu = d * q[Wl]
    # Theorem-1 direction 1/2(Ad_W - Ad_W'), W' = (other Pauli) U with Tr W^dag W' = 0
    Wp = pauli_string('I' * n) * U
    wp = vec(Wp)
    dirs = [('Thm1 worst: 1/2(Ad_W-Ad_W\'), ||.||_dia=1', herm((outer(w, w) - outer(wp, wp)) / 2))]
    for k in range(3):
        dirs.append(('random trace-annihilating #%d' % k, traceless_Y_direction(d, d, rng)))
    for dname, Xd in dirs:
        def fq(e):
            Je = JPsi + e * Xd
            sig = Je / d
            return (Dmax(JPhi, Je), DBS(JPhi, Je, d, d), relent(outer(wh, wh), sig), -mp.log(mp.re(inner(wh, sig * wh))))
        fp, fm = fq(h), fq(-h)
        ders = [(a - b) / (2 * h) for a, b in zip(fp, fm)]
        pred = -mp.re(inner(wh, Xd * wh)) / mu
        dev = max(abs(x - pred) for x in ders) / abs(pred)
        log('-- %s | %s' % (name, dname))
        log('   d/de D_max = %s, D_BS = %s, Umegaki(lower bd zeta) = %s, -ln<w|s|w>(lower bd xi) = %s' %
            tuple(mp.nstr(x, 18) for x in ders))
        log('   predicted -<w^,J(Delta) w^>/mu = %s ; max rel dev = %s' % (mp.nstr(pred, 18), mp.nstr(dev, 3)))
        rec = dict(case=name, direction=dname, pred=float(pred), maxreldev=float(dev))
        if dname.startswith('Thm1'):
            rec['slope_times_2qW'] = float(abs(pred) * 2 * q[Wl])
            log('   Thm-1 direction: |slope| * 2 q_W = %s (expected 1: modulus 1/(2 lambda~))' % mp.nstr(abs(pred) * 2 * q[Wl], 15))
        RES['B'].append(rec)

# exact non-perturbative statement within Pauli perturbations
log('\n   Pauli-preserving perturbation q_W -> q_W + e/2, q_W\' -> q_W\' - e/2 (exact for all e):')
p = mp.mpf('1e-3'); q = depol_rates(p, 1); U = rand_unitary(2, rng)
for e in (mp.mpf('1e-4'), mp.mpf('-1e-4'), mp.mpf('2e-4')):
    q2 = dict(q); q2['Z'] = q['Z'] + e / 2; q2['I'] = q['I'] - e / 2
    JPsi = choi_pauli(q2, U); w = vec(pauli_string('Z') * U); JPhi = outer(w, w)
    log('   e=%s: D_BS=%s  ln(1/(qZ+e/2))=%s  ln(1/qZ)+ln-change slope=%s' % (
        mp.nstr(e, 3), mp.nstr(DBS(JPhi, JPsi, 2, 2), 20), mp.nstr(mp.log(1 / q2['Z']), 20),
        mp.nstr((mp.log(1 / q2['Z']) - mp.log(1 / q['Z'])) / e, 12)))

# ------------------------------------------------------------------ C
log('\n=== C. Thompson-Lipschitz: |q(Phi,Psi\') - q(Phi,Psi)| <= d_T(Psi,Psi\') ===')
RES['C'] = []
p = mp.mpf('1e-2')
for trial in range(8):
    U = rand_unitary(2, rng)
    JPsi = choi_pauli(depol_rates(p, 1), U)
    # random channel Phi: Stinespring isometry from a random unitary on C^2 (x) C^4
    V = rand_unitary(8, rng)
    K = [mp.matrix([[V[a * 4 + k, b * 4 + 0] for b in range(2)] for a in range(2)]) for k in range(4)]
    JPhi = mp.zeros(4, 4)
    for Kk in K:
        v = vec(Kk); JPhi += outer(v, v)
    if trial % 2 == 1:  # rank-one Phi (unitary) for half of the trials
        Wr = rand_unitary(2, rng); v = vec(Wr); JPhi = outer(v, v)
    Xd = traceless_Y_direction(2, 2, rng)
    lam = eigvals(JPsi)[0]
    e = lam / (4 * max(abs(x) for x in eigvals(Xd)))
    JPsi2 = JPsi + e * Xd
    dT = thompson(JPsi, JPsi2)
    q1 = (DBS(JPhi, JPsi, 2, 2), -2 * mp.log(Fhat(JPhi, JPsi, 2, 2)), Chat(JPhi, JPsi, 2, 2))
    q2 = (DBS(JPhi, JPsi2, 2, 2), -2 * mp.log(Fhat(JPhi, JPsi2, 2, 2)), Chat(JPhi, JPsi2, 2, 2))
    ratios = [abs(a - b) / dT for a, b in zip(q1, q2)]
    log('   trial %d (%s Phi): d_T=%s ; |dD_BS|/d_T=%s |d(-2lnF)|/d_T=%s |dChat|/d_T=%s' % (
        trial, 'unitary' if trial % 2 else 'rank-4', mp.nstr(dT, 6), *[mp.nstr(r, 6) for r in ratios]))
    RES['C'].append([float(r) for r in ratios])
log('   max ratio = %s (must be <= 1)' % max(max(r) for r in RES['C']))

# ------------------------------------------------------------------ D
log('\n=== D. Pauli-diagonal pairs: converses are classical and exact ===')
RES['D'] = []
for n in (1, 2):
    labs = pauli_labels(n); d = 2 ** n
    a = [mp.mpf(rng.random()) for _ in labs]; a[1] = mp.mpf(0)  # singular a (one zero rate)
    b = [mp.mpf(rng.random()) + mp.mpf('0.01') for _ in labs]
    sa, sb = mp.fsum(a), mp.fsum(b)
    a = [x / sa for x in a]; b = [x / sb for x in b]
    U = rand_unitary(d, rng)
    Ja = choi_pauli(dict(zip(labs, a)), U); Jb = choi_pauli(dict(zip(labs, b)), U)
    KL = mp.fsum(x * mp.log(x / y) for x, y in zip(a, b) if x > 0)
    dbs = DBS(Ja, Jb, d, d)
    s = mp.mpf('0.37')
    Qs = Qhat_s(Ja, Jb, s, d, d); Qcl = mp.fsum(x ** s * y ** (1 - s) for x, y in zip(a, b) if x > 0)
    F = Fhat(Ja, Jb, d, d); Fcl = mp.fsum(mp.sqrt(x * y) for x, y in zip(a, b))
    log('   n=%d: D_BS-KL=%s, Qhat_s-classical=%s (s=0.37), Fhat-sum sqrt(ab)=%s' % (
        n, mp.nstr(dbs - KL, 3), mp.nstr(Qs - Qcl, 3), mp.nstr(F - Fcl, 3)))
    RES['D'].append(dict(n=n, dDBS=float(abs(dbs - KL)), dQ=float(abs(Qs - Qcl)), dF=float(abs(F - Fcl))))

# ------------------------------------------------------------------ E
log('\n=== E. First-order spurious gap: preparation channels phi pure vs sigma = diag(1-lam, lam), |<1|phi>|^2 = y lam ===')
RES['E'] = []
for lam in ('1e-2', '1e-3', '1e-4'):
    lam = mp.mpf(lam)
    for y in ('0.5', '1', '2', '8'):
        y = mp.mpf(y)
        s2 = y * lam; c2 = 1 - s2
        sv = [1 - lam, lam]
        phi = mp.matrix([mp.sqrt(c2), mp.sqrt(s2)])
        P = outer(phi, phi)
        sig = mp.diag(sv)
        Ninv = c2 / sv[0] + s2 / sv[1]
        conv = mp.log(Ninv)
        zeta = -(c2 * mp.log(sv[0]) + s2 * mp.log(sv[1]))
        xi = -mp.log(c2 * sv[0] + s2 * sv[1])
        # exact first-order moduli over traceless Delta with ||Delta||_1 = 1
        vnorm2 = c2 / sv[0] ** 2 + s2 / sv[1] ** 2
        mod_conv = vnorm2 / (2 * Ninv)
        L = mp.log(sv[0] / sv[1]) / (sv[0] - sv[1])
        G = mp.matrix([[c2 / sv[0], mp.sqrt(c2 * s2) * L], [mp.sqrt(c2 * s2) * L, s2 / sv[1]]])
        ev = eigvals(G)
        mod_zeta = (ev[1] - ev[0]) / 2
        mod_xi = 1 / (2 * (c2 * sv[0] + s2 * sv[1]))
        # numerical check of moduli: finite differences along the optimal directions and random ones
        def quantities(sg):
            Ni = mp.re(inner(phi, mp.inverse(sg) * phi))
            return (mp.log(Ni), -mp.re(inner(phi, logm(sg) * phi)), -mp.log(mp.re(inner(phi, sg * phi))))
        hh = mp.mpf('1e-18')
        best = [mp.mpf(0)] * 3
        dirs = []
        for M in (outer(mp.inverse(sig) * phi, mp.inverse(sig) * phi), G, P):
            E_, Q_ = eigh(M)
            a_ = Q_[:, 1]; b_ = Q_[:, 0]
            dirs.append(herm((outer(a_, a_) - outer(b_, b_)) / 2))
        for k in range(20):
            H = rand_herm(2, rng); H = H - tr(H) / 2 * mp.eye(2)
            nrm = mp.fsum(abs(x) for x in eigvals(H)); dirs.append(H / nrm)
        for Dd in dirs:
            qp, qm = quantities(sig + hh * Dd), quantities(sig - hh * Dd)
            for i in range(3):
                best[i] = max(best[i], abs(qp[i] - qm[i]) / (2 * hh))
        log('   lam=%s y=%s: converse(D_BS=Chat=-2lnF)=%s zeta=%s xi=%s | Mod: conv*lam=%s zeta=%s xi=%s | FD sup: %s %s %s' % (
            mp.nstr(lam, 2), mp.nstr(y, 2), mp.nstr(conv, 8), mp.nstr(zeta, 8), mp.nstr(xi, 8),
            mp.nstr(mod_conv * lam, 8), mp.nstr(mod_zeta, 8), mp.nstr(mod_xi, 8),
            mp.nstr(best[0] * lam, 8), mp.nstr(best[1], 8), mp.nstr(best[2], 8)))
        RES['E'].append(dict(lam=float(lam), y=float(y), conv=float(conv), zeta=float(zeta), xi=float(xi),
                             mod_conv=float(mod_conv), mod_zeta=float(mod_zeta), mod_xi=float(mod_xi),
                             fd=[float(b) for b in best]))

# ------------------------------------------------------------------ F
log('\n=== F. Value gap at the absolute-slope maximiser: Phi = Ad_{Rz(theta)}, Psi = D_p, sin^2(theta/2) = p/2 ===')
RES['F'] = []
from fractions import Fraction
from sharp_alpha import Dsharp_channel, certify, T_GRID
for p in ('1e-2', '1e-3', '1e-4'):
    p = mp.mpf(p)
    qI, qZ = 1 - 3 * p / 4, p / 4
    s2 = p / 2; c2 = 1 - s2
    th = 2 * mp.asin(mp.sqrt(s2))
    JPsi = choi_pauli(depol_rates(p, 1))
    w = vec(Rz(th)); JPhi = outer(w, w)
    conv = mp.log(c2 / qI + s2 / qZ)
    conv_mat = DBS(JPhi, JPsi, 2, 2)
    Fh = Fhat(JPhi, JPsi, 2, 2)
    # lower bounds on zeta: sequential k uses, max-ent input, Umegaki of pure vs depolarised
    best_z = (mp.mpf(0), 0); best_x = (mp.mpf(0), 0)
    kmax = int(4 * mp.pi / th) + 2
    for k in range(1, kmax + 1):
        qk = 1 - (1 - p) ** k
        ck2 = mp.cos(k * th / 2) ** 2; sk2 = 1 - ck2
        Dk = -(ck2 * mp.log(1 - 3 * qk / 4) + sk2 * mp.log(qk / 4))
        Ck = -mp.log(ck2 * (1 - 3 * qk / 4) + sk2 * qk / 4)
        if Dk / k > best_z[0]:
            best_z = (Dk / k, k)
        if Ck / k > best_x[0]:
            best_x = (Ck / k, k)
    # upper bound U3 (Ref. JonesShadows Cor 7.5) at the base pair (id, D_p), where zeta(id||D_p) = ln(1/qI) (Prop M1.2)
    eps = mp.sqrt(s2)
    xx = eps / (1 + eps)
    hb = -xx * mp.log(xx) - (1 - xx) * mp.log(1 - xx)
    U3 = mp.log(1 / qI) + 2 * eps * mp.log(2) + (1 + eps) * hb + eps * mp.log((4 - 3 * p) / p)
    # Fawzi-Fawzi D^#_alpha (certified in mpmath), one copy
    JN = np.array([[complex(JPhi[i, j]) for j in range(4)] for i in range(4)])
    JM = np.array([[complex(JPsi[i, j]) for j in range(4)] for i in range(4)])
    bestD = (mp.inf, None)
    for t in T_GRID:
        Dv, Qv, st, B = Dsharp_channel(JN, JM, 2, 2, t, return_B=True)
        if B is None or not np.isfinite(Dv):
            continue
        dc, c = certify(JN, JM, 2, 2, t, B, dps=40)
        if dc < bestD[0]:
            bestD = (dc, t)
    mp.mp.dps = 40
    log('   p=%s: converse ln<w|s^-1|w>=%s (matrix D_BS %s, -2lnFhat %s; Fhat=%s)' % (
        mp.nstr(p, 2), mp.nstr(conv, 12), mp.nstr(conv_mat, 12), mp.nstr(-2 * mp.log(Fh), 12), mp.nstr(Fh, 8)))
    log('       zeta >= %s (sequential, k=%d);  xi >= %s (k=%d)' % (mp.nstr(best_z[0], 8), best_z[1], mp.nstr(best_x[0], 8), best_x[1]))
    log('       zeta <= U3 = %s [Shadows Cor 7.5];  zeta <= D^#_alpha = %s (t=1/alpha=%s, certified)' % (
        mp.nstr(U3, 8), mp.nstr(bestD[0], 8), bestD[1]))
    zu = min(U3, bestD[0])
    log('       certified converse/zeta >= %s (with U3), >= %s (D^# only); converse/zeta <= %s' % (
        mp.nstr(conv / zu, 6), mp.nstr(conv / bestD[0], 6), mp.nstr(conv / best_z[0], 6)))
    RES['F'].append(dict(p=float(p), converse=float(conv), zeta_lower=float(best_z[0]), k=best_z[1], xi_lower=float(best_x[0]),
                         U3=float(U3), Dsharp=float(bestD[0]), t=str(bestD[1]),
                         ratio_cert=float(conv / zu), ratio_cert_sharp_only=float(conv / bestD[0]), ratio_upper=float(conv / best_z[0])))

json.dump(RES, open(Path(__file__).resolve().parent / 'results_first_order.json', 'w'), indent=1)
log('\ndone')
