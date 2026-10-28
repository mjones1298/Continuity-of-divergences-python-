"""T4: exact second-order (delta -> 0) coefficients at the depolarized qubit gate, and the RLD/BKM/SLD identification.
Choi state rho_delta = J(N_{p,delta})/2 has spectrum {l1 = 1-3p/4 (on |Omega>), l2 = p/4 (x3)}; the family is unitary,
rho_delta = e^{-i delta G} rho_0 e^{i delta G}, G = (Z/2)(x)1, with |G_12|^2 = 1/4 between |Omega> and one l2-eigenvector, G_11 = 0.
Monotone-metric QFIs of a unitary family:  F_f = sum_{ij} |G_ij|^2 (l_i - l_j)^2 * c_f(l_i, l_j)
   SLD: c = 2/(l_i+l_j);  BKM: c = (ln l_i - ln l_j)/(l_i - l_j);  RLD (as a scalar tr(rho' rho^{-1} rho')): c = 1/l_j.
Expected: D_BS ~ (delta^2/2) F_RLD; -2 ln F_hat ~ (delta^2/4) F_RLD; D_Umegaki ~ (delta^2/2) F_BKM; 1 - F_Uhl ~ (delta^2/8) F_SLD.
"""
import sys, json
import sympy as sp
import numpy as np, mpmath as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # common.py, sharp_alpha.py
from common import *

p, d = sp.symbols('p delta', positive=True)
l1, l2 = 1 - 3 * p / 4, p / 4
G2 = sp.Rational(1, 4)
F_SLD = sp.simplify(2 * G2 * (l1 - l2) ** 2 * 2 / (l1 + l2))
F_BKM = sp.simplify(2 * G2 * (l1 - l2) * (sp.log(l1) - sp.log(l2)))
F_RLD = sp.simplify(G2 * (l1 - l2) ** 2 * (1 / l1 + 1 / l2))
s = sp.symbols('s', positive=True)   # s = sin(delta/2); s^2 = delta^2/4 + O(delta^4)
Kx = 4 * (1 - p) ** 2 / (p * (4 - 3 * p))
x = Kx * s ** 2
DBS = (2 - p) * sp.sqrt(Kx) * s / sp.sqrt(1 + x) * sp.asinh(sp.sqrt(Kx) * s)
Fhat = p / 2 + (1 - p / 2) / sp.sqrt(1 + x)
m2lnF = -2 * sp.log(Fhat)
DU = (1 - p) * s ** 2 * sp.log(1 + 4 * (1 - p) / p)
# Uhlmann root fidelity of the max-ent Choi states: common l2 part (2-dim) + 2x2 block; for 2x2 PSD A,B:
# (tr|sqrtA sqrtB|)^2 = tr(AB) + 2 sqrt(detA detB).  Block: A = diag(l1,l2), B = U A U^dag with |<v1,v2>|^2 = cos^2(delta/2).
c2 = 1 - s ** 2
trAB = l1 ** 2 * c2 + 2 * l1 * l2 * (1 - c2) + l2 ** 2 * c2
FU = 2 * l2 + sp.sqrt(trAB + 2 * l1 * l2)
Finf = 2 * (1 - p) ** 2 / (p * (3 - 2 * p))


def coef2(expr):
    ser = sp.series(expr, s, 0, 3).removeO()
    return sp.simplify(sp.factor(ser.coeff(s, 2) / 4))   # coefficient of delta^2


res = {}
cDBS = coef2(DBS); cF = coef2(m2lnF); cDU = coef2(DU); cFU = coef2(1 - FU)
checks = {
    'D_BS / (F_RLD/2)': sp.simplify(cDBS / (F_RLD / 2)),
    '(-2lnFhat) / (F_RLD/4)': sp.simplify(cF / (F_RLD / 4)),
    'D_Umegaki / (F_BKM/2)': sp.simplify(cDU / (F_BKM / 2)),
    '(1-F_Uhl) / (F_SLD/8)': sp.simplify(cFU / (F_SLD / 8)),
}
res['coefficients_of_delta2'] = {'D_BS': str(cDBS), '-2lnFhat': str(cF), 'D_Umegaki_maxent': str(cDU), '1-F_Uhlmann_maxent': str(cFU)}
res['QFIs'] = {'F_SLD': str(F_SLD), 'F_BKM': str(F_BKM), 'F_RLD': str(sp.factor(F_RLD)), 'F_inf_ZJ': str(Finf)}
res['identification_ratios'] = {k: str(v) for k, v in checks.items()}
asym = {}
for name, e in [('p*F_RLD', p * F_RLD), ('F_SLD', F_SLD), ('F_BKM', F_BKM), ('p*F_inf', p * Finf), ('F_inf/F_RLD', Finf / F_RLD)]:
    asym[name] = str(sp.series(e, p, 0, 2).removeO())
res['small_p_expansions'] = asym
res['ratio_Finf_FRLD_exact'] = str(sp.factor(sp.simplify(Finf / F_RLD)))
res['ratio_FRLD_Finf_exact'] = str(sp.factor(sp.simplify(F_RLD / Finf)))
print(json.dumps(res, indent=1))

# ---------------- independent numerical confirmation from matrices (mpmath, 60 digits) ----------------
mp.mp.dps = 60
num = []
for ps in ['1e-1', '1e-2', '1e-3', '1e-4']:
    pm = mp.mpf(ps); dl = mp.mpf('1e-12')
    J1, J0 = choi_qubit(pm, dl), choi_qubit(pm, 0)
    r1, r0 = J1 / 2, J0 / 2
    vals = dict(
        DBS=DBS_channel(J1, J0, 2, 2) / dl ** 2,
        m2lnF=-2 * mp.log(Fhat_channel(J1, J0, 2, 2)) / dl ** 2,
        DU=relent(r1, r0) / dl ** 2,
        oneminusFU=(1 - mp.re(tr(sqrtm(herm(sqrtm(r0) * r1 * sqrtm(r0)))))) / dl ** 2)
    sub = {p: pm}
    pred = dict(DBS=cDBS, m2lnF=cF, DU=cDU, oneminusFU=cFU)
    row = {'p': ps}
    for k in vals:
        pv = mp.mpf(str(sp.N(pred[k].subs(p, sp.Rational(ps.replace('e', 'E')) if False else sp.nsimplify(ps)), 40)))
        row[k] = mp.nstr(vals[k], 15); row[k + '_pred'] = mp.nstr(pv, 15); row[k + '_relerr'] = mp.nstr(abs(vals[k] / pv - 1), 3)
    num.append(row)
    print(row)
res['numerical_check_delta_1e-12'] = num

# ---------------- input optimisation (Schmidt family; phase covariance makes it sufficient) ----------------
# BKM and SLD QFI of (N_theta (x) id)(psi_q), psi_q = sqrt(q)|00> + sqrt(1-q)|11>, maximised over q.
def qfis_for_input(pm, q):
    q = mp.mpf(q)
    psi = mp.matrix([mp.sqrt(q), 0, 0, mp.sqrt(1 - q)])
    rin = psi * psi.H
    # output rho = sum_K (K(x)1) rin (K(x)1)^dag
    P = [mpmat(s) for s in PAULI]
    wts = [1 - 3 * pm / 4, pm / 4, pm / 4, pm / 4]
    rho = mp.zeros(4, 4)
    for i in range(4):
        KK = kron(P[i], mp.eye(2)); rho += wts[i] * KK * rin * KK.H
    Gm = kron(mpmat(Z) / 2, mp.eye(2))
    E, Q = eigh(rho)
    Gt = Q.H * Gm * Q
    Fs = Fb = Fr = mp.mpf(0)
    for i in range(4):
        for j in range(4):
            g2 = abs(Gt[i, j]) ** 2
            li, lj = E[i], E[j]
            if g2 < mp.mpf(10) ** (-50) or abs(li - lj) < mp.mpf(10) ** (-50):
                continue
            Fs += g2 * (li - lj) ** 2 * 2 / (li + lj)
            Fb += g2 * (li - lj) * (mp.log(li) - mp.log(lj))
            Fr += g2 * (li - lj) ** 2 / lj
    return Fs, Fb, Fr

opt = []
for ps in ['1e-1', '1e-2', '1e-3', '1e-4']:
    pm = mp.mpf(ps)
    best = {}
    for name, idx in (('SLD', 0), ('BKM', 1), ('RLD', 2)):
        f = lambda qq: -qfis_for_input(pm, qq)[idx]
        qs = [mp.mpf(k) / 40 for k in range(1, 40)]
        vals = [(-f(qq), qq) for qq in qs]
        vbest, qb = max(vals)
        # refine by golden section on [qb-1/40, qb+1/40]
        a, b = max(qb - mp.mpf(1) / 40, mp.mpf(1) / 10 ** 6), min(qb + mp.mpf(1) / 40, 1 - mp.mpf(1) / 10 ** 6)
        for _ in range(60):
            c1 = b - (b - a) / mp.phi; c2 = a + (b - a) / mp.phi
            if f(c1) < f(c2):
                b = c2
            else:
                a = c1
        qopt = (a + b) / 2
        best[name] = dict(q_opt=mp.nstr(qopt, 8), F_opt=mp.nstr(-f(qopt), 15), F_maxent=mp.nstr(qfis_for_input(pm, mp.mpf(1) / 2)[idx], 15))
    opt.append(dict(p=ps, **best))
    print(ps, best)
res['input_optimisation_schmidt_family'] = opt
json.dump(res, open(Path(__file__).resolve().parent / 'results' / 't4.json', 'w'), indent=1, default=str)
