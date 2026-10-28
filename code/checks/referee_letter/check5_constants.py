"""Referee check 5: small-delta Stein constants of the explicit strategies (SM Sec. S9),
certification constants c0 = 0.0246 (Prop. N34a) and 0.0589 (Prop. N12b), and the
-2 ln mu_* entry at g'=0.49 to more digits.
"""
import mpmath as mp

mp.mp.dps = 40


def golden_max(f, a, b, it=200):
    gr = (mp.sqrt(5) - 1) / 2
    for _ in range(it):
        c, d = b - gr * (b - a), a + gr * (b - a)
        if f(c) > f(d):
            b = d
        else:
            a = c
    return (a + b) / 2


# Strategy A (sequential, max-ent): p*zeta/delta^2 >= u e^{-u} ln((1+3e^{-u})/(1-e^{-u}))/4, u = kp
fA = lambda u: u * mp.e**(-u) * mp.log((1 + 3 * mp.e**(-u)) / (1 - mp.e**(-u))) / 4
uA = golden_max(fA, mp.mpf('0.05'), mp.mpf(3))
print(f"Strategy A limit: {mp.nstr(fA(uA),10)} at u*={mp.nstr(uA,8)}  (SM: 0.149587)")
# Strategy B (heralded code): D_k/k ~ (1-p/2)^k r artanh(r) k delta^2/2, r=(1-2a)^k, a ~ p/4 -> u e^{-u} artanh(e^{-u/2})/2
fB = lambda u: u * mp.e**(-u) * mp.atanh(mp.e**(-u / 2)) / 2
uB = golden_max(fB, mp.mpf('0.05'), mp.mpf(3))
print(f"Strategy B limit: {mp.nstr(fB(uB),10)} at u*={mp.nstr(uB,8)}  (SM: 0.158119)")
# finite-p strategy A sup over integer k (p = 1e-3)
p = mp.mpf('1e-3')
best = max((p * k * (1 - (1 - (1 - p)**k)) * mp.log((4 - 3 * (1 - (1 - p)**k)) / (1 - (1 - p)**k)) / 4, k) for k in range(1, 3000))
print(f"Strategy A finite p=1e-3: {mp.nstr(best[0],8)} at k={best[1]}  (theorems.md: 0.149512)")
# 'rates' quoted in SM: p x SLD rate 0.2334 and p x BKM rate 0.2992 for strategy A (Fisher per use, not /2)
fS = lambda u: 2 * u * mp.e**(-2 * u) / (1 + mp.e**(-u))
uS = golden_max(fS, mp.mpf('0.05'), mp.mpf(3))
print(f"Strategy A: p*SLD-rate={mp.nstr(fS(uS),6)} at u={mp.nstr(uS,4)}; p*BKM-rate={mp.nstr(2*fA(uA),6)}  (SM: 0.2334, 0.2992)")

# c0: min over eta in [0.005,0.1], p in [1e-6,0.04] of N*eta^2*lt with N = (ln2/3)/KL(q0||q1), n = 1
def Nbound(eta, p):
    lt = p / 4
    k2 = ((1 + eta) / (1 - eta))**2
    qP0, qI0 = lt, 1 - 3 * p / 4
    qP1 = qP0 * k2
    qI1 = qI0 - (qP1 - qP0)
    KL = qP0 * mp.log(qP0 / qP1) + qI0 * mp.log(qI0 / qI1)
    return (mp.log(2) / 3) / KL * eta**2 * lt


vals = [(Nbound(mp.mpf(e), mp.mpf(10)**lp), e, lp) for e in ['0.005', '0.01', '0.02', '0.05', '0.08', '0.1']
        for lp in [-6, -4, -3, -2, mp.log10(0.04)]]
m = min(vals)
print(f"c0: min N eta^2 lt = {mp.nstr(m[0],5)} at eta={m[1]}, log10 p={mp.nstr(m[2],3)}  (SM: 0.0246/0.02465)")
print(f"small-eta, small-p limit: {mp.nstr(Nbound(mp.mpf('1e-4'),mp.mpf('1e-8')),6)} (analytic (ln2/3)/8 = {mp.nstr(mp.log(2)/24,6)})")
print(f"0.0589 = (1/2) ln(9/8) = {mp.nstr(mp.log(mp.mpf(9)/8)/2,6)}")
g, gp = mp.mpf('0.5'), mp.mpf('0.49')
mu = mp.sqrt(g * gp) / (1 - mp.sqrt((1 - g) * (1 - gp)))
print(f"-2 ln mu_*(0.5,0.49) = {mp.nstr(-2*mp.log(mu),8)}  (SM table: 2.021e-4)")
