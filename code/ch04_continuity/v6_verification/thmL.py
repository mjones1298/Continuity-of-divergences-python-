# Theorem L: (1 - eta/lam) F1 <= F2 <= (1 + eta/lam) F1, lam = Choi floor of (Phi1,Psi1) only,
# eta = max ||J(Phi2)-J(Phi1)||, ||J(Psi2)-J(Psi1)||  <=  (dX/2) * diamond distance for channels.
import random, numpy as np, mlib as m, mpmath as mp
from npl import diamond_ta
mp.mp.dps = 40
rng = random.Random(4)
def tonp(A): return np.array(A.tolist(), dtype=complex)
def depol_mix(J, p, dX, dY):
    return (1-p)*J + p*m.eye(dX*dY)/dY
worst_up = worst_lo = worst_eta = 0
for trial in range(60):
    dX, dY = rng.choice([(2,2),(2,3),(3,2)])
    p = mp.mpf(10)**(-rng.uniform(0.3, 2.5))
    Jp = depol_mix(m.choi_kraus(m.rand_channel_kraus(dX, dY, rng.randint(1,2), rng), dX, dY), p, dX, dY)
    Jq = depol_mix(m.choi_kraus(m.rand_channel_kraus(dX, dY, rng.randint(1,2), rng), dX, dY), p, dX, dY)
    lam = min(m.lmin(Jp), m.lmin(Jq))
    eps = lam*mp.mpf(rng.uniform(0.01, 0.9))
    # channel perturbations: mix with random channels and rotate the output
    def perturb(J):
        Jr = m.choi_kraus(m.rand_channel_kraus(dX, dY, 2, rng), dX, dY)
        U = m.kron(m.expm_herm(m.rand_herm(dY, rng), eps/3), m.eye(dX))
        return U*((1-eps/3)*J + (eps/3)*Jr)*m.dag(U)
    Jp2, Jq2 = perturb(Jp), perturb(Jq)
    eta = max(m.opnorm(Jp2-Jp), m.opnorm(Jq2-Jq))
    d = max(diamond_ta(tonp(Jp2-Jp), dY, dX), diamond_ta(tonp(Jq2-Jq), dY, dX))
    worst_eta = max(worst_eta, eta/(mp.mpf(dX)/2*d))
    F1 = m.Fhat(Jp, Jq, dY, dX); F2 = m.Fhat(Jp2, Jq2, dY, dX)
    if eta < lam:
        worst_up = max(worst_up, (F2-F1)/(eta/lam*F1))
        worst_lo = max(worst_lo, (F1-F2)/(eta/lam*F1))
print('max eta / ((dX/2) diamond)   (must be <= 1):', mp.nstr(worst_eta, 6))
print('max (F2-F1)/((eta/lam) F1)   (must be <= 1):', mp.nstr(worst_up, 6))
print('max (F1-F2)/((eta/lam) F1)   (must be <= 1):', mp.nstr(worst_lo, 6))

# Qubit gate: closed form and sharpness of the 1/lambda scaling (incoherent X-weight transfer)
def Rx(d): return m.M([[mp.cos(d/2), -1j*mp.sin(d/2)], [-1j*mp.sin(d/2), mp.cos(d/2)]])
X = m.M([[0,1],[1,0]])
def Jgate(p, R): v = m.vec(R); return (1-p)*v*m.dag(v) + (p/2)*m.eye(4)
print('\nqubit gate pair (D_p o R_X(delta), D_p):')
for p in [mp.mpf('1e-2'), mp.mpf('1e-3'), mp.mpf('1e-4')]:
    # choose delta with x = 2
    x = mp.mpf(2); s2 = x*p*(4-3*p)/(4*(1-p)**2); delta = 2*mp.asin(mp.sqrt(s2))
    R = Rx(delta); J1 = Jgate(p, R); J2 = Jgate(p, m.eye(2))
    F = m.Fhat(J1, J2, 2, 2); Fcf = p/2 + (1-p/2)/mp.sqrt(1+x)
    a = m.vec(R)/mp.sqrt(2); xh = m.vec(X*R)/mp.sqrt(2)
    Dir = a*m.dag(a) - xh*m.dag(xh)
    e = p*mp.mpf('1e-6')
    slope = (m.Fhat(J1, J2, 2, 2) - m.Fhat(J1 + e*Dir, J2, 2, 2))/e
    dn = diamond_ta(tonp(Dir), 2, 2)
    print(f'  p={mp.nstr(p,2)}: Fhat={mp.nstr(F,10)} closed form={mp.nstr(Fcf,10)}; -dF/de * p = {mp.nstr(slope*p,6)} '
          f'(predicted 1/(3 sqrt3)={mp.nstr(1/(3*mp.sqrt(3)),6)}); diamond(Dir)={dn:.6f}; ThmL slope bound * p = {mp.nstr(2*F,6)}')
