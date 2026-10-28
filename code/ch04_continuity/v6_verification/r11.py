# R11: rate at mutually singular pairs.  Independent mpmath test of the core inequality
#   F(Phi',Psi') <= tr(J'#J'')/dX <= dY (6 sqrt2/sin th + sqrt2 + 1) sqrt(a eta),
# with eta = max ||J'-J||_inf, ||J''-K||_inf  (<= (dX/2) d for channel perturbations, <= dX d for CP).
import random, mlib as m, mpmath as mp
mp.mp.dps = 50
rng = random.Random(21)
def orth(Mx):
    return Mx*m.fun(m.dag(Mx)*Mx, lambda t: 1/mp.sqrt(t))
def theta_min(Q1, Q2):
    C = m.dag(Q1)*Q2
    s = mp.sqrt(max(m.eigh(m.dag(C)*C)[0]))
    return mp.acos(min(s, 1))
worst = 0; worst_full = 0
for trial in range(300):
    dX, dY = rng.choice([(1,2),(1,3),(2,2),(2,3),(3,2)])
    n = dX*dY
    r1 = rng.randint(1, n-1); r2 = rng.randint(1, n-r1)
    Q1 = orth(m.rand_c(n, r1, rng))
    # second range: random, optionally tilted close to the first to get small angles
    B = m.rand_c(n, r2, rng)
    if rng.random() < 0.5:
        tilt = mp.mpf(10)**(-rng.randint(1, 3))
        B = Q1*m.rand_c(r1, r2, rng) + tilt*B
    Q2 = orth(B)
    th = theta_min(Q1, Q2)
    G1 = m.rand_c(r1, r1, rng); J = Q1*(G1*m.dag(G1) + m.eye(r1)/10)*m.dag(Q1)
    G2 = m.rand_c(r2, r2, rng); K = Q2*(G2*m.dag(G2) + m.eye(r2)/10)*m.dag(Q2)
    J = m.herm(J); K = m.herm(K)
    a = max(m.lmax(J), m.lmax(K))
    eps = a*mp.mpf(10)**(-rng.randint(1, 8))
    E1 = m.rand_c(n, n, rng); E1 = E1*m.dag(E1); E1 = eps*E1/m.lmax(E1)
    E2 = m.rand_c(n, n, rng); E2 = E2*m.dag(E2); E2 = eps*E2/m.lmax(E2)
    if rng.random() < 0.5:   # indefinite perturbation keeping positivity: J' = (1-x)J + E
        J1 = (1-eps/a)*J + E1; K1 = (1-eps/a)*K + E2
    else:
        J1 = J + E1; K1 = K + E2
    eta = max(m.opnorm(J1-J), m.opnorm(K1-K))
    if eta > a: continue
    G = m.gmean(J1, K1)
    F = m.lmin(m.ptrace_Y(G, dY, dX))
    trG = sum(G[i, i] for i in range(n)).real
    core = dY*(6*mp.sqrt(2)/mp.sin(th) + mp.sqrt(2) + 1)*mp.sqrt(a*eta)
    if core == 0: print("core0", a, eta, th, r1, r2, dX, dY); continue
    worst = max(worst, (trG/dX)/core)
    assert F <= trG/dX + mp.mpf(10)**-30
print('max [tr(J#K)/dX] / core bound over random mutually singular instances:', mp.nstr(worst, 6))

# 2x2 block identity: tr((a uu*+eta)#(a vv*+eta)) = sqrt(eta(a+eta)) (2a+4eta)/sqrt(a^2 sin^2 th + 4 a eta + 4 eta^2)
for a, eta, th in [(1, mp.mpf('1e-6'), mp.mpf('0.3')), (2, mp.mpf('0.01'), mp.mpf('1.2'))]:
    u = M = m.M([[1], [0]]); v = m.M([[mp.cos(th)], [mp.sin(th)]])
    A = a*u*m.dag(u) + eta*m.eye(2); B = a*v*m.dag(v) + eta*m.eye(2)
    G = m.gmean_pd(A, B); tr = G[0, 0]+G[1, 1]
    fm = mp.sqrt(eta*(a+eta))*(2*a+4*eta)/mp.sqrt(a**2*mp.sin(th)**2 + 4*a*eta + 4*eta**2)
    print('2x2 block formula check:', mp.nstr(tr - fm, 5))

# Sharpness: replacement channels rho=aa*, sigma=bb* at angle th; rho'=(1-e)aa*+e ee*, diamond = 2e
print('sharpness (F * sin th / sqrt(d), d = diamond distance = 2e; claimed -> 1/sqrt2 = 0.70711):')
for th in [mp.mpf('0.5'), mp.mpf('0.05'), mp.mpf('0.005')]:
    for e in [mp.mpf('1e-8'), mp.mpf('1e-12')]:
        av = m.M([[1], [0]]); ev = m.M([[0], [1]]); bv = m.M([[mp.cos(th)], [mp.sin(th)]])
        rho1 = (1-e)*av*m.dag(av) + e*ev*m.dag(ev); sig = bv*m.dag(bv)
        dX = 2
        J1 = m.kron(rho1, m.eye(dX)); K = m.kron(sig, m.eye(dX))
        F = m.Fhat(J1, K, 2, dX)
        print(f'   th={mp.nstr(th,3)}, e={mp.nstr(e,2)}: F={mp.nstr(F,8)}, F sin(th)/sqrt(2e)={mp.nstr(F*mp.sin(th)/mp.sqrt(2*e),8)}')
