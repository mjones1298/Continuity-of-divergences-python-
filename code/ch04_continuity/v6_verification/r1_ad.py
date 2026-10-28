import random, mlib as m, mpmath as mp
from r1 import short, RH_basis, orth_basis
mp.mp.dps = 60
rng = random.Random(9)
# Lemma SG with nontrivial ker B_t: rank-3 channel on qubits
dX, dY = 2, 2
Ks = m.rand_channel_kraus(dX, dY, 3, rng); J = m.choi_kraus(Ks, dX, dY)
H = m.rand_herm(dY, rng); t = mp.mpf('0.3')
U = m.kron(m.expm_herm(H, t), m.eye(dX)); Jt = U*J*m.dag(U)
W = m.frame(2*J); Wp = m.pinv_frame(W); n = J.rows
Rb = orth_basis(W); Ups = orth_basis(m.eye(n)-Rb*m.dag(Rb)); S = 1/mp.sqrt(2)
A = Wp*U*W*S; B = m.dag(Ups)*U*W*S
E, Q = m.eigh(m.dag(B)*B); cols = [i for i, e in enumerate(E) if e < mp.mpf(10)**(-25)]
Pi = m.zeros(W.cols)
for i in cols: Pi += Q[:, i]*m.dag(Q[:, i])
G2 = W*m.gmean(m.eye(W.cols)/2, A*Pi*m.dag(A))*m.dag(W)
print('Lemma SG (rank 3, dim ker B_t =', len(cols), '): err', mp.nstr(m.opnorm(m.gmean(J, Jt)-G2), 5))
# regularisation cross-check of the ARN mean for singular input
eps = mp.mpf(10)**-30
print('ARN mean vs regularised PD mean:', mp.nstr(m.opnorm(m.gmean(J, Jt) - m.gmean_pd(J+eps*m.eye(n), Jt+eps*m.eye(n))), 5))

def AD(g):
    g = mp.mpf(g)
    return [M([[1,0],[0,mp.sqrt(1-g)]]), M([[0,mp.sqrt(g)],[0,0]])]
M = m.M
def JAD(g): return m.choi_kraus(AD(g), 2, 2)
print('\nAmplitude damping, Fhat(AD_g, AD_g\'):')
for g, gp in [('0.3','0.3'),('0.3','0.30001'),('0.3','0.3000000001'),('0','1e-6'),('0.5','1'),('0.999','1')]:
    J1, J2 = JAD(g), JAD(gp)
    F = m.Fhat(J1, J2, 2, 2)
    D = J1 - J2
    dJ1 = m.tracenorm_h(D)
    g_, gp_ = mp.mpf(g), mp.mpf(gp)
    formula = abs(g_-gp_) + mp.sqrt((g_-gp_)**2 + 4*(mp.sqrt(1-g_)-mp.sqrt(1-gp_))**2)
    print(f'  g={g}, g\'={gp}: Fhat={mp.nstr(F,6)}, ||J-J\'||_1={mp.nstr(dJ1,8)}, formula={mp.nstr(formula,8)}')
# range intersection of J(AD_g) and J(AD_g') : dimension
J1, J2 = JAD('0.3'), JAD('0.4')
W1, W2 = m.frame(J1), m.frame(J2)
Mx = mp.zeros(4, 4)
for r in range(4):
    for c in range(2): Mx[r, c] = W1[r, c]; Mx[r, c+2] = -W2[r, c]
E, Q = m.eigh(m.dag(Mx)*Mx)
print('dim(range J(AD_.3) cap range J(AD_.4)) =', sum(1 for e in E if e < 1e-30))
Pauli = {'X': M([[0,1],[1,0]]), 'Y': M([[0,-1j],[1j,0]]), 'Z': M([[1,0],[0,-1]])}
for g in ['0.3', '0.9']:
    J = JAD(g)
    for nm, Hm in Pauli.items():
        RH = RH_basis(J, Hm, 2)
        FN = m.lmin(m.ptrace_Y(short(J, RH), 2, 2))
        U = m.kron(m.expm_herm(Hm, mp.mpf('1e-6')), m.eye(2))
        F = m.Fhat(J, U*J*m.dag(U), 2, 2)
        print(f'  AD_{g}, H={nm}: dim R_H={RH.cols}, F_N={mp.nstr(FN,8)}, Fhat(AD, Ad_U AD) at t=1e-6: {mp.nstr(F,8)}')
