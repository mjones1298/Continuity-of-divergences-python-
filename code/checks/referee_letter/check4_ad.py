"""Referee check 4: amplitude damping (main text Eq. 4, SM Sec. S7, Table S-AD).

(a) Fhat(A_g, A_g') = 0: regularised geometric mean (J+eps 1) # (J'+eps 1) in mpmath, eps -> 0.
(b) mu_* SDP: max |c| s.t. sum_{k,l} C_lk L_l^dag K_k = c 1, ||C||_op <= 1 (cvxpy), vs closed form.
(c) Asymptotics -2 ln mu_* ~ Delta^2/(4 g^2 (1-g)); product-input Chernoff ~ Delta^2/(8 g (1-g)).
(d) Table S-AD values at g = 1/2.
"""
import numpy as np
import cvxpy as cp
import mpmath as mp

mp.mp.dps = 60


def kraus(g):
    return [np.array([[1, 0], [0, np.sqrt(1 - g)]]), np.array([[0, np.sqrt(g)], [0, 0]])]


def choi_mp(g):
    g = mp.mpf(g)
    v0 = mp.matrix([1, 0, 0, mp.sqrt(1 - g)])   # vec(K0): e00 + c e11
    v1 = mp.matrix([0, mp.sqrt(g), 0, 0])        # vec(K1): sqrt(g) e01
    return v0 * v0.T + v1 * v1.T


def hf(A, f):
    E, Q = mp.eighe((A + A.T) / 2)
    return Q * mp.diag([f(e) for e in E]) * Q.T


def gmean(A, B):
    As = hf(A, mp.sqrt)
    Ai = hf(A, lambda x: 1 / mp.sqrt(x))
    return As * hf(Ai * B * Ai, mp.sqrt) * As


def ptrY(J):
    R = mp.matrix(2, 2)
    for a in range(2):
        for b in range(2):
            R[a, b] = J[a, b] + J[2 + a, 2 + b]
    return R


print("=== (a) Fhat(A_0.5, A_0.4) with regularisation eps ===")
for eps in ['1e-6', '1e-10', '1e-14', '1e-20']:
    e = mp.mpf(eps)
    J1, J2 = choi_mp('0.5') + e * mp.eye(4), choi_mp('0.4') + e * mp.eye(4)
    T = ptrY(gmean(J1, J2))
    E, _ = mp.eighe(T)
    print(f"eps={eps}: eig Tr_Y(J#J') = {[mp.nstr(x,6) for x in E]}")
print("   (identical channels as control) eps=1e-20:",
      [mp.nstr(x, 6) for x in mp.eighe(ptrY(gmean(choi_mp('0.5') + mp.mpf('1e-20') * mp.eye(4),
                                                     choi_mp('0.5') + mp.mpf('1e-20') * mp.eye(4))))[0]])


def mu_star_sdp(g, gp):
    K, L = kraus(g), kraus(gp)
    C = cp.Variable((2, 2), complex=True)
    c = cp.Variable()
    expr = sum(C[l, k] * (L[l].conj().T @ K[k]) for k in range(2) for l in range(2))
    cons = [expr == c * np.eye(2), cp.bmat([[np.eye(2), C], [C.H, np.eye(2)]]) >> 0]
    cp.Problem(cp.Maximize(c), cons).solve(solver=cp.CLARABEL)
    return c.value


def mu_star(g, gp):
    return mp.sqrt(g * gp) / (1 - mp.sqrt((1 - g) * (1 - gp)))


print("\n=== (b) mu_* SDP vs closed form ===")
for g, gp in [(0.5, 0.4), (0.5, 0.2), (0.3, 0.1), (0.9, 0.85)]:
    print(f"g={g} g'={gp}: SDP={mu_star_sdp(g,gp):.9f} closed={float(mu_star(mp.mpf(g),mp.mpf(gp))):.9f}")


def chernoff_bern(g, gp):
    f = lambda s: mp.log(g**s * gp**(1 - s) + (1 - g)**s * (1 - gp)**(1 - s))
    s = mp.findroot(lambda s: mp.diff(f, s), 0.5)
    return -f(s)


print("\n=== (c) asymptotics (Delta = g - g' small) ===")
for g in [mp.mpf('0.5'), mp.mpf('0.2')]:
    D = mp.mpf('1e-5')
    gp = g - D
    up = -2 * mp.log(mu_star(g, gp))
    lo = chernoff_bern(g, gp)
    f = mp.sqrt(g * gp) + mp.sqrt((1 - g) * (1 - gp))
    print(f"g={g}: -2ln mu*/D^2={mp.nstr(up/D**2,8)} vs 1/(4g^2(1-g))={mp.nstr(1/(4*g**2*(1-g)),8)} | "
          f"Chernoff/D^2={mp.nstr(lo/D**2,8)} vs 1/(8g(1-g))={mp.nstr(1/(8*g*(1-g)),8)} | "
          f"-2ln f/D^2={mp.nstr(-2*mp.log(f)/D**2,8)} vs 1/(4g(1-g))={mp.nstr(1/(4*g*(1-g)),8)} | ratio up/lo={mp.nstr(up/lo,6)} (2/g={mp.nstr(2/g,6)})")

print("\n=== (d) Table S-AD, g = 1/2 ===")
g = mp.mpf('0.5')
for gp in ['0.40', '0.47', '0.49']:
    gp = mp.mpf(gp)
    f = mp.sqrt(g * gp) + mp.sqrt((1 - g) * (1 - gp))
    print(f"g'={gp}: product-input={mp.nstr(chernoff_bern(g,gp),4)}  -2ln f={mp.nstr(-2*mp.log(f),4)}  -2ln mu*={mp.nstr(-2*mp.log(mu_star(g,gp)),4)}")
