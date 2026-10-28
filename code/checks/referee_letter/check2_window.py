"""Referee check 2: multi-qubit window law (Eqs. 8-9), constant 4 sqrt(x_eta), quoted numbers.

(a) Brute-force 16x16 Choi matrices for n=2 local depolarizing noise, generator ZZ (w=2) and
    Z(x)1 (w=1), compared with the Pauli-pair closed form of Theorem S(B).
(b) Window delta_0.1 by bisection on the closed form for n=1,2,3, compared with SM Table S-window.
(c) The main-text numbers 4.8e-4 (w=2), 1.2e-7 (w=4), 1.5e-2 (global, n=2) at p=1e-3.
"""
import itertools
import mpmath as mp

mp.mp.dps = 30
P1 = {'I': mp.matrix([[1, 0], [0, 1]]), 'X': mp.matrix([[0, 1], [1, 0]]),
      'Y': mp.matrix([[0, -1j], [1j, 0]]), 'Z': mp.matrix([[1, 0], [0, -1]])}


def kron(A, B):
    C = mp.matrix(A.rows * B.rows, A.cols * B.cols)
    for i in range(A.rows):
        for j in range(A.cols):
            for k in range(B.rows):
                for l in range(B.cols):
                    C[i * B.rows + k, j * B.cols + l] = A[i, j] * B[k, l]
    return C


def pauli(s):
    M = P1[s[0]]
    for c in s[1:]:
        M = kron(M, P1[c])
    return M


def vec(A):
    v = mp.matrix(A.rows * A.cols, 1)
    for a in range(A.rows):
        for b in range(A.cols):
            v[a * A.cols + b] = A[a, b]
    return v


def hf(A, f):
    E, Q = mp.eighe((A + A.H) / 2)
    return Q * mp.diag([f(e) for e in E]) * Q.H


def gmean(A, B):
    As = hf(A, mp.sqrt)
    Ai = hf(A, lambda x: 1 / mp.sqrt(x))
    return As * hf(Ai * B * Ai, mp.sqrt) * As


def ptrY(J, dY, dX):
    R = mp.matrix(dX, dX)
    for a in range(dX):
        for b in range(dX):
            R[a, b] = sum(J[y * dX + a, y * dX + b] for y in range(dY))
    return R


def local_rates(p, n):
    q = {}
    for s in itertools.product('IXYZ', repeat=n):
        w = sum(c != 'I' for c in s)
        q[''.join(s)] = (1 - 3 * p / 4) ** (n - w) * (p / 4) ** w
    return q


def choi_pauli(q, V):
    d = V.rows
    J = mp.matrix(d * d, d * d)
    for s, qs in q.items():
        v = vec(pauli(s) * V)
        J += qs * (v * v.H)
    return J


def expmP(Pstr, delta):
    P = pauli(Pstr)
    d = P.rows
    return mp.cos(delta / 2) * mp.eye(d) - 1j * mp.sin(delta / 2) * P


def mult(a, b):
    # product of single-qubit Pauli labels up to phase
    t = {('I', x): x for x in 'IXYZ'}
    t.update({(x, 'I'): x for x in 'IXYZ'})
    t.update({(x, x): 'I' for x in 'XYZ'})
    for a1, b1, c1 in [('X', 'Y', 'Z'), ('Y', 'Z', 'X'), ('Z', 'X', 'Y')]:
        t[(a1, b1)] = c1
        t[(b1, a1)] = c1
    return t[(a, b)]


def fhat_closed(q, Pstr, delta):
    seen, tot = set(), 0
    for s in q:
        sp = ''.join(mult(a, b) for a, b in zip(s, Pstr))
        key = frozenset([s, sp])
        if key in seen:
            continue
        seen.add(key)
        a, b = q[s], q[sp]
        x = (a - b) ** 2 * mp.sin(delta / 2) ** 2 / (4 * a * b)
        tot += (a + b) / mp.sqrt(1 + x)
    return tot


print("=== (a) Theorem B vs brute-force 16x16 (n=2, local depolarizing, CNOT frame) ===")
CNOT = mp.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
for p, Pstr, d in [(mp.mpf('0.01'), 'ZZ', mp.mpf('0.02')), (mp.mpf('0.01'), 'ZX', mp.mpf('0.02')),
                   (mp.mpf('0.03'), 'ZI', mp.mpf('0.1'))]:
    q = local_rates(p, 2)
    J0 = choi_pauli(q, CNOT)
    J1 = choi_pauli(q, expmP(Pstr, d) * CNOT)
    T = ptrY(gmean(J1, J0), 4, 4)
    E, _ = mp.eighe((T + T.H) / 2)
    print(f"p={p} P={Pstr} delta={d}: brute lmin={mp.nstr(min(E),15)} (spread of Tr_Y eigs {mp.nstr(max(E)-min(E),3)}); closed={mp.nstr(fhat_closed(q,Pstr,d),15)}")

print("\n=== (b) window delta_0.1 (Fhat = 0.9) by bisection on the closed form ===")
eta = mp.mpf('0.1')
xeta = (1 - eta) ** -2 - 1
print("x_eta =", mp.nstr(xeta, 10), " 4 sqrt(x_eta) =", mp.nstr(4 * mp.sqrt(xeta), 10))
sm_table = {('Z', 1): {'1e-1': 1.0719, '1e-2': 0.9780, '1e-3': 0.9696, '1e-5': 0.9687},
            ('ZZ', 2): {'1e-1': 0.5734, '1e-2': 0.4923, '1e-3': 0.4851, '1e-5': 0.4843},
            ('ZZZ', 3): {'1e-1': 0.3120, '1e-2': 0.2482, '1e-3': 0.2428, '1e-5': 0.2422}}


def window(q, Pstr):
    lo, hi = mp.mpf(0), mp.pi
    for _ in range(120):
        mid = (lo + hi) / 2
        if fhat_closed(q, Pstr, mid) > 1 - eta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


for (Pstr, w), vals in sm_table.items():
    n = len(Pstr)
    for ps, sm in vals.items():
        p = mp.mpf(ps)
        dl = window(local_rates(p, n), Pstr)
        print(f"n={n} P={Pstr}: p={ps} delta_0.1/p^(w/2)={mp.nstr(dl/p**(mp.mpf(w)/2),5)}  SM={sm}")
    print(f"   predicted limit 1.9373*2^-w = {mp.nstr(4*mp.sqrt(xeta)*2**-w,6)}")
# weight-1 error on two qubits
p = mp.mpf('1e-3')
dl = window(local_rates(p, 2), 'ZI')
print(f"Z(x)I on n=2, p=1e-3: delta_0.1/sqrt(p) = {mp.nstr(dl/mp.sqrt(p),6)} (SM: 0.9686)")

print("\n=== (c) main-text numbers at p=1e-3, eta=0.1 ===")
p = mp.mpf('1e-3')
for w in [2, 4]:
    asym = 4 * mp.sqrt(xeta) * (p / 4) ** (mp.mpf(w) / 2)
    print(f"w={w}: asymptotic 4 sqrt(x_eta)(p/4)^(w/2) = {mp.nstr(asym,4)}")
dl4 = window(local_rates(p, 4), 'ZZZZ')
print(f"w=n=4 exact closed-form bisection: {mp.nstr(dl4,5)} (text: 1.2e-7)")
dl2 = window(local_rates(p, 2), 'ZZ')
print(f"w=n=2 exact: {mp.nstr(dl2,5)} (text: 4.8e-4)")
# global depolarizing n=2: q_I = 1-p+p/d^2, others p/d^2
d = 4
qg = {''.join(s): (p / d**2) for s in itertools.product('IXYZ', repeat=2)}
qg['II'] = 1 - p + p / d**2
print(f"global n=2 exact window: {mp.nstr(window(qg,'ZZ'),5)}; formula 4 sqrt(x_eta p)/d = {mp.nstr(4*mp.sqrt(xeta*p)/d,5)} (text: 1.5e-2)")
print(f"worst-case modulus (1/4)(4/p)^n at p=1e-3: n=1 {mp.nstr((4/p)/4,5)}, n=2 {mp.nstr((4/p)**2/4,5)}")
