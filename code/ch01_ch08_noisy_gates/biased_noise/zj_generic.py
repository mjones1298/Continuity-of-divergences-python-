"""Zhou-Jiang asymptotic QFI for N_theta = P_q o Ad_{exp(-i theta P/2)} on n qubits (any Pauli channel, any Pauli string).
ZJ (PRX Quantum 2, 010343 (2021)) Thm 2:  F_inf = 4 min_{h: beta(h)=0} ||alpha(h)||,  alpha = sum_i A_i^dag A_i,
A_i = Kdot_i - i sum_j h_ij K_j,  beta = i sum_i K_i^dag A_i.
Certification:
  upper: an explicit Hermitian h with beta(h) = 0 to working precision, ||alpha(h)|| evaluated in mpmath;
  lower: ||alpha|| >= tr(alpha)/d, and min_{h: beta=0} tr(alpha(h))/d is an equality-constrained least-squares problem
         (solved in mpmath on the full h for n = 1, in double on the full h for n = 2).
"""
import itertools
import numpy as np, cvxpy as cp, mpmath as mp

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
P1 = {'1': I2, 'X': PX, 'Y': PY, 'Z': PZ}


def pstring(s):
    M = np.array([[1]], dtype=complex)
    for c in s:
        M = np.kron(M, P1[c])
    return M


def tomp(A):
    return mp.matrix([[mp.mpc(complex(A[i, j])) for j in range(A.shape[1])] for i in range(A.shape[0])])


def kraus_mp(rates, gen):
    """rates: dict pauli-string -> mp rate (zeros dropped); gen: Pauli string of the miscalibration."""
    G = tomp(pstring(gen)) * mp.mpc(0, -0.5)
    K, Kd, labels = [], [], []
    for s, q in rates.items():
        if q == 0:
            continue
        k = mp.sqrt(q) * tomp(pstring(s))
        K.append(k); Kd.append(k * G); labels.append(s)
    return K, Kd, labels


def alpha_beta(K, Kd, h):
    r = len(K); d = K[0].rows
    al = mp.zeros(d, d); be = mp.zeros(d, d)
    for i in range(r):
        A = Kd[i].copy()
        for j in range(r):
            if h[i, j] != 0:
                A = A - 1j * h[i, j] * K[j]
        al += A.H * A
        be += 1j * K[i].H * A
    return al, be


def herm_basis(r, entries):
    """Hermitian basis restricted to the given index pairs (i<=j)."""
    B = []
    for (i, j) in entries:
        if i == j:
            E = mp.zeros(r, r); E[i, i] = 1; B.append(E)
        else:
            E = mp.zeros(r, r); E[i, j] = 1; E[j, i] = 1; B.append(E)
            E = mp.zeros(r, r); E[i, j] = -1j; E[j, i] = 1j; B.append(E)
    return B


def hvec(M):
    d = M.rows
    v = []
    for a in range(d):
        v.append(mp.re(M[a, a]))
        for b in range(a + 1, d):
            v += [mp.re(M[a, b]), mp.im(M[a, b])]
    return v


def trace_min_mp(K, Kd, entries):
    """min tr(alpha(h))/d s.t. beta(h)=0 over h in span of the restricted Hermitian basis (mpmath, KKT via SVD).
    Returns value, h, constraint residual."""
    r = len(K); d = K[0].rows
    B = herm_basis(r, entries); n = len(B)

    def vecA(h):
        v = []
        for i in range(r):
            A = Kd[i].copy()
            for j in range(r):
                if h[i, j] != 0:
                    A = A - 1j * h[i, j] * K[j]
            for a in range(d):
                for c in range(d):
                    v += [mp.re(A[a, c]), mp.im(A[a, c])]
        return mp.matrix(v) / mp.sqrt(d)
    Z0 = mp.zeros(r, r)
    v0 = vecA(Z0)
    _, b0 = alpha_beta(K, Kd, Z0); b0v = hvec(b0)
    M = mp.matrix(len(v0), n); N = mp.matrix(len(b0v), n)
    for k, E in enumerate(B):
        vk = vecA(E) - v0
        for q in range(len(v0)):
            M[q, k] = vk[q]
        _, bE = alpha_beta(K, Kd, E); bv = hvec(bE)
        for q in range(len(b0v)):
            N[q, k] = bv[q] - b0v[q]
    m = len(b0v)
    KKT = mp.zeros(n + m, n + m); rhs = mp.zeros(n + m, 1)
    Q = 2 * M.T * M; c = 2 * M.T * v0
    for i in range(n):
        for j in range(n):
            KKT[i, j] = Q[i, j]
        for qq in range(m):
            KKT[i, n + qq] = N[qq, i]; KKT[n + qq, i] = N[qq, i]
        rhs[i] = -c[i]
    for qq in range(m):
        rhs[n + qq] = -b0v[qq]
    U_, S_, V_ = mp.svd_r(KKT)
    smax = max(S_)
    Sinv = [1 / s if s > smax * mp.mpf(10) ** (-(mp.mp.dps - 8)) else 0 for s in S_]
    sol = V_.T * mp.diag(Sinv) * U_.T * rhs
    x = mp.matrix([sol[i] for i in range(n)])
    h = mp.zeros(r, r)
    for k, E in enumerate(B):
        h += x[k] * E
    val = mp.norm(v0 + M * x) ** 2
    res = mp.norm(N * x + mp.matrix(b0v))
    return val, h, res


def pair_entries(labels, gen):
    """index pairs (i,i), (j,j), (i,j) for Kraus i,j forming a Theorem-B pair {Q, QP}."""
    G = pstring(gen)
    idx = {s: k for k, s in enumerate(labels)}
    ent = set()
    for s, i in idx.items():
        ent.add((i, i))
        M = pstring(s) @ G
        for t, j in idx.items():
            if j > i and abs(abs(np.trace(pstring(t).conj().T @ M)) - M.shape[0]) < 1e-9:
                ent.add((i, j))
    return sorted(ent)


def sdp_double(K, Kd, solver='CLARABEL'):
    r = len(K); d = K[0].rows
    Kn = [np.array(k.tolist(), dtype=complex) for k in K]
    Kdn = [np.array(k.tolist(), dtype=complex) for k in Kd]
    h = cp.Variable((r, r), hermitian=True); t = cp.Variable()
    A = [Kdn[i] - 1j * sum(h[i, j] * Kn[j] for j in range(r)) for i in range(r)]
    Ast = cp.vstack(A)
    beta = 1j * sum(Kn[i].conj().T @ Kdn[i] for i in range(r)) + sum(h[i, j] * (Kn[i].conj().T @ Kn[j]) for i in range(r) for j in range(r))
    Mx = cp.bmat([[t * np.eye(d), Ast.H], [Ast, np.eye(d * r)]])
    prob = cp.Problem(cp.Minimize(t), [Mx >> 0, beta == 0])
    prob.solve(solver=solver)
    return 4 * t.value, prob.status


def trace_min_double_full(K, Kd):
    """min tr(alpha)/d s.t. beta=0 over the FULL Hermitian h, double precision (cvxpy QP)."""
    r = len(K); d = K[0].rows
    Kn = [np.array(k.tolist(), dtype=complex) for k in K]
    Kdn = [np.array(k.tolist(), dtype=complex) for k in Kd]
    h = cp.Variable((r, r), hermitian=True)
    obj = sum(cp.sum_squares(Kdn[i] - 1j * sum(h[i, j] * Kn[j] for j in range(r))) for i in range(r)) / d
    beta = 1j * sum(Kn[i].conj().T @ Kdn[i] for i in range(r)) + sum(h[i, j] * (Kn[i].conj().T @ Kn[j]) for i in range(r) for j in range(r))
    prob = cp.Problem(cp.Minimize(obj), [beta == 0])
    prob.solve(solver='CLARABEL')
    return 4 * prob.value, prob.status


def hnks_rank(K, gen):
    """Is H = gen/2 in span_H{K_i^dag K_j}? Returns (rank of span, rank with H appended)."""
    d = K[0].rows
    vecs = []
    for i in range(len(K)):
        for j in range(len(K)):
            M = np.array((K[i].H * K[j]).tolist(), dtype=complex)
            for Mh in ((M + M.conj().T) / 2, (M - M.conj().T) / 2j):
                vecs.append(np.concatenate([Mh.real.ravel(), Mh.imag.ravel()]))
    A = np.array(vecs)
    H = pstring(gen) / 2
    r1 = np.linalg.matrix_rank(A, tol=1e-10)
    r2 = np.linalg.matrix_rank(np.vstack([A, np.concatenate([H.real.ravel(), H.imag.ravel()])]), tol=1e-10)
    return int(r1), int(r2)


def choi_mp(rates, gen, theta):
    """Choi of P_q o Ad_{R(theta)}: J = sum_Q q_Q vec(Q R) vec(Q R)^dag (row-major vec, output first)."""
    Gm = pstring(gen); d = Gm.shape[0]
    R = mp.cos(theta / 2) * mp.eye(d) - 1j * mp.sin(theta / 2) * tomp(Gm)
    J = mp.zeros(d * d, d * d)
    for s, q in rates.items():
        if q == 0:
            continue
        A = tomp(pstring(s)) * R
        v = mp.matrix([A[a, i] for a in range(d) for i in range(d)])
        J += q * v * v.H
    return J


def F_RLD_direct(rates, gen):
    """lambda_max and lambda_min of Tr_Y[dJ J^{-1} dJ] at theta = 0, with dJ exact."""
    Gm = pstring(gen); d = Gm.shape[0]
    G = tomp(Gm) * mp.mpc(0, -0.5)
    J = mp.zeros(d * d, d * d); dJ = mp.zeros(d * d, d * d)
    for s, q in rates.items():
        if q == 0:
            continue
        A = tomp(pstring(s)); Ad = A * G
        v = mp.matrix([A[a, i] for a in range(d) for i in range(d)])
        vd = mp.matrix([Ad[a, i] for a in range(d) for i in range(d)])
        J += q * v * v.H
        dJ += q * (vd * v.H + v * vd.H)
    M = dJ * mp.inverse(J) * dJ
    M = (M + M.H) / 2
    R = mp.zeros(d, d)
    for x in range(d):
        for z in range(d):
            R[x, z] = mp.fsum(M[y * d + x, y * d + z] for y in range(d))
    ev = sorted([mp.re(e) for e in mp.eighe((R + R.H) / 2)[0]]) if hasattr(mp, 'eighe') else sorted([mp.re(e) for e in mp.eigh((R + R.H) / 2)[0]])
    return ev[-1], ev[0]


def pairs_from_rates(rates, gen):
    """Theorem-B pairs {Q, QP} as (q_Q, q_QP)."""
    G = pstring(gen); seen = set(); out = []
    labs = list(rates.keys())
    for s in labs:
        if s in seen:
            continue
        M = pstring(s) @ G
        for t in labs:
            if abs(abs(np.trace(pstring(t).conj().T @ M)) - M.shape[0]) < 1e-9:
                seen.add(s); seen.add(t); out.append((rates[s], rates[t])); break
    return out


def local_rates(single, n):
    """single: dict '1','X','Y','Z' -> rate; returns product rates on n qubits."""
    out = {}
    for tup in itertools.product('1XYZ', repeat=n):
        q = mp.mpf(1)
        for c in tup:
            q *= single[c]
        out[''.join(tup)] = q
    return out
