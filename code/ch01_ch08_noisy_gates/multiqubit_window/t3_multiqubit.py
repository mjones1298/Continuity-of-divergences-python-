"""T3: multi-qubit fragility window. N_delta = D o exp(-i delta P0/2) o G  vs  N_0 = D o G, with
   G = gate (CNOT, CNOT(x)1, Toffoli): an INPUT-side unitary, which leaves F_hat invariant (verified by brute force below),
   D = local depolarizing D_p^{(x)n} or global depolarizing on d = 2^n, P0 = Z^{(x)n} (also Z(x)1 for comparison).
F_hat = lambda_min Tr_Y (J0 # J_delta)  (ARN formula; both Chois invertible, so W g(T) W^* = J0 # J_delta).
Exact reduction: for a Pauli channel, J0 = sum_P lam_P |vecP><vecP|/d is diagonal in the Pauli basis, and (R(x)1) vec P =
cos(d/2) vec P - i sin(d/2) vec(P0 P). Hence J0, J_delta and their geometric mean are block diagonal in 2x2 blocks {P, P0 P}.
Each block mean is computed in mpmath with the 2x2 formula A#B = sqrt(ab)(A+B)/sqrt(det(A+B)) (det A = det B = ab);
Tr_Y(|vecA><vecB|) = A^T conj(B). Brute-force 16x16 and 64x64 mpmath geometric means confirm the reduction.
"""
import sys, json, itertools, time
import numpy as np, mpmath as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # common.py, sharp_alpha.py
from common import *

mp.mp.dps = 50
LOG = open(Path(__file__).resolve().parent / 'results' / 't3.log', 'w')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


PM = [mpmat(s) for s in PAULI]


def pauli_string(idx):
    M = PM[idx[0]]
    for i in idx[1:]:
        M = kron(M, PM[i])
    return M


def pauli_weights_local(p, n):
    p = mp.mpf(p)
    w1 = [1 - 3 * p / 4, p / 4, p / 4, p / 4]
    return {idx: mp.fprod([w1[i] for i in idx]) for idx in itertools.product(range(4), repeat=n)}


def pauli_weights_global(p, n):
    p = mp.mpf(p); d = 2 ** n
    return {idx: ((1 - p + p / d ** 2) if all(i == 0 for i in idx) else p / d ** 2) for idx in itertools.product(range(4), repeat=n)}


def mult_index(a, b):
    """Pauli product sigma_a sigma_b = phase * sigma_c (single qubit)."""
    A, B = PM[a], PM[b]
    M = A * B
    for c in range(4):
        ph = tr(PM[c] * M) / 2
        if abs(ph) > 0.5:
            return c, ph
    raise ValueError


def Fhat_block(weights, P0, delta, n):
    """Exact F_hat via 2x2 Pauli blocks. Returns (F_hat, lambda_max(Tr_Y G), offdiag weight)."""
    d = 2 ** n
    delta = mp.mpf(delta); c = mp.cos(delta / 2); s = mp.sin(delta / 2)
    lam = {P: d * w for P, w in weights.items()}
    mats = {P: pauli_string(P) for P in weights}
    seen = set(); TY = mp.zeros(d, d)
    for P in weights:
        if P in seen:
            continue
        # Q = P0 P up to phase
        Qi = []; ph = mp.mpc(1)
        for a, b in zip(P0, P):
            ci, phi = mult_index(a, b); Qi.append(ci); ph *= phi
        Q = tuple(Qi)
        seen.add(P); seen.add(Q)
        if Q == P:
            raise ValueError
        a, b = lam[P], lam[Q]
        # basis e1 = vecP/sqrt d, e2 = vecQ/sqrt d ; vec(P0 P) = ph vecQ ; vec(P0 Q) = ph' vecP with ph' = conj... compute
        Qi2 = []; ph2 = mp.mpc(1)
        for a_, b_ in zip(P0, Q):
            ci, phi = mult_index(a_, b_); Qi2.append(ci); ph2 *= phi
        u1 = mp.matrix([c, -1j * s * ph])       # (R(x)1) e1
        u2 = mp.matrix([-1j * s * ph2, c])      # (R(x)1) e2
        A = mp.matrix([[a, 0], [0, b]])
        B = a * u1 * u1.H + b * u2 * u2.H
        if a == b:
            Gm = A
        else:
            S = A + B
            det = mp.re(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
            Gm = mp.sqrt(a * b) * S / mp.sqrt(det)
        basis = [mats[P], mats[Q]]
        for i in range(2):
            for j in range(2):
                if Gm[i, j] != 0:
                    TY += Gm[i, j] * (basis[i].T * conj_mat(basis[j])) / d
    ev = eigvals(TY)
    off = mp.norm(TY - (tr(TY) / d) * mp.eye(d))
    return ev[0], ev[-1], off


def conj_mat(M):
    R = M.copy()
    for i in range(M.rows):
        for j in range(M.cols):
            R[i, j] = mp.conj(M[i, j])
    return R


def choi_full(weights, P0, delta, n, gate=None):
    """Brute-force Choi of D o exp(-i delta P0/2) o gate (mpmath, d^2 x d^2)."""
    d = 2 ** n
    R = expm_pauli(P0, delta)
    V = R * gate if gate is not None else R
    mats = {P: pauli_string(P) for P in weights}

    def f(E):
        out = mp.zeros(d, d)
        X_ = V * E * V.H
        for P, w in weights.items():
            out += w * mats[P] * X_ * mats[P].H
        return out
    return choi_from_map_mp(f, d)


def expm_pauli(P0, delta):
    M = pauli_string(P0); d = M.rows
    return mp.cos(mp.mpf(delta) / 2) * mp.eye(d) - 1j * mp.sin(mp.mpf(delta) / 2) * M


def cnot():
    C = mp.zeros(4, 4); C[0, 0] = 1; C[1, 1] = 1; C[2, 3] = 1; C[3, 2] = 1
    return C


def toffoli():
    T = mp.eye(8); T[6, 6] = 0; T[7, 7] = 0; T[6, 7] = 1; T[7, 6] = 1
    return T


def delta_star(weights, P0, n, target=mp.mpf('0.9')):
    lo, hi = mp.mpf(0), mp.mpf(1)
    while Fhat_block(weights, P0, hi, n)[0] > target:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if Fhat_block(weights, P0, mid, n)[0] > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == '__main__':
    t0 = time.time()
    out = {'verification': [], 'lambda_min': [], 'window': {}, 'curves': {}}
    # ---- verification: block formula vs brute-force geometric mean, including input gates ----
    mp.mp.dps = 40
    tests = [(1, (3,), None, 'local'), (2, (3, 3), cnot(), 'local'), (2, (3, 0), cnot(), 'local'), (2, (3, 3), cnot(), 'global'),
             (3, (3, 3, 3), toffoli(), 'local'), (3, (3, 3, 3), kron(cnot(), mp.eye(2)), 'global')]
    for n, P0, gate, model in tests:
        for p, dl in [('0.01', '0.004'), ('0.001', '0.0007')]:
            W = pauli_weights_local(p, n) if model == 'local' else pauli_weights_global(p, n)
            fb, fbmax, off = Fhat_block(W, P0, dl, n)
            tb = time.time()
            J1 = choi_full(W, P0, dl, n, gate); J0 = choi_full(W, P0, 0, n, gate)
            ev0 = eigvals(J0)
            ff = Fhat_channel(J1, J0, 2 ** n, 2 ** n)
            rec = dict(n=n, P0=P0, gate=(gate is not None), model=model, p=p, delta=dl, Fhat_block=mp.nstr(fb, 25),
                       Fhat_bruteforce=mp.nstr(ff, 25), absdiff=mp.nstr(abs(fb - ff), 3), TrY_offscalar=mp.nstr(off, 3),
                       lambda_min_J=mp.nstr(ev0[0], 15), lambda_max_J=mp.nstr(ev0[-1], 15), secs=round(time.time() - tb, 1))
            out['verification'].append(rec); log('VERIFY', rec)
            if n == 3:
                break
    mp.mp.dps = 50
    # ---- lambda_min scaling ----
    for p in ['1e-1', '1e-2', '1e-3', '1e-4']:
        for n in (1, 2, 3):
            for model in ('local', 'global'):
                W = pauli_weights_local(p, n) if model == 'local' else pauli_weights_global(p, n)
                lmin = min(2 ** n * w for w in W.values())
                pm = mp.mpf(p)
                pred = (pm / 2) ** n if model == 'local' else pm / 2 ** n
                out['lambda_min'].append(dict(p=p, n=n, model=model, lambda_min=mp.nstr(lmin, 15), predicted=mp.nstr(pred, 15),
                                              note='local: (p/2)^n ; global: p/d (unnormalised Choi, tr J = d); p/d^2 for J/d'))
    log('lambda_min', out['lambda_min'])
    # ---- window delta*(F_hat = 0.9) vs p and exponent fit ----
    plist = ['1e-1', '3e-2', '1e-2', '3e-3', '1e-3', '3e-4', '1e-4', '1e-5']
    configs = [('n1_local_Z', 1, (3,), 'local'), ('n2_local_ZZ', 2, (3, 3), 'local'), ('n2_local_ZI', 2, (3, 0), 'local'),
               ('n3_local_ZZZ', 3, (3, 3, 3), 'local'), ('n2_global_ZZ', 2, (3, 3), 'global'), ('n3_global_ZZZ', 3, (3, 3, 3), 'global')]
    for name, n, P0, model in configs:
        rows = []
        for p in plist:
            W = pauli_weights_local(p, n) if model == 'local' else pauli_weights_global(p, n)
            ds = delta_star(W, P0, n)
            lam = sorted(2 ** n * w for w in W.values())
            rows.append(dict(p=float(mp.mpf(p)), delta_star=float(ds), kappa=float(lam[-1] / lam[0])))
        lp = np.log([r['p'] for r in rows]); ld = np.log([r['delta_star'] for r in rows])
        slope_all = np.polyfit(lp, ld, 1)[0]
        slope_small = np.polyfit(lp[-4:], ld[-4:], 1)[0]
        pref = [r['delta_star'] / r['p'] ** (n / 2) for r in rows]
        out['window'][name] = dict(rows=rows, slope_all=slope_all, slope_p_le_1em3=slope_small, prefactor_delta_over_p_pow_n2=pref)
        log('WINDOW', name, 'slope(all)', round(slope_all, 5), 'slope(p<=1e-3)', round(slope_small, 6), 'delta*/p^{n/2}', [round(x, 5) for x in pref])
    # ---- curves for the collapse figure ----
    for name, n, P0, model in configs:
        for p in ['1e-2', '1e-3', '1e-4']:
            W = pauli_weights_local(p, n) if model == 'local' else pauli_weights_global(p, n)
            pm = float(mp.mpf(p))
            scale = pm ** (n / 2) if (model == 'local' and P0 == tuple([3] * n)) else pm ** 0.5
            us = np.logspace(-2, 2, 49)
            F = [float(Fhat_block(W, P0, u * scale, n)[0]) for u in us]
            out['curves'][f'{name}_p{p}'] = dict(u=list(us), scale=scale, Fhat=F, n=n, model=model, P0=P0, p=pm)
        log('curves', name, 'done')
    json.dump(out, open(Path(__file__).resolve().parent / 'results' / 't3.json', 'w'), indent=1, default=str)
    log('done', time.time() - t0)
