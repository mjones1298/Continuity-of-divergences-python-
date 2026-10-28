"""Fawzi-Fawzi D^#_alpha channel divergence (arXiv:2007.12576, Quantum 5, 387 (2021); statements checked via WebFetch 27 Oct 2026):
  Q^#_alpha(rho||sigma) = min tr A  s.t.  rho <= sigma #_{1/alpha} A,  A >= 0;   D^# = log(Q)/(alpha-1).
  Channels (Thm 4.1): Q^#_alpha(N||M) = min || Tr_Y A ||_inf  s.t.  J_N <= J_M #_{1/alpha} A.
  Chain rule (Prop 4.5) => D~_alpha^reg <= (1/m) D^#_alpha(N^{(x)m}||M^{(x)m}) for every m (Thm 5.1); zeta = D^reg <= D~^reg_alpha.
  Hence zeta <= inf_{alpha>1} D^#_alpha(N||M)   (certified: solver point re-checked in mpmath, see certify()).
Implementation: t = 1/alpha dyadic in (1/2,1). With J_M invertible, put C = J_M^{-1/2} J_N J_M^{-1/2}, B = J_M^{-1/2} A J_M^{-1/2};
J_M #_t A = J_M^{1/2} B^t J_M^{1/2}, so the constraint is C <= B^t. B^t for dyadic t = (t_lo+t_hi)/2 is represented recursively by
T <= X # Y, X <= B^{t_lo}, Y <= B^{t_hi}, using Ando's B#S = max{T: [[B,T],[T,S]] >= 0} and monotonicity of #.
"""
import numpy as np, cvxpy as cp
from fractions import Fraction
import mpmath as mp


def ptrace_Y_cp(Mexpr, dY, dX):
    return sum(Mexpr[y * dX:(y + 1) * dX, y * dX:(y + 1) * dX] for y in range(dY))


def _hypo(B, t, n, cons, cache):
    """return an expression E with E <= B^t enforced (t Fraction in [0,1])."""
    if t == 0:
        return np.eye(n)
    if t == 1:
        return B
    if t in cache:
        return cache[t]
    k = t.denominator.bit_length() - 1
    lo = Fraction(int(t * 2 ** (k - 1)), 2 ** (k - 1))
    hi = lo + Fraction(1, 2 ** (k - 1))
    Xe = _hypo(B, lo, n, cons, cache)
    Ye = _hypo(B, hi, n, cons, cache)
    T = cp.Variable((n, n), hermitian=True)
    cons.append(cp.bmat([[Xe, T], [T, Ye]]) >> 0)
    cache[t] = T
    return T


def Dsharp_channel(JN, JM, dY, dX, t, solver='CLARABEL', return_B=False):
    t = Fraction(t)
    n = JN.shape[0]
    w, V = np.linalg.eigh(JM)
    Mh = V @ np.diag(np.sqrt(w)) @ V.conj().T
    Mih = V @ np.diag(1 / np.sqrt(w)) @ V.conj().T
    C = Mih @ JN @ Mih; C = (C + C.conj().T) / 2
    B = cp.Variable((n, n), hermitian=True)
    s = cp.Variable()
    cons = [B >> 0]
    E = _hypo(B, t, n, cons, {})
    cons.append(E - C >> 0)
    TY = ptrace_Y_cp(Mh @ B @ Mh, dY, dX)
    cons.append(s * np.eye(dX) - TY >> 0)
    prob = cp.Problem(cp.Minimize(s), cons)
    try:
        prob.solve(solver=solver)
    except Exception as e:
        return np.nan, np.nan, 'error', None
    alpha = 1 / float(t)
    D = np.log(prob.value) / (alpha - 1) if prob.value is not None and prob.value > 0 else np.nan
    return D, prob.value, prob.status, (B.value if return_B else None)


def certify(JN, JM, dY, dX, t, Bval, dps=40):
    """Rigorous (up to mp rounding) upper bound on D^#_alpha from a candidate B: find minimal scale c >= 1 with
    C <= (c B)^t, then Q <= c * lambda_max Tr_Y(J_M^{1/2} B J_M^{1/2}). Everything in mpmath."""
    from common import mpmat, sqrtm, isqrtm, funm, herm, eigvals, ptrace_Y
    mp.mp.dps = dps
    t = mp.mpf(Fraction(t).numerator) / Fraction(t).denominator
    JNm, JMm, Bm = mpmat(JN), mpmat(JM), herm(mpmat(Bval))
    Mh = sqrtm(JMm); Mih = isqrtm(JMm)
    C = herm(Mih * JNm * Mih)
    ev = eigvals(Bm)
    if ev[0] <= 0:
        Bm = Bm + (abs(ev[0]) + mp.mpf(10) ** (-30)) * mp.eye(Bm.rows)
    Bt = funm(Bm, lambda x: x ** t)
    lo, hi = mp.mpf(1), mp.mpf(1)
    while eigvals(hi ** t * Bt - C)[0] < 0:
        hi *= mp.mpf(1) + mp.mpf(10) ** (-6) if hi == 1 else hi
        hi = hi if hi > 1 else mp.mpf(1) + mp.mpf(10) ** (-9)
    if eigvals(Bt - C)[0] >= 0:
        c = mp.mpf(1)
    else:
        lo = mp.mpf(1)
        for _ in range(80):
            mid = (lo + hi) / 2
            if eigvals(mid ** t * Bt - C)[0] >= 0:
                hi = mid
            else:
                lo = mid
        c = hi
    Q = c * eigvals(ptrace_Y(herm(Mh * Bm * Mh), dY, dX))[-1]
    alpha = 1 / t
    return mp.log(Q) / (alpha - 1), c


T_GRID = [Fraction(1, 2), Fraction(9, 16), Fraction(5, 8), Fraction(11, 16), Fraction(3, 4), Fraction(13, 16),
          Fraction(7, 8), Fraction(15, 16), Fraction(31, 32)]

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    rng = np.random.default_rng(1)
    # self-test on channels: N = M gives 0; commuting (classical) channel pair gives the classical Renyi channel value
    from common import choi_qubit
    JM = np.array(choi_qubit('0.01', 0).tolist(), dtype=complex)
    for t in (Fraction(1, 2), Fraction(5, 8), Fraction(7, 8)):
        D, Q, st, B = Dsharp_channel(JM, JM, 2, 2, t, return_B=True)
        print('N=M  t', t, 'D#', D, st, 'certified', certify(JM, JM, 2, 2, t, B))
    # classical channels: binary symmetric with flip a vs b (diagonal Chois)
    def bsc(a):
        J = np.zeros((4, 4), complex)
        # J = sum_ij Phi(E_ij) (x) E_ij; classical channel: Phi(E_ii) = diag, Phi(E_ij)=0 for i!=j
        for i in range(2):
            out = np.diag([1 - a, a]) if i == 0 else np.diag([a, 1 - a])
            for y in range(2):
                J[y * 2 + i, y * 2 + i] = out[y, y]
        return J
    for t in (Fraction(1, 2), Fraction(3, 4), Fraction(15, 16)):
        al = 1 / float(t)
        a, b = 0.1, 0.3
        ex = np.log(max(((1 - a) ** al * (1 - b) ** (1 - al) + a ** al * b ** (1 - al)), (a ** al * b ** (1 - al) + (1 - a) ** al * (1 - b) ** (1 - al)))) / (al - 1)
        D, Q, st, B = Dsharp_channel(bsc(a), bsc(b), 2, 2, t, return_B=True)
        print('BSC t', t, 'D#', D, 'classical Renyi', ex, 'cert', certify(bsc(a), bsc(b), 2, 2, t, B)[0])
