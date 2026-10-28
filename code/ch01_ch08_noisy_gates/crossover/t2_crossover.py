"""T2: bracket the adaptive Stein exponent zeta(N_{p,delta} || N_p) for delta/(p ln(1/p)) in [1e-2, 1e2], p in {1e-2,1e-3,1e-4}.
By symmetry zeta(N_{p,d}||N_p) = zeta(N_p||N_{p,d}) (post-rotation by Rz(-d), then conjugation by X; D_p is unitarily covariant).
Lower bounds (all certified: exact Umegaki relative entropy of an explicit k-use strategy, repeated i.i.d.; zeta >= D_k/k):
  (A) sequential, max-entangled input, U^dag between uses: D_k = (1-q) sin^2(k d/2) ln(1 + 4(1-q)/q), q = 1-(1-p)^k.
  (B) sequential with a heralded 2-qubit code {|00>,|11>} (probe+ancilla): X/Y errors are detected (flag), Z errors are logical.
      No-flag probability (1-p/2)^k, logical Bloch length r = (1-2a)^k, a = (p/4)/(1-p/2); flagged branch carries no information.
      D_k = (1-p/2)^k * r artanh(r) * (1-cos k d).
  Both formulas are re-verified below by explicit density-matrix simulation in mpmath (k = 1,2,3).
Upper bounds:
  (U3) paper3-shadows Cor stein(b) with Phi=Psi=N_p, Phi'=N_{p,d}, Psi'=N_p (d_T = 0, zeta(N_p,N_p)=0):
       zeta <= 2 eps ln 2 + (1+eps) h(eps/(1+eps)) + eps ln kappa,  eps = (1-p)|sin(d/2)|, kappa = (4-3p)/p.
  (U0) Cor stein(b): zeta <= ln(d_X/lambda_min) = ln(4/p).
  (BS) zeta <= D_BS (Kubo-Ando converse; closed form, verified against the lambda_max Tr_Y formula).
  (#)  zeta <= min_alpha D^#_alpha(N||M) (Fawzi-Fawzi; SDP, certified in mpmath).
"""
import sys, json, time
import numpy as np, mpmath as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # common.py, sharp_alpha.py
from common import *
from sharp_alpha import Dsharp_channel, certify, T_GRID

mp.mp.dps = 40
LOG = open(Path(__file__).resolve().parent / 'results' / 't2.log', 'w')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


# ---------------- closed-form strategies (double precision is safe: no cancellations) ----------------
def seq_maxent(p, d, kmax):
    k = np.arange(1, kmax + 1, dtype=float)
    one_minus_q = np.exp(k * np.log1p(-p)); q = -np.expm1(k * np.log1p(-p))
    D = one_minus_q * np.sin(k * d / 2) ** 2 * np.log1p(4 * one_minus_q / q)
    r = D / k; i = int(np.argmax(r))
    return r[i], int(k[i])


def seq_herald(p, d, kmax):
    k = np.arange(1, kmax + 1, dtype=float)
    a = (p / 4) / (1 - p / 2)
    surv = np.exp(k * np.log1p(-p / 2))
    one_minus_r = -np.expm1(k * np.log1p(-2 * a)); r = 1 - one_minus_r
    at = 0.5 * np.log((1 + r) / one_minus_r)
    D = surv * r * at * 2 * np.sin(k * d / 2) ** 2
    rr = D / k; i = int(np.argmax(rr))
    return rr[i], int(k[i])


def U3(p, d):
    eps = (1 - p) * abs(np.sin(d / 2))
    x = eps / (1 + eps)
    h = -x * np.log(x) - (1 - x) * np.log1p(-x)
    return 2 * eps * np.log(2) + (1 + eps) * h + eps * np.log((4 - 3 * p) / p)


# ---------------- verification of the strategy formulas by explicit simulation ----------------
def verify_strategies():
    mp.mp.dps = 40
    p = mp.mpf('0.03'); d = mp.mpf('0.2')
    P = [mpmat(s) for s in PAULI]
    wts = [1 - 3 * p / 4, p / 4, p / 4, p / 4]
    out = []
    # (A): k uses of D_p o Rz(d) on half of a max-entangled pair (U^dag U cancels)
    for k in (1, 2, 3):
        def run(dd):
            psi = mp.matrix([1, 0, 0, 1]) / mp.sqrt(2)  # system (x) reference
            rho = psi * psi.H
            R = kron(Rz_mp(dd), mp.eye(2))
            for _ in range(k):
                rho = R * rho * R.H
                rho = sum((wts[i] * (kron(P[i], mp.eye(2)) * rho * kron(P[i], mp.eye(2)).H) for i in range(4)), mp.zeros(4, 4))
            return rho
        Dsim = relent(run(d), run(0))
        q = 1 - (1 - p) ** k
        Dform = (1 - q) * mp.sin(k * d / 2) ** 2 * mp.log(1 + 4 * (1 - q) / q)
        out.append(('A', k, mp.nstr(Dsim, 20), mp.nstr(Dform, 20)))
    # (B): probe (x) ancilla, code {|00>,|11>}, input |+_L>; per use: Rz on probe, Pauli noise on probe, syndrome measure,
    # flag branch: apply X on probe (correction) -- flagged branch kept as a separate classical block.
    for k in (1, 2, 3):
        def run(dd):
            psi = mp.matrix([1, 0, 0, 1]) / mp.sqrt(2)  # |0_L>+|1_L> on probe(x)ancilla
            blocks = {0: psi * psi.H, 1: mp.zeros(4, 4)}
            R = kron(Rz_mp(dd), mp.eye(2))
            Pc = mp.zeros(4, 4); Pc[0, 0] = 1; Pc[3, 3] = 1
            Pe = mp.eye(4) - Pc
            Xc = kron(P[1], mp.eye(2))
            for _ in range(k):
                nb = {0: mp.zeros(4, 4), 1: mp.zeros(4, 4)}
                for f, rho in blocks.items():
                    rho = R * rho * R.H
                    rho = sum((wts[i] * (kron(P[i], mp.eye(2)) * rho * kron(P[i], mp.eye(2)).H) for i in range(4)), mp.zeros(4, 4))
                    nb[f] += Pc * rho * Pc
                    nb[1] += Xc * Pe * rho * Pe * Xc.H
                blocks = nb
            return blocks
        b1, b0 = run(d), run(0)
        Dsim = mp.mpf(0)
        for f in (0, 1):
            # restrict to code space (support), relent on 2x2 block of |00>,|11>
            idx = [0, 3]
            A = mp.matrix([[b1[f][i, j] for j in idx] for i in idx]); Bm = mp.matrix([[b0[f][i, j] for j in idx] for i in idx])
            if mp.re(tr(A)) > mp.mpf(10) ** (-30):
                # regularise exact zero eigenvalues (fully dephased block is full rank anyway)
                Dsim += relent(A, Bm)
        a = (p / 4) / (1 - p / 2); r = (1 - 2 * a) ** k
        Dform = (1 - p / 2) ** k * r * mp.atanh(r) * (1 - mp.cos(k * d))
        out.append(('B', k, mp.nstr(Dsim, 20), mp.nstr(Dform, 20)))
    return out


def verify_closed_forms():
    """D_BS and F_hat closed forms vs direct channel formulas with a nontrivial input unitary U=H."""
    mp.mp.dps = 50
    Hd = mpmat(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    s = 1 / mp.sqrt(2); Hd = mp.matrix([[s, s], [s, -s]])
    res = []
    for ps, ds in [('0.01', '0.001'), ('0.001', '0.03'), ('0.0001', '0.0003'), ('0.0001', '0.5')]:
        p = mp.mpf(ps); d = mp.mpf(ds)
        def ch(dd):
            R = Rz_mp(dd) * Hd
            return choi_from_map_mp(lambda E: (1 - p) * R * E * R.H + p * tr(E) * mp.eye(2) / 2, 2)
        J1, J0 = ch(d), ch(0)
        res.append(dict(p=ps, d=ds,
                        DBS_direct=mp.nstr(DBS_channel(J1, J0, 2, 2), 25), DBS_closed=mp.nstr(DBS_qubit_closed(p, d), 25),
                        DBS_direct_rev=mp.nstr(DBS_channel(J0, J1, 2, 2), 25),
                        Fhat_direct=mp.nstr(Fhat_channel(J1, J0, 2, 2), 25), Fhat_closed=mp.nstr(Fhat_qubit_closed(p, d), 25)))
    return res


def verify_diamond():
    import cvxpy as cp
    out = []
    for p, d in [(0.01, 0.003), (0.001, 0.05), (0.1, 1.0)]:
        J1 = np.array(choi_qubit(p, d).tolist(), complex); J0 = np.array(choi_qubit(p, 0).tolist(), complex)
        Jd = J1 - J0
        W = cp.Variable((4, 4), hermitian=True); rho = cp.Variable((2, 2), hermitian=True)
        cons = [W >> 0, cp.kron(np.eye(2), rho) - W >> 0, rho >> 0, cp.real(cp.trace(rho)) == 1]
        pr = cp.Problem(cp.Maximize(cp.real(cp.trace(Jd @ W))), cons); pr.solve(solver='CLARABEL')
        out.append(dict(p=p, d=d, diamond_sdp=2 * pr.value, exact=2 * (1 - p) * np.sin(d / 2)))
    return out


if __name__ == '__main__':
    t0 = time.time()
    ver = dict(strategies=verify_strategies(), closed=verify_closed_forms(), diamond=verify_diamond())
    for k, v in ver.items():
        for row in v:
            log('VERIFY', k, row)
    results = {}
    for p in [1e-2, 1e-3, 1e-4]:
        L = p * np.log(1 / p)
        xs = np.logspace(-2, 2, 41)
        rows = []
        for x in xs:
            d = x * L
            if d > np.pi:
                continue
            kmax = int(min(20 / p, 50 / d + 10)) + 10
            kmax = int(max(kmax, 10 / p))
            zA, kA = seq_maxent(p, d, kmax)
            zB, kB = seq_herald(p, d, kmax)
            mp.mp.dps = 40
            dbs = float(DBS_qubit_closed(p, d)); fh = Fhat_qubit_closed(p, d); m2lnF = float(-2 * mp.log(fh))
            u3 = U3(p, d); u0 = np.log(4 / p)
            D1 = float(umegaki_maxent_qubit(p, d))
            dsharp = np.nan; tbest = None; dsharp_raw = np.nan
            if x >= 0.2:
                JN = np.array(choi_qubit(p, d).tolist(), complex); JM = np.array(choi_qubit(p, 0).tolist(), complex)
                best = (np.inf, None, None)
                for t in T_GRID:
                    Dv, Q, st, B = Dsharp_channel(JN, JM, 2, 2, t, return_B=True)
                    if B is not None and np.isfinite(Dv) and Dv < best[0]:
                        best = (Dv, t, B)
                if best[2] is not None:
                    dc, c = certify(JN, JM, 2, 2, best[1], best[2], dps=30)
                    dsharp = float(dc); tbest = str(best[1]); dsharp_raw = best[0]
            zl = max(zA, zB)
            cands = {'U3': u3, 'BS': dbs, 'ln4/p': u0}
            if np.isfinite(dsharp):
                cands['sharp'] = dsharp
            which = min(cands, key=cands.get); zu = cands[which]
            row = dict(x=x, delta=d, zeta_lower=zl, zeta_lower_seq_maxent=zA, k_maxent=kA, zeta_lower_seq_herald=zB, k_herald=kB,
                       zeta_upper=zu, upper_from=which, U3=u3, ln4p=u0, Dsharp_cert=dsharp, Dsharp_sdp=dsharp_raw, t_best=tbest,
                       DBS=dbs, m2lnFhat=m2lnF, Umegaki_1use_maxent=D1,
                       DBS_over_zu=dbs / zu, DBS_over_zl=dbs / zl, F_over_zu=m2lnF / zu, F_over_zl=m2lnF / zl,
                       zu_over_zl=zu / zl)
            rows.append(row)
            log(f"p={p:g} x={x:.3g} d={d:.3g} zl={zl:.4g}(A {zA:.4g} k{kA}, B {zB:.4g} k{kB}) zu={zu:.4g}[{which}] "
                f"U3={u3:.4g} D#={dsharp:.4g}(t={tbest}) DBS={dbs:.4g} -2lnF={m2lnF:.4g} | DBS/zu={dbs/zu:.3f} DBS/zl={dbs/zl:.3f} zu/zl={zu/zl:.3f}")
        results[str(p)] = rows
    json.dump(dict(verification=ver, results=results), open(Path(__file__).resolve().parent / 'results' / 't2.json', 'w'), indent=1, default=str)
    log('done in', time.time() - t0, 's')
