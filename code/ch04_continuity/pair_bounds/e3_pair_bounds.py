"""E3': validation of different-sum continuity bounds (candidate bound; Jones v5 Thm 10.4(b),
Cor 10.7, Lemma 10.5) and same-sum Thm 4.6, on >= 2000 instances.

Frame: Omega' = Phi1+Psi1+Phi2+Psi2 with a linearly independent Kraus frame W obtained stably
from the SVD of the stacked Choi factors [R_Phi1, R_Psi1, R_Phi2, R_Psi2] = U S V^*:
W = U_r S_r, and the derivative of the j-th map is V_j^* V_j (V_j = row block j of V_r).
"""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
from qlib import *

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261027
rng = np.random.default_rng(SEED)
SV_TOL = 1e-10
CE_TOL = 1e-6        # ratio > 1 + CE_TOL is a counterexample candidate
ABS_TOL = 1e-9       # ... and LHS - RHS > ABS_TOL (numerical floor of Fhat ~1e-12..1e-10)


def frame4(Js):
    Rs = [psd_factor(J) for J in Js]
    W0 = np.hstack(Rs)
    U, S, Vh = np.linalg.svd(W0, full_matrices=False)
    keep = S > SV_TOL * S.max()
    W = U[:, keep] * S[keep]
    V = Vh[keep, :].conj().T
    ders, o = [], 0
    for R in Rs:
        Vb = V[o:o + R.shape[1], :]
        ders.append(herm(Vb.conj().T @ Vb))
        o += R.shape[1]
    return W, ders


def sqrtm_psd(A):
    return psd_fn(A, np.sqrt)


def analyse(JP1, JQ1, JP2, JQ2, dY, dX, check_identities=False):
    W, (A1, B1, A2, B2) = frame4([JP1, JQ1, JP2, JQ2])
    r = W.shape[1]
    C1, C2 = A1 + B1, A2 + B2
    Om_star = ptrace_Y(JP1 + JQ1 + JP2 + JQ2, dY, dX)  # = Omega'^*(1)^T
    nOmp = opnorm(Om_star)          # ||Omega'^*(1)||
    nOmbar = nOmp / 2               # ||Omega-bar^*(1)||
    Tt, gTt = [], []
    for A, B, C in ((A1, B1, C1), (A2, B2, C2)):
        # T~ = C^{+1/2} A C^{+1/2}; g(T~) evaluated endpoint-accurately from t = <u,T~u> and
        # s = <u, C^{+1/2} B C^{+1/2} u> (= 1 - t on ran C) instead of from eigvalsh(T~)
        Cih = pinv_sqrt(C, 1e-13)
        Tm, Sm = herm(Cih @ A @ Cih), herm(Cih @ B @ Cih)
        w, U = np.linalg.eigh(Tm)
        t = np.real(np.einsum("ij,ij->j", U.conj(), Tm @ U)).clip(0)
        sv = np.real(np.einsum("ij,ij->j", U.conj(), Sm @ U)).clip(0)
        Tt.append(Tm)
        gTt.append(herm((U * np.sqrt(t * sv)) @ U.conj().T))
    tau = opnorm(Tt[0] - Tt[1])
    c12 = opnorm(C1 - C2)
    delta = max(opnorm(A1 - A2), opnorm(B1 - B2))
    gamma = c12 / 2
    F1 = fhat_arn(JP1, JQ1, dY, dX)
    F2 = fhat_arn(JP2, JQ2, dY, dX)
    lhs = abs(F1 - F2)
    rank1 = np.linalg.matrix_rank(C1, 1e-10)
    rank2 = np.linalg.matrix_rank(C2, 1e-10)
    same_support = (rank1 == r) and (rank2 == r)
    out = dict(F1=F1, F2=F2, lhs=lhs, tau=tau, C1mC2=c12, delta=delta, gamma=gamma, r=int(r),
               same_support=bool(same_support), nOmp=nOmp,
               lminC=[lmin_pos(C1), lmin_pos(C2)], lmin_G=float(np.linalg.eigvalsh(W.conj().T @ W).min()))
    # candidate bound (prompt):  ||Omega'^*(1)|| ( max||C_i|| tau^1/2 + ||C1-C2||^1/2 )
    out["rhs_candidate"] = nOmp * (max(opnorm(C1), opnorm(C2)) * np.sqrt(tau) + np.sqrt(c12))
    # v5 Thm 10.4(b):  ||Omega-bar^*(1)|| ( tau^1/2 + sqrt2 tanh(dT/2) ),  tanh(dT/2) = ||C1-C2||
    out["rhs_thm104b"] = nOmbar * (np.sqrt(tau) + np.sqrt(2) * c12) if same_support else np.inf
    # v5 Cor 10.7: (11/4)||Omega-bar^*(1)|| delta^1/2
    out["rhs_cor107"] = 11 / 4 * nOmbar * np.sqrt(delta)
    # v5 Lemma 10.5 (if delta < 1/2):  tau <= 2 sqrt2 delta / sqrt(1-4 gamma^2) <= 2 sqrt2 delta/sqrt(1-4delta^2)
    if delta < 0.5:
        out["lem105_gamma"] = tau / (2 * np.sqrt(2) * delta / np.sqrt(max(1e-300, 1 - 4 * gamma ** 2))) if delta > 0 else 0.0
        out["lem105_delta"] = tau / (2 * np.sqrt(2) * delta / np.sqrt(1 - 4 * delta ** 2)) if delta > 0 else 0.0
        out["lem105_gamma_le_delta"] = gamma - delta
    if check_identities:
        # (i) A_i#B_i = C_i^1/2 g(T~_i) C_i^1/2   and  (ii) J(Phi_i)#J(Psi_i) = W (A_i#B_i) W^*,
        # with every geometric mean on the left computed independently by mpmath eps-regularisation.
        errs = []
        for A, B, C, gT, JP, JQ in ((A1, B1, C1, gTt[0], JP1, JQ1), (A2, B2, C2, gTt[1], JP2, JQ2)):
            # compress to ran C (cutoff 1e-8 rel.): frame-level A,B inherit ~1e-10 support noise from
            # the SVD frame, which otherwise makes the regularised mean pick up a spurious direction
            wc, Uc = np.linalg.eigh(C)
            Pc = Uc[:, wc > 1e-8 * wc.max()]
            Pp = Pc @ Pc.conj().T
            AB = gmean_mp(herm(Pp @ A @ Pp), herm(Pp @ B @ Pp))
            Ch = sqrtm_psd(C)
            rhs1 = Ch @ gT @ Ch
            JJ = gmean_mp(JP, JQ)
            errs.append((float(opnorm(AB - rhs1)), float(opnorm(JJ - W @ rhs1 @ W.conj().T)),
                         float(opnorm(JJ - W @ AB @ W.conj().T))))
        out["identity_errs"] = errs
    return out


# ----------------------------------------------------------------------------- instance families
def perturb_kraus(Ks, delta, rng):
    dY, dX = Ks[0].shape
    Kp = [K + delta * (rng.normal(size=K.shape) + 1j * rng.normal(size=K.shape)) / np.sqrt(2 * dY * dX) for K in Ks]
    return tp_renormalize(Kp)


def rand_channel(rng, d, r=None):
    r = r if r is not None else int(rng.integers(1, d * d + 1))
    return random_kraus(rng, d, d, r)


def family_instances(rng):
    inst = []
    deltas = np.logspace(-6, 0, 13)
    # F1: random channels, Kraus perturbation (both maps perturbed)
    for d in (2, 3):
        for _ in range(45 if d == 2 else 30):
            KP, KQ = rand_channel(rng, d), rand_channel(rng, d)
            for dl in deltas:
                inst.append(("rand_kraus_pert", d, KP, KQ, perturb_kraus(KP, dl, rng), perturb_kraus(KQ, dl, rng), dl))
    # F2: random channels, mixing perturbation  Phi2 = (1-dl)Phi1 + dl Xi
    for d in (2, 3):
        for _ in range(15):
            KP, KQ = rand_channel(rng, d), rand_channel(rng, d)
            Xi, Xi2 = rand_channel(rng, d), rand_channel(rng, d)
            for dl in deltas:
                inst.append(("rand_mix", d, KP, KQ, ("mix", KP, Xi, dl), ("mix", KQ, Xi2, dl), dl))
    # F3: low Kraus rank (unitary pairs and rank-2), perturbed
    for d in (2, 3):
        for _ in range(10):
            U1, U2 = random_unitary(rng, d), random_unitary(rng, d)
            for dl in deltas:
                inst.append(("unitary_pair_pert", d, [U1], [U2], perturb_kraus([U1], dl, rng), perturb_kraus([U2], dl, rng), dl))
            KP, KQ = rand_channel(rng, d, 2), rand_channel(rng, d, 1)
            for dl in deltas:
                inst.append(("lowrank_pert", d, KP, KQ, perturb_kraus(KP, dl, rng), perturb_kraus(KQ, dl, rng), dl))
    # F4: replacement/reset channels: (R_a, R_b) vs (R_a', R_b') with nearby states (+ small noise)
    for _ in range(10):
        for dl in deltas:
            a, b = rng.uniform(0, np.pi), rng.uniform(0, np.pi)
            p = 10 ** rng.uniform(-6, -1) if rng.random() < 0.5 else 0.0

            def R(th, p):
                k = np.array([np.cos(th), np.sin(th)], complex)
                return choi_replacement((1 - p) * np.outer(k, k.conj()) + p * np.eye(2) / 2, 2)
            inst.append(("reset", 2, R(a, p), R(b, p), R(a + dl, p), R(b - dl, p), dl))
    # F5: amplitude damping gamma -> 1
    for g in (1.0, 1 - 1e-3, 0.9, 0.5):
        for dl in deltas:
            g2 = min(1.0, max(0.0, g - dl))
            inst.append(("ampdamp", 2, kraus_ampdamp(g), kraus_ampdamp(min(1, g + 0.0)), kraus_ampdamp(g2), kraus_ampdamp(g2 if rng.random() < .5 else g), dl))
            inst.append(("ampdamp_vs_reset", 2, kraus_ampdamp(g), kraus_ampdamp(1.0), kraus_ampdamp(g2), kraus_ampdamp(1.0), dl))
    # F6: near-unitary channels (U composed with weak noise) vs miscalibrated versions
    for d in (2, 3):
        for _ in range(8):
            U = random_unitary(rng, d)
            Hh = herm(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
            p = 10 ** rng.uniform(-5, -1)
            for dl in deltas:
                Ut = sla.expm(-1j * dl * Hh) @ U
                JU = choi_depolarize_after(choi_unitary(U), p, d, d)
                JUt = choi_depolarize_after(choi_unitary(Ut), p, d, d)
                inst.append(("near_unitary", d, JU, JU, JU, JUt, dl))
                inst.append(("near_unitary_both", d, choi_unitary(U), JU, choi_unitary(Ut), JUt, dl))
    # F7: mixtures near singular: Phi = (1-e)U1 + e U2, Psi = U2 etc.
    for d in (2, 3):
        for _ in range(8):
            U1, U2 = random_unitary(rng, d), random_unitary(rng, d)
            e = 10 ** rng.uniform(-6, -1)
            for dl in deltas:
                JA = (1 - e) * choi_unitary(U1) + e * choi_unitary(U2)
                JB = choi_unitary(U2)
                U1t = sla.expm(-1j * dl * herm(rng.normal(size=(d, d)))) @ U1
                JAt = (1 - e) * choi_unitary(U1t) + e * choi_unitary(U2)
                inst.append(("mix_near_singular", d, JA, JB, JAt, JB, dl))
    return inst


def to_choi(obj, d):
    if isinstance(obj, tuple) and obj[0] == "mix":
        _, K, Xi, dl = obj
        return (1 - dl) * choi_from_kraus(K) + dl * choi_from_kraus(Xi)
    if isinstance(obj, list):
        return choi_from_kraus(obj)
    return obj


# Jones v5 Example 10.6 at derivative level (sharpness of Lemma 10.5) + four CP maps via a frame
def example_106(dl, rng):
    Z = np.diag([1., -1.]); Xp = np.array([[0., 1.], [1., 0.]])
    U0, U1 = (Z + Xp) / np.sqrt(2), (Z - Xp) / np.sqrt(2)
    T0 = np.diag([dl, 1 - dl])
    # A_{1,2} = (T0 +- dl U0)/2,  B_{1,2} = (1 - T0 +- dl U1)/2  (so ||A1-A2|| = ||B1-B2|| = dl)
    A1, A2 = 0.5 * (T0 + dl * U0), 0.5 * (T0 - dl * U0)
    B1, B2 = 0.5 * (np.eye(2) - T0 + dl * U1), 0.5 * (np.eye(2) - T0 - dl * U1)
    V = [random_kraus(rng, 2, 2, 1)[0], random_kraus(rng, 2, 2, 1)[0]]
    W = np.stack([vec(v) for v in V], 1)
    return [herm(W @ M @ W.conj().T) for M in (A1, B1, A2, B2)]


def main():
    t0 = time.time()
    inst = family_instances(rng)
    rows = []
    for k, (fam, d, P1, Q1, P2, Q2, dl) in enumerate(inst):
        Js = [to_choi(o, d) for o in (P1, Q1, P2, Q2)]
        res = analyse(*Js, d, d, check_identities=(k % 25 == 0))
        res.update(family=fam, d=d, pert=float(dl), idx=k)
        rows.append(res)
    for dl in np.logspace(-6, np.log10(0.49), 30):
        Js = example_106(dl, rng)
        res = analyse(*Js, 2, 2, check_identities=(len(rows) % 5 == 0))
        res.update(family="jones_ex10.6_CPmaps", d=2, pert=float(dl), idx=len(rows))
        rows.append(res)
    print("instances:", len(rows), "time", time.time() - t0)

    # ----------- same-sum Thm 4.6: |dF| <= ||Omega^*(1)|| ||T1-T2||^1/2 with common Omega
    ss = []
    for k in range(400):
        d = 2 if k % 2 == 0 else 3
        JOm = choi_from_kraus(rand_channel(rng, d)) + choi_from_kraus(rand_channel(rng, d))
        W = arn_frame(JOm)
        r = W.shape[1]
        Hh = herm(rng.normal(size=(r, r)) + 1j * rng.normal(size=(r, r)))
        w, U = np.linalg.eigh(Hh)
        T1 = herm((U * rng.uniform(0, 1, r) ** rng.choice([1, 4, 0.25])) @ U.conj().T) if k % 3 else herm((U * (rng.random(r) < .5)) @ U.conj().T)
        dl = 10 ** rng.uniform(-6, 0)
        E = herm(rng.normal(size=(r, r)) + 1j * rng.normal(size=(r, r)))
        T2 = T1 + dl * E / opnorm(E)
        w2, U2 = np.linalg.eigh(T2)
        T2 = herm((U2 * np.clip(w2, 0, 1)) @ U2.conj().T)
        J1, J2 = herm(W @ T1 @ W.conj().T), herm(W @ T2 @ W.conj().T)
        F1 = fhat_arn(J1, JOm - J1, d, d); F2 = fhat_arn(J2, JOm - J2, d, d)
        rhs = opnorm(ptrace_Y(JOm, d, d)) * np.sqrt(opnorm(T1 - T2))
        ss.append(dict(d=d, lhs=abs(F1 - F2), rhs=rhs, ratio=abs(F1 - F2) / rhs if rhs > 0 else 0.0, dT=opnorm(T1 - T2)))

    # ----------- statistics
    def stat(key, rows_):
        vals = [(r["lhs"] / r[key] if r[key] > 0 and np.isfinite(r[key]) else (0.0 if r["lhs"] < ABS_TOL else np.inf), r) for r in rows_]
        rat = np.array([v for v, _ in vals])
        fin = rat[np.isfinite(rat)]
        best = max(vals, key=lambda x: x[0] if np.isfinite(x[0]) else -1)
        viol = [r for v, r in vals if v > 1 + CE_TOL and r["lhs"] - r[key] > ABS_TOL]
        return dict(n=len(rat), n_finite=int(len(fin)), max_ratio=float(fin.max()) if len(fin) else None,
                    quantiles={q: float(np.quantile(fin, q)) for q in (0.5, 0.9, 0.99)} if len(fin) else None,
                    argmax=dict(family=best[1]["family"], d=best[1]["d"], pert=best[1]["pert"], lhs=best[1]["lhs"],
                                rhs=best[1][key], idx=best[1]["idx"], F1=best[1]["F1"], F2=best[1]["F2"]),
                    n_violations=len(viol), violations=[v["idx"] for v in viol][:50]), viol

    s_cand, v_cand = stat("rhs_candidate", rows)
    s_104, v_104 = stat("rhs_thm104b", [r for r in rows if r["same_support"]])
    s_107, v_107 = stat("rhs_cor107", rows)
    l105 = [r for r in rows if "lem105_gamma" in r]
    lem = dict(n=len(l105), max_ratio_gamma_form=max(r["lem105_gamma"] for r in l105),
               max_ratio_delta_form=max(r["lem105_delta"] for r in l105),
               max_gamma_minus_delta=max(r["lem105_gamma_le_delta"] for r in l105),
               ex106_ratio_vs_delta=[(r["delta"], r.get("lem105_delta"), r["tau"] / r["delta"]) for r in rows if r["family"] == "jones_ex10.6_CPmaps"])
    ids = [e for r in rows if "identity_errs" in r for e in r["identity_errs"]]
    ident = dict(n=len(ids), method="all means by mpmath eps-regularisation (140 digits, eps=1e-30..1e-70, Aitken)",
                 max_err_AsharpB_vs_C12gTC12=max(e[0] for e in ids),
                 max_err_JsharpJ_vs_W_C12gTC12_W=max(e[1] for e in ids),
                 max_err_JsharpJ_vs_W_AsharpB_W=max(e[2] for e in ids))
    ss_rat = np.array([s["ratio"] for s in ss])
    same = dict(n=len(ss), max_ratio=float(ss_rat.max()), median=float(np.median(ss_rat)),
                n_viol=int(np.sum([(s["lhs"] - s["rhs"] > ABS_TOL) and s["ratio"] > 1 + CE_TOL for s in ss])))
    # per-family maxima
    fam = {}
    for f in sorted(set(r["family"] for r in rows)):
        rr = [r for r in rows if r["family"] == f]
        fam[f] = dict(n=len(rr),
                      max_cand=max(r["lhs"] / r["rhs_candidate"] for r in rr if r["rhs_candidate"] > 0),
                      max_104b=max([r["lhs"] / r["rhs_thm104b"] for r in rr if r["same_support"] and r["rhs_thm104b"] > 0] or [None], key=lambda x: -1 if x is None else x),
                      max_107=max(r["lhs"] / r["rhs_cor107"] for r in rr if r["rhs_cor107"] > 0),
                      frac_same_support=float(np.mean([r["same_support"] for r in rr])))

    # double-check top instances (and all violations) with SDP + mp regularisation
    checks = []
    top = sorted(rows, key=lambda r: -(r["lhs"] / r["rhs_thm104b"]) if r["same_support"] and r["rhs_thm104b"] > 0 else 0)[:8]
    top += sorted(rows, key=lambda r: -(r["lhs"] / r["rhs_candidate"]))[:5]
    top += [rows[i] for i in set(v["idx"] for v in v_cand + v_104 + v_107)]
    seen = set()
    for r in top:
        if r["idx"] in seen:
            continue
        seen.add(r["idx"])
        if r["family"] == "jones_ex10.6_CPmaps":
            continue
        f, d, P1, Q1, P2, Q2, dl = inst[r["idx"]]
        Js = [to_choi(o, d) for o in (P1, Q1, P2, Q2)]
        a1 = fhat_all(Js[0], Js[1], d, d); a2 = fhat_all(Js[2], Js[3], d, d)
        checks.append(dict(idx=r["idx"], family=f, F1=a1, F2=a2, lhs_sdp=abs(a1["sdp"] - a2["sdp"]),
                           lhs_arn=r["lhs"], rhs_thm104b=r["rhs_thm104b"], rhs_candidate=r["rhs_candidate"],
                           rhs_cor107=r["rhs_cor107"]))
    counterexamples = []
    for v in v_cand + v_104 + v_107:
        f, d, P1, Q1, P2, Q2, dl = inst[v["idx"]] if v["family"] != "jones_ex10.6_CPmaps" else (None,) * 7
        if f is None:
            continue
        Js = [to_choi(o, d) for o in (P1, Q1, P2, Q2)]
        counterexamples.append(dict(row={k: v[k] for k in v if k != "identity_errs"},
                                    choi=[[J.real.tolist(), J.imag.tolist()] for J in Js]))

    summary = dict(seed=SEED, n_instances=len(rows), n_same_support=int(sum(r["same_support"] for r in rows)),
                   tolerances=dict(CE_TOL=CE_TOL, ABS_TOL=ABS_TOL, SV_TOL=SV_TOL),
                   candidate=s_cand, thm104b=s_104, cor107=s_107, lemma105=lem, identities=ident,
                   thm46_same_sum=same, per_family=fam, sdp_crosschecks=checks,
                   n_counterexamples=len(counterexamples), runtime_s=time.time() - t0)
    json.dump(dict(summary=summary, rows=[{k: v for k, v in r.items()} for r in rows], same_sum=ss,
                   counterexamples=counterexamples),
              open(os.path.join(HERE, "results/e3.json"), "w"), indent=1,
              default=lambda o: float(o) if np.isscalar(o) else str(o))
    print(json.dumps({k: summary[k] for k in summary if k not in ("per_family", "sdp_crosschecks")}, indent=1, default=float))
    print(json.dumps(fam, indent=1, default=float))
    for c in checks:
        print(c["family"], "lhs_arn %.3e lhs_sdp %.3e  spreads %.1e %.1e  rhs104b %.3e" % (c["lhs_arn"], c["lhs_sdp"], c["F1"]["spread"], c["F2"]["spread"], c["rhs_thm104b"]))
    return rows


def adversarial(n_starts=24, seed=SEED + 1):
    """Nelder-Mead maximisation of LHS/RHS over channel quadruples (qubit, Kraus ranks rP,rQ),
    Phi2/Psi2 = TP-renormalised Kraus perturbations of Phi1/Psi1 of size 10^s (s optimised)."""
    from scipy.optimize import minimize
    rng_a = np.random.default_rng(seed)
    d = 2
    results = {}
    for key in ("rhs_thm104b", "rhs_candidate", "rhs_cor107"):
        best = []
        for st in range(n_starts):
            rP, rQ = [(1, 1), (1, 2), (2, 2), (2, 4), (4, 4), (1, 4)][st % 6]
            nP, nQ = 2 * rP * d * d, 2 * rQ * d * d

            def unpack(x):
                def ks(v, r):
                    M = (v[:r * d * d] + 1j * v[r * d * d:]).reshape(r * d, d)
                    return [M[k * d:(k + 1) * d] for k in range(r)]
                o = 0
                KP = tp_renormalize(ks(x[o:o + nP], rP)); o += nP
                KQ = tp_renormalize(ks(x[o:o + nQ], rQ)); o += nQ
                DP = ks(x[o:o + nP], rP); o += nP
                DQ = ks(x[o:o + nQ], rQ); o += nQ
                sc = 10 ** np.clip(x[o], -6, 0)
                KP2 = tp_renormalize([K + sc * D for K, D in zip(KP, DP)])
                KQ2 = tp_renormalize([K + sc * D for K, D in zip(KQ, DQ)])
                return [choi_from_kraus(k) for k in (KP, KQ, KP2, KQ2)]

            def neg(x):
                try:
                    r = analyse(*unpack(x), d, d)
                except Exception:
                    return 0.0
                if not np.isfinite(r[key]) or r[key] <= 0:
                    return 0.0
                return -r["lhs"] / r[key]

            x0 = np.concatenate([rng_a.normal(size=2 * nP + 2 * nQ), [rng_a.uniform(-4, -0.5)]])
            res = minimize(neg, x0, method="Nelder-Mead", options=dict(maxiter=1500, xatol=1e-6, fatol=1e-9))
            r = analyse(*unpack(res.x), d, d)
            best.append(dict(ratio=-res.fun, ranks=(rP, rQ), lhs=r["lhs"], rhs=r[key], delta=r["delta"], tau=r["tau"],
                             C1mC2=r["C1mC2"], same_support=r["same_support"], log10_scale=float(np.clip(res.x[-1], -6, 0)),
                             choi=[J.tolist() for J in unpack(res.x)] if -res.fun > 1 + CE_TOL else None))
        best.sort(key=lambda b: -b["ratio"])
        results[key] = dict(max_ratio=best[0]["ratio"], top=[{k: v for k, v in b.items() if k != "choi"} for b in best[:5]],
                            violations=[b for b in best if b["ratio"] > 1 + CE_TOL and b["lhs"] - b["rhs"] > ABS_TOL])
    return results


if __name__ == "__main__":
    main()
    t0 = time.time()
    adv = adversarial()
    d = json.load(open(os.path.join(HERE, "results/e3.json")))
    d["summary"]["adversarial"] = adv
    d["summary"]["adversarial_runtime_s"] = time.time() - t0
    json.dump(d, open(os.path.join(HERE, "results/e3.json"), "w"), indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
    print(json.dumps({k: dict(max_ratio=v["max_ratio"], n_viol=len(v["violations"]), top=v["top"][:2]) for k, v in adv.items()}, indent=1, default=float))
