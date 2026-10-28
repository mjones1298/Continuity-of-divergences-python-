"""E7: family comparison of noise-floor laws near noisy gates N_p = D_p o U  (U = Hadamard),
miscalibrated by delta (U_delta = e^{-i delta Z} U), plus paper-2 'fragile' example and a
finite-shot tomography-cost experiment.  Produces the 4-panel main figure.

Families:
 (i)  reverse test: geometric fidelity Fhat (1-Fhat), geometric Chernoff C-hat, BS
 (ii) Umegaki channel relative entropy D(N_{p,d} || N_p)  (input-optimised)
 (iii) testing region: channel hockey-stick E_gamma (SDP), gamma in {1, 1.05, 2}; HT squared Hellinger
 (iv) Uhlmann channel fidelity (SDP)
Local sensitivity = divergence/delta^2 (hockey: /delta) at delta = 1e-2 sqrt(p).
"""
import sys, os, json, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
from qlib import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 7
rng = np.random.default_rng(SEED)


def gate_pair(p, dl):
    J0 = choi_depolarize_after(choi_unitary(H2), p, 2, 2)
    J1 = choi_depolarize_after(choi_unitary(sla.expm(-1j * dl * Z2) @ H2), p, 2, 2)
    return J0, J1


# ----------------------------------------------------------------------------- (b) slopes per family
def family_sensitivity(ps):
    rows = []
    for p in ps:
        dl = 1e-2 * np.sqrt(p)
        J0, J1 = gate_pair(p, dl)
        C, s_star = chernoff_geo(J1, J0, 2, 2)
        r = dict(p=p, delta=dl,
                 geo_1mF=(1 - fhat_arn(J0, J1, 2, 2)) / dl ** 2,
                 geo_1mF_sdp=(1 - fhat_sdp(J0, J1, 2, 2)[0]) / dl ** 2,
                 chernoff=C / dl ** 2, chernoff_s=s_star,
                 BS=dbs_arn(J1, J0, 2, 2) / dl ** 2,
                 umegaki=umegaki_channel(J1, J0, 2, 2, rng, n_starts=3) / dl ** 2,
                 umegaki_choi_state=rel_ent(J1 / 2, J0 / 2) / dl ** 2,
                 ht_hellinger=ht_hellinger_channel(J1, J0, 2, 2, rng, n_starts=1) / dl ** 2,
                 ht_hellinger_choi_state=ht_hellinger_state(J1 / 2, J0 / 2) / dl ** 2,
                 uhlmann_1mF=(1 - uhlmann_channel_sdp(J0, J1, 2, 2)) / dl ** 2,
                 hockey={g: hockey_sdp(J1, J0, g, 2, 2) / dl for g in (1.0, 1.05, 2.0)})
        rows.append(r)
        print("p", p, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "hockey"}, flush=True)
    x = np.log([1 / r["p"] for r in rows])
    slopes = {}
    for k in ("geo_1mF", "chernoff", "BS", "umegaki", "umegaki_choi_state", "ht_hellinger", "ht_hellinger_choi_state", "uhlmann_1mF"):
        slopes[k] = float(np.polyfit(x, np.log([r[k] for r in rows]), 1)[0])
    for g in (1.0, 1.05, 2.0):
        y = np.array([r["hockey"][g] for r in rows])
        slopes[f"hockey_{g:g}"] = float(np.polyfit(x, np.log(np.maximum(y, 1e-300)), 1)[0]) if np.all(y > 1e-12) else None
    um = np.polyfit(x, [r["umegaki"] for r in rows], 1)
    ht = np.polyfit(x, [r["ht_hellinger"] for r in rows], 1)
    return rows, slopes, dict(umegaki_vs_ln1p=dict(slope=float(um[0]), intercept=float(um[1])),
                              ht_hellinger_vs_ln1p=dict(slope=float(ht[0]), intercept=float(ht[1])))


# ----------------------------------------------------------------------------- (a) paper-2 fragile example
def fragile_example(ts):
    rows = []
    e = np.eye(4)
    phi = 0.5 * (np.outer(e[0], e[0]) + np.outer(e[1], e[1]))
    for t in ts:
        v1 = (e[0] + np.sqrt(t) * e[2]) / np.sqrt(1 + t); v2 = (e[0] - np.sqrt(t) * e[2]) / np.sqrt(1 + t)
        w1 = (e[1] + np.sqrt(t) * e[3]) / np.sqrt(1 + t); w2 = (e[1] - np.sqrt(t) * e[3]) / np.sqrt(1 + t)
        phit = 0.5 * (np.outer(v1, v1) + np.outer(w1, w1)); psit = 0.5 * (np.outer(v2, v2) + np.outer(w2, w2))
        dX = 2  # replacement channels from a qubit input to C^4
        JP, JQ = choi_replacement(phit.astype(complex), dX), choi_replacement(psit.astype(complex), dX)
        C, _ = chernoff_geo(JP, JQ, 4, dX)
        F = fhat_arn(JP, JQ, 4, dX)
        Fsdp = fhat_sdp(JP, JQ, 4, dX)[0]
        xi, s = xi_qc(phit, psit)
        rows.append(dict(t=t, trace_dist_to_phi=0.5 * np.sum(np.abs(np.linalg.eigvalsh(phit - phi))),
                         diamond_Phi_t_vs_Phi=diamond_sdp(JP - choi_replacement(phi.astype(complex), dX), 4, dX),
                         Fhat=F, Fhat_sdp=Fsdp, C_geo=C, fidelity_converse=(np.inf if F <= 1e-12 else -2 * np.log(F)),
                         xi_QC=xi, xi_QC_s=s, xi_QC_closed=2 * np.log((1 + t) / (1 - t)),
                         tr_sqrt_sqrt=float(np.real(np.trace(psd_sqrt(phit) @ psd_sqrt(psit)))),
                         tr_sqrt_sqrt_closed=((1 - t) / (1 + t)) ** 2,
                         uhlmann_root_fid=uhlmann_fid(phit, psit),
                         hockey1=hockey_state(phit, psit, 1.0)))
    return rows


# ----------------------------------------------------------------------------- (d) tomography cost
PA = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}


def eigbasis(s):
    w, V = np.linalg.eigh(PA[s])
    return V[:, ::-1]


def tomo_choi_state(rho, N, rng):
    """Pauli tomography of a 2-qubit state (9 local Pauli settings, N shots each), linear inversion,
    then projection to the nearest density matrix (eigenvalue clipping + renormalisation)."""
    ex = {}
    sa = np.array([1, 1, -1, -1]); sb = np.array([1, -1, 1, -1])
    for a, b in itertools.product("XYZ", "XYZ"):
        U = np.kron(eigbasis(a), eigbasis(b))
        pr = np.clip(np.real(np.diag(U.conj().T @ rho @ U)), 0, None); pr /= pr.sum()
        f = rng.multinomial(N, pr) / N
        for (pa, pb, s) in ((a, b, sa * sb), (a, "I", sa), ("I", b, sb)):
            ex.setdefault(pa + pb, []).append(f @ s)
    R = np.eye(4, dtype=complex) / 4
    for k, v in ex.items():
        R = R + np.mean(v) * np.kron(PA[k[0]], PA[k[1]]) / 4
    w, V = np.linalg.eigh(herm(R)); w = np.clip(w, 0, None); w /= w.sum()
    return herm((V * w) @ V.conj().T)


def tomography_cost(ps, Ns, n_trials=60, tol_rel=0.1):
    """Two independent tomographies of the pair (N_p, N_{p,delta}), delta = sqrt(p) (true Fhat ~ 0.7).
    Plug-in estimates of Fhat (ARN), Choi-state Uhlmann fidelity, Choi-state hockey-stick E_1.
    Cost N* = smallest N (shots/setting) with >= 90% of runs within tol_rel (relative) of the truth.
    Note: the Choi matrix here is 2 x (state of Y (x) X); estimates are renormalised to TP-free Choi states."""
    out = {}
    for p in ps:
        dl = np.sqrt(p)
        J0, J1 = gate_pair(p, dl)
        tF = fhat_arn(J0, J1, 2, 2); tU = uhlmann_fid(J0 / 2, J1 / 2); tH = hockey_state(J0 / 2, J1 / 2, 1.0)
        res = dict(true=dict(F=tF, uhl=tU, hockey=tH), curves={})
        for N in Ns:
            F, U, H = [], [], []
            for _ in range(n_trials):
                a = 2 * tomo_choi_state(J0 / 2, N, rng); b = 2 * tomo_choi_state(J1 / 2, N, rng)
                F.append(fhat_arn(a, b, 2, 2)); U.append(uhlmann_fid(a / 2, b / 2)); H.append(hockey_state(a / 2, b / 2, 1.0))
            F, U, H = map(np.array, (F, U, H))
            res["curves"][N] = dict(F_med=float(np.median(F)), F_q90_abs_err=float(np.quantile(np.abs(F - tF), 0.9)),
                                    U_q90_abs_err=float(np.quantile(np.abs(U - tU), 0.9)),
                                    H_q90_abs_err=float(np.quantile(np.abs(H - tH), 0.9)),
                                    F_ok=float(np.mean(np.abs(F - tF) <= tol_rel * tF)),
                                    U_ok=float(np.mean(np.abs(U - tU) <= tol_rel * tU)),
                                    H_ok=float(np.mean(np.abs(H - tH) <= tol_rel * tH)))
        for key in ("F", "U", "H"):
            ok = [N for N in Ns if res["curves"][N][f"{key}_ok"] >= 0.9]
            # require it to stay ok for all larger N
            Nstar = None
            for N in Ns:
                if all(res["curves"][M][f"{key}_ok"] >= 0.9 for M in Ns if M >= N):
                    Nstar = N
                    break
            res[f"Nstar_{key}"] = Nstar
        out[p] = res
        print("tomo p", p, {k: res[k] for k in res if k.startswith("Nstar")}, flush=True)
    return out


def main():
    t0 = time.time()
    ps = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    sens, slopes, lin = family_sensitivity(ps)
    print("slopes", slopes, lin, time.time() - t0, flush=True)
    frag = fragile_example(np.logspace(-6, -1, 11))
    print("fragile done", time.time() - t0, flush=True)
    Ns = [int(10 ** k) for k in np.arange(2, 10.5, 0.5)]
    tomo = tomography_cost([1e-2, 1e-3, 1e-4], Ns, n_trials=60)
    print("tomo done", time.time() - t0, flush=True)
    fr = frag[len(frag) // 2]
    summary = dict(seed=SEED, sensitivity_rows=sens, loglog_slopes_vs_1_over_p=slopes, linear_in_ln_1_over_p=lin,
                   sensitivity_times_p=dict(geo=[r["geo_1mF"] * r["p"] for r in sens], chernoff=[r["chernoff"] * r["p"] for r in sens],
                                            BS=[r["BS"] * r["p"] for r in sens]),
                   fragile_rows=frag,
                   fragile_check=dict(max_Fhat=max(r["Fhat"] for r in frag), max_Fhat_sdp=max(r["Fhat_sdp"] for r in frag),
                                      all_C_inf=all(not np.isfinite(r["C_geo"]) for r in frag),
                                      max_rel_err_xi_vs_closed=max(abs(r["xi_QC"] / r["xi_QC_closed"] - 1) for r in frag),
                                      xi_over_4t=[r["xi_QC"] / (4 * r["t"]) for r in frag],
                                      diamond_over_sqrt_t=[r["diamond_Phi_t_vs_Phi"] / np.sqrt(r["t"]) for r in frag]),
                   tomography={str(p): dict(true=v["true"], Nstar_F=v["Nstar_F"], Nstar_uhl=v["Nstar_U"], Nstar_hockey=v["Nstar_H"],
                                            curves={str(N): c for N, c in v["curves"].items()}) for p, v in tomo.items()},
                   runtime_s=time.time() - t0)
    json.dump(summary, open(os.path.join(HERE, "results/e7.json"), "w"), indent=1,
              default=lambda o: float(o) if np.isscalar(o) else str(o))
    print(json.dumps({k: summary[k] for k in ("loglog_slopes_vs_1_over_p", "linear_in_ln_1_over_p", "fragile_check")}, indent=1,
                     default=lambda o: float(o) if np.isscalar(o) else str(o)))
    print({p: (v["Nstar_F"], v["Nstar_U"], v["Nstar_H"]) for p, v in tomo.items()})


if __name__ == "__main__":
    main()
