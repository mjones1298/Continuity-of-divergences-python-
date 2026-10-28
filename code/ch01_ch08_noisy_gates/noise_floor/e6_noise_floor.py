"""E6': near-unitary gates with a noise floor (headline experiment).

Pairs (N_p, N_{p,theta}) with N_p = Noise_p o U and N_{p,theta} = Noise_p o (e^{-i theta H} U).
Computes: Fhat (ARN; SDP + mp-regularisation cross-checks on a subgrid), Qhat_alpha, diamond
distance, lambda_min(J(Omega)), nu(Omega)^2 (optimised upper value + SDP-relaxation lower value
+ Lemma 8.2 bounds), Choi-state and Uhlmann channel fidelity, BS, Umegaki (qubit), hockey-stick.
Fits 1 - Fhat = L theta^beta, data collapse vs theta/sqrt(p), local sensitivities vs 1/p,
v5 Thm 12.4(a) bound comparison, referee checks, noise models, and a two-qubit CNOT.
"""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # qlib.py in code/common
from qlib import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 6
rng = np.random.default_rng(SEED)
NHAT = (X2 + Y2 + Z2) / np.sqrt(3)


# ----------------------------------------------------------------------------- noise models (Choi level)
def noisy_gate_choi(U, model, p, d):
    JU = choi_unitary(U)
    if model == "depol":
        return choi_depolarize_after(JU, p, d, d)
    if model == "depol_local2":  # D_p (x) D_p after a 2-qubit gate
        Ks = [np.kron(A, B) for A in kraus_depol(p) for B in kraus_depol(p)]
        return choi_from_kraus([K @ U for K in Ks])
    if model == "gad":           # generalised amplitude damping, gamma = p, thermal N = 0.05
        return choi_from_kraus([K @ U for K in kraus_gad(p, 0.05)])
    if model == "gad_lowT":      # gamma = 1e-2 fixed, thermal population N = p
        return choi_from_kraus([K @ U for K in kraus_gad(1e-2, p)])
    if model == "t1t2":          # zero-temperature AD (gamma=p) followed by dephasing (p)
        return choi_from_kraus([K @ U for K in compose_kraus(kraus_dephase(p), kraus_ampdamp(p))])
    if model == "t1":
        return choi_from_kraus([K @ U for K in kraus_ampdamp(p)])
    if model == "pauli":         # unequal Pauli rates (px,py,pz) = p (0.1, 0.2, 0.7)
        return choi_from_kraus([K @ U for K in kraus_pauli(0.1 * p, 0.2 * p, 0.7 * p)])
    raise ValueError(model)


def sweep(U, Herr, model, p, thetas, d, heavy=True, sdp_every=3, mp_every=10, nu=True, umegaki=False):
    rows = []
    J0 = noisy_gate_choi(U, model, p, d)
    for k, th in enumerate(thetas):
        Ut = sla.expm(-1j * th * Herr) @ U
        J1 = noisy_gate_choi(Ut, model, p, d)
        F = fhat_arn(J0, J1, d, d)
        r = dict(model=model, p=p, theta=float(th), F=F,
                 Q={a: qhat_alpha_arn(J1, J0, a, d, d) for a in (0.25, 0.5, 0.75)},
                 lam_min_JOm=float(np.linalg.eigvalsh(J0 + J1).min()),
                 lam_min_J0=float(np.linalg.eigvalsh(J0).min()),
                 choi_state_fid=choi_state_fid(J0, J1, d))
        if heavy:
            r["diamond"] = diamond_sdp(J0 - J1, d, d)
            r["hockey"] = {g: hockey_sdp(J1, J0, g, d, d) for g in (1.0, 1.05, 2.0)}
            r["uhlmann_channel"] = uhlmann_channel_sdp(J0, J1, d, d) if d == 2 else None  # d=4 SDP ~90 s/point
            r["D_BS"] = dbs_arn(J1, J0, d, d)
            if umegaki:
                r["D_umegaki"] = umegaki_channel(J1, J0, d, d, rng, n_starts=3)
        if heavy and k % sdp_every == 0:
            r["F_sdp"] = fhat_sdp(J0, J1, d, d)[0]
        if heavy and k % mp_every == 0:
            r["F_mpreg"] = fhat_reg_mp(J0, J1, d, d)[0]
        if nu and p > 0:
            JOm = J0 + J1
            nb = nu_bounds(JOm, d, d, rng, n_restarts=8 if d == 2 else 4)
            r["nu"] = nb
            # v5 Thm 12.4(a) at the pair (N_p, N_p) [J(Omega)=2J(N_p) invertible] perturbed to (N_p, N_{p,theta}):
            #   |dF| <= (11/4) ||Omega_bar^*(1)|| nu(Omega)^-1 d_cb^1/2,  Omega_bar = (2N_p + N_p + N_{p,th})/2 -> ||.||=2
            if k == 0:
                nb0 = nu_bounds(2 * J0, d, d, rng, n_restarts=8 if d == 2 else 4)
                r["nu_base"] = nb0
            if heavy:
                r["thm124a_rhs_nu_base"] = None  # filled below
        rows.append(r)
    if nu and p > 0 and heavy:
        nu_base = np.sqrt(rows[0]["nu_base"]["nu2_opt"])
        for r in rows:
            r["nu_base_value"] = float(nu_base)
            r["thm124a_rhs_nu_base"] = 11 / 4 * 2 / nu_base * np.sqrt(r["diamond"])
            r["thm124a_rhs_nu_pair"] = 11 / 4 * 2 / np.sqrt(r["nu"]["nu2_opt"]) * np.sqrt(r["diamond"])
            r["thm124a_ratio"] = (1 - r["F"]) / r["thm124a_rhs_nu_base"]
    return rows


def fit_modulus(rows, lo=1e-9, hi=1e-2):
    th = np.array([r["theta"] for r in rows]); y = np.array([1 - r["F"] for r in rows])
    m = (y > lo) & (y < hi)
    if m.sum() < 3:
        return dict(beta=None, L=None, n=int(m.sum()), max_1mF=float(y.max()), min_1mF=float(y.min()))
    beta, logL = np.polyfit(np.log(th[m]), np.log(y[m]), 1)
    return dict(beta=float(beta), L=float(np.exp(logL)), n=int(m.sum()))


def theta_at(rows, level=0.9):
    th = np.array([r["theta"] for r in rows]); F = np.array([r["F"] for r in rows])
    idx = np.where(F < level)[0]
    if len(idx) == 0 or idx[0] == 0:
        return None
    i = idx[0]
    return float(np.exp(np.interp(level, [F[i], F[i - 1]], [np.log(th[i]), np.log(th[i - 1])])))


def ser(o):
    if isinstance(o, dict):
        return {str(k): ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [ser(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def cnot_stage():
    t0 = time.time()
    ZZ = np.kron(Z2, Z2)
    cn_rows, cn_fit = {}, {}
    thetas_cn = np.logspace(-6, 0, 13)
    for model in ("depol", "depol_local2"):
        for p in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
            rows = sweep(CNOT, ZZ, model, p, thetas_cn, 4, heavy=True, nu=(p > 0), sdp_every=4, mp_every=100)
            cn_rows[(model, p)] = rows
            f = fit_modulus(rows)
            J0 = noisy_gate_choi(CNOT, model, p, 4)
            f.update(lam_min_J0=float(np.linalg.eigvalsh(J0).min()), F_smallest_theta=rows[0]["F"],
                     theta_F0p9=theta_at(rows, 0.9), diamond_smallest_theta=rows[0]["diamond"],
                     choi_state_fid_smallest_theta=rows[0]["choi_state_fid"],
                     max_abs_arn_minus_sdp=max(abs(r["F"] - r["F_sdp"]) for r in rows if "F_sdp" in r and np.isfinite(r["F_sdp"])),
                     sdp_solvers_ok=all(np.isfinite(r["F_sdp"]) for r in rows if "F_sdp" in r))
            if p > 0:
                f["L_times_p"] = f["L"] * p if f.get("L") else None
                f["theta_F0p9_over_sqrt_p"] = f["theta_F0p9"] / np.sqrt(p) if f["theta_F0p9"] else None
                f["nu2_base"] = rows[0]["nu_base"]["nu2_opt"]
                f["nu2_base_sdp_lower"] = rows[0]["nu_base"]["nu2_sdp_lower"]
                f["nu2_base_over_p"] = f["nu2_base"] / p
                f["lemma82"] = [rows[0]["nu_base"]["lemma82_lower"], rows[0]["nu_base"]["lemma82_upper"]]
                f["thm124a_max_ratio"] = max(r["thm124a_ratio"] for r in rows)
            cn_fit[f"{model}_p{p:g}"] = f
            print("cnot", model, p, time.time() - t0, flush=True)
    D = dict(summary=dict(cnot=ser(cn_fit)), rows_cnot=ser({f"{m}_p{p:g}": r for (m, p), r in cn_rows.items()}))
    json.dump(D, open(os.path.join(HERE, "results/e6_cnot.json"), "w"), indent=1)
    print(json.dumps(ser(cn_fit), indent=1))


def main(stage="qubit"):
    t0 = time.time()
    out = {}
    main_rows, nm_rows = {}, {}
    thetas = np.logspace(-6, 0, 37)
    ps = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # ------------------------------------------------ 1. qubit gates, depolarising floor
    gates = dict(H=H2, X=X2) if stage == "qubit" else {}
    for gname, U in gates.items():
        for p in ps:
            main_rows[(gname, p)] = sweep(U, Z2 if gname == "H" else NHAT, "depol", p, thetas, 2,
                                          heavy=True, nu=(gname == "H"), umegaki=False)
        print("gate", gname, "done", time.time() - t0, flush=True)

    fits = {}
    for (g, p), rows in main_rows.items():
        f = fit_modulus(rows)
        f["theta_F0.9"] = theta_at(rows, 0.9)
        f["theta_F0.5"] = theta_at(rows, 0.5)
        if p > 0:
            f["L_times_p"] = f["L"] * p if f["L"] else None
            f["theta_F0.9_over_sqrt_p"] = f["theta_F0.9"] / np.sqrt(p) if f["theta_F0.9"] else None
            f["lam_min_J0"] = rows[0]["lam_min_J0"]
        else:
            f["F_at_smallest_theta"] = rows[0]["F"]
            f["choi_state_fid_at_smallest_theta"] = rows[0]["choi_state_fid"]
            f["diamond_at_smallest_theta"] = rows[0]["diamond"]
        f["max_abs_arn_minus_sdp"] = max(abs(r["F"] - r["F_sdp"]) for r in rows if "F_sdp" in r)
        f["max_abs_arn_minus_mpreg"] = max(abs(r["F"] - r["F_mpreg"]) for r in rows if "F_mpreg" in r)
        fits[f"{g}_p{p:g}"] = f
    out["depol_fits"] = fits

    # nu checks and Thm 12.4(a) comparison (gate H)
    nu_tab, thm = {}, {}
    for p in (ps[1:] if stage == "qubit" else []):
        rows = main_rows[("H", p)]
        nb0 = rows[0]["nu_base"]
        nu_tab[f"{p:g}"] = dict(
            nu2_base_opt=nb0["nu2_opt"], nu2_base_sdp_lower=nb0["nu2_sdp_lower"],
            lam_min_pos_base=nb0["lam_min_pos"], lemma82_lower=nb0["lemma82_lower"],
            nu2_pair_small_theta=rows[0]["nu"]["nu2_opt"], nu2_pair_theta1=rows[-1]["nu"]["nu2_opt"],
            nu2_over_p=nb0["nu2_opt"] / p,
            lemma82_ok=all(r["nu"]["lemma82_lower"] - 1e-12 <= r["nu"]["nu2_opt"] <= r["nu"]["lemma82_upper"] + 1e-12 for r in rows),
            opt_minus_sdp_lower_max=max(r["nu"]["nu2_opt"] - r["nu"]["nu2_sdp_lower"] for r in rows))
        # tightness of Thm 12.4(a) : ratio (1-F)/RHS over theta; and where RHS < 1
        rat = np.array([r["thm124a_ratio"] for r in rows])
        rhs = np.array([r["thm124a_rhs_nu_base"] for r in rows])
        dcb = np.array([r["diamond"] for r in rows])
        th = np.array([r["theta"] for r in rows])
        nonvac = rhs < 1
        y = 1 - np.array([r["F"] for r in rows])
        m = (y > 1e-9) & (y < 1e-2)
        slope_vs_dcb = float(np.polyfit(np.log(dcb[m]), np.log(y[m]), 1)[0])
        thm[f"{p:g}"] = dict(max_ratio=float(rat.max()), ratio_at_largest_nonvacuous=float(rat[nonvac].max()) if nonvac.any() else None,
                             dcb_max_nonvacuous=float(dcb[nonvac].max()) if nonvac.any() else None,
                             dcb_nonvac_over_p=float(dcb[nonvac].max() / p) if nonvac.any() else None,
                             slope_log1mF_vs_logdcb=slope_vs_dcb,
                             const_1mF_over_dcb2_times_p=float(np.median(y[m] / dcb[m] ** 2 * p)))
    out["nu"] = nu_tab
    out["thm124a"] = thm

    # ------------------------------------------------ 2. local sensitivities vs 1/p (qubit H, H_err = Z)
    sens = []
    for p in ([] if stage != "qubit" else [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]):
        th = 1e-2 * np.sqrt(p)
        J0 = noisy_gate_choi(H2, "depol", p, 2)
        J1 = noisy_gate_choi(sla.expm(-1j * th * Z2) @ H2, "depol", p, 2)
        h = 1e-7
        Dj = (noisy_gate_choi(sla.expm(-1j * h * Z2) @ H2, "depol", p, 2) - noisy_gate_choi(sla.expm(1j * h * Z2) @ H2, "depol", p, 2)) / (2 * h)
        s = dict(p=p, theta=th,
                 geo=(1 - fhat_arn(J0, J1, 2, 2)) / th ** 2,
                 geo_sdp=(1 - fhat_sdp(J0, J1, 2, 2)[0]) / th ** 2,
                 BS=dbs_arn(J1, J0, 2, 2) / th ** 2,
                 umegaki=umegaki_channel(J1, J0, 2, 2, rng, n_starts=2) / th ** 2,
                 uhlmann=(1 - uhlmann_channel_sdp(J0, J1, 2, 2)) / th ** 2,
                 choi_state=(1 - choi_state_fid(J0, J1, 2)) / th ** 2,
                 hockey1_over_theta=hockey_sdp(J1, J0, 1.0, 2, 2) / th,
                 hockey105_over_theta=hockey_sdp(J1, J0, 1.05, 2, 2) / th,
                 hockey105_at_theta_0p1=hockey_sdp(noisy_gate_choi(sla.expm(-1j * 0.1 * Z2) @ H2, "depol", p, 2), J0, 1.05, 2, 2),
                 I_RLD=rld_fisher_channel(J0, Dj, 2, 2))
        sens.append(s)
    slopes = {}
    if stage != "qubit":
        sens = None
    x = np.log([1 / s["p"] for s in sens]) if sens else None
    for key in (("geo", "BS", "umegaki", "uhlmann", "choi_state", "hockey1_over_theta", "hockey105_over_theta", "I_RLD") if sens else ()):
        y = np.log(np.maximum(1e-300, [s[key] for s in sens]))
        slopes[key] = float(np.polyfit(x, y, 1)[0])
    # Umegaki vs log(1/p): linear fit in log(1/p)
    um_lin = np.polyfit(x, [s["umegaki"] for s in sens], 1) if sens else [0, 0]
    if sens: out["sensitivity"] = dict(rows=sens, loglog_slopes_vs_1_over_p=slopes,
                              umegaki_linear_in_log1p=dict(slope=float(um_lin[0]), intercept=float(um_lin[1])),
                              geo_times_p=[s["geo"] * s["p"] for s in sens], BS_times_p=[s["BS"] * s["p"] for s in sens],
                              geo_over_IRLD=[s["geo"] / s["I_RLD"] for s in sens], BS_over_IRLD=[s["BS"] / s["I_RLD"] for s in sens])
    print("sensitivity done", time.time() - t0, flush=True)

    # ------------------------------------------------ 3. referee checks (their exact setup: Rx(pi/2) gate, over-rotation delta)
    def rx(t):
        return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])
    ref = {}
    for p in ([1e-2, 1e-3, 1e-4, 1e-5] if stage == "qubit" else []):
        J0 = choi_depolarize_after(choi_unitary(rx(np.pi / 2)), p, 2, 2)
        ds = np.logspace(np.log10(np.sqrt(p)) - 1, np.log10(np.sqrt(p)) + 1, 400)
        Fs = np.array([fhat_arn(J0, choi_depolarize_after(choi_unitary(rx(np.pi / 2 + dl)), p, 2, 2), 2, 2) for dl in ds])
        i = np.where(Fs < 0.9)[0][0]
        dstar = float(np.exp(np.interp(0.9, [Fs[i], Fs[i - 1]], [np.log(ds[i]), np.log(ds[i - 1])])))
        J1 = choi_depolarize_after(choi_unitary(rx(np.pi / 2 + dstar)), p, 2, 2)
        ref[f"{p:g}"] = dict(delta_star_F0p9=dstar, delta_star_over_sqrt_p=dstar / np.sqrt(p),
                             diamond_at_delta_star=diamond_sdp(J0 - J1, 2, 2),
                             F_sdp_at_delta_star=fhat_sdp(J0, J1, 2, 2)[0],
                             closed_form_delta_star_over_sqrt_p=None)
    if ref: out["referee_delta_star"] = ref

    # ------------------------------------------------ 4. noise models (gate H, H_err = n.sigma)
    models = ["gad", "gad_lowT", "t1t2", "t1", "pauli"] if stage == "models" else []
    nm_rows, nm_fit = {}, {}
    thetas_nm = np.logspace(-6, 0, 25)
    for model in models:
        for p in [1e-4, 1e-3, 1e-2, 1e-1]:
            rows = sweep(H2, NHAT, model, p, thetas_nm, 2, heavy=True, nu=False, sdp_every=4, mp_every=6)
            nm_rows[(model, p)] = rows
            f = fit_modulus(rows)
            J0 = noisy_gate_choi(H2, model, p, 2)
            f.update(lam_min_J0=float(np.linalg.eigvalsh(J0).min()), F_smallest_theta=rows[0]["F"],
                     theta_F0p9=theta_at(rows, 0.9), diamond_smallest_theta=rows[0]["diamond"],
                     max_abs_arn_minus_sdp=max(abs(r["F"] - r["F_sdp"]) for r in rows if "F_sdp" in r),
                     max_abs_arn_minus_mpreg=max(abs(r["F"] - r["F_mpreg"]) for r in rows if "F_mpreg" in r),
                     sdp_values_smallest_theta=rows[0].get("F_sdp"))
            if f.get("L") and f["lam_min_J0"] > 1e-14:
                f["L_times_lam_min_J0"] = f["L"] * f["lam_min_J0"]
            nm_fit[f"{model}_p{p:g}"] = f
        print("model", model, time.time() - t0, flush=True)
    # within-channel jumps (zero temperature): rotations before the noise, three axes, 3 gammas
    jumps = []
    for model in (("t1", "t1t2") if stage == "models" else ()):
        for g in (1e-3, 0.1, 0.5):
            for an, Ha in (("X", X2), ("Y", Y2), ("Z", Z2), ("n", NHAT)):
                for th in (1e-8, 1e-5, 1e-2):
                    J0 = noisy_gate_choi(I2, model, g, 2)
                    J1 = noisy_gate_choi(sla.expm(-1j * th * Ha), model, g, 2)
                    jumps.append(dict(model=model, gamma=g, axis=an, theta=th, F_arn=fhat_arn(J0, J1, 2, 2),
                                      F_mpreg=fhat_reg_mp(J0, J1, 2, 2)[0], F_sdp=fhat_sdp(J0, J1, 2, 2)[0],
                                      diamond=diamond_sdp(J0 - J1, 2, 2),
                                      zero_eigs_J0=int(np.sum(np.linalg.eigvalsh(J0) < 1e-12))))
    if stage == "models":
        out["noise_models"] = nm_fit
        out["zero_temperature_within_channel"] = jumps

    out["runtime_s"] = time.time() - t0

    json.dump(ser(dict(summary=out,
                       rows_depol={f"{g}_p{p:g}": r for (g, p), r in main_rows.items()},
                       rows_models={f"{m}_p{p:g}": r for (m, p), r in nm_rows.items()})),
              open(os.path.join(HERE, f"results/e6_{stage}.json"), "w"), indent=1)
    print(json.dumps(ser({k: v for k, v in out.items() if k not in ("zero_temperature_within_channel",)}), indent=1))
    zt = {}
    for j in jumps:
        zt.setdefault((j["model"], j["axis"]), []).append((j["gamma"], j["theta"], round(j["F_arn"], 6), round(j["F_mpreg"], 6), round(j["F_sdp"], 6), "%.1e" % j["diamond"]))
    for k, v in zt.items():
        print(k, v)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "qubit"
    if stage == "cnot":
        cnot_stage()
    elif stage == "merge":
        D = {"summary": {}}
        for st in ("qubit", "models", "cnot"):
            E = json.load(open(os.path.join(HERE, f"results/e6_{st}.json")))
            D["summary"].update(E["summary"])
            for k, v in E.items():
                if k != "summary" and v:
                    D[k] = v
        json.dump(D, open(os.path.join(HERE, "results/e6.json"), "w"), indent=1)
        print("merged")
    else:
        main(stage)
