# Code and data for the numerical checks of the book

This directory holds the Python code behind the numbers, tables and figures of *Continuity, Testing Regions and Fragility of Quantum Channel Divergences*: the noisy-gate study of Chapters 1 and 8, the amplitude-damping Chernoff computations of Section 7.6.1, numerical verifications of results of Chapters 4 and 5, and independent re-computations by two checkers and a referee. Section, theorem, figure and table numbers below are those of the built book (`main.pdf`).

- [Installation](#installation)
- [Running the scripts](#running-the-scripts)
- [Smoke test](#smoke-test)
- [Layout](#layout)
- [Book claims and the scripts behind them](#book-claims-and-the-scripts-behind-them)
- [The folders in detail](#the-folders-in-detail)
- [Runtimes](#runtimes)
- [Random seeds](#random-seeds)
- [What is not here](#what-is-not-here)

## Installation

Python 3.13 (tested with 3.13.17) and the packages in `requirements.txt`:

| package | version | used for |
|---|---|---|
| numpy | 2.5.3 | everything |
| scipy | 1.18.1 | matrix functions, local optimization |
| cvxpy | 1.9.3 | semidefinite programs (diamond norms, fidelities, Zhou–Jiang, Fawzi–Fawzi, testers, confidence regions) |
| clarabel | 0.11.1 | default SDP solver (`solver='CLARABEL'`) |
| scs | 3.3.1 | cross-check and fallback SDP solver (`solver='SCS'`) |
| mpmath | 1.3.0 | 30–140-digit arithmetic |
| sympy | 1.14.0 | series expansions (`t4_hierarchy.py`, `crosscheck_theorems.py`, `b1_closed_forms.py`, `sym_incoh.py`) |
| matplotlib | 3.11.2 | figures |

```sh
python3.13 -m venv .venv && . .venv/bin/activate
python3 -m pip install -r code/requirements.txt
```

## Running the scripts

- Every script can be run from any working directory: `python3 code/<folder>/<script>.py [args]`. Scripts find the shared modules in `code/common/` and their data through paths relative to their own file.
- Results are written next to the script, as in the original working folders: to `results/` in the folder of the script (the early experiments also write PNG figures to `figures/`; the figure scripts write PDFs into `ch01_ch08_noisy_gates/figures/`). Running a script therefore **overwrites** the shipped result file with a freshly computed one. To keep the shipped files, run on a copy of `code/` (this is what the smoke test does).
- Conventions are the same in all scripts: J(Φ) = Σ_ij Φ(E_ij) ⊗ E_ij with the output first, row-major `vec`, F̂ = λ_min Tr_Y(J₁ # J₂) unsquared. Chapter 3 of the book lists where the book's chapters differ (e.g. the Choi ordering).
- The libraries in `checks/` are deliberately independent re-implementations (`checks/independent_C1_C7/qlib.py` is *not* `common/qlib.py`); do not put `code/common` on `PYTHONPATH` when running them.

## Smoke test

```sh
make code-test           # from the repository root; or: bash code/smoke_test.sh
bash code/smoke_test.sh -k proof_support    # only tests whose path contains "proof_support"
```

`smoke_test.sh` calls `smoke_test.py`, which copies `code/` to a temporary directory, runs one or more fast scripts from every folder there (from an unrelated working directory), and compares what they produce with the shipped JSON files and outputs or with the numbers printed in the book. It takes about 8 minutes on 2 cores (no single test over 2 minutes) and exits non-zero if any test fails. Its test list shows which numbers each test checks.

## Layout

```
code/
├── README.md  EXCLUDED.md  DATA_NOT_INCLUDED.md  requirements.txt  smoke_test.sh  smoke_test.py
├── audit_ch04/                  independent audit of Chapter 4: seven checkers (A–G), reports, findings and scripts
├── common/                     shared modules: qlib.py, glib.py, common.py, sharp_alpha.py, certlib.py
├── ch01_ch08_noisy_gates/      Chapters 1 and 8: noisy gates
│   ├── noise_floor/            early experiments E6–E8: noise-floor law, family comparison, gap, moduli
│   ├── amplitude_damping/      E2: reset channels with noise, amplitude damping, controls
│   ├── adaptive_qfi/           T1: asymptotic adaptive QFI F_∞ (Zhou–Jiang), local Stein bracket
│   ├── crossover/              T2, T2b: Stein-exponent bracket at finite δ, two-copy Fawzi–Fawzi bound
│   ├── multiqubit_window/      T3: multi-qubit windows, weight rule
│   ├── hierarchy/              T4: second-order coefficients and Fisher-information hierarchy
│   ├── proof_support/          numerical checks of the proofs of Chapter 8 (Theorems L1, B, W, Prop. L2, C, N…)
│   ├── first_order/            M1: first-order fragility and the Chernoff side
│   ├── biased_noise/           biased Pauli noise, one and two qubits
│   ├── certification/          simulated Pauli tomography and certified converses
│   └── figures/                scripts for Figures 1.1–1.6, 8.1, 8.2
├── ch04_continuity/            Chapter 4
│   ├── pair_bounds/            E1, E3, E4: Cor. 4.3.3, different-sum bounds of §4.10, fragility predictors
│   ├── dichotomy/              channel-valued discontinuity (§4.12) and rates at mutually singular pairs
│   └── v6_verification/        independent verification of the results new in the revised paper (§§4.4, 4.8, 4.10, 4.12)
├── ch05_shadows/hockey_lipschitz/   E5: diamond-Lipschitz continuity of channel hockey-stick divergences
├── ch07_chernoff/              Section 7.6.1: amplitude damping, exact adaptive errors and fidelities
└── checks/                     independent checks and referee re-computations
    ├── independent_C1_C7/      claims C1–C7 (continuity examples, noise floor, Thm 4.10.4, hockey stick)
    ├── independent_K1_K4/      claims K1–K4 (Choi-floor law, closed form, worst-case slope, Stein rate)
    ├── referee_letter/         the referee's re-computation of the Chapter 1/8 constants (closed forms, windows, F_∞, AD)
    └── referee_prereview/      the referee's prototypes on fragility, tomography cost and moduli
```

## Audit of Chapter 4

`audit_ch04/` holds an independent, computer-assisted audit of every result of Chapter 4 (*Continuity of certain quantum channel divergences*). Seven checkers (A–G) re-derived the proofs and tested every formula and constant numerically. They used 50–110-digit arithmetic near singular points and computed diamond and Schur-multiplier norms exactly by SDP. No statement was found false and no proof has a gap. The minor corrections are incorporated in the book. `audit_ch04/README.md` explains the layout. The findings are in `audit_ch04/AUDIT_FINDINGS.md`, and the per-checker reports are in `audit_ch04/audit-reports/`.

## Book claims and the scripts behind them

Paths are relative to `code/`; `nf` = `ch01_ch08_noisy_gates`. "Output" is the shipped file the numbers are read from (or "stdout" for scripts that print them). Runtime classes: **s** (under a minute), **min** (1–10 min), **long** (10–60 min), **heavy** (hours, or large memory); see [Runtimes](#runtimes).

### Chapter 1 (statements) and Chapter 8 (proofs and further results)

| Book item | What is computed | Script(s) | Output | Time |
|---|---|---|---|---|
| Fig. 1.1 | schematic of the three regimes (no data) | `nf/figures/make_fig0_schematic.py` | `fig0_schematic.pdf` | s |
| Fig. 1.2(a), §1.3, §8.7, Table 8.2 | amplitude damping 𝒜_γ vs 𝒜_γ′: F̂ = 0, converses infinite; ξ_prod, −2 ln f, −2 ln μ_* | `ch07_chernoff/ad_basic.py`, `ad_table.py`; checks: `nf/amplitude_damping/e2_reset_noise.py` (F̂ = 0 down to diamond distance 2e-6), `checks/referee_letter/check4_ad.py` | stdout; `nf/amplitude_damping/results/e2.json`; `checks/referee_letter/check4.out` | s |
| Fig. 1.2(b), Thm. "fragile" example (end of Ch. 8) | replacement channels: F̂ = 0, ξ_QC ≈ 4t | `nf/noise_floor/e7_family.py` (`fragile_rows`); figure `nf/figures/make_fig1_regimes.py` | `nf/noise_floor/results/e7.json` | long; figure s |
| §1.4, §8.3 (Thm. L1, Cor.) | Choi-floor law: log-slope d/(4λ) at Φ = Ad_P; ratios 1.0000000, 1.0000034, 0.9999816, 0.99999995 | `nf/proof_support/p1_modulus_mp.py`; 60-digit check d²/(4pⁿ) for n = 1, 2, 3 in `nf/proof_support/crosscheck_theorems.py` | stdout; `nf/proof_support/results/crosscheck_theorems.json` | s |
| §1.4, §8.3 (Prop. L2) | largest absolute slope 2/(3√3 p) ≈ 0.385/p for unitary Φ: pΣ_unit = 0.384904, 0.384900; SDP check | `nf/proof_support/p2_unitary.py`; `checks/referee_letter/check1b_L2.py` | stdout; `check1b.out` | s |
| §8.3, after Prop. L2 | search over mixed Φ (no value above Σ_unit; the 5.34 artifact and its 80-digit resolution, 0.384602) | `nf/proof_support/p2_search_v2.py` (+ `p2_mpcheck.py`), `p2_resolve.py p2_restart0_x.npy 1e-3` | `p2v2_a.out`, `p2v2_b.out`, `p2_resolve_restart0.out` | min each restart |
| §8.2 (Lemma J), Lemma 4.8.7 | ‖J(Δ)‖_∞ ≤ (d_X/2)‖Δ‖_⋄ and its equality cases | `nf/proof_support/lemJ.py`; `ch04_continuity/v6_verification/cconst.py` | stdout | s |
| §8.4 (Thm. B) | Pauli-block closed forms (F̂, D_BS, Tr_Y ∝ 1) vs direct 40-digit matrices (deviation ≤ 3e-35); 2×2 identities | `nf/proof_support/crosscheck_theorems.py`, `multiq.py`, `block2x2.py`, `chk_qubit_closed.py` | `crosscheck_theorems.json`; stdout | s |
| Table 1.1, §1.5, §8.6, Fig. 8.1 | δ² coefficients F_RLD/8, F_RLD/2, F_BKM/2, F_SLD/8 (sympy, exact), Fisher-information hierarchy | `nf/hierarchy/t4_hierarchy.py`, `nf/proof_support/p3_hier.py`; `checks/referee_letter/check1_closed_forms.py`; figure `nf/figures/make_t2_t3_t4_figures.py hierarchy` | `nf/hierarchy/results/t4.json`; `t4_hierarchy.pdf` | s |
| Fig. 1.3, §1.6, §8.5, Table 8.1 | windows δ_0.1 (50-digit bisection), slopes 0.500/1.000/1.500, constants 4√x_η = 1.9373, 0.97√p for weight one | `nf/multiqubit_window/t3_multiqubit.py`; bounds of Thm. W: `nf/proof_support/crosscheck_theorems.py`; `checks/referee_letter/check2_window.py`; figure `nf/figures/make_fig2_fig3.py` | `nf/multiqubit_window/results/t3.json`; `fig2_window.pdf` | long; rest s |
| §1.4–1.6 | noise-floor law 1 − F̂ ≈ θ²/(2p), F̂ = 0.9 at 0.97√p (R_X), λ_min(J) as the floor for GAD/Pauli/2-qubit noise | `nf/noise_floor/e6_noise_floor.py {qubit,models,cnot,merge}`, `e6_figures.py`; `checks/independent_K1_K4/k2.py`, `checks/independent_C1_C7/c4.py`, `c5.py` | `nf/noise_floor/results/e6*.json` | long |
| Fig. 1.4, §8.12, Tables 8.8, 8.9 | biased Pauli noise: closed forms (≤ 1.2e-33), 60 windows, F_∞ at 29 points (≤ 1e-46), F_RLD/F_∞ < 2 on 22 952 points, two qubits | `nf/biased_noise/b1_closed_forms.py`, `b2_adaptive_qfi.py`, `b3_two_qubit.py`; figure `nf/figures/make_fig_biased.py` | `nf/biased_noise/results/b1.json`, `b2.json`, `b3.json`, `table_pz1e-3.json`, `b2_ratio_scan.json`†, `fig_data.json`; `biased_window_ratio.pdf` | s (b3: min) |
| §1.6, §8.8, Table 8.3 | F_∞ = 2(1−p)²/(p(3−2p)) certified to 50 digits; F_RLD/F_∞ < 3/2; local Stein bracket | `nf/adaptive_qfi/t1_adaptive_qfi.py`; `checks/referee_letter/check3_Finf_sdp.py` | `nf/adaptive_qfi/results/t1.json`; `check3.out` | s |
| §8.11 (strategies), §8.3 constants | f*/4 = 0.149587125, finite-p suprema 0.148832/0.149512/0.149580, c₀ = 0.0246, 0.0589, 4√x_η | `nf/proof_support/consts.py`; `checks/referee_letter/check5_constants.py` | stdout; `check5.out` | s |
| Fig. 1.5, Table 1.2, §8.11, Table 8.7, Fig. 8.2 | ζ bracket at finite δ: certified overshoot 1.81/3.74/9.15, bracket [1, 3.16], paper-3-only 1.29/3.06/7.70 | `nf/crossover/t2_crossover.py` → `t2b_sharp_m2.py <p>` (and `t2b_fill.py <p>`) → `t2b_integrate.py`; figures `nf/figures/make_fig3_crossover.py` (Fig. 1.5 and 8.2) | `nf/crossover/results/t2.json`, `t2b_m2_<p>.json`, `t2b_merged.json`, `summary.json` | t2: min; t2b: long |
| Table 1.2, §8.9, §8.10, Tables 8.4, 8.5, 8.6 | first-order sense of fragility; Chernoff side: Ĉ/ξ up to 1.13/2.48/6.00, local constants, μ_* | `nf/first_order/m1_first_order.py`, `m1_chernoff.py` | `nf/first_order/results_first_order.json`, `results_chernoff.json`, `m1_*.out` | s |
| Fig. 1.6, §1.7, §8.13, Table 8.10 | certification from simulated Pauli tomography: 3.6e11–3.6e15 vs 1.2e6–1.2e8 uses, slopes −2.000 and −1.00, coverage 1000/1000, ratio ≈ 750/λ̃; opposite side (4.3e10, 1.3e9, 9.4e6, 2.4e5 uses at p = 1e-3; exponents 1.98, 1.93, 0.96, 0.99; plug-in overclaim 21.8/8.2/0.6/31.9 %); CNOT p^{-4} | `nf/certification/s1_estimators.py`, `s2_certified.py`, `s2b_pipelines.py`, `s3_cnot.py`, `validate_hp.py` → `analyze_s1.py`, `analyze_s2.py`, `make_tables.py`, `figures.py`; figure `nf/figures/make_fig4_cert.py` | `nf/certification/results/s1_summary.json`, `s2_summary.json`, `s2b_summary.json`, `tables.json`, `s3_cnot.json`, raw `s*_raw/` | simulations heavy; analysis s |
| §8.13 (Thm. C, Prop. N) | Thompson-ball converse on 200 random ball members; two-point constants; exact diamond norm 2(1−p) sin(δ/2) | `nf/proof_support/p4_cert.py`, `consts.py` | stdout | s |
| §8.3, Prop. "incoherent" | incoherent X-weight family: −dF̂/de → x/(2p(1+x)^{3/2}), max 1/(3√3 p); first-order worst direction by SDP | `nf/proof_support/chk_incoh_exact.py`, `sym_incoh.py`, `chk_localmod.py`, `chk_worstdir.py`, `chk_incoherent.py`, `chk_misc.py` | stdout | s (sym_incoh: min) |
| §8.11 and Thm. 4.8.8 (supporting; not quoted) | Theorem L on 60 random noisy instances (max ratio 0.046); sequential-strategy bound ζ ≥ 0.146 δ²/p for p ≤ 1e-2, δ ≤ p | `nf/proof_support/chk_thmL.py` (uses `scale_law.py`) | stdout | s |
| §1.4 interior modulus (supporting) | worst-case interior modulus |ΔF̂| ≈ 0.22‖ΔJ‖₁/√p, Lipschitz | `nf/noise_floor/e8c_interior_modulus.py`; referee prototype `checks/referee_prereview/worst4.py` | `nf/noise_floor/results/e8c.json` | s |
| §1.8 (supporting) | converses vs D^(1), D^(2)/2 and paper-3 bound (precursor of the crossover study) | `nf/noise_floor/e8_gap_moduli.py` → `e8b_fix_d2.py`, `e8_figure.py` | `nf/noise_floor/results/e8.json` | long |
| §8.14 | numerical methods (ARN factorization, SDP cross-checks, mpmath) | `common/qlib.py`, `common/common.py`, `common/sharp_alpha.py`, `common/certlib.py` | — | — |

† `b2_ratio_scan.json` and `table_pz1e-3.json` were produced interactively with the functions of `pauli_biased.py`; no separate script exists for them.

### Chapter 4 (continuity in the Arveson–Radon–Nikodym derivative)

| Book item | What is checked | Script(s) | Output | Time |
|---|---|---|---|---|
| Cor. 4.3.3 | replacement pair with the same sum: F̂(Φ₁,Ψ₁) = 0 while F̂(Φ₂,Ψ₂) = 1 at distance sin θ (three independent F̂ methods) | `ch04_continuity/pair_bounds/e1_jones_cor33.py`; `checks/independent_C1_C7/c1.py` | `ch04_continuity/pair_bounds/results/e1.json` | s |
| Thm. 4.4.6 and the remark after it | the Hölder-½ bound is asymptotically attained (ratio √(1−s)) | `ch04_continuity/v6_verification/r2.py`; `checks/independent_C1_C7/c6_scan.py` | stdout | s |
| Lemma 4.8.2 | ν(Ω)² = p/2 at depolarized gates (lower bound attained) | `nf/noise_floor/e6_noise_floor.py` | `e6.json` | long |
| Lemma 4.8.7, Thm. 4.8.8, Cor. 4.8.10, Ex. 4.8.11 | Choi vs cb norm constants d_X/2, d_X; Choi-floor Lipschitz law; sharpness of λ^{-1} | `ch04_continuity/v6_verification/cconst.py`, `thmL.py`; `nf/proof_support/chk_thmL.py`; `checks/independent_K1_K4/k1.py` | stdout | s |
| Lemma 4.10.3, Cor. 4.10.7, Cor. 4.10.8 | sharper sandwich lemma (max ratio 0.99992), constant K = 2.5273 (δ₀ = 0.15657), Schur-multiplier conjecture | `ch04_continuity/v6_verification/r6.py`, `r6b.py`, `r6_schur.py`; `ch04_continuity/pair_bounds/chk_laws.py` | stdout | s–min |
| Thm. 4.10.4, Lemma 4.10.5, Ex. 4.10.6 | different-sum bounds on 2773 instances + adversarial search (0 violations); τ ≤ 2√2δ/√(1−4γ²) and its finite-δ slack; candidate bound | `ch04_continuity/pair_bounds/e3_pair_bounds.py`, `e3b_lemma105_finite_delta.py`, `e3_figure.py`, `chk_pair_bound.py`; `checks/independent_C1_C7/c6.py`, `c6_adv.py`, `c6_scan.py` | `pair_bounds/results/e3.json`, `e3b.json`; `checks/independent_C1_C7/c6_results_*.npy`, `*.log` | e3, e3b: ~7 min each |
| §4.10 discussion | which quantity predicts fragility (1/λ⁺_min(C_i), Spearman 0.76) | `ch04_continuity/pair_bounds/e4_fragility.py` | `pair_bounds/results/e4.json` | long |
| Ex. 4.12.8 | amplitude damping: F̂(𝒜_γ,𝒜_γ′) = 0 for γ ≠ γ′, ranges meet in C·e₀₁, distance formula | `ch04_continuity/v6_verification/r1_ad.py`; `nf/amplitude_damping/e2_reset_noise.py`; `checks/independent_C1_C7/c3.py` | stdout; `e2.json` | s |
| Thm. 4.12.9, Cor. 4.12.10 | output-rotation limit bound via shorts, attained to 12 digits on 9 random channels; off-diagonal and trace-preserving constructions | `ch04_continuity/v6_verification/r1.py`; `ch04_continuity/dichotomy/chk_channel_jump.py`, `chk_p1.py`, `chk_p1_general.py`, `chk_p1_offdiag.py`, `chk_p1_tp.py` | stdout | s–min |
| Thm. 4.12.11, Ex. 4.12.12 | rate at mutually singular pairs, constant 3d_Xd_Y√d / sin θ, sharpness F̂ sin θ/√d → 1/√2 | `ch04_continuity/v6_verification/r11.py`; `ch04_continuity/dichotomy/chk_p3b.py` | stdout | s |

### Chapter 5 (shadows)

| Book item | What is checked | Script(s) | Output | Time |
|---|---|---|---|---|
| §5.4 (Thm. 5.4.1), channel version | channel hockey-stick divergences are diamond-Lipschitz: |ΔE_γ| ≤ (‖ΔΦ‖_⋄ + γ‖ΔΨ‖_⋄)/2, 736 checks, 0 violations, tight to solver tolerance | `ch05_shadows/hockey_lipschitz/e5_hockey.py`; `checks/independent_C1_C7/c7.py` (with proof and trace-decreasing counterexample) | `ch05_shadows/hockey_lipschitz/results/e5.json` | min |
| Cor. 5.7.5 (used in Ch. 1/8) | paper-3 continuity bound U₃ on ζ, ε = (1−p)|sin(δ/2)| checked against the diamond-norm SDP | `nf/crossover/t2_crossover.py` (verification block) | `t2.json` | min |

The numerical checks summarized in Section 5.9 (and in Section 6.11 of Chapter 6) are not part of this directory; see [What is not here](#what-is-not-here).

### Chapter 7 (adaptive Chernoff)

| Book item | What is computed | Script(s) | Output | Time |
|---|---|---|---|---|
| §7.6.1, Table 7.1 | exponents for γ = ½: ξ_prod, Choi product, −2 ln f, −2 ln μ_* (μ_* closed form and SDP) | `ch07_chernoff/ad_basic.py`, `ad_table.py` | stdout | s |
| §7.6.1, Table 7.2 | optimal n-use adaptive / parallel / product errors (SDP over testers), n ≤ 3; e.g. 0.2815 / 0.2854 / 0.3020 at γ′ = 0.2, n = 3 | `ch07_chernoff/ad_sdp.py [Ds] [ns]`, then `ad_table.py` | `ch07_chernoff/sdp_results_0.1_0.03_0.01_0.3.json`, `sdp_results_0.1.json` | min |
| §7.6.1, Conj. on fidelity | minimal adaptive root fidelity = fⁿ to 1e-8 for n ≤ 4 | `ch07_chernoff/ad_fid.py <Ds> <ns>`, `fid_inspect.py` | `fid_results_*.json`, `fid_n4.log` | n ≤ 3: min; n = 4: heavy |
| §7.6.1, amortization fails | √F ratio 0.948315 < f(½,⅕) = 0.948683 (40-digit check) | `ch07_chernoff/amort_test.py` (search), `amort_check.py` | stdout | long (amort_test); s |

Section 7.7 (Tables 7.3–7.5) was computed with code that is not in this directory.

### Independent checks and referee re-computations (`checks/`)

Grouped by what they verify. Each folder is self-contained (its own library); the reports that discuss them are not included.

| What is verified | Book items | Scripts |
|---|---|---|
| Continuity examples and the different-sum bound | Cor. 4.3.3; Thm. 4.10.4 (bound tight, 0 violations in ~1000 instances + 36 adversarial searches); Thm. 4.4.6 sharpness | `independent_C1_C7/c1.py`, `c6.py <seed>`, `c6_adv.py <seed> <full\|mix> <r> <starts>`, `c6_scan.py`, `validate_sub.py` |
| Closed forms at noisy reset / gates and the noise floor | §1.4 (θ₀.₉ = 0.4843√p, ν² = p/2, D/θ² ≈ ln(1/p)); Ex. 4.12.8 (amplitude damping) | `independent_C1_C7/c2.py`, `c3.py`, `c4.py`, `c5.py`; `independent_K1_K4/k2.py` (closed form to 1e-56, δ*/√p → 0.968644) |
| Hockey-stick Lipschitz property | §5.4 | `independent_C1_C7/c7.py` |
| Choi-floor law and ‖ΔJ‖_∞ ≤ d_X‖Δ‖_⋄ | Thm. 4.8.8, Lemma 4.8.7 | `independent_K1_K4/k1.py` |
| Worst-case slope at gates (Φ = N_p, coherent and Pauli-X families) | §8.3 | `independent_K1_K4/k3a.py` |
| Stein rate of parallel vs sequential strategies (two-use parallel stays at ~ln(1/p)) | §8.11 | `independent_K1_K4/k4.py <p> <n> <seeds>`, `k4v.py`, `k4v2.py` (`k4_n2.out`, `k4_best_*.npy`) |
| Referee: closed forms, Table 1.1 coefficients, sharp constant 1/(4λ̃) | Table 1.1, §8.3, §8.4 | `referee_letter/check1_closed_forms.py`, `check1b_L2.py` |
| Referee: window law and the quoted windows 4.8e-4, 1.2e-7, 1.5e-2 | §1.6, Table 8.1 | `referee_letter/check2_window.py` |
| Referee: F_∞ by an own Zhou–Jiang SDP | §8.8, Table 8.3 | `referee_letter/check3_Finf_sdp.py` |
| Referee: amplitude damping (F̂ = 0, μ_*, asymptotics, Table 8.2) | §1.3, §8.7 | `referee_letter/check4_ad.py` |
| Referee: Stein constants 0.149587, 0.158119, c₀ = 0.0246, 0.0589 | §8.11, §8.13 | `referee_letter/check5_constants.py`; `fit_opp.py` refits the opposite-side exponents of §8.13 (1.98, 1.93, 0.96, 0.99 over p ∈ [1e-4, 1e-2]) on two p-ranges |
| Referee prototypes: reset fragility, metric hierarchy, misspecification, tomography cost, a gate against itself, fragility scale and moduli, RLD Fisher value, overshoot of D_BS over paper 3's bound | §1.3–1.8 (supporting) | `referee_prereview/frag.py`, `hier.py`, `mc.py`, `tomo.py`, `tomo2.py`, `budget.py`, `self.py`, `lowrank.py`, `floor.py`, `worst3.py`, `worst4.py`, `moduli.py`, `moduli2.py`, `rld.py`, `gap.py`, `gate.py` (library `lib.py`) |

The referee scripts print their results; the shipped `referee_letter/*.out` files are their outputs. The prototypes in `referee_prereview/` are qubit-only, use linear inversion and local optimization, and their worst cases are lower bounds.

## The folders in detail

### `common/`
| Module | Used by | Contents |
|---|---|---|
| `qlib.py` | early experiments (E1–E8), `certlib.py` | ARN/SVD route for F̂, Q̂_α, D_BS; ε-regularized geometric mean in mpmath; diamond norm, hockey stick, Uhlmann SDPs; ν(Ω) |
| `glib.py` | `proof_support/`, `ch04_continuity/pair_bounds/`, `dichotomy/` | minimal helpers for the proof checks (Choi, frames, geometric means, Tr_Y, diamond norm) |
| `common.py` | T1–T4, `crosscheck_theorems.py`, `b1_closed_forms.py` | high-precision helpers of the phase-2 study |
| `sharp_alpha.py` | `crossover/`, `first_order/m1_first_order.py` | certified Fawzi–Fawzi D^#_α SDP; `python3 common/sharp_alpha.py` runs its self-test |
| `certlib.py` | `certification/` | Pauli process tomography, estimators, confidence regions, certified bounds |

### `ch01_ch08_noisy_gates/`
- `noise_floor/` (from `numerics/`): `e6_noise_floor.py qubit|models|cnot|merge` (run the four stages in this order; `merge` writes `e6.json`), `e6_figures.py`, `e7_family.py`, `main_figure.py` (four-panel overview figure from `e6.json`, `e7.json`), `e8_gap_moduli.py` then `e8b_fix_d2.py` (recomputes the two-use values with the support-safe relative entropy and patches `e8.json`), `e8_figure.py`, `e8c_interior_modulus.py`.
- `amplitude_damping/`: `e2_reset_noise.py` (reset channels with depolarizing noise, closed form to 1e-14; amplitude damping near γ = 1; dephasing/depolarizing controls; Q̂_α, BS, Hellinger).
- `adaptive_qfi/`: `t1_adaptive_qfi.py`.
- `crossover/`: `t2_crossover.py`; `t2b_sharp_m2.py <p>` for p in 0.01, 0.001, 0.0001 (about 40 s per row, 9–11 rows per p); `t2b_fill.py <p> …` reruns rows whose SDP returned no point (it appends to `results/t2b_fill_<p>.log`, which `t2b_integrate.py` reads); `t2b_integrate.py` merges into `t2b_merged.json` and updates `summary.json`.
- `multiqubit_window/`: `t3_multiqubit.py` (verification against brute-force 16×16 and 64×64 means, λ_min table, windows, curves).
- `hierarchy/`: `t4_hierarchy.py`.
- `proof_support/`: from the author's working code (`lemJ.py`, `multiq.py`, `block2x2.py`, `p1_modulus_mp.py`, `p2_common.py`, `p2_unitary.py`, `p2_mpcheck.py`, `p2_search_v2.py <p> <restarts> <seed> [warm]`, `p2_resolve.py <x.npy> <p>`, `p3_hier.py`, `p4_cert.py`, `consts.py`), from `proofs/` (`chk_qubit_closed.py`, `chk_incoh_exact.py`, `chk_incoherent.py`, `chk_localmod.py`, `chk_worstdir.py`, `chk_misc.py`, `chk_thmL.py`, `scale_law.py`, `sym_incoh.py`) and `crosscheck_theorems.py` (from the author's working code). `p2_common.py` and `p2_unitary.py` are the two valid parts of the original `p2_worst.py`, unchanged (see `EXCLUDED.md`). `scale_law.py` is kept because `chk_thmL.py` imports its `zeta_lower`; its printed tables are superseded by `crossover/`.
- `first_order/`: `m1lib.py`, `m1_first_order.py`, `m1_chernoff.py` (reads `crossover/results/t2.json` and `t2b_merged.json`).
- `biased_noise/`: `pauli_biased.py`, `zj_generic.py`, `b1_closed_forms.py`, `b2_adaptive_qfi.py`, `b3_two_qubit.py`. `b2` and `b3` are checkpointed: they skip points already present in `results/b2.json` / `b3.json`, so delete those files to recompute.
- `certification/` (from the author's working code): run order and arguments as in its original README:
  ```sh
  python3 s1_estimators.py 0 2 & python3 s1_estimators.py 1 2          # S1 raw    -> results/s1_raw/*.npz
  python3 s2_certified.py coverage 0 1; python3 s2_certified.py shots 0 1  # S2 -> results/s2_raw/
  python3 s2b_pipelines.py lcb_scan; python3 s2b_pipelines.py lcb_coverage; python3 s2b_pipelines.py lcb_contrast  # converse side
  python3 s2b_pipelines.py scan; python3 s2b_pipelines.py coverage       # opposite side
  python3 s3_cnot.py; python3 validate_hp.py
  python3 analyze_s1.py; python3 analyze_s2.py; python3 figures.py; python3 make_tables.py
  ```
  `make_tables.py` fills `README_template.md` into a report `README.md` in that folder (generated, not shipped).
- `figures/`: `figstyle.py` (shared style), `make_fig0_schematic.py` (Fig. 1.1), `make_fig1_regimes.py` (Fig. 1.2), `make_fig2_fig3.py` (Fig. 1.3; its Fig. 3 code is superseded and not called), `make_fig_biased.py` (Fig. 1.4; also writes `biased_noise/results/fig_data.json`), `make_fig3_crossover.py` (Fig. 1.5 and Fig. 8.2), `make_fig4_cert.py` (Fig. 1.6), `make_t2_t3_t4_figures.py [crossover|window|hierarchy]` (Fig. 8.1 = `t4_hierarchy.pdf`; also the supplementary `t2_crossover.pdf`, `t3_window_collapse.pdf`). The scripts write the PDFs into this folder; the book uses copies in the repository's `figures/`.

### `ch04_continuity/`, `ch05_shadows/`, `ch07_chernoff/`
See the tables above. `ch04_continuity/v6_verification/` is self-contained (`mlib.py` in mpmath, `npl.py` in double precision); its scripts print their checks.

## Runtimes

Measured on the 2-core machine used to assemble this directory (Python 3.13, the versions above), or taken from the `runtime_s` fields of the result files.

| Class | Scripts |
|---|---|
| seconds (< 1 min) | all figure scripts (1–5 s); `common/sharp_alpha.py` (2 s); `t1_adaptive_qfi.py` (7 s); `t4_hierarchy.py` (8 s); `t2b_integrate.py`; `crosscheck_theorems.py` (45 s); `analyze_s1.py` (2 s), `analyze_s2.py` (12 s), `validate_hp.py` (5 s), `make_tables.py`, `figures.py`; `b1_closed_forms.py` (8 s), `b2_adaptive_qfi.py` (13 s); `m1_first_order.py` (34 s), `m1_chernoff.py` (9 s); `e1_jones_cor33.py` (31 s), `e2_reset_noise.py` (43 s), `e8c_interior_modulus.py` (21 s); `proof_support/` (`p1_modulus_mp.py` 17 s, `p3_hier.py` 17 s, `p4_cert.py` 12 s, `lemJ.py` 7 s, `p2_resolve.py` 5 s, the rest ≤ 6 s); `ch04_continuity/` proof checks and `v6_verification/` (≤ 15 s each, `r6.py` 55 s); `ch07_chernoff/ad_basic.py` (17 s), `ad_table.py` (1 s); `checks/referee_letter/` (≤ 1 s each); `checks/referee_prereview/` (most ≤ 17 s; `worst3.py`, `worst4.py`, `moduli.py`, `moduli2.py` 28–39 s); the other scripts of `checks/` (≤ 20 s) |
| minutes (1–10 min) | `sym_incoh.py` (1.5 min); `t2_crossover.py` (1–2 min); `e5_hockey.py` (1–1.5 min); `b3_two_qubit.py` (1.5 min); `p2_search_v2.py` (minutes per restart); `e3_pair_bounds.py` (~7 min), `e3b_lemma105_finite_delta.py` (~7 min); `ad_sdp.py` with its default γ′ (1.5 min); `checks/independent_C1_C7/c4.py` (10 min) |
| long (10–60 min) | `t3_multiqubit.py` (~25 min); `t2b_sharp_m2.py` (~40 s per row, 6–8 min per p); `e4_fragility.py` (20 min); `e6_noise_floor.py qubit` (11 min), `models` (36 min), `cnot` (~35 min); `e7_family.py` (17 min); `e8_gap_moduli.py` (49 min); `ch07_chernoff/amort_test.py` (13 min) |
| heavy (hours or large memory) | the certification simulations `s1_estimators.py`, `s2_certified.py`, `s2b_pipelines.py`, `s3_cnot.py` (tens of minutes to hours per mode on 2 cores; the original runs were checkpointed per p and pipeline and needed several restarts); `ad_fid.py` at n = 4 (a large SDP; n = 5 needs about 138 GB of memory and was not completed) |

## Random seeds

All randomness uses fixed seeds set in the scripts:

| Seed(s) | Script |
|---|---|
| `SEED = 20261027` | `e3_pair_bounds.py` (instance families) |
| `SEED = 44`, `55`, `6`, `7`, `8`; `0`, `81`, `83`, `105` | `e4_fragility.py`, `e5_hockey.py`, `e6_noise_floor.py`, `e7_family.py`, `e8_gap_moduli.py`; `e1_jones_cor33.py`, `e8b_fix_d2.py`, `e8c_interior_modulus.py`, `e3b_lemma105_finite_delta.py` |
| `1000·i_p + i_N + 12345` | `s1_estimators.py` (one stream per (p, N) cell, independent of the worker split) |
| `777 + i`, `31 + 7 i_p (+100)` | `s2_certified.py` (coverage, shots) |
| `10⁶ p + 5`, `777000 + i_p`, `4242 + 17 i_p + 1000 i_w`, `9090 + i_p`, `55500 + i_p`, `88800 + 10 i_p + i_w`, `66600 + i_p` | `s2b_pipelines.py` (one per mode and grid point) |
| `2026`, `99`, `20261027`, `7` | `s3_cnot.py`, `validate_hp.py`, `analyze_s2.py` (bootstrap), `certlib.py` |
| `20261027` | `m1_first_order.py` |
| command-line seed | `c6.py <seed>` (logs for seeds 0, 1 shipped), `c6_adv.py <seed> …`, `k4.py <p> <n> <seeds>`, `p2_search_v2.py <p> <restarts> <seed>` (outputs for seeds 7 and 8 shipped) |
| fixed small integers (0–21) | every other script (`np.random.default_rng(k)` or `random.Random(k)` at the top) |

Scripts without randomness (closed forms, bisections, exact SDPs) are deterministic. SDP solver output can differ in the last digits between platforms; the smoke test allows for this.

## What is not here

- Excluded scripts and files, with reasons (values later shown wrong, superseded or duplicate versions, scratch): `EXCLUDED.md`.
- Raw data and outputs not included, and how to regenerate them: `DATA_NOT_INCLUDED.md`.
- The code for the numerical checks of Section 5.9 (Chapter 5), Section 6.11 (Chapter 6) and Section 7.7 (Tables 7.3–7.5) was not among the working folders this directory was assembled from.

