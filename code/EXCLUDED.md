# What was left out of `code/`, and why

`code/` was assembled from the author's working code. Everything that the book's numbers come from is included (see `README.md`). The items below were left out. Text files (notes, reports, paper sources, `.tex`, `.bib`, `.md` reports), LaTeX build products, `__pycache__/` and rendered figure images were never candidates and are not listed one by one.

## Scripts and outputs that produced a value later shown wrong

| Excluded | Reason | Replaced by |
|---|---|---|
| The mixed-Φ search in `p2_worst.py` (lines after `# global search over Phi`) and its output `p2_worst.out` | Double-precision, thresholded finite-difference search over Kraus-rank-4 Φ. At p = 1e-3 its warm start returned p·Σ = 5.34, which violates the proved bound p·Σ ≤ 1 (Proposition L2(a), Chapter 8). Recomputed at 80 digits the same point gives 0.384602: a double-precision artifact from square roots of round-off-level Choi eigenvalues ("Resolution" in `theorems.md`). | `proof_support/p2_search_v2.py` (exact-subspace square root, every restart re-verified at 50 digits by `p2_mpcheck.py`) and `proof_support/p2_resolve.py` (the 80-digit resolution). The first, correct part of `p2_worst.py` is kept unchanged, split into `p2_common.py` (helpers) and `p2_unitary.py` (exact unitary supremum 0.384904 / 0.384900 and its SDP check). |
| `p2_repro.py`, `p2_repro.out`, `p2_restart0_x0.npy` | Re-ran the artifact search above to save its optimum. | The saved point itself, `proof_support/p2_restart0_x.npy`, is kept as the input of `p2_resolve.py`. |
| `p1_modulus.py` | Double-precision version of the Theorem L1 sharpness check; its own header calls it unreliable at n = 2, p = 1e-3 ("kept as a warning"). | `proof_support/p1_modulus_mp.py` (40 digits; the ratios 1.0000000, 1.0000034, 0.9999816, 0.99999995 quoted in Section 8.3). |
| `k4seq.py`, `k4seq2.py` | Sequential-strategy Stein constant on a coarse grid of use numbers; the claimed limit 0.149574 is an artifact of the grid step (`theorems.md`, Thm S, "Corrections"). The correct limit is 0.1495871. | `proof_support/consts.py`, `adaptive_qfi/t1_adaptive_qfi.py`, `checks/referee_letter/check5_constants.py`. |
| `k3b.py`, `k3c.py`, `k3r.py`, `k3v.py` and their optimizers `k3_best_*.npy`, `k3c_best_*.npy` | Double-precision local searches for the worst-case absolute slope over Φ (K3). The resulting "≥ 0.32/p" is superseded by the exact unitary value 0.3849/p (Proposition L2(b)); `theorems.md` §3.3 notes that such searches near rank-deficient Φ are unreliable in double precision. | `proof_support/p2_unitary.py`, `p2_search_v2.py`; `checks/referee_letter/check1b_L2.py`. |
| `results/e8_firstrun_buggy_relent.json` | First E8 run, made while `rel_ent` returned +∞ whenever σ was singular; it corrupted the two-use Umegaki values. | `noise_floor/e8b_fix_d2.py` recomputes them with the support-safe `rel_ent_on_input`; the shipped `e8.json` is the corrected file. |

## Superseded or duplicate scripts

| Excluded | Reason |
|---|---|
| `chk_finf.py` | Zhou–Jiang F_∞ "as recalled", in a double-precision SDP. Superseded by `adaptive_qfi/t1_adaptive_qfi.py` (certified to 50 digits) and `checks/referee_letter/check3_Finf_sdp.py`. |
| `chk_p3.py` | Preliminary form of the mutually-singular rate check. Superseded by `ch04_continuity/dichotomy/chk_p3b.py` (the statement as proved) and `ch04_continuity/v6_verification/r11.py` (improved constant). |
| `t2b_resume.py` and its checkpoint logs `results/t2b_<p>.log`, `t2b_resume_*.out` | Resume wrapper for an interrupted run; it performs exactly the computation of `crossover/t2b_sharp_m2.py`, whose outputs `t2b_m2_<p>.json` are included. |
| `c6_seed1.log` | Log of a run of an earlier version of `c6.py` (it predates the last edit of `c6.py`); superseded by `checks/independent_C1_C7/c6_seed0_v2.log` and `c6_seed1_v2.log`, the logs of the shipped `c6.py`. |
| `results/summary_pre_t2b.json` | Snapshot of `summary.json` before the two-copy update; superseded by `crossover/results/summary.json`. |
| `make_fig1_regimes_v1.py`, `make_fig2_fig3_v1.py` | First-submission versions of the figure scripts; superseded by the revision-1 scripts in `ch01_ch08_noisy_gates/figures/`. |
| `make_figures_numerics_orig.py` | Byte-identical copy of `A_numerics/make_figures.py`, which is included as `figures/make_t2_t3_t4_figures.py`. |
| `glib.py` | Identical to `proofs/glib.py`; one copy is in `common/`. |
| `p2_xunit_1e-3.npy` | Byte-identical to `p2_xunit_0.001.npy` (included; the warm start read by `p2_search_v2.py`). |

## Dead ends and scratch

| Excluded | Reason |
|---|---|
| `chk_triangle.py` | Search for triangle-inequality violations of √(1 − tr ρ#σ); a side question not used in the book. |
| `chk_grad.py` | Finite-difference test of the gradient of tr(A#B); exploratory, not cited. |
| `chk_gm_lip.py` | Exploratory scan of the operator-Lipschitz constant of #; not used in the book. |
| `validate_sub2.py` | Second validation of the exact-subspace F̂ against mpmath on random rank-deficient pairs. It stops with a `ZeroDivisionError` in the ε-regularized mpmath mean at its third case, also in its original folder, so only two of its six cases ever ran; the comparison is covered by `checks/independent_C1_C7/validate_sub.py`. |
| `sdp_test.py` | Scratch test of the diamond-ball SDP (the same SDP is in `p2_common.py`). |
| `p2v2_x_*.npy` | Optimizer points saved by `p2_search_v2.py`; regenerated by it, and its results are in `p2v2_a.out`, `p2v2_b.out`. |
| `c6adv_*.npy` | Optimizer points saved by `c6_adv.py`; the adversarial ratios are in `adv_*.log`. |
| `fid_n5.log` | The n = 5 fidelity SDP ran out of memory (about 138 GB requested); no result. |

## Logs and other non-code files

- Run logs whose numbers are in the JSON results: `numerics/results/*.log`; `A_numerics/results/{t2,t3,t2b_*}.log`, `t3.out`, `t2b_fill.out`, `crosscheck_theorems.out`; `A_cert/logs/*`; `A_letter/m1/run_*.log` (the documented outputs `m1_*.out` are kept); `A_letter/biased/results/*.log`. Kept logs are those a script reads (`crossover/results/t2b_fill_<p>.log`) or that a report cites as the output (`checks/independent_C1_C7/*.log`, `ch07_chernoff/fid_n4.log`, and the `.out` files of `checks/referee_letter`, `checks/independent_K1_K4/k4_n2.out`, `proof_support/p2*.out`, `first_order/m1_*.out`).
- Rendered figures (`*.png`, `*.pdf` under `numerics/figures`, `*/figures`): regenerated by the figure scripts. The PDFs used by the book are in the repository's `figures/`.
- `A_cert/README.md`: a generated report; `certification/make_tables.py` regenerates it from the included `README_template.md`.
- Raw data not included: see `DATA_NOT_INCLUDED.md`.
