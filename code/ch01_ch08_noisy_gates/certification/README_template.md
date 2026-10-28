# A_cert: can divergence-based converses near noisy gates be *certified* from finite tomography data?

Phase 2, workstream A_cert, 27 Oct 2026. Every number below comes from `results/*.json`, which the scripts in this directory produce. Seeds are fixed in each script.
Environment: Python 3.13, numpy 2.5.3, scipy 1.18.1, cvxpy 1.9.3 with Clarabel 0.11.1 (SCS as fallback), mpmath 1.3.0 and matplotlib 3.11.
The runs used a 2-core machine; the runtime limits that follow from this are listed in §6.

```
python3 s1_estimators.py 0 2 & python3 s1_estimators.py 1 2     # S1 raw    -> results/s1_raw/*.npz
python3 s2_certified.py coverage 0 1; python3 s2_certified.py shots 0 1   # S2 -> results/s2_raw/
python3 s2b_pipelines.py lcb_scan; python3 s2b_pipelines.py lcb_coverage; python3 s2b_pipelines.py lcb_contrast   # S2b converse side
python3 s2b_pipelines.py scan; python3 s2b_pipelines.py refine; python3 s2b_pipelines.py coverage    # S2b opposite side (supplement)
python3 s3_cnot.py                                                     # S3 -> results/s3_cnot.json
python3 validate_hp.py                                                 # high-precision check of F-hat on singular estimates
python3 analyze_s1.py; python3 analyze_s2.py; python3 figures.py; python3 make_tables.py   # summaries, figures, this README
```

Labels used below: **CERTIFIED** means a bound that holds with probability ≥ 0.95 by construction (Hoeffding or Clopper–Pearson with a union bound, then exact optimisation or Theorem C). **SIMULATED** means a Monte Carlo statistic of such bounds or of plug-in estimates (medians, quantiles, coverage). **FITTED** means a regression on simulated statistics (slopes, prefactors).

{{HEADLINE}}

---

## 0. Setting and conventions

- **Channels.** The gate is N_p = D_p∘H, with D_p(ρ) = (1−p)ρ + p·tr(ρ)·1/2. The miscalibrated gate is N_{p,δ} = D_p∘(e^{−iδZ/2}H).
  - The main pair is **δ = 2√p**. Its true F̂ is 0.711 / 0.708 / 0.707 at p = 1e-2 / 1e-3 / 1e-4, the same scale as E7. This is the regime where the geometric quantities are O(1).
  - "Self" means the gate compared with itself, estimated by two independent tomographies. Its true values are F̂ = F_U = 1 and D_BS = E₁ = 0.
  - S2 also uses a fixed pair with δ = 1 rad.
- **Choi convention.** Watrous: J = Σ Φ(E_ij)⊗E_ij, output first, so Tr_Y J = 1 for trace-preserving maps. λ_min(J(N_p)) = p/2, and the Choi-*state* floor is λ̃ = p/4.
- **Quantities.**
  - F̂ = λ_min Tr_Y(J₁#J₂), computed by the stable ARN/SVD route of `numerics/qlib.py`.
  - D_BS(N_{p,δ}‖N_p) = λ_max Tr_Y(...). It is +∞ when supp J₁ ⊄ supp J₂.
  - E₁ = ½‖N₁−N₂‖_⋄, from a concave maximisation over the input state, checked against the Watrous SDP to ≤ 2e-11.
  - Uhlmann channel fidelity F_U = min_σ F(...), from a convex minimisation over the input state, checked against the SDP to ≤ 5e-9.
- **High-precision check of the singular cases** (`validate_hp.py`, `results/validate_hp.json`). On 36 *estimated* pairs (linear inversion and rank-2 fits, ranks 2–4), ARN F̂ agrees with the 120-digit ε-regularised geometric mean (mpmath, ε → 0 by Aitken extrapolation) to **3.2e-14**, and with the SDP to 8.3e-11. In particular, the zeros produced by low-rank fits are exact: mp gives |F̂| ≤ 2.5e-43.
- **Tomography.**
  - Standard Pauli process tomography uses 6 inputs (±X, ±Y, ±Z eigenstates) and 3 measurement bases, so there are 18 binary settings per channel with N shots each.
  - Counts are sampled from the exact binomial distribution.
  - A pair estimate uses two (for "self", three) independent tomographies.
- **Direction of an invalid converse.** The definition used here is the one below and report signed errors, which lets either reading be recovered.
  - A converse is an *upper* bound on distinguishability. Examples are ξ_adapt ≤ −2 ln F̂ (Chernoff-type), the Stein exponent ≤ D_BS, P_succ ≤ ½(1+E₁), and error lower bounds through F_U.
  - It becomes **invalid when the plug-in fidelity is too HIGH, or the plug-in divergence too LOW.** That direction claims *less* distinguishability than the truth.
  - An F̂ estimate *lower* than the truth claims more distinguishability. It is therefore a looser but still valid converse.
  - This matches the pre-review (K6: "an estimated F̂ can only be too small (loose bound)").
  - Throughout, **overclaim** = F̂ or F_U above the truth, or D_BS or E₁ below the truth. **Loose** is the opposite.
  - For the "self" pair overclaiming is impossible, because F̂, F_U ≤ 1 and D, E₁ ≥ 0.
- **Which side of a confidence interval certifies a converse (correction, added 27 Oct, 18:35).**
  - By the definition above, a converse is certified by a **lower** confidence bound on F̂ and F_U and an **upper** confidence bound on D_BS and E₁. This is also the side that Theorem C bounds: F̂(true) ≥ (1−κ)F̂(model), i.e. an upper bound on −ln F̂.
  - The SDPs of §2 and of §3.5 compute the opposite, convex-tractable side: max_R F̂, max_R F_U, min_R D_BS and min_R E₁. Those are valid confidence bounds, but they bound distinguishability **from below**. They do not certify any of the converses listed above. (For E₁ and F_U they are achievability statements, e.g. P_succ ≥ ½(1+E₁-LCB).)
  - §3.1–3.4 therefore compute the converse side directly. Pipeline (a) is Theorem C on the same Hoeffding trace-norm ball. Pipeline (b) is the exact minimum of F̂ (maximum of D_BS) over the Pauli polytope, by vertex enumeration. On the opposite side, the shot-cost slopes are the same as on the converse side (§3.5), but the prefactors differ.

## 1. S1: plug-in estimators (200 repetitions per (p, N); E₁ and F_U on the first 20)

**Estimators.**
- **LI:** linear inversion (least squares over all 36 outcome frequencies), then the Frobenius projection onto CPTP. The projection uses Dykstra (PSD cone / TP affine) and finishes with an exact-TP congruence J ↦ (1⊗S^{-1/2})J(1⊗S^{-1/2}), which preserves exact zero eigenvalues.
- **MLE:** CPTP-constrained maximum likelihood (cvxpy + Clarabel, exponential cone).
  - Interior-point solutions carry eigenvalue fuzz of about 1e-9 to 4e-8 at the boundary. I polish them: eigenvalues below 1e-7 are set to 0, followed by the exact-TP congruence.
  - The sensitivity to this threshold is reported below. **F̂ is insensitive to it, BS is not.**
- **Hedged MLE** (Blume-Kohout): maximise L(J)·det(J)^β, i.e. Σ f log p + (β/N) log det J, with β = 1/2 and β = 0.05.
- **Low-rank (LR1/LR2):** maximum-likelihood fit over CPTP maps of Kraus rank ≤ r, using L-BFGS over an isometry parametrisation initialised from the top-r eigenvectors of the LI estimate.

Cells give the **median [10%, 90%]** of the plug-in estimate, followed by **o** = overclaim fraction and **l** = loose fraction. True values:

{{S1_truth}}

### 1a. Geometric fidelity F̂, pair δ = 2√p
{{S1_delta_F}}

### 1b. Belavkin–Staszewski D_BS(N_{p,δ}‖N_p), pair δ = 2√p (fraction of runs with D_BS = +∞ in parentheses)
{{S1_delta_BS}}

### 1c. Hockey-stick E₁ (= ½ diamond distance), pair δ = 2√p (20 runs; LI, MLE, HMLE0.5, LR2 only)
{{S1_delta_E1}}

### 1d. Uhlmann channel fidelity F_U, pair δ = 2√p (20 runs)
{{S1_delta_FU}}

### 1e. Gate vs itself (true F̂ = 1, D_BS = 0)
F̂:
{{S1_self_F}}

D_BS:
{{S1_self_BS}}

### 1f. Bias and signed error of F̂ (pair δ = 2√p; LI, MLE, hedged β=1/2)
"rank-def." is the fraction of runs in which either estimated Choi matrix has an eigenvalue below 1e-12.

{{S1_bias_F}}

### 1g. MLE polishing threshold (pair δ = 2√p)
{{S1_polish}}

{{S1_REFEREE}}

## 2. S2: certified one-sided bounds on the opposite (lower-distinguishability) side

> These bounds are valid at 95%, but they are UCBs on F̂ and F_U and LCBs on D_BS and E₁, so they bound distinguishability from below (see §0, last bullet). The converse-side certificates are in §3.1–3.4. The coverage, region and solver statements below are unaffected.

### 2.1 Confidence region (valid, deliberately simple)
- **Hoeffding bound.** For each binary setting k of each channel, P(|f_k − p_k| > t) ≤ 2e^{−2Nt²}. A union bound over the 36 settings of the two channels gives simultaneous validity at level 1−α = 0.95 with **t = √(ln(72/α)/(2N))**.
- **Box region.** R_box = {(J₁,J₂) CPTP : |tr(A_k J_i) − f_ik| ≤ t ∀k,i}. It contains the truth on that event.
- **Trace-norm ball.**
  - Linear inversion is exactly linear, J_LI − J = Σ_k (f_k − p_k) D_k, where D_k is the dual-frame element for setting k, with ‖D_k‖₁ = 2 and ‖D_k‖_∞ = 2/3.
  - Hence ‖J_LI − J‖₁ ≤ ε_tr = 36t and ‖J_LI − J‖_∞ ≤ ε_op = 12t on the same event.
  - R_ball = {CPTP : ‖J_i − J_LI,i‖₁ ≤ ε_tr}. The trace norm of the Choi difference upper-bounds the diamond norm.
- **Weyl floor.** λ_min(J) ≥ λ_min(J_LI) − ε_op.
- **Certified bounds by direct convex optimisation over the region.** These are small qubit SDPs, solved with Clarabel at its default tolerance of 1e-8; values agree with tight-tolerance Clarabel and with SCS at 1e-10 to ≤ 4e-6. A safety margin of 1e-5 is added on the conservative side.
  - **F̂:** UCB = max_R F̂. F̂ is jointly concave, so this is the exact maximum, an SDP with the KW block LMI.
  - **D_BS:** LCB = max(−2 ln max_R F̂, min_R D̂_{7/8}). This is valid because the geometric Rényi divergence is monotone in α and D̂_1 = D_BS.
    - Q̂_{7/8} has an exact SDP hypograph via three nested 2×2 block LMIs, since (A#B)#B = A#_{3/4}B, and so on.
    - At these pairs, D̂_{7/8}/D_BS = 0.913, so a factor-2 certificate of BS is reachable.
  - **E₁:** LCB = min_R ½‖J₁−J₂‖_⋄ (Watrous SDP, jointly convex).
  - **F_U:** UCB = max_R F_U (Uhlmann SDP, jointly concave).
- **The "theorem route" for F̂, for comparison.**
  - It combines v5 Cor 10.7 (|ΔF̂| ≤ (11/2)δ_rel^{1/2}), Cor 10.8 (δ_rel ≤ diamond error / ν(Ω′)²), Lemma 8.2 (ν² ≥ λ_min J(Ω′)/2) and Weyl, applied to the CPTP-projected estimate.
  - The Frobenius projection onto CPTP is non-expansive, so the radius is 2ε_fro, with ε_fro = 18.97t.
  - The paper-2 Cor "uncertain" route for Ĉ was not implemented. It needs the same Thompson ball, so it inherits the same p⁻² radius requirement.

### 2.2 Coverage (1000 runs each; pair δ = 2√p; (p, N) chosen at the factor-2 threshold of §2.3)
{{S2_cov}}

The 95% binomial half-width of a coverage estimate at 0.95 is ±0.014 for 1000 runs. Coverage is ≥ 0.95 everywhere and is in fact **1.000 for every quantity**, as expected for a union-bound Hoeffding region. The region is conservative, not tight.

**Medians of the certified values:**
{{S2_covmed}}

{{S2_COVTEXT}}

### 2.3 Shots per setting for a certified bound within a factor 2 of the truth
**Criterion.**
- In divergence form: −2 ln F̂_UCB ≥ ½(−2 ln F̂_true), D_BS-LCB ≥ ½ D_BS, E₁-LCB ≥ ½ E₁, and −2 ln F_U,UCB ≥ ½(−2 ln F_U,true).
- The criterion is applied to the median over 10 runs per point, on a half-decade grid N = 10³ … 10¹².
- N_req is log-interpolated and is quoted only once the criterion holds for all larger N.

{{S2_nreq}}

![ratio, scaled pair](figures/s2_ratio_scaled.png)
![ratio, fixed pair](figures/s2_ratio_fixed.png)
![N_req vs p](figures/s2_nreq_vs_p.png)

{{S2_TEXT}}

## 3. S2b: what does it cost to certify the converses, and which part of the p⁻² is intrinsic?

Proved predictions (A_proofs/theorems.md §7, Theorem C and Props N1–N4), with λ̃ = p/4 the Choi-state floor and η the target accuracy in −ln F̂:
- Certifying through a diamond-norm ball needs radius ε ≤ 4ηλ̃, so it costs Θ(1/(ηλ̃)²). For incoherent diamond learners, N ≥ 0.0589(1−p)²/ε² (N2).
- Any strategy needs Ω(1/(η²λ̃)), with c₀ = 0.0246 (N3). For Pauli channels in a known frame this is matched by 3 ln(2/δ_f)/(η²λ̃) (N4).

### 3.1 Pipelines and criterion (pair δ = 2√p, true F̂ ≈ 0.71; all regions at 95% simultaneous validity, δ_f = 0.05)
- **(a) Tomography + diamond-norm ball (Theorem C).** Standard Pauli tomography uses 18 settings. The Hoeffding union bound gives the trace-norm ball ‖J_i − J_LI,i‖₁ ≤ ε_tr = 36t around the linear-inversion estimate. J_LI is exactly trace-preserving (checked to 4e-16), so Lemma J applies to J − J_LI, and Theorem C gives the closed form
  F̂_LCB = (1−κ)·F̂(J_LI,1, J_LI,0), with κ = ε_tr/(2λ̂) and λ̂ = min_i λ_min(J_LI,i). It is vacuous (0) if κ ≥ 1.
- **(b) Direct Pauli-error (Pauli-eigenvalue) estimation in the known gate frame.**
  - Model assumption: N = Pauli∘U, with U known for both channels.
  - There are 3 settings per channel: input U†|P+⟩, then measure P. The flip probability s_P is the sum of the two anticommuting error rates (s_P = (1 − Pauli eigenvalue)/2).
  - Clopper–Pearson intervals at α/6 give a polytope for (q_X, q_Y, q_Z) of each channel.
  - F̂ is jointly concave and D_BS is jointly convex, and the Choi pair is affine in (q₁, q₀). So F̂_LCB = min F̂ and D_BS,UCB = max D_BS over the product polytope are attained at pairs of vertices, and are computed **exactly** by vertex enumeration. Spot check at p = 1e-3, N = 1e5 per setting, against 3000 random interior points of the product polytope: the vertex minimum of F̂ is 5.5e-4, below the interior minimum of 0.516. The vertex maximum of D_BS is ∞, above the interior maximum of 2.50.
- **Criterion.** Accuracy η is reached when ln(F̂_true/F̂_LCB) ≤ η in the median run, with η = 0.05 unless stated otherwise. N_req is log–log interpolated on a 1/8-decade grid, with 100 runs per point for (a) and 50 for (b), and is quoted only once the criterion holds at every larger grid N.
- **Cost unit.** The cost is **N = total channel uses per channel** (settings × shots per setting), the same unit as N2–N4. Slope CIs are 95%: "OLS" uses t with n−2 dof on the 5 points, and "boot" resamples runs within each grid point (2000 replicates).

### 3.2 Shot cost versus floor: converse side
{{S2b_main}}

**Fitted slopes, d log N / d log p** (FITTED; expected −2 for (a) and −1 for (b)):
{{S2b_slopes}}

**Against the proved bounds** (bounds are analytic; N values SIMULATED):
{{S2b_bounds}}

### 3.3 Contrast: hockey-stick and Uhlmann through the same diamond-norm ball (converse side)
- Same tomography and Hoeffding event. The pair is δ = 1 rad, where E₁ ≈ 0.47 and F_U ≈ 0.88 are O(1) at every p. (At δ = 2√p both vanish with p, so an additive target would be met by the trivial bound.)
- **E₁-UCB** = ½‖J_LI,1 − J_LI,0‖_⋄ + ε_tr. This is the triangle inequality, i.e. §7.3 with γ = 1. The criterion is E₁-UCB − E₁ ≤ η.
- **F_U-LCB** = √(1 − (P̂ + 2√(2ε_fro))²). Here P = √(1 − F_U²) is the purified distance, which is a metric, and P̂ is computed from the CPTP projections of J_LI. The criterion is ln(F_U/F_U-LCB) ≤ η.
- 20 runs per point on a 1/4-decade grid.

{{S2b_contrast}}

### 3.4 Coverage of the converse-side certificates (SIMULATED; 1000 runs per p at the η = 0.05 budget of each pipeline)
{{S2b_cov}}

### 3.5 Supplement: the opposite side (SDP max F̂ / min D over the region, as in §2)
- Same four pipelines as originally planned: (a) trace-norm ball, (a′) Hoeffding box, (a″) Clopper–Pearson box in probability space (variance-adaptive, no model assumption) and (b) the Pauli polytope.
- The η-criterion is ln(F̂_UCB/F̂_true) ≤ η in the median run. Shot costs are total channel uses per channel.
- All four use the coarse scan (1/2 decade, 10 runs; (a′) from §2). A refined scan (1/8 decade, 50 runs) completed only for p = 1e-2 before the run was interrupted (`results/s2b_raw/refine_p1e-02.json`) and is not used, so that every p is treated alike.

{{S2b_opp}}

Factor-2 criterion (median certified divergence ≥ ½ of the truth; the original S2b table):
{{S2b_fits}}

Coverage on this side (500 runs per (p, pipeline), at the η = 0.05 budget from the coarse scan):
{{S2b_covopp}}

![main figure](figures/cert_pipelines.png)

{{S2b_TEXT}}

## 4. S3: two-qubit CNOT with local depolarizing noise (reduced S1)

**Setup.**
- U_θ = e^{−iθ Z⊗Z}·CNOT, followed by D_p⊗D_p. θ is chosen so that the true F̂ = 0.7071.
- Product Pauli tomography: 36 inputs × 9 bases, 4 outcomes each.
- Estimator: LI followed by a CPTP projection (16×16 Dykstra).
- 40 repetitions per point, at p = 0.1, 0.03 and 0.01 (only p = 1e-2 was requested; the other two are added to fit the exponent).

{{S3_table}}

{{S3_nreq}}

![CNOT](figures/s3_cnot.png)

{{S3_TEXT}}

## 5. Figures
- `figures/s1_plugin_fhat.png`: plug-in F̂ (median and 10–90% band) against N for the six estimators. Top row: the δ = 2√p pair. Bottom row: the gate with itself.
- `figures/s1_bs_inf_overclaim.png`: fraction of runs with D_BS = +∞, and the overclaim fraction of F̂.
- `figures/s2_ratio_{scaled,fixed}.png`: certified/true ratio against N for the four quantities and the v5 theorem route.
- `figures/s2_nreq_vs_p.png`: N_req against p with fitted exponents.
- `figures/cert_pipelines.{pdf,png}`: **main letter figure** (one column, 3.4 in). Panel (a): total channel uses per channel for a converse-side certificate of −ln F̂ to η = 0.05, against p, for (a) the diamond-norm ball (Theorem C) and (b) direct Pauli-error estimation. Also shown: the hockey-stick and Uhlmann certificates through the same ball, and the N3 lower bound c₀/(η²λ̃). Panel (b): empirical coverage of the F̂ lower bound.
- `figures/s2b_pipelines_slopes.png`: opposite-side (SDP) pipelines, factor-2 criterion.
- `figures/s3_cnot.png`: CNOT against the qubit.

![S1 F](figures/s1_plugin_fhat.png)
![S1 BS](figures/s1_bs_inf_overclaim.png)

## 6. Validity, limitations and runtime (honest list)
1. **Statistical validity.**
   - Hoeffding, Clopper–Pearson and the union bound are exact, non-asymptotic statements. The regions are valid for every N, but conservative: coverage is 1.000 against a nominal 0.95.
   - A sharper valid region (empirical-Bernstein, likelihood-ratio or Bayesian credible regions *with* frequentist calibration) would shift the constants but not the slope of the variance-adaptive pipelines (a″) and (b).
   - The p⁻² of (a)/(a′) comes from variance-blind width t ∝ N^{-1/2} in the directions that pin the floor. Those directions are binomials with mean ≈ p/2, whose width is really √(p/N).
2. **Model assumption in (b).** (b) is valid only if the noise is a Pauli channel in the *known* gate frame. A coherent error not in the model would invalidate it. (a″) needs no model assumption.
3. **SDP tolerance.**
   - Clarabel's default tolerance of 1e-8 often reports `optimal_inaccurate` at tighter settings. Tight Clarabel, default Clarabel and SCS at 1e-10 agree to ≤ 4e-6 on spot checks; the margin is 1e-5.
   - No coverage failure was observed, but the bounds are not rigorous at the 1e-6 level without a dual certificate.
4. **MLE boundary fuzz.** The plug-in BS of MLE estimates is ill-posed. Without polishing, the IPM fuzz of about 1e-9 turns +∞ into finite values of 4–9 (§1g). F̂ is unaffected to 1e-3.
5. **Sample sizes.**
   - S1 uses 200 repetitions for F̂ and D_BS, but only 20 repetitions (and only 4 estimators) for E₁ and F_U, because their input optimisation is the slow step on 2 shared cores. Their quantiles are therefore coarse.
   - N_req uses 10 runs per grid point on a half-decade grid. Its resolution is about ±0.25 decade, and the fitted exponents carry roughly ±0.1.
6. **Fixed-pair N_req for E₁ and F_U is at the grid floor** (≤ 10³ for every p). The statement "p⁰" therefore means "≤ 10³ shots/setting for all p ≥ 1e-4".
7. **The v5 theorem route** uses the constants as stated in `jones_v5/DIGEST.md` (Cor 10.7, 11/2; Cor 10.8; Lemma 8.2). It never becomes non-vacuous below N = 10¹². Its deterministic requirement at p = 1e-3 is N ≈ 1.6e16 shots/setting.
8. **Not done.**
   - Bayesian or bootstrap intervals. Their coverage would be interesting precisely because the plug-in is biased.
   - GST.
   - Ĉ via paper-2 Cor "uncertain".
   - A certified CNOT pipeline (S3 is plug-in only), and MLE or hedged estimators for the CNOT.
   - S1 cells at N = 1e7–1e9 (planned originally, about 6.5 min each). The S1 grid ends at N = 1e6 for every p.
9. **Converse side, scope** (§3.1–3.4).
   - (a) is Theorem C applied to the Hoeffding trace-norm ball. It is a valid, closed-form LCB, but not the exact minimum of F̂ over the ball (a non-convex problem).
   - (b) is exact over its polytope.
   - No converse-side certificate was computed for the model-free Clopper–Pearson box (a″). That would require a concave minimisation over an 18-dimensional polytope ∩ CPTP. Whether model-free tomography reaches the 1/λ̃ law for the converse is therefore open here.
   - The D_BS converse for (a) (Theorem C item 4, which has extra continuity terms) was not evaluated.
10. **N3 as plotted.**
    - N3 is proved for estimating F̂(Ad_{PU}, Ψ) to 1 ± η with success probability 2/3.
    - Our estimand is F̂(N_{p,δ}, N_p), and the criterion is the median run.
    - The line c₀/(η²λ̃) in the figure is therefore the proved intrinsic scale, not a rigorous lower bound for this pair. A certifier at accuracy η with success probability ≥ 0.72 would give an estimator at η/2, i.e. the 4× higher line in §3.2. All measured points lie above both lines.