# auditB — Sections 4–5 (source lines 402–935)
COUNTS: FATAL 0, MAJOR 0, MINOR 3, TYPO 2. No statement in the region could be broken.
Note: there is no printed "Lemma 5.7" (5.7 is the Definition of operator convex function); the lemma with part (3) and the Okayasu/Kian/Bhatia–Sinha neighbourhood is Lemma 6.4(3) (with Lemma 6.9, Remark 6.10) — checked under "Extra", all correct.

## MINOR
M1. Thms 4.5, 4.6, 5.5 and display (1.1) are false as literally read: "T_i is the ARN derivative of Φ_i with respect to Ω" never fixes one Kraus decomposition of Ω by linearly independent operators; ‖T_1−T_2‖ only makes sense in a common decomposition (T' = conj(U) T U^T).
 Counterexample (t13_ambiguity.py): X=Y=ℂ², V_1=|0⟩⟨0|, V_2=1, ‖Ω*(1)‖=2; T_1=diag(.5,.1), T_2=diag(.1,.5) in (V_1,V_2): F̂_1=0.3, F̂_2=0.5. In the permuted decomposition (V_2,V_1) the derivative of Φ_2 is diag(.5,.1)=T_1, so the literal RHS is 0 < 0.2. In a common decomposition the bound is 2√.4=1.26 ≥ 0.2.
 Fix: add "relative to one fixed Kraus decomposition of Ω by linearly independent operators, the same for both pairs" to Thms 4.5, 4.6, 5.5, the last paragraph of Prop 4.3 and (1.1); then the norm is decomposition independent.
M2. Cor 5.14 closing sentence (l.~930 "every bounded maximal f-divergence exhibits the phenomenon") and Intro l.58 ("jumps by f(0+)+f'(∞)") fail for affine f=c(t−1): operator convex, f(1)=0, f(0+)+f'(∞)=0, D_f=c(trA−trB) continuous. Jump h(1)=f(0+)+f'(∞) > 0 iff f is not c(t−1). Fix: add "f not of the form c(t−1)".
M3 (cosmetic). Lemma 5.12, Thm 5.13, Cor 5.14 still carry f(1)=0 though Remark 5.15 says it is unused (Thm 5.13(a) confirmed numerically for f=−√t and 2−√t). Fix: drop f(1)=0 there; keep only for the last sentence of Cor 5.14.

## TYPO
1. Lemma 4.1 (l.407): "fixed Kraus decomposition of Φ" needs "by linearly independent operators".
2. Remark 4.4: say the sharpness also covers the last display of Prop 4.3.

## CHECKED AND CORRECT
- Lemma 4.1 both inequalities (25 random, SDP diamond norm; equality occurs, constant 1 sharp).
- Prop 4.3(1),(2): Ando/Bhatia X.1.1 hypotheses met; sup ratio 0.9999999995; matrix sup = scalar √(δ(1−δ)) at δ=.5,.2. Remark 4.4: sup 1, not attained for δ>0.
- Thm 4.6: proof correct; 300 random (incl. eigenvalues exactly 0,1) max ratio 0.990; saturation r=1, V isometry, T_1=δ, T_2=0 gives √(1−δ)→1, so ‖Ω*(1)‖ and exponent ½ sharp as a supremum; worked example correct.
- Thm 4.5 (g_σ continuous on [0,1]); Remark 4.7; Lemma 4.8 (no circularity with Thm 5.13); Prop 4.9 (40 SDP instances); Lemma 4.10.
- Defs 5.1–5.2; Lemma 5.3 (argument order; direct minimisation over pure inputs matches λ_min formula to ~4e-6); Lemma 5.4 (exponent and leading constant sharp, ratio 0.99999994 at a=0.2; at a=½ constant 2 non-optimal but not wrong); Thm 5.5 (2000 random, max ratio 0.97); Thm 5.6 (constants a,b,m; rank(A#B)=dim(ranA∩ranB); Kian criterion re-proved).
- Defs 5.8–5.10, Remark 5.11 (D_f and h_f formulas, boundary values, operator convexity of f_H, f_Ha, f_LC, f_JS); Lemma 5.12 (Bhatia V.2.5 hypotheses met); Thm 5.13(a) tested from the DEFINITION in 50-digit mpmath (Hellinger, Hellinger_a, Vincze–Le Cam, JS, −√t, 2−√t; T with eigenvalues exactly 0 and 1; singular Choi); Thm 5.13(b) (2000 instances, max ratio 0.9999); Cor 5.14, Rem 5.15 apart from M2/M3.
- Operator-Hölder: only x^β on PSD operators used in this region.
- Attributions vs arXiv abstracts: Kian 2605.11701, Kian 2609.09915, Okayasu 2605.06019, Okayasu 2609.00554, Bhatia–Sinha (IAS abstract) match. Internal theorem numbers of external papers (KW Prop 44, FF Lemma 5, Okayasu Prop 4.11/Thm 4.7/Def 3.1, Kian Prop 3.13/Thm 5.1/Thm 4.3, Hiai–Petz 5.11, Paulsen 3.6, JKP Thm 3, Bhatia X.1.1) not verified.
## EXTRA (outside region)
Lemma 6.4(1)–(3) correct; Lemma 6.9(1)–(4) correct (constants 1/ε, βε^{−β−1}, 3−β; ratios 0.87, 0.9998); Remark 6.10 deduction valid (relies on Cor 9.7(d), checked by auditD).
Scripts: p1audit/auditB/ (lib.py, dn.py diamond SDP, mplib.py 50-digit, t1..t13*.py).
