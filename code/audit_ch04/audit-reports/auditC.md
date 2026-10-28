# auditC — Sections 6–8 (source lines 936–1422)
COUNTS: FATAL 0, MAJOR 0, MINOR 3, TYPO 3. Every statement re-derived and tested; none false, no proof gaps.

## MINOR
F1. Classical formula after Thm 7.5 (l.1272–1274) double-counts the f(0) term. Printed: D_f(P‖Q) = ∫ f(dP_ac/dQ) dQ + f'(∞)P_s(S) + f(0)Q_s(S). The density dP_ac/dQ is 0 on {p=0}, so the integral already contains f(0)Q_s(S). Counterexample: S={1,2}, P=(1,0), Q=(0,1), f(t)=2max(1−t,0)+3max(t−1,0) (f(0)=2, f'(∞)=3, f(1)=0): true value 5, printed 7; on random pairs overshoots by exactly f(0)Q_s(S) (t20_classical.py). Thm 7.5(c) itself correct (P_a, P_1, P_0 orthogonal). Fix: ∫ f(dP_ac/dQ_ac) dQ_ac + f'(∞)P_s(S) + f(0)Q_s(S), with P_ac, Q_ac restricted to {p>0,q>0}; or drop the third term. [Coordinator re-derived the counterexample by hand: confirmed.]
F4. Cor 7.7 (l.1308) omits Ω ≠ 0 and endpoint conventions (ν(Ω) needs Ω ≠ 0; D_max can be ±∞; h_f can be +∞). Fix: add "Ω ≠ 0" and "with θ(−∞)=0, θ(+∞)=1 and right-hand side +∞ where h_f is +∞". BS numerics right (h_f(θ(x)) = θ(x)x ≥ x/2; m_f = −0.27846 > −0.28).
F5. §8 intro (l.1322) and Cor 8.6 (l.1416) overreach. Intro says "all continuity estimates of the previous sections" transfer; Cor 8.6 lists only Thms 4.5, 4.6, 5.5, 5.13(b). Replacing ‖T1−T2‖ by δ' = |||Φ1−Φ2|||/ν² is false for Cor 6.11 once δ' > 1: its RHS ψ(d) = d(log(4/(εd)) + 1/ε) increases only on d ≤ 1 and turns negative for d > (4/ε)e^{1/ε} (ε=½: d > 59.1; ψ(100) = −52.6 while LHS ≥ 0; t21_cor611_monot.py). "δ^α + δ^{1−α} increasing" holds only for α ∈ (0,1) (Thm 5.5's range; state it). Fix: replace δ by min{δ,1} (valid since ‖T1−T2‖ ≤ 1); list Thm 6.7(b), Cors 6.11, 6.12, Thm 7.6(a) in Cor 8.6; intro: "the estimates listed in Corollary 8.6".

## TYPO
- F2 (l.948–949): "For positive definite A,B we write P_f" later used for A ≥ 0, B > 0 → "A ≥ 0 and B positive definite".
- F3 (Lemma 8.1, l.1328): text says "every unitarily invariant norm" but display has ‖·‖_∞ only; identity holds in every u.i. norm (W=U|W| ⇒ middle operator UKU*; Schatten 1,2,∞ to 2e-13). Write ‖·‖ on both sides.
- F6 (l.1275): Okayasu "Definition 4.10" cited for mutual absolute continuity; characterisation is Prop 4.11 → "Prop 4.11 and the definition preceding it".

## OBSERVATIONS
- 3−β in Lemma 6.9(4) not sharp (optimal 2 at β=1); Lemma 6.9(3) valid but loose; Rem 6.10 (Bhatia–Sinha) confirmed.
- ε ≤ 1/3 threshold in the χ² remark is real (ε=0.36: Lip 3.7187 > ε^{−2}−4 = 3.7160).
- Abstract's "explicit dimension-free moduli" for all operator convex f needs Thm 9.8's interval condition (outside region).
- x² term of the Loewner/Hiai–Mosonyi representation correctly allowed; ‖κ_s‖_Lip = 1/(s(1+s)) on [0,1] to 8 digits (consistent with Rem 6.13).

## CHECKED AND CORRECT
Lemma 6.4 (Def 6.2) incl. singular pairs; Lemma 6.5 (D_max formula; mpmath at eigenvalue 1−1e-10); Thm 6.7(a) closed form vs η-regularised definition (t log t, t^α, (t−1)², −log t); Thm 6.7(b), Cors 6.11, 6.12 (200 random); Lemma 6.9(1)–(4); Rem 6.10, 6.13, 6.14; Prop 7.1 (Ando decomposition); Lemmas 7.3, 7.4; Thm 7.5(a,b,c) (all four finite/infinite combinations); Thm 7.6(a,b,c); Lemma 8.1; Lemma 8.2 (λ⁺_min/min(dX,dY) ≤ ν² ≤ λ⁺_min); Thm 8.4 (ν²‖K‖ ≤ |||Ψ_K|||_1 ≤ ‖Ω*(1)‖‖K‖, Watrous dual SDP, both constants attained); Cor 8.6; read-only: Effros / Ebadian–Nikoufar–Gordji hypotheses, Hahn–Banach/polar step in Thm 8.4, Okayasu Def 3.1, Prop 4.11, Thm 4.7.
Scripts: p1audit/auditC/.
