# auditF — Section 12 (l.2219–2358) and Appendix A (l.2377–2489)
COUNTS: FATAL 0, MAJOR 0, MINOR 4, TYPO 5. No claimed statement broken.

## MINOR
M1. Thm 12.4(d)(iv) (p.73; l.2306, proof 2328) incomplete and its hypothesis superfluous. (iv) covers only f'(∞)=+∞; operator convex f with f(0⁺)=+∞, f'(∞)<∞ (e.g. −log) is in neither (iii) nor (iv). "Φ, Ψ not mutually singular" excludes cases where the conclusion holds (Φ=0, Ψ=Ω, c_Ω≥1: D̂_f(0‖Ψ) finite, nearby pairs +∞). Printed statement true but not sharp. Check mirror_case.py (state level Y=ℂ⁴, ρ1=ρ2=(e1e1*+e2e2*)/2; D̂_{−log}(ρ1‖ρ2+te3e3*) and D̂_{x log x}(ρ1+te3e3*‖ρ2) grow like t log(1/η); D̂_{x log x}(0‖ρ2)=0 while D̂_{x log x}(te3e3*‖ρ2) diverges). Fix: replace (iv) by "If f'(∞)=+∞ or f(0⁺)=+∞, and c_Ω ≥ 1, every neighbourhood of (Φ,Ψ) contains a pair with D̂_f=+∞." Proof: if D̂_f(Φ‖Ψ)=+∞ nothing to prove; otherwise Lemma 12.3 with T̂ = T ⊕ 1_{c_Ω} (f'(∞)=∞) or T ⊕ 0_{c_Ω} (f(0⁺)=∞); then J(Φ_t)=J(Φ)+tΥΥ* (resp. J(Ψ_t)=J(Ψ)+tΥΥ*), pair converges; W_t injective and T̂ has eigenvalue 1 (resp. 0), so by Prop 7.1 Φ_t (resp. Ψ_t) has nonzero singular part and Thm 7.5(a) gives +∞. [Independently matches an earlier observation that the "not mutually singular" hypothesis is superfluous at finite-valued pairs.]
M2. Thm 12.4(c) (l.2297): "c_Ω ≥ rank T(1−T)" admits c_Ω=0, but (b) and Lemma 12.3 need c_Ω ≥ 1. For c_Ω=0, T is forced to be a projection (pair already mutually singular), conclusions hold trivially with Φ_t=Φ, Ψ_t=Ψ. Fix: "(c) If c_Ω ≥ 1 and c_Ω ≥ rank T(1−T), …" and add "If c_Ω=0 the hypothesis forces T to be a projection; the pair is then already mutually singular and the assertions hold with Φ_t=Φ, Ψ_t=Ψ."
M3. Cor 12.6 (l.2345–2352): U_n=e^{iH/n} gives U_nV ∉ ℂV only for n ≥ n_0, yet F̂(Φ,Φ_n)=0 asserted for all n. Counterexample Y=ℂ², X=ℂ, V=e1, H=πσ_x: U_1=−1, Φ_1=Φ, F̂=1 (cor126_check.py). Fix: "take U_n=e^{iH/n} for n ≥ n_0" (n_0 exists: e^{iH/n}V=V+(i/n)HV+O(n^{−2}), ℂV closed). [Same as auditG M16.]
M4. Thm A.7 (l.2479): "the completely bounded versions of the p→p norms" undefined; proof covers ‖Ξ‖_{cb,p→p} := sup_V ‖Ξ⊗1_{L(V)}‖_{p→p} only (diamond at p=1, usual cb at p=∞; Pisier / DJKR conventions for 1<p<∞ not covered). Fix: add that definition after ‖Ξ‖_{p→p} (l.2379) and state Thm A.7 for it. (appendix_final.py, worst ratio 0.954.)

## TYPO
T1. Remark A.5 (l.2455; also l.341): Nuradha–Wilde eq. (20) is the SDP for the SQUARE ROOT of their geometric fidelity ([F̂_b]^{1/2} = sup{λ : λI ≤ Tr_B W, [[Γ^N, W],[W, Γ^M]] ≥ 0}) — say "square root of".
T2. §12 opening (l.2223) "each quantity is continuous … exactly when …" overstates D̂_f (part (iii) needs 1 ≤ c_Ω and rank T(1−T) ≤ c_Ω for "only if"). [= auditG M14]
T3. Example after Cor 12.5 (l.2343): v, w = e1 ± √t e3 not unit vectors; ρ_{i,t} have trace 1+t/2; value ½ is before normalisation.
T4. Cor 12.5 (l.2336) includes ρ1=ρ2=0 where Ω=0 is excluded; add "ρ1+ρ2 ≠ 0".
T5. l.2227 "Hermitian-preserving" → "Hermiticity-preserving".

## OBSERVATIONS
O1. "If" direction of Thm 12.4(d)(iii) needs no rank hypothesis (lsc + Lemma 12.1(c)); rank needed only for "continuity ⇒ D̂_f = E_f".
O2. PSD-restriction fact at l.2383 (Watrous 2005 Thm 1) confirmed; self-contained proof for q=p via [[|X*|,X],[X*,|X|]] ≥ 0 and Lemma A.6(2).
O3. Numerics trap at eigenvalues exactly 0/1 (rounding amplified by non-Lipschitz means); two early "violations" of Prop A.1 were this artifact.

## CHECKED AND CORRECT
§12: φ_σ = ℓ_σ + η_σ, η_σ>0 on (0,1) iff m_σ≠0; mutual singularity ⇔ T projection ⇔ J(Φ)#J(Ψ)=0; Lemma 12.1(a)–(c); Rem 12.2; Lemma 12.3 (100-digit, 8 connections); Halmos dilation (12.1)/(12.2); Thm 12.4(a) (worst LHS/RHS 0.0091), (b), (c) (amplitude damping, Hellinger E_f, rank-3 Pauli limits 0.25 and 0.375 as predicted), (d)(i)–(iv) both directions; Rem 5.15; Cor 12.5 + ℂ⁴ example; Cor 12.6 (unitary qubit: ‖J(Φ_n)−J(Φ)‖_1 → 0 while F̂ ≤ 1e-6); Rem 12.7 GAD.
App A: Prop A.1 (norm bound, worst ratio 1+2e-15; p=1 trace formula 4.9e-13); Rem A.2; Lemma A.3 Ando maximality; Prop A.4; Rem A.5 (mod T1); Lemma A.6 block Cauchy–Schwarz; Thm A.7 (p→p and cb versions; (A⊗C)#(B⊗C)=(A#B)⊗C).
External: verified online Bluhm et al. 2023, Matsumoto 2018 intro, Watrous 2005, Nuradha–Wilde eq. (20). Not verified: Halmos 1969, Horn–Mathias 1990, Bhatia–Davis 1995, Ando 1979, Okayasu 2026, Kian 2026, Hiai 2019, Mosonyi–Hiai 2024, Luo–Song–Xu 2018.
Scripts: p1audit/auditF/.
