# Proof audit of "Continuity of certain quantum channel divergences"

**Version audited:** the pre-revision manuscript (`main.tex`, 2495 lines; 84 pp.). It is identical to v5 apart from the new Acknowledgements section.
**Line numbers** refer to that `main.tex`. Result numbers are the printed ones, taken from `main.aux`.

## Summary

| Checker | Region | Fatal | Major | Minor | Typo |
|---|---|---|---|---|---|
| A | §2–3 (ARN theorem, fundamental principle, fidelity formula) | 0 | 0 | 4 | 4 |
| B | §4–5 (Hölder estimates, closed form for bounded f) | 0 | 0 | 3 | 2 |
| C | §6–8 (unbounded f, all operator convex f, diamond norm) | 0 | 0 | 3 | 3 |
| D | §9 (Loewner, Schur multipliers, sharp constants) | 0 | 0 | 6 | 5 |
| E | §10–11 (different sums, different Choi supports) | 0 | 0 | 5 | 4 |
| F | §12 and App. A (dichotomy, variational characterisation) | 0 | 0 | 4 | 5 |
| G | Whole paper: claims, cross-references, citations | 0 | 0 | 18 | 10 |

After removing overlaps between checkers, about 39 distinct minor issues remain. No statement was found to be false in substance, and no proof has a real gap.

Every issue below is one of the following:
- a missing hypothesis or edge case;
- one wrong display (§7);
- a statement that claims more than the theorem it describes;
- a slip in an attribution.

The checkers re-derived the proofs, tested every formula and constant numerically, and used 50–110-digit arithmetic near singular points. They computed diamond norms and Schur-multiplier norms exactly by SDP.

**Caveat.** The checkers are AI agents, and their searches for counterexamples were local. Treat this as a strong independent check, not a substitute for an expert reader.

Groups 1–3 below are ranked by how much they matter.

---

## 1. Things that are wrong as written (fix before the next arXiv version)

**1.1 Display after Theorem 7.5 (l.1272–1274) counts the f(0) term twice.**
- **Problem.**
  - The display reads D_f(ℙ‖ℚ) = ∫ f(dℙ_ac/dℚ) dℚ + f′(∞)ℙ_s(S) + f(0)ℚ_s(S).
  - The density dℙ_ac/dℚ is 0 on {p = 0}, so the integral already contains f(0)ℚ_s(S).
  - Example: P = (1,0), Q = (0,1) and f(t) = 2max(1−t,0) + 3max(t−1,0). The true value is 5; the display gives 7.
  - I re-derived this by hand. Theorem 7.5(c) itself is correct.
- **Fix.** Write ∫ f(dℙ_ac/dℚ_ac) dℚ_ac + f′(∞)ℙ_s(S) + f(0)ℚ_s(S), with ℙ_ac and ℚ_ac restricted to {p>0, q>0}. Alternatively, drop the last term.
- (C-F1)

**1.2 Theorems 4.5, 4.6, 5.5 and display (1.1): the frame of Ω is never fixed.**
- **Problem.** ‖T₁−T₂‖ only makes sense when T₁ and T₂ are taken in the same Kraus decomposition, because the derivative changes by T′ = Ū T Uᵀ.
- **Counterexample (literal reading).**
  - Take X = Y = ℂ², V₁ = |0⟩⟨0|, V₂ = 𝟙.
  - Take T₁ = diag(.5,.1) and T₂ = diag(.1,.5). Then F̂₁ = 0.3 and F̂₂ = 0.5.
  - Computed in the permuted frame, T₂ equals T₁, so the right-hand side is 0.
  - Lemma 4.1 already says "fixed Kraus decomposition".
- **Fix.** In Theorems 4.5, 4.6, 5.5, the last paragraph of Proposition 4.3 and (1.1), add "relative to one fixed Kraus decomposition of Ω by linearly independent operators, the same for both pairs".
- (B-M1)

**1.3 Theorem 9.9 and Corollary 9.11 fail at ε = ½ (l.1660, 1670, 1716).**
- **Problem.** At ε = ½ the interval is I = {½}, so Lip^I is a supremum over an empty set. For f = x log x, h_f′(½) = 2, and max_I|h_f′| ≤ Lip^I(h_f) fails.
- **Fix.** Assume 0 < ε < ½; at ε = ½ the modulus is trivially 0. Also l.1929, "I = [ε, 1−ε]", needs ε ∈ (0, ½].
- (D-M2, E-typo)

**1.4 Theorem 11.3: the case Q_R = 0 (orthogonal Choi supports, or one sum zero).**
- **Problem.**
  - Here Ω_i^ac = 0, and Lemma 10.1 (which assumes Ω′ ≠ 0) is used for d_ac = d_T(0, 0).
  - With log 0 = −∞ this gives tanh = −1, so R_φ can be negative.
  - Example: dim X = 1, Φ₁ = Ψ₁ = e₁e₁*, Φ₂ = Ψ₂ = e₂e₂*. Then R_{g#} = −0.83 < 0 = LHS.
  - The intended reading tanh(d_ac/2) = 2γ_ac = 0 is fine.
- **Fix.** Add "if Q_R = 0 then γ_ac = h_ac = 0 and we set d_ac := 0". In Corollary 11.4 write "d_ac = 0 (by this convention if Ω₁ = 0)".
- (E-M1)

**1.5 Corollary 12.6 (l.2345–2352): U_n = e^{iH/n} works only for n ≥ n₀.**
- **Problem.** Example: V = e₁ and H = πσ_x give U₁ = −𝟙. Then Φ₁ = Φ and F̂(Φ, Φ₁) = 1, while the corollary asserts F̂(Φ, Φₙ) = 0 for all n.
- **Fix.** Choose Hermitian H with HV ∉ ℂV, then take U_n = e^{iH/n} for n ≥ n₀. Such n₀ exists: e^{iH/n}V = V + (i/n)HV + O(n⁻²), and ℂV is closed.
- (F-M3, G-M16)

**1.6 "Every bounded maximal f-divergence jumps by f(0⁺)+f′(∞)" (Intro l.60; sentence after Corollary 5.14, l.930).**
- **Problem.** The jump is zero exactly when f is affine, f = c(t−1). Such f are operator convex with f(1) = 0, and D̂_f = c(trA − trB) is continuous.
- **Fix.**
  - l.60: "...jumps by $f(0^+)+f'(\infty)$, which is positive unless $f$ is affine (Corollary~\ref{maxf impossibility})".
  - l.930: "Thus every bounded maximal $f$-divergence with $f$ not affine exhibits the phenomenon of Corollary~\ref{impossibility corollary}...".
- (B-M2, G-M3)

**1.7 §8 opening (l.1322): "All continuity estimates of the previous sections therefore hold in the cb trace norm".**
- **Problem.**
  - Corollary 8.6 covers only Theorems 4.5, 4.6, 5.5 and 5.13(b).
  - Substituting δ′ = |||Φ₁−Φ₂|||/ν² naively breaks Corollary 6.11 once δ′ > 1. Its right-hand side ψ(d) = d(log(4/(εd)) + 1/ε) decreases for d > 1 and becomes negative for d > (4/ε)e^{1/ε}; at ε = ½, ψ(100) = −52.6.
  - "δ^α + δ^{1−α} increasing" needs α ∈ (0,1).
- **Fix.**
  - In Corollary 8.6, replace δ by min{δ, 1}. This is valid because ‖T₁−T₂‖ ≤ 1.
  - Then list Theorem 6.7(b), Corollaries 6.11 and 6.12, and Theorem 7.6(a) in it.
  - Reword l.1322 as "The continuity estimates of Sections~\ref{sec:norms} and~\ref{sec:general} for the means, the fidelity, the R\'enyi quantities of order in $(0,1)$ and the bounded maximal $f$-divergences therefore hold in the completely bounded trace norm on the splittings of a fixed map (Corollary~\ref{cor:cb continuity})" (or the extended list).
- (C-F5, G-M10)

**1.8 Abstract overclaims (l.54).**
- **(a) Moduli.** "In T, these quantities and all maximal f-divergences of operator convex f are continuous, with explicit dimension-free moduli" is too strong: unbounded f needs a spectral bound on T, and Theorem 7.6(c) is only qualitative. Replace with:
  "In $T$, the fidelity, the geometric R\'enyi quantities of order in $(0,1)$ and the bounded maximal $f$-divergences are continuous with explicit dimension-free moduli; so are the Belavkin--Staszewski relative entropy and the maximal $f$-divergences of operator convex $f$ whose derivative has spectrum in a compact interval on which $h_f$ is finite; and every maximal $f$-divergence of an operator convex $f$ is continuous in $T$ with values in $(-\infty,+\infty]$;"
- **(b) Unitary gates.** "Unitary gates are not [continuity points]" is too broad. (U, Ψ) with Ψ of full Kraus rank is a continuity point (Theorem 12.4(a)); only (U, U) fails (Corollary 12.6). Replace with: "pairs containing a noisy channel of full Kraus rank are continuity points, whereas the pair $(U,U)$ of an ideal unitary gate with itself is not."
- (G-M1, M2)

**1.9 Proposition 3.2 (l.354) and Corollary 3.3 (l.375) say "completely positive maps".**
- **Problem.** The proofs build channels, and the abstract and introduction say channels.
- **Fix.** Use "pairs of channels from $\mathcal{X}$ to $\mathcal{Y}$". As a check, |||Φ₁−Φ₂|||₁ = ½‖φ−ψ‖₁ = sin θ.
- (G-M6)

## 2. Incomplete statements (true, but a case is missing or a hypothesis is stronger than needed)

**2.1 Theorem 12.4(d)(iv) (l.2306, proof l.2328).**
- **Problem.**
  - It covers only f′(∞) = +∞. Operator convex f with f(0⁺) = +∞ and f′(∞) < ∞ (for example −log) falls under neither (iii) nor (iv).
  - The hypothesis "not mutually singular" is superfluous. For example, Φ = 0 and Ψ = Ω with c_Ω ≥ 1 also jumps to +∞.
- **Fix.** Replace (iv) by: "If $f'(\infty)=+\infty$ or $f(0^+)=+\infty$, and $c_\Omega\ge1$, every neighbourhood of $(\Phi,\Psi)$ contains a pair with $\widehat D_f=+\infty$."
- **Proof of the replacement.**
  - If D̂_f(Φ‖Ψ) = +∞ there is nothing to prove.
  - Otherwise apply Lemma 12.3 with T̂ = T ⊕ 𝟙_{c_Ω} (when f′(∞) = ∞) or T̂ = T ⊕ 0 (when f(0⁺) = ∞).
  - Then J(Φ_t) = J(Φ) + tΥΥ* (respectively J(Ψ_t) = J(Ψ) + tΥΥ*).
  - T̂ has eigenvalue 1 (respectively 0), so by Proposition 7.1 there is a nonzero singular part, and Theorem 7.5(a) gives +∞.
- (F-M1)

**2.2 Theorem 12.4(c) (l.2297): the hypothesis allows c_Ω = 0, but (b) and Lemma 12.3 need c_Ω ≥ 1.**
- **Fix.** Write "If $c_\Omega\ge1$ and $c_\Omega\ge\operatorname{rank}T(\mathbf 1-T)$, …". Add that for c_Ω = 0 the hypothesis forces T to be a projection, so the assertions hold with Φ_t = Φ and Ψ_t = Ψ.
- (F-M2)

**2.3 Lemma 9.3 / definition of Lip^I (l.1477–1487) vs Remark 6.10 (l.1140).**
- **Problem.** The lemma assumes g is C¹ on a neighbourhood of I. Remark 6.10 applies it to t^{1+β} on [0,1], which is not defined left of 0. The conclusion is still true; the SDP reproduces the Bhatia–Sinha √2 threshold.
- **Fix.** State the lemma for g ∈ C¹(I) with one-sided derivatives at the endpoints. Lip^I depends only on g|_I, which can be extended by tangent lines.
- (D-M1)

**2.4 Remark 10.9 (l.2053–2055).**
- **(i) The commuting-case bound.** "‖T₁−T₂‖ ≤ 2δ/(1−2δ), as a scalar computation shows" is unproved, and the sharp value is 2δ.
  - Write c₁,₂ = ½ ± γ and d = t₁ − t₂ ≥ 0.
  - If γ ≥ 0, then c₁d = (A₁−A₂) − 2γt₂ ≤ δ, so d ≤ 2δ.
  - If γ < 0, then c₂d = 2γ(1−t₁) − (B₁−B₂) ≤ δ, so d ≤ 2δ.
  - Equality holds for equal sums with t₂ = 0 and t₁ = 2δ.
- **(ii) The constant.** It "depends only on ‖Ω̄*‖ and ν(Φ₁+Ψ₁)", but Ω̄ involves the second pair. Use ‖Ω̄*‖ ≤ ½(‖Ω₁*‖ + ‖Ω₂*‖).
- (E-M5)

**2.5 Corollary 11.6 and the example after Remark 11.5.**
- **(a) l.2206.** The equal-support hypothesis reduces to "τ→0 and min{d_T, h_ac}→0", which is weaker than "τ→0, d_T→0". Example: dim X = 1, ½diag(1,ε) vs ½diag(1,ε/4). Here d_T = log 4 stays fixed while h_ac→0.
- **(b) l.2195.** "Unit vectors y₁ ≠ y₂" must be "linearly independent unit vectors with |⟨y₁,y₂⟩| close to 1".
- **(c) l.2213.** "None of the hypotheses can be dropped" needs examples:
  - Corollary 3.3 for τ_ac;
  - Corollary 11.4 with Θ_Φ = Θ_Ψ for m_s;
  - Example 11.7 for κ;
  - for min{d_ac, h_ac}: dim X = 1, J(Φ₁) = J(Ψ₁) = ½𝟙 and J(Φ₂) = J(Ψ₂) = ½diag(1,s). All other quantities vanish but ΔF̂ = (1−s)/2.
- (E-M2–M4)

**2.6 Theorem A.7 (l.2479): "the completely bounded versions of the p→p norms" is undefined.**
- **Problem.** The proof covers ‖Ξ‖_{cb,p→p} := sup_V ‖Ξ⊗𝟙_{L(V)}‖_{p→p}, which is the diamond norm at p = 1. It does not cover operator-space conventions for 1 < p < ∞.
- **Fix.** Define it after l.2379 and state Theorem A.7 for that norm.
- (F-M4)

**2.7 Standing hypotheses and conventions.**
- **(a) Kubo–Ando formula (Thm 2.12, l.230–239; Thm 2.15 display, l.273).** f_σ is evaluated at singular operators.
  - Add "f_σ(0) := f_σ(0⁺)".
  - In the second expression of Theorem 2.15, add "for T and 𝟙−T invertible".
  - (A-M1)
- **(b) Ω ≠ 0.** Add it where ν(Ω) > 0 is asserted (l.141) and in Corollary 7.7. Corollary 7.7 also needs the conventions θ(−∞) = 0, θ(+∞) = 1, with the right-hand side +∞ where h_f = +∞. (A-M4, C-F4)
- **(c) Corollary 12.5 (l.2336).** Add "ρ₁ + ρ₂ ≠ 0". (F-T4)
- **(d) Lemma 5.12, Theorem 5.13, Corollary 5.14.** They still carry f(1) = 0, which Remark 5.15 says is unused. Drop it except for the last sentence of Corollary 5.14. This is cosmetic. (B-M3)

**2.8 Missing one-line justifications.**
- **(a) §2.4 (l.258–262).** Say that Φ σ Ψ does not depend on the Kraus frame (from Theorem 2.15 and injectivity of J). Say also that it is CP but, for channels, only trace non-increasing: Φ#Ψ ≤ (Φ+Ψ)/2. For two random qubit channels, Tr_Y J(Φ#Ψ) has eigenvalues (0.062, 0.609). The title "mean of channels" could mislead. (A-M2)
- **(b) Proposition 3.1 proof (l.341).** Joint concavity of F_M is cited for states but used on unnormalised pairs. It holds on Pos × Pos: the direct-sum property and the transformer inequality with C* = (𝟙, 𝟙) give (AσB) + (A′σB′) ≤ (A+A′)σ(B+B′). Cite Kubo–Ando/Ando or add this. (A-M3)
- **(c) Lemma 9.5(c) proof (l.1556).** The off-diagonal entry q̃ is used as real, but Y₁ is only Hermitian. Since M is real, replace (Y₁, Y₂) by their entrywise real parts. (D-M4)
- **(d) Lemma 9.14 (l.1819).** Show the maximiser: η_s′(t) = (1+s)(1−2t−(s−1)t²)/(1+(s−1)t)², whose only zero in (0,1) is t₀ = 1/(1+√s). Then η_s(t₀) = (1+s)/(1+√s)² ≤ 1. (D-M5)
- **(e) Definition of ‖M‖_S (l.1481).** Say "Z ranges over all complex matrices". (D-M6)
- **(f) Corollary 9.7(c), s = 1 (l.1637).** "diag(2,−2) has ∞-norm 2" should say Schur norm. Also state ‖diag(d)‖_S = max|d_i|. (D-M3)
- **(g) Lemma 4.8 (l.563) ↔ Theorem 5.13 (l.849).** These cite each other. It isn't circular, but say so: "...continuous on $[0,1]$ by the first paragraph of the proof of Theorem~\ref{thm:maxf} (up to the sentence `Hence $h_f$ is continuous on $[0,1]$'), which does not use the present lemma." (G-M8)

## 3. Statements broader than the results they describe

- **G-M9. §9 title (l.1423).**
  - Replace with "Explicit moduli for all operator convex functions and all means, with sharp or near-sharp constants".
  - Reason: Theorem 9.9(a) is sharp only to within a factor 3.
- **G-M14 / F-T2. §12 opening (l.2221–2223).**
  - Change to "for the fidelity, $\widehat{Q}_\alpha$ and every mean other than a weighted arithmetic mean, the quantity is continuous at a pair exactly when …".
  - Add "For bounded maximal $f$-divergences this holds under the rank condition of Theorem~\ref{thm:dichotomy}(d)(iii)."
- **G-M15. l.2333.**
  - Replace "comparable to the square root of the smallest eigenvalue" with "$\lambda_{\min}(J(\Omega))/\min\{\dim\mathcal{X},\dim\mathcal{Y}\}\le\nu(\Omega)^2\le\lambda_{\min}(J(\Omega))$ for invertible $J(\Omega)$ (Lemma~\ref{lem:nu}(2))".
- **G-M17. §13, first sentence (l.2361).**
  - Restrict to what is proved: "Channel divergences of the form $\mathsf{H}(J(\Phi)\,\sigma\,J(\Psi))$ with $\mathsf{H}(0)=0$ and $f_\sigma(0^+)=f_\sigma'(\infty)=0$, such as the geometric mean, vanish on pairs with trivially intersecting Choi ranges (Theorem~\ref{range theorem}); the fidelity, $\widehat{Q}_\alpha$ and the bounded maximal $f$-divergences with $f$ not affine are not continuous in the diamond norm, even for pairs with the same sum (Corollaries~\ref{impossibility corollary} and~\ref{maxf impossibility})."
- **G-M18. Open problem 5 (l.2373).**
  - Proposition 4.3(1) needs only ‖AB‖_ui ≤ ‖A‖_ui‖B‖_∞, not Ando.
  - Rephrase: "Proposition~\ref{K6}(1) holds in every unitarily invariant norm, since $\|AB\|_{\mathrm{ui}}\le\|A\|_{\mathrm{ui}}\|B\|_\infty$, and, by Ando's inequality \cite{Ando1988}, Proposition~\ref{K6}(2) becomes …".
- **G-M4. Intro (l.72), generalised amplitude damping.**
  - Write "with damping $p\in(0,1]$ and thermal parameter $N\in(0,1)$".
  - Reason: N = 0 and N = 1 give Kraus rank 2.
- **C-F3 / G-M11. Lemma 8.1 (l.1327–1329).**
  - The text says "every unitarily invariant norm", but the display shows only ‖·‖_∞.
  - Write ‖·‖ on both sides. The proof gives equal singular values, so the general statement is what's proved.

## 4. Citations and attributions

- **G-M5(a). Raginsky (l.79).**
  - Eq. (17) concerns derivatives relative to a fixed reference map A ↦ τ(A)𝟙, not relative to the pair's sum. The remark at l.1408 already says this correctly.
  - Replace with "and, for derivatives taken with respect to a fixed reference map, a lower bound with the dimensional factor $(\dim\mathcal{H})^{-2}$ \cite[eq.~(17)]{Raginsky2003}".
- **G-M5(b). Okayasu Prop. 4.11 (l.79).**
  - It gives criteria "for a dense domain and for boundedness", not "for finiteness".
- **G-M12 / C-F6. l.1275.**
  - arXiv:2609.00554 Def. 4.10 is one-sided absolute continuity; mutual absolute continuity is Prop. 4.11(3).
  - Replace with "mutually absolutely continuous, that is $\Phi\ll\Psi$ and $\Psi\ll\Phi$ \cite[Proposition~4.11(3), Definition~4.10]{Okayasu2026maxf}".
- **G-M7. l.630–632, Katariya–Wilde.**
  - Lemma 5.3 is stated for α ∈ (0,1). The range (1, 2] is Fang–Fawzi Lemma 5, which the checker read.
  - KW21 "Prop. 44" could not be located because the PDF fetch was truncated. Please check the number.
  - Suggested text: "For $\alpha\in(0,1)$ this quantity admits a closed form in terms of the Choi matrices \cite{KatariyaWilde2021}, and for $\alpha\in(1,2]$ one is given in \cite[Lemma~5]{FangFawzi2021}."
- **G-M13. l.1187, Hiai–Mosonyi.**
  - Cor. 3.4 and Prop. 3.7 concern the standard S_f, not the maximal divergence.
  - Suggested text: "(the case $\dim\mathcal{X}=1$ of Theorem~\ref{thm:all f}(a); for the standard divergence see \cite[Corollary~3.4]{HiaiMosonyi2017})" and "(immediate from the reverse-test definition \cite{Matsumoto2018newf}; for the standard divergence see \cite[Proposition~3.7]{HiaiMosonyi2017})".
- **F-T1. l.341 and Remark A.5 (l.2455), Nuradha–Wilde eq. (20).**
  - It is the SDP for the square root of their geometric fidelity. Say "square root of".
- **A. l.130, Okayasu2026.**
  - It is cited for "commutant elements are PSD operators in finite dimensions". The checker could not find this in the paper (low confidence); the fact itself is true.
- **A. l.262/268, Kian.**
  - Prop. 3.13 is a Heisenberg-picture Kraus formula. The Choi identity is Kian Thm 5.1, and Okayasu Prop. 4.1 for #. Citing both is acceptable.
- **Optional.**
  - Kueng–Long–Doherty–Flammia (l.60, l.1322): the abstract does not mention the diamond norm, so consider "see, e.g.,".
  - Bhatia–Gaubert–Jain 2019 has a published Correction about the barycentre claim.
  - LuoSongXu2018 is the 2019 journal version.
- **Verified against sources the checkers read:**
  - Kian 2605.11701 (Rem. 3.3, Prop. 3.5, 3.13, Rem. 4.2, Thm 4.3, Thm 5.1);
  - Kian 2609.09915;
  - Okayasu 2605.06019 (Prop. 4.1, 3.5, Thm 7.3);
  - Okayasu 2609.00554 (Def. 3.1, Thm 4.7, Prop. 4.11);
  - Cree–Sikora (eq. 1.5, Lemmas 3.3, 3.13, 3.17);
  - Matsumoto; Fang–Fawzi Lemma 5;
  - Nuradha–Wilde (eqs. 14, 19, 20);
  - Hiai–Mosonyi eq. (2.2);
  - Bluhm et al. 2023; Bhatia–Sinha; Johnston et al.; Watrous 2005; Khatri–Sharma–Wilde; Maciejewski et al.; Reeb–Kastoryano–Wolf; Pitrik–Virosztek.
- **Not accessible:** book locators (Bhatia, Hiai–Petz, Watrous, Paulsen) and about 22 classical references.

## 5. Typos and small wording

- **l.282** — the garbled sentence becomes "We first note that the transformer inequality also holds for rectangular $C$, by embedding in $\mathcal V_1\oplus\mathcal V_2$ with zero extension."
- **l.407 (Lemma 4.1)** — "fixed Kraus decomposition of Φ by linearly independent operators".
- **Remark 4.4** — say that sharpness also covers the last display of Proposition 4.3.
- **l.948–949** — "For $A\ge0$ and $B$ positive definite we write $\mathsf P_f$".
- **l.1080** — clumsy "by considering … that this corresponds".
- **l.1428** — "many of which can be found in the paper".
- **l.1445** — delete the meaningless sentence "...use that these have to converge to zero when positive semidefinite operators have close operator norms".
- **l.1462 (Theorem 9.2)** — "the derivative of $t\mapsto f(A+tB)$ at $t=0$".
- **t_±** — means the open domain in Theorem 9.2 but the endpoints of a compact I in Theorem 9.6, Proposition 9.10 and Theorem 9.9.
- **l.1727** — "follows once we identify".
- **Corollary 9.7(a)** — punctuation of displays.
- **l.1935** — A_i = a_i T_i a_i, B_i = a_i(𝟙−T_i)a_i (it currently reads a₁).
- **l.2007** — "Finally γ ≤ δ" duplicates l.1985.
- **l.2227** — "Hermiticity-preserving".
- **l.2343** — v, w = e₁ ± √t e₃ are not unit vectors. The value ½ is before normalisation.
- **Definition 2.8** — defined on Pos(X) for one fixed X, but used on several spaces and with rectangular C. The proof handles this.
- **Stale label** — eq:3.16 (prints as (9.3), never referenced).
- **Build** — the last pass logs "Label(s) may have changed". One more pdflatex pass clears it; the labels are final.

**Acknowledgements:** left to the author.

## 6. Confirmed positively (worth knowing)

**Constants confirmed sharp or attained:**
- Theorem 4.6: the constant ‖Ω*(𝟙)‖ and the exponent ½ are sharp as a supremum (r = 1, V an isometry, T₁ = δ, T₂ = 0).
- Lemma 4.1: constant 1.
- Proposition 4.3: the matrix supremum equals the scalar one.
- Lemma 5.4: the exponent and leading constant are sharp.
- Theorem 8.4: both constants, ν² and ‖Ω*(𝟙)‖, are attained (diamond norm computed exactly by SDP).
- Lemma 10.5: the 2√2 is optimal (Example 10.6 approaches it).
- Theorem 11.3: the coefficient ½ of m_s is attained.
- Corollary 10.7: the Hölder exponent ½ is sharp, and the optimal constant lies in [√2, 11/4].

**Section 9:**
- All constants in Section 9 survived exact SDP computation of the Schur norms.
- The † entries in Table 2 (BS, reverse BS, Rényi) are exact, not just lower bounds.
- L_f/Lip is 1.00–1.08 in the examples, so the factor 3 in Theorem 9.9(a) is a worst case.
- Remark 9.4 is correctly phrased: the Schur norm itself is not basis invariant.

**Foundations:**
- The fundamental principle (Theorem 2.15) was verified at 80 digits on singular sums for 11 connections, including those with nonzero endpoint values. Only injectivity of W is needed.
- The closed forms (Theorems 5.13(a), 6.7(a), 7.5) were checked against the definition at 50 digits, including with T having eigenvalues exactly 0 and 1.

**Dichotomy and references:**
- Every direction of every "iff" in Theorem 12.4 was confirmed on explicit constructions: amplitude damping, a rank-3 Pauli channel and unitary qubit channels.
- All 499 cross-references point to the intended result, and none is circular.

## Files

- Per-checker reports, with the full list of results checked: `audit-reports/audit{A–G}.md` (inside `audit-scripts.zip`).
- Scripts that reproduce each counterexample and test: `audit-scripts.zip`.
