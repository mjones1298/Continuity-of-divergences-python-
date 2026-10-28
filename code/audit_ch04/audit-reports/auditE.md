# auditE — Sections 10–11 (source lines 1835–2218)
Evidence: t20_findings.py, t20_findings.log and scripts in p1audit/auditE/.

COUNTS: FATAL 0, MAJOR 0, MINOR 5, TYPO 4. No statement falsified.

## MINOR
E-M1 (l.2099–2102; affects Thm 11.3 at 2126/2136, Cor 11.4 at 2184, Cor 11.6): "tanh(d_ac/2)=2γ_ac, d_ac=d_T(Ω_1^ac,Ω_2^ac)<∞" relies on Lemma 10.1, which assumes Ω'≠0. If Q_R=0 (orthogonal Choi supports, or one sum zero), Ω_i^ac=0; with Lemma 6.5's convention log0=−∞ this gives d_ac=−∞, tanh=−1, and the min in R_φ goes negative. Counterexample: dimX=1, Y=ℂ², Φ_1=Ψ_1=e_1e_1*, Φ_2=Ψ_2=e_2e_2*: LHS=0, ‖Ω̄*‖=2, m_s=4, R_{g#}=2+min{−2√2,0}=−0.83. Fix: add "if Q_R=0 then γ_ac=h_ac=0 and we set d_ac:=0, so tanh(d_ac/2)=2γ_ac always"; in Cor 11.4 write "d_ac=0 (by this convention if Ω_1=0)".
E-M2 (l.2206): last sentence of Cor 11.6 says the equal-support hypothesis "reduces to τ→0, d_T→0"; by Lemma 11.2(a) it reduces to τ→0 and min{d_T,h_ac}→0 (strictly weaker). Example dimX=1, J(Φ_i)=J(Ψ_i)=½diag(1,ε) vs ½diag(1,ε/4): d_T=log4 fixed, h_ac~√ε→0, ΔF̂=0.375ε→0. Fix: "reads τ→0 and min{d_T,h_ac}→0; implied by τ→0, d_T→0 (Thm 10.4)".
E-M3 (l.2195, example after Remark 11.5): "unit vectors y_1≠y_2" must be "linearly independent" (y_2=−y_1 or e^{iθ}y_1 gives F̂_2=1 and W not injective). Fix: "linearly independent unit vectors with |⟨y_1,y_2⟩| close to 1".
E-M4 (l.2213): "None of the hypotheses can be dropped" unsupported for min{d_ac,h_ac}; τ_ac and m_s shown only jointly. Fix: list Cor 3.3 (τ_ac), Cor 11.4 with Θ_Φ=Θ_Ψ (m_s), Ex 11.7 (κ); for min{d_ac,h_ac}: dimX=1, J(Φ_1)=J(Ψ_1)=½·1, J(Φ_2)=J(Ψ_2)=½diag(1,s) — all other quantities vanish, ΔF̂=(1−s)/2.
E-M5 (Remark 10.9, l.2053–2055): (i) "commuting case ‖T_1−T_2‖≤2δ/(1−2δ), as a scalar computation shows" unproved; sharp value is 2δ. Proof: c_{1,2}=½±γ, d=t_1−t_2≥0 wlog. If γ≥0, c_1d=(A_1−A_2)−2γt_2≤δ ⇒ d≤2δ. If γ<0, c_2d=2γ(1−t_1)−(B_1−B_2)≤δ ⇒ d≤2δ. Equality at equal sums, t_2=0, t_1=2δ (t4_scalar.py). (ii) "constant depends only on ‖Ω̄*‖ and ν(Φ_1+Ψ_1)" imprecise since Ω̄ involves the second pair; use ‖Ω̄*‖≤½(‖Ω_1*‖+‖Ω_2*‖).

## TYPO
- l.1935: "A_i=a_1T_ia_1 … for i=1" → A_i=a_iT_ia_i, B_i=a_i(1−T_i)a_i.
- l.1929: "I=[ε,1−ε]" needs ε∈(0,½] (Cor 9.11).
- l.2007: "Finally γ≤δ" duplicates l.1985.
- Remark 10.9 dependence phrase (E-M5 ii).

## Specific points
Frames: one frame W of Ω'=Ω_1+Ω_2 with C_1+C_2=1; T_i is the derivative in frame WC_i^{1/2}; a_1↔a_2 change costs 2√2γ‖φ‖ (Lemma 10.3); W→WU leaves norms invariant. ν, λ_min^+: only Cor 10.8 uses them (via Thm 8.4, Lemma 8.2(3)); hypotheses met; SDP tests. Thompson: tanh(d_T/2)=2γ to 7e-15; use of Cor 9.11 in Thm 10.4(f) valid. No Davis–Kahan used (exact spectral projections of C_1, no gap condition); κ_i≤½ proved and attained (Ex 11.7). Hölder ½ sharp for Cor 10.7/Thm 10.4(b) and the τ_ac term (optimal constant in [√2, 11/4]; search 1.38); coefficient ½ of m_s attained; 2√2 in Lemma 10.5 optimal (Ex 10.6). Near-singular sums, rank changes, one map zero all pass except the Q_R=0 degeneracy (E-M1).

## Checked and correct
Lemmas 10.1–10.3 (17263 random, worst 0.9999); Thm 10.4(a)–(f) (4000 near-singular instances); Lemma 10.5 (max 0.924), Ex 10.6; Cor 10.7 (2.746<11/4, 2.7386<2.74); Cor 10.8 (SDP); (11.1), Lemmas 11.1(a–d), 11.2(a–c); Thm 11.3 (1669 structured instances worst 0.84; adversarial ES worst 0.94; φ-variants worst 0.9997; 24 degenerate cases); Cor 11.4 (300 trials); Rem 11.5; rank-one example (mod E-M3); Cor 11.6 (mod E-M2/E-M4); Ex 11.7; all \ref targets in region.
