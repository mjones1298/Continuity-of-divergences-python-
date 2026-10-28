# auditA — Sections 2–3 (source lines 83–401)
COUNTS: FATAL 0, MAJOR 0, MINOR 4, TYPO/citation 4. Every statement true; proofs correct apart from small repairs.

## MINOR
1. Thm 2.12 (Kubo–Ando, l.230–239) and display in Thm 2.15 (l.273): f_σ defined only on (0,∞) but evaluated at singular operators (B ≥ 0 in A^{1/2}f(A^{−1/2}BA^{−1/2})A^{1/2}; T^{−1/2}, f(T^{−1/2}(1−T)T^{−1/2}) when T or 1−T singular). Fix: add "f_σ(0):=f_σ(0⁺)" (finite); in Thm 2.15 write "for T and 1−T invertible" in the second expression. Later uses go through g_σ(T) with endpoint limits, so nothing breaks.
2. §2.4 (l.258–262): never says Φ σ Ψ is independent of the Kraus frame of Ω, nor that it is CP but not a channel. Frame independence follows from Thm 2.15 (J(ΦσΨ)=J(Φ)σJ(Ψ)=Wg(T)W*, J injective); CP since g(T) ≥ 0; for channels only trace non-increasing since Φ#Ψ ≤ (Φ+Ψ)/2. Two random qubit channels: Tr_Y J(Φ#Ψ) eigenvalues (0.062, 0.609) (test_chan.py). Add one sentence; title "mean of channels" could mislead.
3. Prop 3.1 proof (l.341): joint concavity of F_M cited from Cree–Sikora for states but used on unnormalised pairs (Φ_ρ, Ψ_ρ). True on Pos×Pos: direct-sum property + transformer inequality with C*=(1,1) gives (AσB)+(A'σB') ≤ (A+A')σ(B+B'). Cite Kubo–Ando/Ando or add this.
4. Ω=0 (r=0): ν(Ω) (l.141) is a min over the empty set yet asserted >0. Say "Ω ≠ 0". Thms 2.2, 2.15 trivially fine for r=0.

## TYPO / citation
- l.282: sentence "We will show that transformer inequality ... holds (for more suitable matrices this already holds ...)" garbled → "We first note that the transformer inequality also holds for rectangular C, by embedding in V1⊕V2 with zero extension." (argument correct).
- Def 2.8 defines a connection on Pos(X) for one fixed X but it is used on Pos(ℂ^r), Pos(Y⊗X) and with rectangular C (proof handles it).
- l.130: [Okayasu2026] cited for "commutant elements are PSD operators in finite dimensions" — not found in that paper (low confidence; fact true).
- l.262/268: Kian Prop 3.13 is a Heisenberg-picture Kraus-coefficient formula without a Choi identity; the Choi identity is Kian Thm 5.1 (and Okayasu Prop 4.1 for #). Citing 3.13 and 5.1 together is acceptable. Index convention T_ij V_i X V_j* agrees (no lost transpose).

## OBSERVATION
Infimum in F̂ generally not attained: F(u)=tr((AA*)^T Tr_Y(J#J)) holds only for invertible A; F jumps up at singular limit (test_attain.py: λ_min=2.5464, rank-one input 3.958, F(u_s)=2.5465 at s=1e-4). Proof correct (uses usc + density, not attainment).

## CHECKED AND CORRECT
Lemma 2.1 (Douglas); Thm 2.2 (ARN Kraus form: existence, uniqueness, order bijection, J(Ψ_T)=WTW*, WW*=J(Ω), W injective iff V_k lin. indep.; 1e-15 / 1e-79 mpmath); Def 2.3, Rem 2.4 (T'=conj(U)TU^T; ν(Ω) frame invariant); Thm 2.5(1)–(3) and |||Ψ_T|||_1 formula; Rem 2.6; Defs 2.7–2.10; Thms 2.11–2.12; examples; endpoint limits g(0)=f'(∞), g(1)=f(0⁺); Thm 2.15 (fundamental principle) + Rem 2.16 — only injectivity of W needed, singular J(Ω) fine; 80-digit mpmath, 3 seeds × 24 instances, 11 connections (incl. left/right trivial, arithmetic, parallel sum, log mean, power mean, mixtures), transfer identity to 2e-78, regularisation converges at predicted rates; non-isometric injective W; Prop 3.1 (derivation incl. vec identity, (3.1), transpose (AA*)^T, reduction to pure states, usc+density; 24 instances, 1500 random inputs each); Prop 3.2 and Cor 3.3 (F̂=0 and 1, diamond norm = sin θ). Rem 3.4 citations only. Edge cases dX=1, Φ=0, dY=1 fine.
Scripts: p1audit/auditA/ (test_fundamental.py, test_logmean.py, test_transfer.py, test_effect.py, test_prop31.py, test_fhat2.py, test_fhat.py, test_attain.py, test_cor33.py, test_chan.py).
