# auditD — Section 9 (source lines 1423–1834)
COUNTS: FATAL 0, MAJOR 0, MINOR 6, TYPO 5. All explicit constants and sharpness claims survived; Schur norms computed by Haagerup SDP (~1e-10).

## MINOR
1. Lemma 9.3 / Lip^I definition (l.1477–1487): hypothesis "g C^1 on a neighbourhood of I" not met in Remark 6.10 (l.1140), which applies it to t^{1+β} on [0,1]. Conclusion still true (SDP: Lip^{[0,1]}(t^α) ≥ α for α=1.1,1.2,1.3,1.38 with excess .097,.05,.016,.0016; equality for α≥1.42, matching Bhatia–Sinha √2 threshold; t_power.py). Fix: state for g ∈ C^1(I) with one-sided endpoint derivatives (Lip^I depends only on g|_I; extend by tangent lines).
2. Thm 9.9 and Cor 9.11 allow ε=1/2 (l.1660, 1670, 1716): I={1/2}, Lip^I = sup over empty set; f=x log x gives h_f'(1/2)=2 so max_I|h_f'| ≤ Lip^I(h_f) fails. Fix: assume 0<ε<1/2 (at ε=1/2 the modulus is trivially 0).
3. Cor 9.7(c) proof, s=1 (l.1637): "diag(2,−2) has ∞-norm 2" should be Schur norm; ‖diag(d)‖_S = max|d_i| never stated (follows from D∘Z = diag(d_i z_ii) or Lemma 9.5(c) with q=0). SDP confirms 2.
4. Lemma 9.5(c) proof (l.1556): off-diagonal q̃ used as real but Y_1 only Hermitian. Fix: M real ⇒ replace (Y_1,Y_2) by entrywise real parts (still admissible). Formula matches SDP on 60 random triples to 1.6e-11.
5. Lemma 9.14 (l.1819): maximiser 1/(1+√s) of η_s asserted without computation. Fix: η_s'(t) = (1+s)(1−2t−(s−1)t²)/(1+(s−1)t)², unique zero in (0,1) at t_0 = 1/(1+√s), η_s(t_0) = (1+s)/(1+√s)² ≤ 1.
6. Definition of ‖M‖_S (l.1481): field of Z unspecified; proofs need complex Z (dilation in Lemma 9.3 uses complex N; Haagerup's theorem is complex). Fix: "Z ranges over all complex matrices".

## TYPOS (5)
- Thm 9.2 (l.1462): "the derivative t↦f(A+tB)" → derivative at t=0.
- l.1445: sentence "...use that these have to converge to zero when positive semidefinite operators have close operator norms" is meaningless and unused.
- t_± means the open domain in Thm 9.2 but the endpoints of compact I in Thm 9.6, Prop 9.10, Thm 9.9.
- Cor 9.11 proof (l.1727): ungrammatical "follows that we identify".
- Several displays in the proof of Cor 9.7(a) lack punctuation.

## OBSERVATIONS (not errors)
- † entries in Table 2 (BS, reverse BS, Rényi) are numerically exact, not just lower bounds (to 1e-9). L_f/Lip only 1.00–1.08 in examples; factor 3 in Thm 9.9(a) is worst case. Random 3×3 search approaches constants from below (Hellinger 1.49994 vs 1.5; BS 6.3853 vs 6.3863; JS 2.19720 vs 2.19722).
- Prop 9.10's hypothesis holds numerically (40 random + 105 structured g; excess ≤ 3e-11).
- Cor 9.12 constant worse than Lemma 5.4 for −t^α, as stated; ratio → 4/π at α=1/2.
- Remark 9.4 correctly phrased: the Schur norm is NOT basis invariant (t^{1.3} divided differences: 1.3 vs 1.9 after random orthogonal conjugation); only the induced norm of H ↦ U(M∘(U*HU))U* is, which is what Lemma 9.3 uses.

## CHECKED AND CORRECT
Eq (9.1) vs Hiai–Mosonyi (2.2) (fetched arXiv:1604.03089); eq (9.2); Table 1 (11 rows, mpmath 30 digits); Lemma 9.3 + Rem 9.4; Lemma 9.5(a)–(d) (√2 constant, ordering hypothesis needed); Thm 9.6(a)–(d) (80 random one-pole g); Cor 9.7(a)–(e) (ε≤1/3 restriction in (b) necessary); Thm 9.8 (20 cases); Thm 9.9(a)–(c) (40 random f; rank-two decomposition re-derived); Table 2 entries and moduli; Prop 9.10 proof; Cor 9.11; Cor 9.12; §9.4 Kubo–Ando means and Thm 9.15 (uniform over connections, dimension-free). Uses of §9 in Thms 5.13(b), 6.7(b), 7.6(a), 10.4(a),(f): hypotheses met except finding 1. Not checked: r=2 example at l.2011 (outside §9).
Scripts: p1audit/auditD/.
