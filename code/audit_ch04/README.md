# Audit of Chapter 4 (*Continuity of certain quantum channel divergences*)

A computer-assisted audit of the results of Chapter 4: seven independent checkers (A–G), each covering one part of the chapter.

| Checker | Covers (numbers as printed in the paper; in the book prefix them with 4, e.g. Theorem 7.5 → Theorem 4.7.5) |
|---|---|
| A | §2–3: the ARN theorem, the fundamental principle, the fidelity formula |
| B | §4–5: Hölder estimates, closed form for bounded f |
| C | §6–8: unbounded f, all operator convex f, the diamond norm |
| D | §9: Loewner, Schur multipliers, sharp constants |
| E | §10–11: different sums, different Choi supports |
| F | §12 and Appendix A: the dichotomy, the variational characterisation |
| G | the whole paper: claims, cross-references, citations |

## Method

The checkers did the following:
- re-derived the proofs;
- tested every formula and constant numerically, using 50–110-digit arithmetic (mpmath) near singular points;
- computed diamond norms and Schur-multiplier norms exactly by semidefinite programming.

**Results.** No statement was found false in substance, and no proof has a real gap. About 39 minor issues were found: missing hypotheses or edge cases, one wrong display, a few statements broader than the theorem they describe, and attribution slips. All are corrected in the version of Chapter 4 in this book.

**Caveat.** The searches for counterexamples were local. Treat the audit as a strong independent check, not a substitute for an expert reader.

## Contents

- `AUDIT_FINDINGS.md`: the consolidated findings, ranked by importance. Line numbers refer to the pre-audit manuscript.
- `audit-reports/auditA.md` … `auditG.md`: the per-checker reports, listing every result checked.
- `auditA/` … `auditG/`: the scripts. Each folder is self-contained; scripts import helper modules (`lib.py`, `core.py`, `schur.py`, `mplib.py`, …) from their own folder.

## Running

The requirements are the same as for the rest of `code/` (see `../requirements.txt`: numpy, scipy, mpmath, cvxpy). Run any script directly, for example:

```sh
python3 code/audit_ch04/auditA/test_fundamental.py   # fundamental principle at 80 digits, about 40 s
python3 code/audit_ch04/auditD/t_cor97.py            # Corollary 9.7 constants, a few seconds
```

Most scripts finish in seconds to a few minutes. Some adversarial searches in `auditE/` and `auditF/` run for longer; their logs (`*.log`, `*.out`) are included.
