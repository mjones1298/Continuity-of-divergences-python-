# Continuity, Testing Regions and Fragility of Quantum Channel Divergences

*M. Jones*

This repository contains the LaTeX source of a book on quantum channel divergences.

The book opens with noisy quantum gates. Converse bounds on discriminating and estimating a gate are evaluated on a characterized model of it, and the first chapter asks whether their sensitivity to characterization error belongs to the task or only to the bound.

The rest of the book develops the general theory. It asks when converse bounds for quantum channels, and the operational exponents they bound, depend continuously on the channels, and what controls the size of the error when they do. The bounds in question include:
- geometric fidelities;
- geometric Rényi and Belavkin–Staszewski divergences;
- hockey-stick and Umegaki divergences.

The answers are organised around three classical "shadows" of a pair of states or channels:
- the reverse test, through the Arveson–Radon–Nikodym derivative;
- the testing region, through the hockey-stick profile;
- the modular shadow, through Petz divergences.

These are then applied to adaptive channel discrimination. A chapter of proofs returns to the noisy gates of the opening chapter.

Chapter 4 is a revised and extended version of M. Jones, *Continuity of certain quantum channel divergences*, arXiv:2610.16270.

The built PDF has 262 pages: front matter, nine chapters and a bibliography.

**Read the book:** [`monograph.pdf`](monograph.pdf), a 262-page PDF built from this source. It is also rebuilt by the GitHub Actions workflow on every push.

## Contents

| Chapter | File | Summary |
|---|---|---|
| Abstract | `chapters/abstract.tex` | One-page abstract of the whole book. |
| 1. Genuine and spurious fragility of quantum channel divergences near noisy gates | `chapters/ch01_noisy_gates.tex` | Which discontinuities belong to the task and which to the bound; noise-floor laws near noisy gates; certification from tomographic data; a plan of the book. |
| 2. Overview and reading guide | `chapters/ch02_overview.tex` | The general questions, the three shadows, the main results of each chapter, chapter dependencies and a reading guide. |
| 3. Notation and conventions | `chapters/ch03_notation.tex` | Common notation, and tables of the conventions that differ between chapters (Choi ordering, fidelities, vectorisation, symbols). |
| 4. Continuity of certain quantum channel divergences | `chapters/ch04_continuity.tex` | Kubo–Ando channel divergences are continuous in the Arveson–Radon–Nikodym derivative; the continuity dichotomy at singular Choi matrices; Choi-floor estimates. |
| 5. Shadows of quantum divergences: testing regions, modular lifts and continuity | `chapters/ch05_shadows.tex` | Testing regions and modular lifts; uniform continuity of hockey-stick divergences; continuity of the adaptive Stein exponent. |
| 6. Super-multiplicativity of hockey-stick Rényi quasi-entropies | `chapters/ch06_supermult.tex` | The super-multiplicativity conjecture for testing-region Rényi quasi-entropies and the cases in which it is proved. |
| 7. Adaptive quantum channel discrimination as a controlled random walk | `chapters/ch07_chernoff.tex` | The level decomposition, the geometric Chernoff converse, when it is attained, and the fragility of single-letter converses. |
| 8. Noisy gates: proofs and further results | `chapters/ch08_gate_proofs.tex` | Proofs of the results of Chapter 1, closed forms for Pauli channels, amplitude damping, certification bounds and numerical methods. |
| 9. Open problems | `chapters/ch09_open_problems.tex` | The open problems and conjectures of all chapters, grouped by theme. |

Shared files:
- `main.tex`: the book.
- `preamble.tex`: packages, theorem environments and global macros.
- `bib/references.bib`: the bibliography.
- `figures/`: the figures of Chapters 1 and 8.
- `tools/test_chapter.sh`: builds one chapter on its own.
- `code/`: the code and data of the numerical checks (see "Code and data").

## Building

Requirements:
- A TeX Live 2022 or later installation (or an equivalent MiKTeX) with `pdflatex`, `bibtex` and `latexmk`.
- The packages loaded in `preamble.tex`: amsmath, amssymb, amsthm, mathtools, mathrsfs, bm, cancel, xfrac, tikz-cd (which loads TikZ), xcolor, graphicx, booktabs, array, longtable, multirow, enumitem, natbib, hyperref, bookmark, geometry and lmodern. A full TeX Live installation (`scheme-full`) contains all of them.

```sh
make            # latexmk -pdf main.tex  ->  main.pdf
make test       # build every chapter on its own (tools/test_chapter.sh) and report problems
make code-test  # smoke-test the numerical code in code/ (Python; about 8 minutes)
make clean      # remove auxiliary files and the per-chapter test builds
```

Without `make`, run either of:
- `latexmk -pdf main.tex`;
- `pdflatex main`, `bibtex main`, `pdflatex main`, `pdflatex main`.

The `.latexmkrc` file selects pdflatex and bibtex.

To build one chapter on its own, run `bash tools/test_chapter.sh ch04_continuity` (or any other chapter name). The output goes to `build_tests/<chapter>/`. References to other chapters show as undefined in such a build; this is expected.

The GitHub Actions workflow in `.github/workflows/build.yml` builds `main.pdf` on every push and pull request. It uploads the PDF as the artifact `monograph-pdf`.

## Code and data

The Python code and data behind the numerical checks are in [`code/`](code/). [`code/README.md`](code/README.md) explains how to install and run them, and has a table that maps each claim, figure and table of the book (by chapter and section number) to the script and the result file it comes from, together with runtimes and random seeds. It covers:
- the noisy-gate study of Chapters 1 and 8, including the scripts for Figures 1.1–1.6, 8.1 and 8.2;
- numerical verifications of results of Chapter 4 and of the hockey-stick continuity of Chapter 5;
- the amplitude-damping computations of Section 7.6.1;
- independent checks and a referee's re-computations (`code/checks/`).
- an independent, computer-assisted audit of every result of Chapter 4 (`code/audit_ch04/`). Seven checkers re-derived the proofs and tested every formula and constant numerically, in 50–110-digit arithmetic and with exact SDP norms. No statement was found false, and the minor corrections are incorporated in the book.

The numerical checks of Sections 5.9, 6.11 and 7.7 were computed with the author's own code, which is not included.

Requirements: Python 3.13 and the pinned packages in `code/requirements.txt` (numpy, scipy, cvxpy with the Clarabel and SCS solvers, mpmath, sympy, matplotlib).

```sh
python3 -m pip install -r code/requirements.txt
make code-test     # smoke test: runs a fast script from every folder of code/ and compares with the shipped results
```

`code/EXCLUDED.md` lists the scripts that were left out (superseded versions, scratch, and runs whose values were later shown to be artifacts), and `code/DATA_NOT_INCLUDED.md` the data that is not shipped and how to regenerate it.

Repository: https://github.com/mjones1298/Continuity-of-divergences-python-

## How to cite

Use the metadata in `CITATION.cff`, or the entry below.

```bibtex
@book{Jones2026book,
  author = {Jones, M.},
  title  = {Continuity, Testing Regions and Fragility of Quantum Channel Divergences},
  url    = {https://github.com/mjones1298/Continuity-of-divergences-python-},
  year   = {2026}
}
```

## License

License: to be chosen by the author.

<!-- TODO(author): choose a license for the text and the LaTeX source, and add a LICENSE file. -->
