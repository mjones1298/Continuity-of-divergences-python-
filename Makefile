# Makefile for the book.
#
#   make          build main.pdf with latexmk (pdflatex + bibtex, see .latexmkrc)
#   make test     build every chapter on its own with tools/test_chapter.sh
#                 (every chapters/ch*.tex: ch01_noisy_gates, ch02_overview, ch03_notation, ch04_continuity,
#                 ch05_shadows, ch06_supermult, ch07_chernoff, ch08_gate_proofs, ch09_open_problems)
#   make code-test  smoke-test the numerical code in code/ (code/smoke_test.sh; Python 3.13 and
#                 code/requirements.txt needed; about 8 minutes)
#   make clean    remove LaTeX auxiliary files and the per-chapter test builds
#   make distclean  also remove main.pdf

MAIN     := main
CHAPTERS := $(sort $(basename $(notdir $(wildcard chapters/ch*.tex))))

.PHONY: all pdf test code-test clean distclean

all: pdf

pdf:
	latexmk -pdf $(MAIN).tex

# A chapter fails the test if its stand-alone build has LaTeX errors or undefined
# citations. Undefined references to other chapters are expected and only reported.
test:
	@status=0; \
	for c in $(CHAPTERS); do \
	  echo "=== $$c"; \
	  out=$$(bash tools/test_chapter.sh $$c); \
	  echo "$$out"; \
	  echo "$$out" | grep -q '^errors: *0$$' || { echo "FAIL: $$c has LaTeX errors"; status=1; }; \
	  echo "$$out" | grep -q '^undefined citations: *0$$' || { echo "FAIL: $$c has undefined citations"; status=1; }; \
	done; \
	exit $$status

# Runs a representative fast script from every folder of code/ in a temporary copy and compares the
# numbers with the shipped results.
code-test:
	bash code/smoke_test.sh

clean:
	latexmk -c $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml $(MAIN).synctex.gz
	# remove only the per-chapter test builds; other files in build_tests/ are kept
	find build_tests -type f -name 'test.*' -delete 2>/dev/null || true

distclean: clean
	rm -f $(MAIN).pdf
