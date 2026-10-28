#!/usr/bin/env bash
# Usage: tools/test_chapter.sh ch04_continuity
# Builds a throwaway document containing only chapters/<name>.tex, in build_tests/<name>/.
# References to OTHER chapters will show as undefined (expected); everything else must be clean.
set -euo pipefail
NAME="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build_tests/$NAME"
mkdir -p "$OUT"
cat > "$OUT/test.tex" <<EOF
\documentclass[11pt,oneside]{book}
\input{preamble}
\begin{document}
\input{chapters/$NAME}
\bibliographystyle{plainnat}
\bibliography{bib/references}
\end{document}
EOF
cd "$ROOT"
export TEXINPUTS="$ROOT//:"
export BIBINPUTS="$ROOT:"
pdflatex -interaction=nonstopmode -output-directory "$OUT" "$OUT/test.tex" >/dev/null || true
( cd "$OUT" && BIBINPUTS="$ROOT:" bibtex test >/dev/null || true )
pdflatex -interaction=nonstopmode -output-directory "$OUT" "$OUT/test.tex" >/dev/null || true
pdflatex -interaction=nonstopmode -output-directory "$OUT" "$OUT/test.tex" >/dev/null || true
LOG="$OUT/test.log"
echo "errors:    $(grep -c '^!' "$LOG" || true)"
grep -A2 '^!' "$LOG" | head -20 || true
echo "undefined refs (other chapters expected): $(grep -c 'Reference .* undefined' "$LOG" || true)"
grep -o "Reference \`[^']*' on page" "$LOG" | sort -u | head -40 || true
echo "undefined citations: $(grep -c 'Citation .* undefined' "$LOG" || true)"
grep -o "Citation \`[^']*'" "$LOG" | sort -u | head -20 || true
echo "overfull >10pt:"; grep -oE 'Overfull \\hbox \(([1-9][0-9]+\.[0-9]+)pt' "$LOG" | head -10 || true
echo "pages: $(pdfinfo "$OUT/test.pdf" 2>/dev/null | awk '/^Pages:/{print $2}' | grep . || echo none)"
