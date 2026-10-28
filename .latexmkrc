# latexmk configuration for the combined monograph.
# pdflatex + bibtex; the bibliography is bib/references.bib.
$pdf_mode = 1;                 # build a PDF with pdflatex
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$bibtex_use = 2;               # run bibtex when needed; delete the .bbl on a full clean
$bibtex = 'bibtex %O %S';
$clean_ext = 'bbl run.xml synctex.gz';
