# Data not included

All results files that the book's numbers come from are included, and so is the raw simulation data of the certification study (`ch01_ch08_noisy_gates/certification/results/s1_raw/*.npz`, `s2_raw/*.json`, `s2b_raw/*.json`, about 6.5 MB). The whole `code/` directory is about 13 MB, and no file exceeds 2 MB.

Left out, with how to regenerate them:

| Not included | Size | Why | How to regenerate |
|---|---|---|---|
| `certification/results/s2b_raw/refine_p1e-02.json` | 0.4 MB | Raw data of the refined (1/8-decade, 50-run) opposite-side scan, which was interrupted after p = 1e-2. No number in the book uses it; `analyze_s2.py` reads it only when `USE_REFINE=1`. | `python3 ch01_ch08_noisy_gates/certification/s2b_pipelines.py refine` (hours; writes `results/s2b_raw/refine_p*.json`). |
| Rendered figures (`*.png`, `*.pdf`) | 2.5 MB | Outputs, not data. The PDFs used by the book are in the repository's `figures/`. | Run the figure scripts listed in `README.md` ("Figures"); each takes a few seconds. They write into their own folder (`figures/` subfolders for the early experiments). |
| Run logs (`*.log`) of the early experiments and of A_numerics | 0.1 MB | Progress logs; the numbers are in the JSON results. | Written again by the scripts (e.g. `crossover/results/t2.log`, `multiqubit_window/results/t3.log`). |
| Optimizer checkpoints (`*.npy`) of local searches | < 0.01 MB | Scratch; the searches print and log their results. | Written again by `proof_support/p2_search_v2.py`, `checks/independent_C1_C7/c6_adv.py`. |
| A data file for Section 7.7 (Tables 7.3–7.5) and the numerical checks of Sections 5.9 and 6.11 | — | The code for these was not among the folders this directory was assembled from. | — (to be added by the author) |
