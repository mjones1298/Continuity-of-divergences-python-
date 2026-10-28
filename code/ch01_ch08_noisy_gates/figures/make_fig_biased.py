"""Figure for the biased-noise supplement: one column (3.4 in), two stacked panels, vector PDF + 300-dpi PNG.
(a) window delta_0.1 of the geometric fidelity (1 - Fhat = 0.1) against the bias ratio b = p_z/(p_x+p_y), at fixed
    p_z = 1e-3, for Z and X miscalibrations: exact values (markers, 30-digit bisection on the Theorem-B closed form) and the
    law delta = 4 sqrt(x_eta q_P / q_1) (lines).
(b) F_RLD / F_inf against b at the same p_z (exact closed forms, markers) with the leading-order laws 1 + 1/(4b) and
    (1 + 4b)/(1 + 2b) (lines), and the depolarizing value 3/2 for reference.
Palette: the letter's validated categorical pair (blue, orange); Z and X also differ by marker (circle / square) and by
direct labels, so colour is never the only cue.  Data written to results/fig_data.json.
"""
import json, sys
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
BIASED = HERE.parent / 'biased_noise'
sys.path.insert(0, str(BIASED))  # pauli_biased.py
import pauli_biased as pb

mp.mp.dps = 30
D = str(BIASED) + '/'
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['cmr10', 'DejaVu Serif'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False, 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4,
    'lines.linewidth': 1.2, 'lines.markersize': 3.6, 'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.constrained_layout.use': True, 'figure.constrained_layout.h_pad': 0.03, 'figure.constrained_layout.w_pad': 0.02})

PZ = mp.mpf('1e-3'); ETA = mp.mpf('0.1'); XETA = pb.ginv(ETA)
bs_line = np.logspace(np.log10(0.5), 4, 200)
bs_mark = np.logspace(np.log10(0.5), 4, 15)

data = dict(pz=float(PZ), eta=float(ETA), rows=[])
for b in bs_mark:
    q = pb.qubit_rates(PZ, mp.mpf(b))
    row = dict(b=float(b))
    for ax_ in 'ZX':
        pr = pb.qubit_pairs(q, ax_)
        row['delta_' + ax_] = float(pb.window_Fhat(pr, ETA))
        row['ratio_' + ax_] = float(pb.ratio(pr))
    data['rows'].append(row)
json.dump(data, open(D + 'results/fig_data.json', 'w'), indent=1)


def law(b, ax_):
    q = pb.qubit_rates(PZ, mp.mpf(b)); (q1, qP), _ = pb.qubit_pairs(q, ax_)
    return float(4 * mp.sqrt(XETA * qP / q1))


fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.3))
bm = np.array([r['b'] for r in data['rows']])
# (a)
for ax_, c, mk, lab in (('Z', BLUE, 'o', r'$Z$ miscalibration: $q_P=p_z$'), ('X', ORANGE, 's', r'$X$ miscalibration: $q_P=p_x$')):
    ax[0].plot(bs_line, [law(b, ax_) for b in bs_line], color=c, lw=1.2, zorder=2)
    ax[0].plot(bm, [r['delta_' + ax_] for r in data['rows']], ls='none', marker=mk, mfc=c, mec='white', mew=0.5, ms=4.2, zorder=3)
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel(r'bias ratio $b=p_z/(p_x+p_y)$')
ax[0].set_ylabel(r'window $\delta_{0.1}$ (rad)')
ax[0].set_ylim(2e-4, 0.2)
ax[0].text(3.0, 0.085, r'$Z$ miscalibration ($q_P=p_z$)', color=INK, fontsize=7, ha='left', va='bottom')
ax[0].text(25, 3.3e-3, r'$X$ miscalibration ($q_P=p_x$)', color=INK, fontsize=7, ha='left', va='top', rotation=-17)
ax[0].text(0.03, 0.04, r'$p_z=10^{-3}$, $p_x=p_y$', transform=ax[0].transAxes, fontsize=7, color=INK2, ha='left', va='bottom')
# (b)
ax[1].axhline(1.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax[1].text(2.3e3, 1.515, r'depolarizing, $3/2$', fontsize=6.5, color=INK2, ha='right', va='bottom')
ax[1].plot(bs_line, 1 + 1 / (4 * bs_line), color=BLUE, lw=1.2, zorder=2)
ax[1].plot(bs_line, (1 + 4 * bs_line) / (1 + 2 * bs_line), color=ORANGE, lw=1.2, zorder=2)
ax[1].plot(bm, [r['ratio_Z'] for r in data['rows']], ls='none', marker='o', mfc=BLUE, mec='white', mew=0.5, ms=4.2, zorder=3)
ax[1].plot(bm, [r['ratio_X'] for r in data['rows']], ls='none', marker='s', mfc=ORANGE, mec='white', mew=0.5, ms=4.2, zorder=3)
ax[1].set_xscale('log')
ax[1].set_ylim(0.95, 2.1)
ax[1].set_xlabel(r'bias ratio $b=p_z/(p_x+p_y)$')
ax[1].set_ylabel(r'$F_{\mathrm{RLD}}/F_\infty$')
ax[1].text(40, 1.06, r'$Z$: $1+1/(4b)$', fontsize=7, color=INK, ha='left', va='bottom')
ax[1].text(40, 1.93, r'$X$: $(1+4b)/(1+2b)$', fontsize=7, color=INK, ha='left', va='top')
for a, tag in zip(ax, '(a)(b)'.replace(')(', ') (').split()):
    a.text(-0.17, 1.0, tag, transform=a.transAxes, ha='left', va='top', fontsize=8.5, color=INK)
fig.savefig(HERE / 'biased_window_ratio.pdf')
fig.savefig(HERE / 'biased_window_ratio.png', dpi=300)
print('saved')
