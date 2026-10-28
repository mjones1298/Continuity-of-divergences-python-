"""Letter figures for A_numerics: one-column (3.4 in wide) vector PDFs plus 300-dpi PNGs of the same figures.
  figures/t2_crossover.{pdf,png}        D_BS/zeta, (-2 ln F_hat)/zeta and the bracket on zeta (uses results/t2b_merged.json:
                                        the zeta upper bound includes the two-copy D^# bound where it is smaller)
  figures/t3_window_collapse.{pdf,png}  F_hat window for local and global depolarizing noise (results/t3.json)
  figures/t4_hierarchy.{pdf,png}        RLD / adaptive / BKM / SLD Fisher informations (exact closed forms)
Encoding: each value of p (or n) has a fixed colour AND a fixed marker, and curves are labelled, so no information is carried
by colour alone. Palette: categorical slots blue / orange / aqua (validated all-pairs for colour-vision deficiency), violet
as fourth line in the hierarchy plot. No internal status words in any label (PROSE_STYLE_GUIDE Sec. 5).
Usage: python3 make_figures.py [crossover] [window] [hierarchy]"""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
HERE = Path(__file__).resolve().parent
R_T2 = str(HERE.parent / 'crossover' / 'results') + '/'          # t2b_merged.json
R_T3 = str(HERE.parent / 'multiqubit_window' / 'results') + '/'  # t3.json
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
W1 = 3.4                                        # PRX one-column width (in)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['cmr10', 'DejaVu Serif'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False, 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4,
    'lines.linewidth': 1.2, 'lines.markersize': 3.6, 'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.handlelength': 2.2, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.constrained_layout.use': True, 'figure.constrained_layout.h_pad': 0.02, 'figure.constrained_layout.w_pad': 0.02})
PSET = [('0.01', r'$p=10^{-2}$', BLUE, 'o'), ('0.001', r'$p=10^{-3}$', ORANGE, 's'), ('0.0001', r'$p=10^{-4}$', AQUA, '^')]


def save(fig, name):
    fig.savefig(HERE / f'{name}.pdf')
    fig.savefig(HERE / f'{name}.png', dpi=300)
    plt.close(fig)


def panel_tag(ax, s):
    ax.text(0.015, 0.97, s, transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color=INK)


def crossover():
    T2 = json.load(open(R_T2 + 't2b_merged.json'))['results']
    fig, ax = plt.subplots(3, 1, figsize=(W1, 6.1), sharex=True)
    for pk, lab, c, mk in PSET:
        rows = T2[pk]; p = float(pk)
        x = np.array([r['x'] for r in rows]); d = np.array([r['delta'] for r in rows])
        zl = np.array([r['zeta_lower'] for r in rows]); zu = np.array([r['zeta_upper'] for r in rows])
        dbs = np.array([r['DBS'] for r in rows]); fh = np.array([r['m2lnFhat'] for r in rows])
        me = (0, 5)
        for axi, num in ((ax[0], dbs), (ax[1], fh)):
            axi.fill_between(x, num / zu, num / zl, color=c, alpha=0.16, lw=0)
            axi.plot(x, num / zu, color=c, ls='-', marker=mk, markevery=me, mfc=c, mec='white', mew=0.5)
            axi.plot(x, num / zl, color=c, ls='--', lw=0.9, marker=mk, markevery=me, mfc='white', mec=c, mew=0.7)
        s = p / d ** 2
        ax[2].fill_between(x, zl * s, zu * s, color=c, alpha=0.16, lw=0)
        ax[2].plot(x, zu * s, color=c, ls='-', marker=mk, markevery=me, mfc=c, mec='white', mew=0.5)
        ax[2].plot(x, zl * s, color=c, ls='--', lw=0.9, marker=mk, markevery=me, mfc='white', mec=c, mew=0.7)
        ax[2].plot(x, dbs * s, color=c, ls=':', lw=1.0)
    for axi in ax:
        axi.set_xscale('log'); axi.set_yscale('log')
    for axi in ax[:2]:
        axi.axhline(1, color=INK2, lw=0.6)
    # delta -> 0 limit of D_BS/zeta: 1 <= lim <= F_RLD/F_inf <= 3/2
    xb = 0.0072
    ax[0].plot([xb, xb], [1, 1.5], color=INK, lw=1.0, solid_capstyle='butt')
    ax[0].plot([xb * 0.9, xb * 1.11], [1.5, 1.5], color=INK, lw=1.0); ax[0].plot([xb * 0.9, xb * 1.11], [1, 1], color=INK, lw=1.0)
    ax[0].annotate(r'$\delta\to0$: $[1,\,3/2]$', (xb, 1.5), xytext=(0.0085, 2.05), fontsize=6.5, color=INK)
    ax[0].set_ylabel(r'$D_{\rm BS}/\zeta$')
    ax[1].set_ylabel(r'$-2\ln\hat F/\zeta$')
    ax[2].set_ylabel(r'$p\,\zeta/\delta^{2}$')
    ax[2].set_xlabel(r'$\delta\,/\,[p\,\ln(1/p)]$')
    ax[0].set_xlim(0.006, 110)
    ax[2].axhline(1 / 3, color=INK2, lw=0.6, ls='-.')
    ax[2].text(0.0075, 0.315, r'$pF_\infty/2\to1/3$', fontsize=6.5, color=INK2, va='top')
    ax[2].axhline(0.5, color=INK2, lw=0.6, ls=(0, (1, 1.5)))
    ax[2].text(0.0075, 0.53, r'$pF_{\rm RLD}/2\to1/2$', fontsize=6.5, color=INK2, va='bottom')
    ax[2].set_ylim(0.004, 1.9)
    for a, t in zip(ax, ('(a)', '(b)', '(c)')):
        panel_tag(a, t)
    hp = [Line2D([], [], color=c, marker=mk, mfc=c, mec='white', mew=0.5, label=lab) for _, lab, c, mk in PSET]
    ax[0].legend(handles=hp, loc='upper left', bbox_to_anchor=(0.06, 1.0), fontsize=6.5)
    hs = [Line2D([], [], color=INK2, ls='-', label=r'divided by $\zeta_+$'), Line2D([], [], color=INK2, ls='--', lw=0.9, label=r'divided by $\zeta_-$')]
    ax[1].legend(handles=hs, loc='upper left', bbox_to_anchor=(0.06, 1.0), fontsize=6.5)
    hz = [Line2D([], [], color=INK2, ls='-', label=r'$\zeta_+$'), Line2D([], [], color=INK2, ls='--', lw=0.9, label=r'$\zeta_-$'),
          Line2D([], [], color=INK2, ls=':', lw=1.0, label=r'$D_{\rm BS}$')]
    ax[2].legend(handles=hz, loc='lower left', fontsize=6.5)
    fig.align_ylabels(ax)
    save(fig, 't2_crossover')


def window():
    T3 = json.load(open(R_T3 + 't3.json'))
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.6))
    NSET = [('n1_local_Z', 1, BLUE, 'o', r'$n=1$, $Z$'), ('n2_local_ZZ', 2, ORANGE, 's', r'$n=2$, $Z\!\otimes\!Z$'),
            ('n3_local_ZZZ', 3, AQUA, '^', r'$n=3$, $Z\!\otimes\!Z\!\otimes\!Z$')]
    styles = {'1e-2': (0, (4, 1.6)), '1e-3': (0, (1.2, 1.2)), '1e-4': '-'}
    for name, n, c, mk, lab in NSET:
        for pk, ls in styles.items():
            cv = T3['curves'][f'{name}_p{pk}']
            u = np.array(cv['u']); F = np.array(cv['Fhat']); m = u * float(cv['scale']) <= np.pi; u, F = u[m], F[m]
            kw = dict(color=c, ls=ls, lw=1.0)
            mkw = dict(marker=mk, markevery=(4, 14), mfc=c, mec='white', mew=0.5) if pk == '1e-4' else {}
            ax[0].plot(u, F, **kw, **mkw)
            ax[1].plot(u * 2.0 ** (n - 2), F, **kw, **mkw)
    for name, n in (('n2_global_ZZ', 2), ('n3_global_ZZZ', 3)):
        for pk, ls in styles.items():
            cv = T3['curves'][f'{name}_p{pk}']
            u = np.array(cv['u']); F = np.array(cv['Fhat']); sc = float(cv['scale']); m = u * sc <= np.pi; u, F = u[m], F[m]
            d = 2 ** n; p = float(cv['p']); kap = (d * (1 - p) + p / d) / (p / d)
            mkw = dict(marker='D' if n == 2 else 'v', markevery=(9, 14), mfc=INK2, mec='white', mew=0.5, ms=3.2) if pk == '1e-4' else {}
            ax[1].plot(u * sc * np.sqrt(kap) / 4, F, color=INK2, ls=ls, lw=0.8, **mkw)
    xi = np.logspace(-2, 2, 300)
    ax[1].plot(xi, (1 + xi ** 2) ** -0.5, color=INK, lw=0.8, ls='-.', zorder=5)
    for a in ax:
        a.set_xscale('log'); a.axhline(0.9, color=INK2, lw=0.5, ls=(0, (1, 1.5))); a.set_ylim(0, 1.13)
    ax[0].set_xlim(1e-2, 1e2); ax[1].set_xlim(1e-2, 1e2)
    ax[1].axvline(0.4843, color=INK2, lw=0.5, ls=(0, (1, 1.5)))
    ax[1].text(0.45, 0.79, r'$\xi_*=0.484$', fontsize=6.5, color=INK2, ha='right')
    ax[0].set_xlabel(r'$\delta\,/\,p^{w/2}$ (rad)')
    ax[0].set_ylabel(r'$\hat F(\mathcal{N}_{p,\delta},\mathcal{N}_p)$')
    ax[1].set_xlabel(r'$\xi=\delta\sqrt{\kappa}/4$')
    ax[1].set_ylabel(r'$\hat F$')
    panel_tag(ax[0], '(a)'); panel_tag(ax[1], '(b)')
    h = [Line2D([], [], color=c, marker=mk, mfc=c, mec='white', mew=0.5, label=lab) for _, _, c, mk, lab in NSET]
    h += [Line2D([], [], color=INK2, ls=ls, lw=1.0, label=lab) for ls, lab in
          ((styles['1e-2'], r'$p=10^{-2}$'), (styles['1e-3'], r'$p=10^{-3}$'), (styles['1e-4'], r'$p=10^{-4}$'))]
    ax[0].legend(handles=h, loc='lower left', bbox_to_anchor=(0.0, 0.03), ncol=1, fontsize=6.2)
    h2 = [Line2D([], [], color=c, marker=mk, mfc=c, mec='white', mew=0.5, label=lab + ' local') for _, _, c, mk, lab in NSET]
    h2 += [Line2D([], [], color=INK2, marker='D', mfc=INK2, mec='white', mew=0.5, ms=3.2, lw=0.8, label=r'$n=2$ global'),
           Line2D([], [], color=INK2, marker='v', mfc=INK2, mec='white', mew=0.5, ms=3.2, lw=0.8, label=r'$n=3$ global'),
           Line2D([], [], color=INK, ls='-.', lw=0.8, label=r'$(1+\xi^2)^{-1/2}$')]
    ax[1].legend(handles=h2, loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=1, fontsize=6.2)
    save(fig, 't3_window_collapse')


def hierarchy():
    p = np.logspace(-5, np.log10(0.3), 300)
    FR = 2 * (2 - p) * (1 - p) ** 2 / (p * (4 - 3 * p))
    Fi = 2 * (1 - p) ** 2 / (p * (3 - 2 * p))
    FB = (1 - p) * np.log((4 - 3 * p) / p) / 2
    FS = 2 * (1 - p) ** 2 / (2 - p)
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    lines = [(FR, r'$F_{\rm RLD}$: $D_{\rm BS}$, $\hat F$', BLUE, 'o', (1.6e-4, 1.6e4)),
             (Fi, r'$F_\infty$: adaptive estimation', ORANGE, 's', (2.5e-4, 12)),
             (FB, r'$F_{\rm BKM}$: $D$', AQUA, '^', (1.3e-5, 7.6)),
             (FS, r'$F_{\rm SLD}$: $F$', VIOLET, 'D', (1.3e-5, 1.35))]
    for i, (y, lab, c, mk, (tx, ty)) in enumerate(lines):
        ax.loglog(p, y, color=c, marker=mk, markevery=(10 + 9 * i, 40), mfc=c, mec='white', mew=0.5, ms=3.4, label=lab)
        ax.text(tx, ty, lab, fontsize=6.5, color=INK, va='bottom')
    ax.set_xlabel(r'depolarizing probability $p$')
    ax.set_ylabel(r'Fisher information (rad$^{-2}$)')
    ax.set_ylim(0.3, 3e5); ax.set_xlim(1e-5, 0.3)
    save(fig, 't4_hierarchy')


if __name__ == '__main__':
    import sys
    for w in sys.argv[1:] or ['crossover', 'window', 'hierarchy']:
        globals()[w]()
        print('made', w)
