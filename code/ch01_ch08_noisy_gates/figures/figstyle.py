"""Common style for all letter figures (revision 1): Computer Modern text and math, one-column width 3.4 in,
labels >= 7 pt, US spelling, palette with fixed colour + marker per series (meaning never carried by colour alone)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
RED, GREY = "#b2182b", "#6b6a66"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
W1 = 3.4
RC = {
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.direction": "in", "ytick.direction": "in", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "lines.markersize": 3.6, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.handlelength": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
plt.rcParams.update(RC)


def panel_tag(ax, s, x=0.015, y=0.97):
    ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK)
