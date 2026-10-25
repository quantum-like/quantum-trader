"""Shared matplotlib style for the fbdmd-hs-movies report figures.
Usage:  import sys; sys.path.insert(0, 'report'); from figstyle import *; apply_style()
Save every figure with savefig(fig, 'name') -> report/figures/name.png (200 dpi) + name.svg.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "report", "figures")
DATADIR = os.path.join(ROOT, "report", "data")

# Colours (match the movies): neutral gray, + red, - blue; integrators get their own palette.
C_NEUTRAL = "#8E8E93"
C_PLUS = "#D92626"
C_MINUS = "#2659E6"
C_INTEG = {"cv-phys": "#1B7F3B", "cv": "#D97A00", "kdk-phys": "#6A4CBF", "naive": "#9A9A9A"}
C_ACCENT = "#1F4E79"
C_GRID = "#DDDDE3"

FULL_W = 7.2     # inches, full text width in the report
HALF_W = 3.55

def apply_style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 1.4,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

def savefig(fig, name, svg=True):
    os.makedirs(FIGDIR, exist_ok=True)
    png = os.path.join(FIGDIR, name + ".png")
    fig.savefig(png)
    if svg:
        fig.savefig(os.path.join(FIGDIR, name + ".svg"))
    plt.close(fig)
    return png

def panel_label(ax, s):
    ax.text(-0.08, 1.04, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")
