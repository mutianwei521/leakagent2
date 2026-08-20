# -*- coding: utf-8 -*-
"""Single source of truth for the Nature-subjournal figure style.

Every make_*.py imports this so the whole figure set shares one palette,
one typography scale and one set of spine/legend conventions. Colours are a
low-saturation register with FIXED semantic roles that are used identically in
every panel; the manuscript legends refer to these roles by name (system =
purple/violet, baselines = grey, floors/anchors = red dashed, sensor noise = amber
dotted), so the hues here are refined but the families are preserved.

This module only controls appearance. It never touches data values.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")

# ---- semantic palette: three vivid mains (blue / red / green) + simple accents ----
# Design rule: blue = system/contract/acted (primary), red = gate/veto/reference,
# green = the second data series (coverage, forced curve, request-evidence).
# Accents: amber thin caution lines, grey neutral. Red and green never encode a
# contrast against each other in the same panel (colour-vision safety).
INK      = "#1a1a1a"   # text / axes
SYS_DEEP = "#1864ab"   # primary system (vivid deep blue)
SYS_MID  = "#339af0"   # mid blue
SYS_LT   = "#a5d8ff"   # light blue fill
GREEN    = "#2f9e44"   # third main: secondary data series
GREEN_LT = "#b2f2bb"   # light green fill
# neutral family: baselines, existence/scalar threshold, abstained, non-confined
GREY     = "#9a9a9a"
GREY_LT  = "#cbcbcb"
GREY_D   = "#5a5a5a"   # dark neutral for de-emphasized secondary text (footnotes)
# reference lines / anchors / drops (vivid red)
RED      = "#e03131"
# sensor-noise floor / caution accent (amber, thin lines only)
AMBER    = "#f59f00"
# Three-colour scheme (design rule): purple = primary/system, red = gate/anchor,
# amber = evidence/noise cue; grey is the non-counting neutral. No other hues.

# back-compat aliases so existing scripts that referenced TEAL/BLUE keep working
TEAL = SYS_DEEP
BLUE = SYS_DEEP

# pastel single-hue sequential for heatmaps (near-white -> deep system violet)
CMAP_SYS = LinearSegmentedColormap.from_list(
    "sys_blue", ["#f1f8ff", "#d0ebff", "#74c0fc", SYS_MID, SYS_DEEP, "#0b4884"])

# ---- canvas proportions (design rule: golden/silver rectangles) ----
GOLDEN, SILVER = 1.618, 1.414
W2 = 7.09                                  # NC double-column width (180 mm)
SIZE_2PANEL = (W2, round(W2 / (2 * SILVER), 2))   # 1x2 grid: each panel a silver landscape
SIZE_1PANEL = (4.6, round(4.6 / GOLDEN, 2))       # single panel: golden landscape


def apply_rc():
    """Install the shared rcParams. Call once at the top of each make_*.py."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "svg.fonttype": "none",   # editable text in SVG
        "pdf.fonttype": 42,       # editable TrueType text in PDF
        "savefig.dpi": 600,
        "figure.dpi": 130,
    })


def panel_tag(ax, letter, title=None, x=-0.16, y=1.06, title_pad=0.028):
    """Nature-style panel tag: bold lowercase letter at top-left, optional light
    title to its right. Keeps titles unobtrusive (regular weight, small)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left")
    if title:
        ax.text(x + title_pad + 0.02, y, title, transform=ax.transAxes,
                fontsize=7.3, fontweight="regular", va="bottom", ha="left")


def lspine(ax):
    """Force the L-shaped frame (drop top+right); use after imshow/twinx which
    can re-enable spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def tint(color, amount=0.62):
    """Lighten a colour by `amount` toward white (for pale bar fills that let the
    individual replicate dots read on top)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def replicate_dots(ax, center, values, color, orient="v", jw=0.12, s=15, seed=0):
    """Overlay the individual per-run values as jittered 'bubbles' around `center`
    (Nature 'show every replicate' convention). Jitter is deterministic (fixed seed)
    so the figure is reproducible. `values` are the real per-seed numbers; nothing is
    synthesized. orient='v' scatters across x at height=value; 'h' scatters across y."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    off = (rng.random(len(values)) - 0.5) * 2 * jw
    pos = np.full(len(values), center) + off
    if orient == "v":
        ax.scatter(pos, values, s=s, facecolor=color, edgecolor="white",
                   linewidths=0.5, zorder=6, clip_on=True)
    else:
        ax.scatter(values, pos, s=s, facecolor=color, edgecolor="white",
                   linewidths=0.5, zorder=6, clip_on=True)


def save(fig, name, formats=("png", "pdf", "svg")):
    """Export to figures/ in the requested formats (PNG for the manuscript embed,
    PDF+SVG as editable vectors)."""
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  saved", name, "->", "/".join(formats))
