# -*- coding: utf-8 -*-
"""Fig. 5 (Methods, schematic): the executor's diagnostic pipeline.
Observe (Eq. 1) -> hypothesize (5 families) -> falsify in the twin (Eq. 2)
-> fuse into a posterior with an Occam penalty (Eq. 3) -> evidence package.
Purely illustrative shapes; no measured values are drawn."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

fs.apply_rc()
BLUE, MID, LT, GREEN, RED, AMB = fs.SYS_DEEP, fs.SYS_MID, fs.SYS_LT, fs.GREEN, fs.RED, fs.AMBER
INK, GREY, GREY_D = fs.INK, fs.GREY, fs.GREY_D

fig = plt.figure(figsize=(7.09, 3.95))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(10.5, 55.5)
ax.axis("off")


def box(x, y, w, h, ec, fc="white", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def head(x, w, text):
    ax.text(x + w / 2, 51.8, text, ha="center", va="center", fontsize=7.6,
            fontweight="bold", color=BLUE)


def arrow(x0, x1, y=27.0, color=MID):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.8, color=color, zorder=6))


# ============ stage 1: observe ============
X1, W1 = 2.0, 20.5
head(X1, W1, "1  Observe")
box(X1, 27.5, W1, 22.0, ec=MID)
rng = np.random.default_rng(7)
nx = np.array([4.5, 8, 12, 16, 19.5, 6, 10.5, 15, 18.5, 5.5, 9.5, 13.5, 17.5, 7.5, 14.5]) + X1 - 2
ny = np.array([46, 47.5, 46.5, 47.5, 46, 42.5, 43.5, 43, 42, 38.5, 39.5, 38.5, 39.5, 35, 35.5]) - 4.5
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (1, 6), (2, 7), (3, 8), (5, 6), (6, 7), (7, 8),
         (5, 9), (6, 10), (7, 11), (8, 12), (9, 10), (10, 11), (11, 12), (10, 13), (11, 14), (13, 14)]
for a, b in edges:
    ax.plot([nx[a], nx[b]], [ny[a], ny[b]], color=GREY, lw=0.7, zorder=3)
ax.scatter(nx, ny, s=10, c="white", edgecolors=GREY_D, linewidths=0.6, zorder=4)
S_IDX = [1, 7, 9, 14]
ax.scatter(nx[S_IDX], ny[S_IDX], s=22, marker="s", c=MID, edgecolors="white",
           linewidths=0.5, zorder=5, label="sensors")
ax.scatter([nx[11]], [ny[11]], s=55, marker="*", c=RED, edgecolors="white",
           linewidths=0.5, zorder=6)
ax.text(nx[11] + 1.0, ny[11] - 1.6, "leak?", color=RED, fontsize=6.0)
ax.text(X1 + W1 / 2, 29.0, "sparse sensors $\\mathcal{S}$, $m\\ll N$", ha="center",
        fontsize=6.0, color=INK)
# observation strip
box(X1, 15.5, W1, 9.5, ec=MID)
bx = np.linspace(X1 + 2.2, X1 + W1 - 2.2, 8)
bh = np.array([1.4, 3.6, 0.9, 4.8, 1.8, 0.7, 2.9, 1.2])
for xi, hi in zip(bx, bh):
    ax.add_patch(plt.Rectangle((xi - 0.7, 17.3), 1.4, hi, fc=fs.tint(MID, 0.55),
                               ec=MID, lw=0.6, zorder=4))
ax.text(X1 + W1 / 2, 23.3, "$\\Delta\\mathbf{p}^{\\mathrm{obs}}=\\Delta\\mathbf{p}^{(h)}_{\\mathcal{S}}+\\boldsymbol{\\epsilon}$",
        ha="center", fontsize=6.4, color=INK)
ax.text(X1 + W1 / 2, 16.3, "field noise $\\epsilon\\sim\\mathcal{N}(0,\\sigma^2)$  ·  Eq. (1)",
        ha="center", fontsize=5.8, color=GREY_D)

# ============ stage 2: hypothesize ============
X2, W2 = 26.5, 19.0
head(X2, W2, "2  Hypothesize")
fams = [("leak in zone $C_k$  (×K)", BLUE), ("zone-wide demand  (×K)", INK),
        ("sensor fault  (×$|\\mathcal{S}^*|$)", INK), ("valve mis-state", INK),
        ("no anomaly  $h_0$", GREY_D)]
for i, (lab, c) in enumerate(fams):
    yb = 43.2 - i * 5.0
    box(X2, yb, W2, 4.0, ec=MID if c is not BLUE else BLUE,
        fc=fs.tint(BLUE, 0.9) if c is BLUE else "white")
    ax.text(X2 + W2 / 2, yb + 2.0, lab, ha="center", va="center", fontsize=6.2, color=c)
box(X2, 15.5, W2, 6.2, ec=GREY)
ax.text(X2 + W2 / 2, 19.6, "leak fitted by best node-rate", ha="center", fontsize=5.9, color=INK)
ax.text(X2 + W2 / 2, 17.6, "pair over grid $\\mathcal{Q}$ (Eq. S4)", ha="center",
        fontsize=5.9, color=INK)

# ============ stage 3: falsify ============
X3, W3 = 49.0, 22.5
head(X3, W3, "3  Falsify in the twin")
cases = [("$h_{\\mathrm{leak}}$: reproduces,  $\\rho\\approx 1$", BLUE, True),
         ("$h_{\\mathrm{dem}}$: strained,  $\\rho>2$", AMB, None),
         ("$h_0$: fails,  $\\rho\\gg 3$", RED, False)]
prof = np.array([1.2, 3.4, 0.8, 4.6, 1.6, 0.6, 2.7, 1.1])
preds = [prof * 0.96 + 0.08, prof * 0.55 + 1.2, prof * 0.0 + 0.15]
for i, ((lab, c, ok), pv) in enumerate(zip(cases, preds)):
    yb = 39.4 - i * 8.6
    box(X3, yb, W3, 7.4, ec=c)
    xs = np.linspace(X3 + 2.2, X3 + W3 - 7.5, 8)
    for xi, ho, hp in zip(xs, prof, pv):
        ax.add_patch(plt.Rectangle((xi - 0.55, yb + 1.0), 1.1, ho * 0.9,
                                   fc=fs.tint(GREY, 0.5), ec=GREY, lw=0.4, zorder=3))
        ax.plot([xi - 0.8, xi + 0.8], [yb + 1.0 + hp * 0.9] * 2, color=c, lw=1.3, zorder=4)
    ax.text(X3 + W3 - 1.2, yb + 5.4, lab, ha="right", fontsize=5.9, color=c)
    mark = "$\checkmark$" if ok else ("$\times$" if ok is False else "...")
    ax.text(X3 + W3 - 2.2, yb + 2.2, mark, ha="center", fontsize=9,
            fontweight="bold", color=c)
ax.text(X3 + W3 / 2, 12.4, "$\\mathbf{r}_h=\\Delta\\mathbf{p}^{\\mathrm{obs}}-\\boldsymbol{\\mu}_h$;  "
        "likelihood collapses if the twin\ncannot reproduce the observation  ·  Eq. (2)",
        ha="center", fontsize=5.8, color=GREY_D)

# ============ stage 4: fuse ============
X4, W4 = 76.0, 22.0
head(X4, W4, "4  Fuse and package")
box(X4, 29.0, W4, 20.4, ec=BLUE)
post = [("leak", 15.2, BLUE), ("demand", 4.6, GREY), ("sensor", 2.6, GREY),
        ("valve", 1.8, GREY), ("none", 1.2, GREY_D)]
for i, (lab, w, c) in enumerate(post):
    yb = 45.6 - i * 3.1
    ax.add_patch(plt.Rectangle((X4 + 5.8, yb), w, 2.1, fc=fs.tint(c, 0.5), ec=c, lw=0.7, zorder=3))
    ax.text(X4 + 5.2, yb + 1.0, lab, ha="right", va="center", fontsize=5.8, color=INK)
ax.text(X4 + W4 / 2, 30.3, "Occam penalty $-\\frac{1}{2}k_h\\log m$  ·  Eq. (3)",
        ha="center", fontsize=5.8, color=GREY_D)
box(X4, 15.5, W4, 11.0, ec=GREEN, fc=fs.tint(GREEN, 0.92))
ax.text(X4 + W4 / 2, 24.4, "numeric evidence package", ha="center", fontsize=6.4,
        fontweight="bold", color="#237032")
for i, line in enumerate(("$\\Pi_{\\mathrm{exist}}$, margin $\\Delta$, region $R$",
                          "per-hypothesis $\\rho$ + falsification status",
                          "fitted node and rate; tool log",
                          "no free text, no self-rated confidence")):
    ax.text(X4 + W4 / 2, 22.4 - i * 1.75, line, ha="center", fontsize=5.7, color=INK)

arrow(X1 + W1 + 0.4, X2 - 0.4)
arrow(X2 + W2 + 0.4, X3 - 0.4)
arrow(X3 + W3 + 0.4, X4 - 0.4)

fs.save(fig, "Fig5_method_pipeline")
print("saved Fig5_method_pipeline -> png/pdf/svg")
