# -*- coding: utf-8 -*-
"""
Fig. 1 (Introduction overview): the accountability gap and the position of this
work. Three bands: Challenge -> Existing approaches and their gaps -> This work
(accountable decision-making under verifiable abstention). Layout follows the
author's three-band overview convention; palette and typography follow
figstyle.py (paper-wide purple/red/amber scheme). No result numbers appear
here, so nothing is read from artifacts.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

W_MM, H_MM = 180.0, 178.0
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

PUR, MID, LT = fs.SYS_DEEP, fs.SYS_MID, fs.SYS_LT
RED, AMB, GREY = fs.RED, fs.AMBER, fs.GREY
GREY_LT = fs.GREY_LT


def band_header(y, h, text, fc):
    ax.add_patch(FancyBboxPatch((2, y), 96, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec="none", zorder=2))
    ax.text(50, y + h / 2, text, ha="center", va="center", fontsize=9.2,
            fontweight="bold", color="white", zorder=3)


def card(x, y, w, h, title, lines, ec, title_c=None, fc="white", lh=2.55,
         title_fs=6.9, body_fs=6.1, align="center", y0=4.6, pad=2.4):
    """align="left" draws the body as a list flush-left at `pad` from the card
    edge; the default centres each line, as the other bands do."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=title_c or ec, zorder=3)
    for i, (txt, c) in enumerate(lines):
        if align == "left":
            ax.text(x + pad, y + h - y0 - i * lh, txt, ha="left", va="center",
                    fontsize=body_fs, color=c, zorder=3)
        else:
            ax.text(x + w / 2, y + h - y0 - i * lh, txt, ha="center", va="center",
                    fontsize=body_fs, color=c, zorder=3)


def down_arrow(x, y0, y1, color):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=13, lw=2.2, color=color, zorder=4))


# ---------------- band 1: challenge (y 82..97) ----------------
band_header(93.4, 3.6, "Challenge: localization is not the bottleneck, accountability is", PUR)
cw, gap, x0, cy, ch = 30.7, 1.6, 2, 82.2, 10.2
card(x0, cy, cw, ch, "Ill-posed inversion",
     [("candidate leak locations vastly", fs.INK), ("outnumber sparse sensors", fs.INK)],
     ec=MID)
card(x0 + cw + gap, cy, cw, ch, "Confusable causes",
     [("demand surges, sensor drift and", fs.INK), ("valve mis-operation mimic leaks", fs.INK)],
     ec=MID)
card(x0 + 2 * (cw + gap), cy, cw, ch, "Costly, regulated action",
     [("a wrong excavation wastes crews;", fs.INK), ("no method proves when NOT to act", RED)],
     ec=MID)
down_arrow(50, 81.2, 78.4, MID)

# ---------------- band 2: existing approaches (y 56..78) ----------------
band_header(74.4, 3.6, "Existing approaches and the gap each leaves", MID)
cy2, ch2 = 58.0, 15.0
B2 = dict(ec=GREY, align="left", lh=3.3, y0=6.0, pad=2.6)
card(x0, cy2, cw, ch2, "Data-driven localizers",
     [("\u2022  graph, sequence, transfer models", fs.INK),
      ("\u2022  physics-informed networks", fs.INK),
      ("gap:  answers every event, uncalibrated", RED)], **B2)
card(x0 + cw + gap, cy2, cw, ch2, "Selective / conformal prediction",
     [("\u2022  reject options and deferral", fs.INK),
      ("\u2022  distribution-free coverage", fs.INK),
      ("gap:  a scalar score, no justification", RED)], **B2)
card(x0 + 2 * (cw + gap), cy2, cw, ch2, "LLM agents reach water",
     [("\u2022  domain-adapted water models", fs.INK),
      ("\u2022  agents drive hydraulic simulation", fs.INK),
      ("gap:  orchestrate, never gate action", RED)], **B2)
down_arrow(50, 57.0, 52.6, MID)

# ---------------- band 3: this work (y 8..52) ----------------
band_header(48.6, 3.6, "This work: accountable decision-making under verifiable abstention", PUR)

# executor -> supervisor -> outcomes
ex_w, sv_w, oc_w = 29.5, 29.5, 29.5
ey, eh = 26.0, 19.4
card(2, ey, ex_w, eh, "Executor agent (deterministic)",
     [("competing hypotheses:", fs.INK), ("leak / demand / sensor / valve / none", PUR),
      ("falsified in a hydraulic digital twin;", fs.INK), ("Bayesian fusion, Occam penalty;", fs.INK),
      ("numeric evidence package only", fs.INK)],
     ec=PUR, lh=2.75)
card(2 + ex_w + 2.7, ey, sv_w, eh, "Supervisor agent (never localizes)",
     [("code-verifiable goal contract:", fs.INK), ("existence, region, margin,", PUR),
      ("alternatives, residual, safety", PUR), ("+ independent LLM auditor", RED),
      ("(may only add a rejection)", fs.INK)],
     ec=RED, title_c=RED, lh=2.75)
ox = 2 + 2 * (ex_w + 2.7)
oh3 = (eh - 2 * 1.3) / 3.0
card(ox, ey + 2 * (oh3 + 1.3), oc_w, oh3, "Act", [], ec=PUR, fc=fs.tint(PUR, 0.88))
ax.text(ox + oc_w / 2, ey + 2 * (oh3 + 1.3) + 1.15, "dispatch with hashed evidence certificate",
        ha="center", va="center", fontsize=5.9, color=fs.INK, zorder=3)
card(ox, ey + oh3 + 1.3, oc_w, oh3, "Request evidence", [], ec=fs.GREEN, title_c="#237032")
ax.text(ox + oc_w / 2, ey + oh3 + 1.3 + 1.15, "active sensing: most discriminative reading",
        ha="center", va="center", fontsize=5.9, color=fs.INK, zorder=3)
card(ox, ey, oc_w, oh3, "Abstain", [], ec=GREY)
ax.text(ox + oc_w / 2, ey + 1.15, "first-class outcome, dossier to a human",
        ha="center", va="center", fontsize=5.9, color=fs.INK, zorder=3)
for xa in (2 + ex_w + 0.2, 2 + ex_w + 2.7 + sv_w + 0.2):
    ax.add_patch(FancyArrowPatch((xa, ey + eh / 2), (xa + 2.4, ey + eh / 2),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.8,
                                 color=MID, zorder=4))

# validation strip
vy, vh = 17.4, 6.4
ax.add_patch(FancyBboxPatch((2, vy), 96, vh, boxstyle="round,pad=0.35,rounding_size=1.0",
                            fc=fs.tint(LT, 0.55), ec=MID, lw=0.9, zorder=2))
ax.text(50, vy + vh - 1.9, "Tiers of increasing realism, one measurement battery",
        ha="center", va="center", fontsize=6.6, fontweight="bold", color=PUR, zorder=3)
ax.text(50, vy + 1.9,
        "four in-silico networks  →  independent benchmark (BattLeDIM L-Town)  →  "
        "audited 194-order field register (City D): excavation tier + district survey tier",
        ha="center", va="center", fontsize=6.1, color=fs.INK, zorder=3)

# question strip
qy, qh = 8.6, 6.2
ax.add_patch(FancyBboxPatch((2, qy), 96, qh, boxstyle="round,pad=0.35,rounding_size=1.0",
                            fc="white", ec=GREY, lw=0.9, zorder=2))
for xq, q, c in ((18, "Is anything wrong?", fs.INK),
                 (50, "Can the evidence carry a dispatch?", fs.INK),
                 (82, "Act, request, or abstain, with proof", PUR)):
    ax.text(xq, qy + qh / 2, q, ha="center", va="center", fontsize=6.8,
            fontweight="bold" if c is PUR else "normal", color=c, zorder=3)
for xa in (30.5, 63.5):
    ax.add_patch(FancyArrowPatch((xa, qy + qh / 2), (xa + 3.0, qy + qh / 2),
                                 arrowstyle="-|>", mutation_scale=10, lw=1.5,
                                 color=GREY, zorder=4))

fs.save(fig, "Fig1_overview")
print("saved Fig1_overview -> png/pdf/svg")
