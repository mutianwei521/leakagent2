# -*- coding: utf-8 -*-
"""Extended Data Fig. 2 (schematic): from evidence to an accountable action.
Evidence package -> goal-contract checklist (Eq. 4, G1-G7) -> three outcomes
(act with a hashed certificate / request evidence via active sensing / abstain
with a dossier), with the independent LLM auditor that can only tighten the
gate. Purely illustrative; no measured values are drawn."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

fs.apply_rc()
BLUE, MID, LT, GREEN, RED, AMB = fs.SYS_DEEP, fs.SYS_MID, fs.SYS_LT, fs.GREEN, fs.RED, fs.AMBER
INK, GREY, GREY_D = fs.INK, fs.GREY, fs.GREY_D
GREEN_D = "#237032"

fig = plt.figure(figsize=(7.09, 4.35))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 61)
ax.axis("off")


def box(x, y, w, h, ec, fc="white", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def harrow(x0, y0, x1, y1, color, lw=1.6, style="-|>", ls="solid"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, linestyle=ls,
                                 mutation_scale=11, lw=lw, color=color, zorder=6))


# ============ left: evidence package ============
XE, WE = 2.0, 20.0
box(XE, 22.0, WE, 26.0, ec=GREEN, fc=fs.tint(GREEN, 0.93))
ax.text(XE + WE / 2, 45.6, "Numeric evidence", ha="center", fontsize=6.8,
        fontweight="bold", color=GREEN_D)
ax.text(XE + WE / 2, 43.4, "package", ha="center", fontsize=6.8,
        fontweight="bold", color=GREEN_D)
for i, line in enumerate(("Posterior over $\\mathcal{H}$", "$\\Pi_{\\mathrm{exist}}$, margin $\\Delta$",
                          "Fitted node and rate", "Per-hypothesis residual $\\rho$",
                          "Candidate region $R$", "Per-tool evidence log")):
    ax.text(XE + WE / 2, 40.6 - i * 2.5, line, ha="center", fontsize=5.9, color=INK)
ax.text(XE + WE / 2, 23.6, "Numbers only", ha="center", fontsize=5.8,
        fontstyle="italic", color=GREY_D)

# ============ middle: the contract checklist ============
XC, WC = 27.5, 30.0
box(XC, 14.0, WC, 39.0, ec=RED)
ax.text(XC + WC / 2, 51.7, "Goal contract  ·  Eq. (4)", ha="center", fontsize=7.0,
        fontweight="bold", color=RED)
preds = [("G1  Existence", "$h_{(1)}=h_{\\mathrm{leak}}$,  $\\Pi_{\\mathrm{exist}}\\geq\\tau_e$"),
         ("G2  Region", "$|R|\\leq R_{\\max}$"),
         ("G3  Margin", "$\\Delta\\geq\\delta$"),
         ("G4  Alternatives", "$P(h_{(2)})\\leq\\alpha$"),
         ("G5  Reserved", "Not evaluated here"),
         ("G6  Residual", "$\\rho_{(1)}\\leq 3$ per dof"),
         ("G7  Safety", "$\\mathcal{U}=\\varnothing$")]
for i, (name, expr) in enumerate(preds):
    yb = 45.6 - i * 4.55
    grey = name.startswith("G5")
    box(XC + 1.2, yb, WC - 2.4, 3.7, ec=GREY if grey else MID,
        fc=fs.tint(GREY, 0.85) if grey else "white", lw=0.8)
    ax.text(XC + 2.6, yb + 1.85, name, ha="left", va="center", fontsize=6.0,
            fontweight="bold", color=GREY_D if grey else BLUE)
    ax.text(XC + WC - 2.6, yb + 1.85, expr, ha="right", va="center", fontsize=5.8,
            color=GREY_D if grey else INK)
ax.text(XC + WC / 2, 15.9, "Deterministic arithmetic; every evaluated check recorded",
        ha="center", fontsize=5.7, fontstyle="italic", color=GREY_D)

# ============ bottom band: LLM auditor ============
XA, WA = 27.5, 30.0
box(XA, 2.0, WA, 7.6, ec=RED, fc=fs.tint(RED, 0.93))
ax.text(XA + WA / 2, 7.4, "Independent LLM auditor", ha="center", fontsize=6.4,
        fontweight="bold", color=RED)
ax.text(XA + WA / 2, 5.3, "Different model family · temperature 0 · strict JSON",
        ha="center", fontsize=5.5, color=INK)
ax.text(XA + WA / 2, 3.4, "Sees the numeric summary only", ha="center", fontsize=5.5, color=INK)
# the arrow sits at the right of the band so the note has the whole gap between
# the auditor box (top 9.6) and the contract box (floor 14.0) to itself
harrow(XA + WA - 6.0, 9.9, XA + WA - 6.0, 13.8, RED, lw=1.3, ls="dashed")
ax.text(XA + 0.6, 11.75, "May only add a rejection,\nnever overturn a hard check",
        ha="left", va="center", fontsize=5.5, color=RED)

# ============ right: three outcomes ============
XO, WO = 68.0, 30.0
# ACT
box(XO, 40.5, WO, 12.5, ec=BLUE, fc=fs.tint(BLUE, 0.92))
ax.text(XO + 2.0, 50.6, "ACT: Dispatch authorised", ha="left", fontsize=6.6,
        fontweight="bold", color=BLUE)
for i, line in enumerate(("Certificate: decision, thresholds, every", "predicate's value and pass/fail,",
                          "accepted hypothesis and region,", "SHA-256 digest: tamper-evident")):
    ax.text(XO + 2.0, 48.4 - i * 1.85, line, ha="left", fontsize=5.7, color=INK)
# REQUEST
box(XO, 26.5, WO, 11.0, ec=GREEN, fc=fs.tint(GREEN, 0.93))
ax.text(XO + 2.0, 35.3, "REQUEST EVIDENCE (active sensing)", ha="left", fontsize=6.4,
        fontweight="bold", color=GREEN_D)
for i, line in enumerate(("Names the failed predicate, reveals the", "three most discriminative hidden nodes",
                          "(Eq. S6), re-fits and re-ranks")):
    ax.text(XO + 2.0, 33.2 - i * 1.85, line, ha="left", fontsize=5.7, color=INK)
# ABSTAIN
box(XO, 13.5, WO, 10.0, ec=GREY, fc=fs.tint(GREY, 0.9))
ax.text(XO + 2.0, 21.4, "ABSTAIN: First-class outcome", ha="left", fontsize=6.4,
        fontweight="bold", color=GREY_D)
for i, line in enumerate(("Dossier: surviving hypotheses and the", "reason each predicate failed;",
                          "deferred to a human, never forced")):
    ax.text(XO + 2.0, 19.4 - i * 1.85, line, ha="left", fontsize=5.7, color=INK)

# ============ arrows ============
harrow(XE + WE + 0.4, 35.0, XC - 0.5, 35.0, MID)
harrow(XC + WC + 0.5, 46.5, XO - 0.5, 46.5, BLUE)
ax.text((XC + WC + XO) / 2, 47.6, "All pass", ha="center", fontsize=5.8, color=BLUE)
harrow(XC + WC + 0.5, 32.0, XO - 0.5, 32.0, GREEN)
ax.text((XC + WC + XO) / 2, 33.1, "Resolvable\nfailure", ha="center", fontsize=5.6, color=GREEN_D)
harrow(XC + WC + 0.5, 18.5, XO - 0.5, 18.5, GREY_D)
ax.text((XC + WC + XO) / 2, 19.6, "Evidence\ncannot carry", ha="center", fontsize=5.6, color=GREY_D)
# active-sensing loop back to the evidence package: routed through the free
# corridor (down between checklist and outcomes, across the bottom, up at left)
loop_x, loop_y = 58.9, 1.0
ax.plot([XO - 0.3, loop_x], [30.4, 30.4], color=GREEN, lw=1.3, ls="dashed", zorder=6)
ax.plot([loop_x, loop_x], [30.4, loop_y], color=GREEN, lw=1.3, ls="dashed", zorder=6)
ax.plot([loop_x, XE + WE / 2], [loop_y, loop_y], color=GREEN, lw=1.3, ls="dashed", zorder=6)
ax.add_patch(FancyArrowPatch((XE + WE / 2, loop_y), (XE + WE / 2, 21.5),
                             arrowstyle="-|>", mutation_scale=11, lw=1.3, color=GREEN,
                             linestyle="dashed", zorder=6))
ax.text(13.5, 12.6, "Augmented sensor set,\none more round", ha="left", fontsize=5.7,
        color=GREEN_D)

fs.save(fig, "Fig6_method_contract")
print("saved Fig6_method_contract -> png/pdf/svg")
