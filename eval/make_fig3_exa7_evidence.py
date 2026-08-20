# -*- coding: utf-8 -*-
"""Fig. 3 (merged): EXA7 in-silico evidence in one figure.
Panels (lettered by first-citation order in the text):
  a  selective risk-coverage frontier (contract vs existence threshold)
  b  differential-diagnosis confusion matrix
  c  trained baselines vs the executor twin-fit
  d  per-severity leak-partition gradient
  e  noise robustness (accuracy and coverage@100% precision vs sigma)
All values read from committed artifacts; panel code carried over from
make_figures.py and make_fig4.py unchanged except axes wiring."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ROOT = fs.ROOT
fs.apply_rc()

R = json.load(open(os.path.join(ROOT, "artifacts", "results_full.json"), encoding="utf-8"))
ms = R["multiseed_sigma0.05"]
base = json.load(open(os.path.join(ROOT, "artifacts", "results_baselines_fair.json")))["forced_top1"]
stats = json.load(open(os.path.join(ROOT, "artifacts", "results_stats.json")))["per_severity_leak_acc"]

fig = plt.figure(figsize=(7.09, 4.9))
# separate gridspecs so the three narrow top panels get a wider gap than the
# two wide bottom panels
gs_top = fig.add_gridspec(1, 3, wspace=0.50, left=0.075, right=0.985,
                          top=0.93, bottom=0.615)
gs_bot = fig.add_gridspec(1, 2, wspace=0.30, left=0.075, right=0.985,
                          top=0.44, bottom=0.10)
axA = fig.add_subplot(gs_top[0, 0])
axB = fig.add_subplot(gs_top[0, 1])
axC = fig.add_subplot(gs_top[0, 2])
axD = fig.add_subplot(gs_bot[0, 0])
axE = fig.add_subplot(gs_bot[0, 1])
# top panels are ~0.22 of figure width vs ~0.40 for the bottom row; scale the
# letter-title pad so the ABSOLUTE gap matches panels d and e
TOP_PAD = 0.065

# ---- a) selective risk-coverage ----
full = sorted(R["risk_coverage_sigma0.05_seed42"], key=lambda c: c["coverage"])
bl = sorted(R["risk_coverage_existence_only_sigma0.05_seed42"], key=lambda c: c["coverage"])
axA.plot([c["coverage"] * 100 for c in full], [c["decision_precision"] * 100 for c in full],
         "-o", color=fs.SYS_DEEP, ms=2.8, lw=1.3, label="full contract")
axA.plot([c["coverage"] * 100 for c in bl], [c["decision_precision"] * 100 for c in bl],
         "-s", color=fs.GREY, ms=2.8, lw=1.0, label="existence only")
forced = ms["forced_retrieval_top1"]["mean"] * 100
axA.axhline(forced, color=fs.RED, ls="--", lw=0.9, label=f"forced retrieval ({forced:.0f}%)")
axA.set_xlabel("Coverage (%)")
axA.set_ylabel("Decision precision (%)")
axA.set_ylim(25, 103)
axA.legend(loc="lower left", handlelength=1.4, fontsize=5.8,
           frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
# guide annotations (values computed from the artifact, not hard-coded)
c100 = max(c["coverage"] * 100 for c in full if c["decision_precision"] >= 1.0)
axA.annotate(f"acts on {c100:.0f}%, all correct", xy=(c100, 100), xytext=(8, 87),
             fontsize=6.0, color=fs.SYS_DEEP,
             arrowprops=dict(arrowstyle="->", color=fs.SYS_DEEP, lw=0.7))
g_cov = min(c["coverage"] * 100 for c in bl)
g_prec = max(c["decision_precision"] * 100 for c in bl)
axA.annotate(f"a single score: {g_prec:.0f}% at best,\nnever all-correct",
             xy=(g_cov, g_prec), xytext=(26, 66), fontsize=6.0, color=fs.GREY_D,
             arrowprops=dict(arrowstyle="->", color=fs.GREY_D, lw=0.7))
fs.panel_tag(axA, "a", "Selective quality", title_pad=TOP_PAD)

# ---- b) confusion matrix ----
dc = R["differential_confusion_sigma0.05_seed42"]
cls = dc["classes"]
mat = np.array(dc["matrix"], dtype=float)
norm = mat / np.clip(mat.sum(axis=1, keepdims=True), 1, None)
im = axB.imshow(norm, cmap=fs.CMAP_SYS, vmin=0, vmax=1)
short = [c.replace("_", "\n") for c in cls]
axB.set_xticks(range(len(cls)))
axB.set_yticks(range(len(cls)))
axB.set_xticklabels(short, fontsize=5.6)
axB.set_yticklabels(short, fontsize=5.6)
axB.set_xlabel("Predicted top hypothesis")
axB.set_ylabel("True class")
axB.tick_params(length=0)
for sp in axB.spines.values():
    sp.set_visible(False)
for i in range(len(cls)):
    for j in range(len(cls)):
        if mat[i, j] > 0:
            axB.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=6.4,
                     color="white" if norm[i, j] > 0.55 else fs.INK)
cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
cb.set_label("Row rate", fontsize=6.0)
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(labelsize=5.6, width=0.5, length=2)
fs.panel_tag(axB, "b", "Differential diagnosis", title_pad=TOP_PAD)

# ---- c) trained baselines ----
order = ["GraphRAG retrieval", "GCN (3-layer, trained)", "MLP (128,64)", "RandomForest (200)",
         "kNN (k=5, cosine)", "Executor-Supervisor (twin-fit)"]
names = [o.split(" (")[0].replace("Executor-Supervisor", "Exec–Super")
          .replace("GraphRAG retrieval", "Retrieval").replace("RandomForest", "RF")
         for o in order]
cols = [fs.RED, fs.GREY, fs.GREY, fs.GREY, fs.GREY, fs.SYS_DEEP]
for i, (o, c) in enumerate(zip(order, cols)):
    m = base[o]["mean"] * 100
    sd = base[o]["std"] * 100
    vals = [x * 100 for x in base[o]["values"]]
    axC.barh(i, m, height=0.62, facecolor=fs.tint(c), edgecolor=c, linewidth=0.8, zorder=2)
    axC.errorbar(m, i, xerr=sd, fmt="none", ecolor=fs.INK, elinewidth=0.7,
                 capsize=2, capthick=0.7, zorder=4)
    fs.replicate_dots(axC, i, vals, c, orient="h", s=7, seed=i)
axC.set_yticks(range(len(order)))
axC.set_yticklabels(names, fontsize=5.9)
axC.set_xlabel("Forced Top-1 (%)")
axC.set_xlim(0, 100)
axC.invert_yaxis()
fs.panel_tag(axC, "c", "Trained baselines", x=-0.34, title_pad=TOP_PAD)

# ---- d) per-severity gradient ----
rates = sorted(float(k.split()[0]) for k in stats)
vbr = [[v * 100 for v in stats[f"{r} L/s"]["values"]] for r in rates]
means = [float(np.mean(vs)) for vs in vbr]
sds = [float(np.std(vs, ddof=1)) for vs in vbr]
axD.axhspan(0, 35, color=fs.RED, alpha=0.06, lw=0)
axD.plot(rates, means, "-", color=fs.SYS_DEEP, lw=1.5, zorder=4)
for r, vs in zip(rates, vbr):
    jr = np.random.default_rng(int(round(r * 13)))
    xj = float(r) * (1 + (jr.random(len(vs)) - 0.5) * 0.14)
    axD.scatter(xj, vs, s=8, facecolor=fs.SYS_MID, edgecolor="white", linewidths=0.4, zorder=3)
axD.errorbar(rates, means, yerr=sds, fmt="o", color=fs.SYS_DEEP, ms=3.8, elinewidth=0.8,
             capsize=2.5, capthick=0.8, zorder=5)
axD.text(2.3, 8, "noise-floor\nregime", color=fs.RED, fontsize=6.2)
axD.set_xlabel("Leak discharge (L s$^{-1}$)")
axD.set_ylabel("Leak-partition accuracy (%)")
axD.set_ylim(0, 112)
axD.set_xscale("log")
axD.set_xticks(rates)
axD.set_xticklabels([int(r) for r in rates])
axD.minorticks_off()
fs.panel_tag(axD, "d", "Accuracy vs severity")

# ---- e) noise robustness ----
ns = R["noise_sweep_multiseed"]
sig = sorted(float(k) for k in ns)
x = np.arange(len(sig))
series = [("leak_acc", -0.2, fs.SYS_DEEP, "leak-partition accuracy"),
          ("coverage_at_100prec", 0.2, fs.GREEN, "coverage @100% precision")]
for j, (key, off, c, lab) in enumerate(series):
    for i, s in enumerate(sig):
        d = ns[str(s)][key]
        m = d["mean"] * 100
        sd = d["std"] * 100
        vals = [v * 100 for v in d["values"]]
        axE.bar(x[i] + off, m, 0.36, facecolor=fs.tint(c), edgecolor=c, linewidth=0.8,
                zorder=2, label=lab if i == 0 else None)
        axE.errorbar(x[i] + off, m, yerr=sd, fmt="none", ecolor=fs.INK, elinewidth=0.7,
                     capsize=2, capthick=0.7, zorder=4)
        fs.replicate_dots(axE, x[i] + off, vals, c, orient="v", jw=0.07, s=8, seed=10 * j + i)
axE.set_xticks(x)
axE.set_xticklabels([f"{s:.2f}" for s in sig])
axE.set_xlabel("Sensor noise σ (m)")
axE.set_ylabel("Percent (%)")
axE.set_ylim(0, 108)
axE.legend(loc="upper right", fontsize=5.8)
fs.panel_tag(axE, "e", "Robustness to noise")

# nudge panel c right so its y-tick labels clear panel b's colorbar
box = axC.get_position()
axC.set_position([box.x0 + 0.018, box.y0, box.width - 0.018, box.height])

fs.save(fig, "Fig3_exa7_evidence")
print("saved Fig3_exa7_evidence -> png/pdf/svg")
