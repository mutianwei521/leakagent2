# -*- coding: utf-8 -*-
"""Fig. 4 (merged): the two real-measurement tiers in one figure.
  a  L-Town real-data risk-coverage frontier + contract operating point
  b  L-Town contract vs scalar threshold at matched coverage
  c  City D register leaks vs the detection floor (acted/abstained)
  d  City D controlled severity sweep + register severity histogram
Panel code carried over from make_fig5.py and make_fig6.py unchanged except
axes wiring; all values read from committed artifacts."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ROOT = fs.ROOT
fs.apply_rc()

lt = json.load(open(os.path.join(ROOT, "artifacts", "results_ltown.json"), encoding="utf-8"))
dg = json.load(open(os.path.join(ROOT, "artifacts", "results_city_d.json"), encoding="utf-8"))
sw = json.load(open(os.path.join(ROOT, "artifacts", "results_city_d_severity.json"),
                    encoding="utf-8"))["sweep"]

fig = plt.figure(figsize=(7.09, 5.4))
gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.48,
                      left=0.085, right=0.95, top=0.94, bottom=0.09)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

# ---- a) L-Town risk-coverage frontier ----
fr = sorted(lt["risk_coverage"]["frontier"], key=lambda p: p["coverage"])
cov = np.array([p["coverage"] * 100 for p in fr])
prec = np.array([p["precision"] * 100 for p in fr])
summ = lt["summary"]
floor = summ["forced_top1_zone_accuracy"] * 100
axA.step(cov, prec, where="post", color=fs.SYS_MID, lw=1.6)
axA.axhline(floor, color=fs.RED, ls="--", lw=0.9)
axA.text(2, floor + 3, f"act-on-all floor ({floor:.0f}%)", color=fs.RED, fontsize=6.4, ha="left")
op_cov = summ["coverage"] * 100
op_prec = summ["decision_precision_on_acted"] * 100
axA.plot([op_cov], [op_prec], "*", color=fs.SYS_DEEP, ms=13, zorder=5)
axA.annotate("contract operating point\n(12% coverage, 100% precision)",
             (op_cov, op_prec), xytext=(op_cov + 10, op_prec - 24), fontsize=6.4,
             arrowprops=dict(arrowstyle="->", color=fs.SYS_DEEP, lw=0.7))
axA.set_xlabel("Coverage (% of leaks acted on)")
axA.set_ylabel("Decision precision (%)")
axA.set_xlim(0, 100)
axA.set_ylim(0, 105)
fs.panel_tag(axA, "a", "Benchmark risk–coverage (L-Town)")

# ---- b) contract vs scalar at matched coverage: the existence-margin plane ----
cvs = lt["contract_vs_scalar_threshold"]
rows_lt = {r["pipe"]: r for r in lt["rows"]}
con_set = set(cvs["contract_acted"])
sca_set = set(cvs["scalar_top_k"])
DELTA = 0.10                     # transfer-preset margin floor (Methods, Eq. 4 G3)
scalar_cut = min(rows_lt[p]["existence"] for p in sca_set)

# context: all 33 leaks in the existence-margin plane
ex_all = [r["existence"] for r in lt["rows"]]
mg_all = [r["margin"] for r in lt["rows"]]
axB.scatter(ex_all, mg_all, s=9, c=fs.GREY_LT, edgecolors="none", zorder=2)

# the two decision boundaries
axB.axvline(scalar_cut, color=fs.GREY_D, ls="--", lw=0.9, zorder=3)
axB.text(scalar_cut - 0.015, 0.62, f"scalar top-4 cut\n(existence ≥ {scalar_cut:.2f})",
         color=fs.GREY_D, fontsize=5.8, ha="right")
axB.axhline(DELTA, color=fs.SYS_DEEP, ls="--", lw=0.9, zorder=3)
axB.text(0.03, DELTA + 0.025, f"contract margin floor δ = {DELTA:g}",
         color=fs.SYS_DEEP, fontsize=5.8, ha="left")

# shared correct dispatches (both rules)
shared = sorted(con_set & sca_set)
axB.scatter([rows_lt[p]["existence"] for p in shared],
            [rows_lt[p]["margin"] for p in shared],
            s=34, c=fs.SYS_DEEP, edgecolors="white", linewidths=0.5, zorder=5)
axB.annotate("both rules dispatch:\n3 of 3 correct",
             xy=(0.995, 0.55), xytext=(0.72, 0.78), fontsize=5.9, color=fs.SYS_DEEP,
             arrowprops=dict(arrowstyle="->", color=fs.SYS_DEEP, lw=0.7))

# the swap that decides the comparison
veto = cvs["vetoed_by_contract"][0]
rv = rows_lt[veto]
axB.scatter([rv["existence"]], [rv["margin"]], s=52, marker="X", c=fs.RED,
            edgecolors="white", linewidths=0.5, zorder=6)
axB.annotate(f"scalar's extra pick: wrong zone\n(existence {rv['existence']:.2f}, "
             f"margin {rv['margin']:.2f}); contract vetoes",
             xy=(rv["existence"], rv["margin"]), xytext=(0.30, 0.24),
             fontsize=5.9, color=fs.RED,
             arrowprops=dict(arrowstyle="->", color=fs.RED, lw=0.7))
extra = sorted(con_set - sca_set)[0]
re_ = rows_lt[extra]
axB.scatter([re_["existence"]], [re_["margin"]], s=44, c=fs.SYS_DEEP,
            edgecolors=fs.INK, linewidths=0.7, zorder=6)
axB.annotate(f"contract's extra pick: correct\n(existence {re_['existence']:.2f}, "
             f"margin {re_['margin']:.2f})",
             xy=(re_["existence"], re_["margin"]), xytext=(0.30, 0.40),
             fontsize=5.9, color=fs.SYS_DEEP,
             arrowprops=dict(arrowstyle="->", color=fs.SYS_DEEP, lw=0.7))

# tallies
cP = cvs["contract_precision"] * 100
sP = cvs["scalar_existence_precision"] * 100
axB.text(0.03, 0.96, f"contract 4/4 = {cP:.0f}%", transform=axB.transAxes,
         fontsize=6.6, fontweight="bold", color=fs.SYS_DEEP, va="top")
axB.text(0.03, 0.88, f"scalar top-4 3/4 = {sP:.0f}%", transform=axB.transAxes,
         fontsize=6.6, fontweight="bold", color=fs.GREY_D, va="top")

axB.set_xlabel("Leak-existence mass (scalar score)")
axB.set_ylabel("Top-1 margin Δ")
axB.set_xlim(0, 1.05)
axB.set_ylim(0, 1.02)
fs.panel_tag(axB, "b", "Why the contract wins at matched coverage")

# ---- c) City D register vs detection floor ----
rows = dg["rows"]
det = dg["detectability"]
dsum = dg["summary"]
afloor = det["actionable_dp_threshold_m"]
noise1s = dsum["sensor_std_m"]
rate = np.array([max(r["rate_Ls"], 1e-4) for r in rows])
dpc = np.array([max(r["max_abs_dp_clean_m"], 1e-5) for r in rows])
acted = np.array([r["outcome"] == "ACTED" for r in rows])
axC.scatter(rate[~acted], dpc[~acted], s=7, c=fs.GREY, alpha=0.40, label="abstained",
            edgecolors="none")
if acted.any():
    axC.scatter(rate[acted], dpc[acted], s=26, c=fs.SYS_DEEP, label="acted",
                edgecolors="white", linewidths=0.5, zorder=5)
axC.axhline(afloor, color=fs.RED, ls="--", lw=0.9)
axC.text(rate.min() * 1.1, afloor * 1.3, f"actionable floor (~3σ = {afloor:g} m)",
         color=fs.RED, fontsize=6.2)
axC.axhline(noise1s, color=fs.AMBER, ls=":", lw=1.0)
axC.text(rate.min() * 1.1, noise1s * 1.3, f"1σ sensor noise ({noise1s:g} m)",
         color=fs.AMBER, fontsize=6.2)
axC.set_xscale("log")
axC.set_yscale("log")
axC.set_xlabel("Real leak rate (L s$^{-1}$, register-derived)")
axC.set_ylabel("Clean max|Δp| at sensors (m)")
axC.legend(loc="lower right")
fs.panel_tag(axC, "c", "Field leaks vs the floor (City D)")

# ---- d) severity sweep + register histogram ----
srate = np.array([s["rate_Ls"] for s in sw])
cover = np.array([s["coverage"] * 100 for s in sw])
forcedv = np.array([s["forced_top1_zone_acc"] * 100 for s in sw])
axD2 = axD.twinx()
bins = np.logspace(np.log10(max(rate.min(), 1e-3)), np.log10(50), 22)
axD2.hist(rate, bins=bins, color=fs.GREY_LT, alpha=0.55, edgecolor="none")
axD2.set_ylabel("Real register leaks (count)", color=fs.GREY, fontsize=6.8)
axD2.tick_params(axis="y", labelcolor=fs.GREY, labelsize=6.2, width=0.5, length=2)
axD2.spines["top"].set_visible(False)
axD.plot(srate, cover, "-o", color=fs.SYS_DEEP, lw=1.5, ms=4.2, label="coverage (acted)", zorder=6)
axD.plot(srate, forcedv, "-s", color=fs.GREEN, lw=1.3, ms=3.6, label="forced top-1 zone acc",
         zorder=6)
axD.axvline(np.median(rate), color=fs.RED, ls="--", lw=0.9)
axD.text(np.median(rate) * 1.15, 64, f"register median\n{np.median(rate):.3f} L s$^{{-1}}$",
         color=fs.RED, fontsize=6.2)
axD.set_xscale("log")
axD.set_xlabel("Leak rate (L s$^{-1}$)")
axD.set_ylabel("Localization (%)")
axD.set_ylim(0, 100)
axD.set_zorder(axD2.get_zorder() + 1)
axD.patch.set_visible(False)
axD.legend(loc="upper left", fontsize=5.8)
fs.panel_tag(axD, "d", "Large leaks localize; real ones are smaller")

fs.save(fig, "Fig4_realworld")
print("saved Fig4_realworld -> png/pdf/svg")
