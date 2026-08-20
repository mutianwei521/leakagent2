"""
Publication figures from the real results JSON (no hard-coded numbers - every
value is read from artifacts/results_full.json). Outputs PNG+PDF+SVG to figures/.
Shared Nature-subjournal style lives in eval/figstyle.py.

Run:  python eval/make_figures.py     (via the wds_rag interpreter)
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ROOT = fs.ROOT
fs.apply_rc()


def main():
    R = json.load(open(os.path.join(ROOT, "artifacts", "results_full.json"), encoding="utf-8"))
    ms = R["multiseed_sigma0.05"]

    # ---- Fig 2: precision-coverage + noise sweep -------------------------
    fig, ax = plt.subplots(1, 2, figsize=fs.SIZE_2PANEL)   # two silver-ratio panels
    fig.subplots_adjust(wspace=0.32)
    full = sorted(R["risk_coverage_sigma0.05_seed42"], key=lambda c: c["coverage"])
    base = sorted(R["risk_coverage_existence_only_sigma0.05_seed42"], key=lambda c: c["coverage"])
    ax[0].plot([c["coverage"] * 100 for c in full], [c["decision_precision"] * 100 for c in full],
               "-o", color=fs.SYS_DEEP, ms=3.2, lw=1.4, label="Executor–Supervisor (full contract)")
    ax[0].plot([c["coverage"] * 100 for c in base], [c["decision_precision"] * 100 for c in base],
               "-s", color=fs.GREY, ms=3.2, lw=1.1, label="existence threshold only")
    forced = ms["forced_retrieval_top1"]["mean"] * 100
    ax[0].axhline(forced, color=fs.RED, ls="--", lw=0.9, label=f"forced retrieval ({forced:.0f}%)")
    ax[0].set_xlabel("Coverage (% of events acted on)")
    ax[0].set_ylabel("Decision precision (%)")
    ax[0].set_ylim(25, 102)
    ax[0].legend(loc="center left", handlelength=1.6)
    fs.panel_tag(ax[0], "a", "Selective decision quality")

    ns = R["noise_sweep_multiseed"]
    sig = sorted(float(k) for k in ns)
    x = np.arange(len(sig))
    series = [("leak_acc", -0.2, fs.SYS_DEEP, "leak-partition accuracy"),
              ("coverage_at_100prec", 0.2, fs.SYS_LT, "coverage @100% precision")]
    for j, (key, off, c, lab) in enumerate(series):
        for i, s in enumerate(sig):
            d = ns[str(s)][key]
            m = d["mean"] * 100
            sd = d["std"] * 100
            vals = [v * 100 for v in d["values"]]      # the 5 real per-seed runs
            ax[1].bar(x[i] + off, m, 0.36, facecolor=fs.tint(c), edgecolor=c, linewidth=0.8,
                      zorder=2, label=lab if i == 0 else None)
            ax[1].errorbar(x[i] + off, m, yerr=sd, fmt="none", ecolor=fs.INK, elinewidth=0.7,
                           capsize=2, capthick=0.7, zorder=4)
            fs.replicate_dots(ax[1], x[i] + off, vals, c, orient="v", jw=0.07, s=9, seed=10 * j + i)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{s:.2f}" for s in sig])
    ax[1].set_xlabel("Sensor noise σ (m)")
    ax[1].set_ylabel("Percent (%)")
    ax[1].set_ylim(0, 108)
    ax[1].legend(loc="upper right")
    fs.panel_tag(ax[1], "b", "Robustness to sensor noise")
    fs.save(fig, "Fig2_selective_and_noise")

    # ---- Fig 3: differential confusion -----------------------------------
    dc = R["differential_confusion_sigma0.05_seed42"]
    cls = dc["classes"]
    mat = np.array(dc["matrix"], dtype=float)
    norm = mat / np.clip(mat.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(4.7, round(4.7 / fs.GOLDEN, 2)))   # golden canvas
    im = ax.imshow(norm, cmap=fs.CMAP_SYS, vmin=0, vmax=1)
    ax.set_xticks(range(len(cls)))
    ax.set_yticks(range(len(cls)))
    short = [c.replace("_", "\n") for c in cls]
    ax.set_xticklabels(short)
    ax.set_yticklabels(short)
    ax.set_xlabel("Predicted top hypothesis")
    ax.set_ylabel("True class")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(cls)):
        for j in range(len(cls)):
            if mat[i, j] > 0:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.55 else fs.INK)
    ax.set_title("Differential diagnosis (event counts)", loc="left", fontsize=7.5, pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalised rate", fontsize=6.8)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6.2, width=0.5, length=2)
    fs.save(fig, "Fig3_differential_confusion")

    # ---- Fig 4: multi-seed robustness (bars + every per-seed run as a dot) ----
    fig, ax = plt.subplots(figsize=fs.SIZE_1PANEL)   # golden-ratio canvas
    labels = ["forced\nretrieval", "leak-acc\n(no active)", "leak-acc\n(+active)",
              "decision\nprecision"]
    keys = ["forced_retrieval_top1", "leak_partition_acc_no_active",
            "leak_partition_acc_with_active", "decision_precision_at_maxcov"]
    cols = [fs.RED, fs.SYS_DEEP, fs.SYS_DEEP, fs.SYS_DEEP]
    for i, (k, c) in enumerate(zip(keys, cols)):
        m = ms[k]["mean"] * 100
        sd = ms[k]["std"] * 100
        vals = [v * 100 for v in ms[k]["values"]]      # the 5 real per-seed runs
        ax.bar(i, m, width=0.62, facecolor=fs.tint(c), edgecolor=c, linewidth=0.8, zorder=2)
        ax.errorbar(i, m, yerr=sd, fmt="none", ecolor=fs.INK, elinewidth=0.8,
                    capsize=2.5, capthick=0.8, zorder=4)
        fs.replicate_dots(ax, i, vals, c, orient="v", seed=i)
        ax.text(i, m + sd + 2.5, f"{m:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 112)
    n = ms["forced_retrieval_top1"]["n"]
    ax.set_title(f"Mean ± s.d. with all {n} per-seed runs (σ = 0.05)",
                 loc="left", fontsize=7.5, pad=6)
    fs.save(fig, "Fig7_multiseed_robustness")   # Fig 7 after renumbering to citation order

    print(f"\nAll figures -> {fs.FIGDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
