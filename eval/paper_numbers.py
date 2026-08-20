# -*- coding: utf-8 -*-
"""Manifest of the numbers the main text quotes from the in-silico artifacts.

Prints each quantity as the paper rounds it, so a regeneration can be diffed
against the manuscript line by line. Optional first argument: an alternative
artifacts directory (e.g. artifacts/archive_pre_margin_fix) to print side by side.

Run:  python eval/paper_numbers.py [old_dir]
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def L(d, name):
    return json.load(open(os.path.join(d, name), encoding="utf-8"))


def ms(x, digits=1):
    """mean +- sd (ddof=1, as the paper reports) in percent."""
    v = np.array(x["values"], dtype=float) * 100
    return f"{v.mean():.{digits}f} +- {v.std(ddof=1):.{digits}f}"


def manifest(d):
    out = {}
    rf = L(d, "results_full.json"); m = rf["multiseed_sigma0.05"]
    out["EXA7 forced retrieval top-1 (%)"] = ms(m["forced_retrieval_top1"])
    out["EXA7 twin-fit before active sensing (%)"] = ms(m["leak_partition_acc_no_active"])
    out["EXA7 twin-fit with active sensing (%)"] = ms(m["leak_partition_acc_with_active"])
    out["EXA7 decision precision at op point (%)"] = ms(m["decision_precision_at_maxcov"])
    out["EXA7 false dispatch at op point (%)"] = ms(m["false_dispatch_at_maxcov"])
    out["EXA7 coverage@100% precision (%)"] = ms(m["coverage_at_100pct_precision"], 0)
    for nz in ("0.02", "0.05", "0.1"):
        n = rf["noise_sweep_multiseed"][nz]
        out[f"noise {nz}: leak acc / precision / cov@100 (%)"] = (
            f"{ms(n['leak_acc'])} / {ms(n['precision_at_maxcov'])} / {np.mean(n['coverage_at_100prec']['values'])*100:.1f}")
    # seed-42 curve annotations (Fig. 4a)
    rc = rf["risk_coverage_sigma0.05_seed42"]
    out["seed42 max coverage at 100% precision (%)"] = f"{max(c['coverage'] for c in rc if c['decision_precision'] >= 0.999)*100:.1f}"
    ex = rf["risk_coverage_existence_only_sigma0.05_seed42"]
    out["seed42 existence-only min coverage / best precision (%)"] = (
        f"{min(c['coverage'] for c in ex)*100:.1f} / {max(c['decision_precision'] for c in ex)*100:.0f}")
    st = L(d, "results_stats.json")
    mc = st["mcnemar_vs_retrieval_pooled"]
    out["pooled McNemar chi2 / n10 / n01"] = f"{mc['chi2_cc']:.0f} / {mc['n10_system_only']} / {mc['n01_retrieval_only']}"
    cm = st["differential_confusion_pooled"]["matrix"]
    out["pooled confusion: leak row / demand row / sensor row / valve row"] = f"{cm[0]} / {cm[1]} / {cm[2]} / {cm[3]}"
    out["confounder false dispatch full / no-diff (of 250)"] = (
        f"{st['ablation_no_differential']['full_confounder_false_dispatch'][0]} / {st['ablation_no_differential']['no_diff_confounder_false_dispatch'][0]}")
    out["loosened contract precision (%) / acted"] = (
        f"{st['ablation_loosened_contract']['precision']*100:.1f} / {st['ablation_loosened_contract']['acted']}")
    for k, v in st["per_severity_leak_acc"].items():
        out[f"severity {k} leak acc (%)"] = f"{v['acc']*100:.0f}"
    bf = L(d, "results_baselines_fair.json")["forced_top1"]
    for k, v in bf.items():
        out[f"forced top-1 {k} (%)"] = ms(v)
    cf = L(d, "results_conformal.json")
    out["ECE raw (n_test)"] = f"{cf['ece']['raw']:.3f} ({cf['provenance']['n_test']})"
    p = cf["precision_at_cov0.30"]
    out["precision at cov>=0.30: contract / calibrated / existence-only (%)"] = (
        f"{p['goal_contract']*100:.1f} / {p['calibrated_threshold']*100:.1f} / {p['existence_only']*100:.1f}")
    for k, v in cf["split_conformal"].items():
        out[f"split conformal {k}: test risk / coverage (%)"] = f"{v['test_risk']*100:.1f} / {v['test_coverage']*100:.1f}"
    sv = L(d, "results_sensitivity.json")
    out["threshold sweep precision min / max (%)"] = f"{sv['precision_min_across_sweeps']*100:.1f} / {sv['precision_max_across_sweeps']*100:.1f}"
    r25 = [s for s in sv["sweeps"]["max_region"] if s["value"] == 25][0]
    out["R_max=25: coverage / precision (%)"] = f"{r25['coverage']*100:.1f} / {r25['precision']*100:.1f}"
    ab = L(d, "results_ablation_components.json")
    out["ablation full: precision / coverage / n_acted / demand->leak"] = (
        f"{ab['full']['decision_precision']*100:.1f} / {ab['full']['coverage']*100:.1f} / {ab['full']['n_acted']} / {ab['full']['demand_to_leak_mistype']}")
    out["ablation no-Occam: demand->leak / precision"] = f"{ab['no_occam']['demand_to_leak_mistype']} / {ab['no_occam']['decision_precision']*100:.1f}"
    out["ablation random reveal: precision / leak acc (%)"] = f"{ab['random_reveal']['decision_precision']*100:.1f} / {ab['random_reveal']['leak_partition_acc']*100:.1f}"
    out["ablation active sensing leak acc: none / discriminative (%)"] = f"{ab['no_active_leak_acc']*100:.1f} / {ab['full']['leak_partition_acc']*100:.1f}"
    for net, fn in (("KY4", "results_ky4.json"), ("City H std", "results_city_h_standard.json"),
                    ("City D std", "results_city_d_standard.json"), ("City D v2 std", "results_city_d_v2_standard.json")):
        try:
            mm = L(d, fn)["multiseed_sigma0.05"]
        except FileNotFoundError:
            continue
        out[f"{net}: forced retrieval / twin-fit / precision / coverage / cov@100 (%)"] = (
            f"{ms(mm['forced_retrieval_top1'])} / {ms(mm['leak_partition_acc_with_active'])} / "
            f"{ms(mm['decision_precision_at_maxcov'])} / {ms(mm['coverage_at_maxcov'])} / {ms(mm['coverage_at_100pct_precision'])}")
    ub = L(d, "results_unified_battery.json")
    for net in ("exa7", "ky4", "city_h", "city_d"):
        pl = ub[net]["pooled"]
        out[f"battery {net}: coverage / contract precision / scalar matched (%)"] = (
            f"{pl['coverage']*100:.1f} / {pl['contract_precision']*100:.1f} / {pl['scalar_existence_precision_matched']*100:.1f}")
    for fn, keyf in (("results_accountability.json", lambda a: f"catch {a['catch_rate_overall_on_corrupted']*100:.0f}% / FA {a['false_alarm_rate_on_clean']*100:.0f}% / stab {a['decision_stability_temp0']*100:.0f}% ({a['supervisor_model']})"),
                     ("results_llm_executor.json", lambda a: f"agreement {a['decision_agreement_llm_vs_deterministic']*100:.0f}% (n={a['n_cases']})")):
        try:
            out[fn] = keyf(L(d, fn))
        except FileNotFoundError:
            pass
    return out


def main():
    new = manifest(os.path.join(ROOT, "artifacts"))
    old = manifest(sys.argv[1]) if len(sys.argv) > 1 else None
    w = max(len(k) for k in new) + 2
    for k, v in new.items():
        if old is not None:
            o = old.get(k, "-")
            flag = "   " if o == v else " * "
            print(f"{flag}{k:<{w}} old: {o:<34} new: {v}")
        else:
            print(f"{k:<{w}} {v}")


if __name__ == "__main__":
    main()
