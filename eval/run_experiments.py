"""
Paper-grade EXA7 experiments (real runs, no fabrication):
  - multi-seed robustness at sigma=0.05  (mean +/- std)
  - noise sweep sigma in {0.02, 0.05, 0.10}
  - active-sensing ablation (with vs without)
  - calibration (ECE + reliability) of the residual-derived confidence
  - paired McNemar: full system (acted) vs forced retrieval on the leak subset

One twin is built ONCE and its leak-response cache is shared across all runs
(leaks are noise-independent), so the whole suite is tractable.

Run:  python eval/run_experiments.py     (via the wds_rag interpreter)
"""
import os
import sys
import json
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context
from schemas.evidence import EvidencePackage
from tools.graphrag_tool import GraphRAGTool
from tools.digital_twin import DigitalTwinTool
from tools.residual_analysis import ResidualTool
from tools.bayesian import BayesianEngine
from tools.active_sensing import ActiveSensingTool
from agents.executor import ExecutorAgent
from anomaly_sim.leak_anomaly import make_leak_scenario
from anomaly_sim.demand_anomaly import make_demand_anomaly_scenario
from anomaly_sim.sensor_fault import make_sensor_fault_scenario
from anomaly_sim.valve_misstate import make_valve_misstate_scenario
import eval.metrics as M

CFG = os.path.join(ROOT, "configs", "exa7.yaml")
PERIOD = "day_normal"
SEEDS = [17, 42, 101, 2025, 31337]
NOISES = [0.02, 0.05, 0.10]


def generate(ctx, gr, rng, noise):
    scens, parts = [], ctx.partitions[:10]
    for pid in parts:
        kb = gr.candidate_nodes(pid, max_nodes=1)
        kb_node = kb[0] if kb else None
        pnodes = ctx.partition_nodes_by_degree.get(pid, [])
        rand_node = str(rng.choice(pnodes)) if pnodes else None
        for node, tag in [(kb_node, "kb"), (rand_node, "rand")]:
            if node is None:
                continue
            for rate in (5.0, 20.0, 50.0):
                s = make_leak_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                       node, rate, PERIOD, noise_std=noise, rng=rng)
                if s:
                    s["case_id"] = f"leak_{tag}_P{pid}_N{node}_{int(rate)}"
                    scens.append(s)
    for pid in parts:
        for f in (1.6, 2.2):
            s = make_demand_anomaly_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                             pid, factor=f, period=PERIOD, noise_std=noise, rng=rng)
            if s:
                s["case_id"] = f"demand_P{pid}_f{f}"
                scens.append(s)
    for i in range(15):
        k = int(rng.integers(1, 3))
        faulty = list(rng.choice(ctx.sensors, size=k, replace=False))
        s = make_sensor_fault_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors, faulty,
                                       mode=str(rng.choice(["bias", "stuck"])),
                                       magnitude=float(rng.choice([0.5, 1.0, 1.5])),
                                       period=PERIOD, noise_std=noise, rng=rng)
        s["case_id"] = f"sensorfault_{i}"
        scens.append(s)
    pipes = list(ctx.twin.wn.pipe_name_list)
    rng.shuffle(pipes)
    got = 0
    for pn in pipes:
        if got >= 15:
            break
        s = make_valve_misstate_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors, pn,
                                         period=PERIOD, noise_std=noise, rng=rng)
        if s and s["max_abs_dp"] < 200.0:
            s["case_id"] = f"valve_{pn}"
            scens.append(s)
            got += 1
    return scens


def stat_row(ev, gr, scen):
    top1 = ev.top1
    best_alt = max((h.posterior for h in ev.hypotheses if h is not top1), default=0.0)
    rr = (top1.params.get("residual") if top1 else {}) or {}
    return {"true_class": scen["true_class"], "true_region": int(scen.get("true_region", -1)),
            "top1_kind": top1.kind if top1 else "none",
            "pred_partition": (int(top1.partition) if top1 and top1.kind == "leak" else None),
            "retrieval_top1": int(gr.rank_partitions(ev.observation, top_k=1)[0][0]),
            "existence": float(ev.existence_mass), "margin": float(ev.margin),
            "top1_mahal": float(rr.get("mahalanobis_per_dof", 999.0)),
            "best_alt": float(best_alt), "region": int(len(ev.candidate_region))}


def run_once(ctx, gr, dt, asens, noise, seed, active=True):
    rt = ResidualTool(sensor_std=noise, pass_threshold=3.0)
    ex = ExecutorAgent(ctx, gr, dt, rt, BayesianEngine(), top_k_leak=len(ctx.partitions),
                       leak_cands=50, noise_std=noise)
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed + 1)
    scens = generate(ctx, gr, rng, noise)
    rows_pre, rows = [], []
    for s in scens:
        ev = EvidencePackage(observation=s["observation"], demand_period=PERIOD, network=ctx.name)
        ex.step(ev, 1)
        rows_pre.append(stat_row(ev, gr, s))
        if active:
            ex.active_sense_verify(ev, s.get("full_dp", {}), PERIOD, arng, asens, n_extra=3, top_k=6)
        rows.append(stat_row(ev, gr, s))
    return rows_pre, rows


def leak_acc(rows):
    lk = [r for r in rows if r["true_class"] == "leak"]
    return (sum(1 for r in lk if r["top1_kind"] == "leak" and r["pred_partition"] == r["true_region"])
            / len(lk)) if lk else 0.0


def op_point(rows):
    """At the residual gate giving max coverage, report precision/coverage/false-dispatch,
    plus the max coverage achievable at 100% precision."""
    grid = [round(x, 3) for x in np.linspace(0.3, 8.0, 40)]
    curve = M.risk_coverage_mahal(rows, grid, min_margin=0.12, alt_max=0.20)
    top = max(curve, key=lambda c: c["coverage"])
    cov100 = max([c["coverage"] for c in curve if c["decision_precision"] >= 0.999], default=0.0)
    return {"coverage": top["coverage"], "precision": top["decision_precision"],
            "false_dispatch": top["false_dispatch_rate"], "coverage_at_100prec": cov100}


def ece(rows, n_bins=10):
    """ECE of a residual-derived confidence c=1/(1+mahal) on leak cases."""
    lk = [r for r in rows if r["true_class"] == "leak" and r["top1_kind"] == "leak"]
    if not lk:
        return None
    conf = [1.0 / (1.0 + r["top1_mahal"]) for r in lk]
    corr = [1.0 if r["pred_partition"] == r["true_region"] else 0.0 for r in lk]
    e, N = 0.0, len(lk)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi or (b == 0 and c == 0))]
        if not idx:
            continue
        acc = np.mean([corr[i] for i in idx])
        cf = np.mean([conf[i] for i in idx])
        e += (len(idx) / N) * abs(acc - cf)
    return float(e)


def mcnemar(rows):
    """Paired McNemar: full system 'acted & correct' vs forced retrieval, leak subset.
    Acted-correct uses the operating point (gates pass + leak top1 + pred==true)."""
    lk = [r for r in rows if r["true_class"] == "leak"]
    n01 = n10 = 0  # n10: system right, retrieval wrong ; n01: opposite
    for r in lk:
        sys_ok = (r["top1_kind"] == "leak" and r["existence"] >= 0.5 and r["margin"] >= 0.12
                  and r["best_alt"] <= 0.20 and r["top1_mahal"] <= 3.0
                  and r["pred_partition"] == r["true_region"])
        ret_ok = (r["retrieval_top1"] == r["true_region"])
        if sys_ok and not ret_ok:
            n10 += 1
        elif ret_ok and not sys_ok:
            n01 += 1
    nd = n10 + n01
    chi2 = ((abs(n10 - n01) - 1) ** 2) / nd if nd > 0 else 0.0
    return {"n10_system_only": n10, "n01_retrieval_only": n01, "chi2_cc": chi2,
            "significant_p<0.05": chi2 > 3.841}


def agg(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a), "values": [round(float(v), 4) for v in a]}


def main():
    print("=" * 74)
    print("EXA7 PAPER-GRADE EXPERIMENTS (real runs, no fabrication)")
    print("=" * 74)
    ctx = build_context(CFG)
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt = DigitalTwinTool()
    asens = ActiveSensingTool()

    # ---- multi-seed at sigma=0.05 ----------------------------------------
    print("\n[1] Multi-seed robustness (sigma=0.05) ...")
    leak_pre, leak_post, prec, cov100, fd, retr = [], [], [], [], [], []
    rows_main = None
    for sd in SEEDS:
        rp, rw = run_once(ctx, gr, dt, asens, 0.05, sd, active=True)
        leak_pre.append(leak_acc(rp)); leak_post.append(leak_acc(rw))
        op = op_point(rw)
        prec.append(op["precision"]); cov100.append(op["coverage_at_100prec"])
        fd.append(op["false_dispatch"])
        retr.append(M.forced_top1_accuracy(rw)["top1_accuracy"])
        if sd == 42:
            rows_main = rw
        print(f"   seed {sd}: leak_acc {leak_acc(rw)*100:.1f}%  "
              f"prec@maxcov {op['precision']*100:.1f}%  cov@100%prec {op['coverage_at_100prec']*100:.1f}%")

    # ---- noise sweep (all seeds) -----------------------------------------
    print("\n[2] Noise sweep (all seeds) ...")
    noise_rows = {}          # seed-42 rows retained for the single-seed keys (ece, confusion)
    noise_multiseed = {}     # per-sigma: per-seed leak_acc / coverage / precision / false-dispatch
    for ns in NOISES:
        la_s, cov_s, prec_s, fd_s = [], [], [], []
        for sd in SEEDS:
            _, rw = run_once(ctx, gr, dt, asens, ns, sd, active=True)
            op = op_point(rw)
            la_s.append(leak_acc(rw)); cov_s.append(op["coverage_at_100prec"])
            prec_s.append(op["precision"]); fd_s.append(op["false_dispatch"])
            if sd == 42:
                noise_rows[ns] = rw
        noise_multiseed[str(ns)] = {"leak_acc": agg(la_s), "coverage_at_100prec": agg(cov_s),
                                    "precision_at_maxcov": agg(prec_s), "false_dispatch_at_maxcov": agg(fd_s)}
        print(f"   sigma={ns:.2f}: leak_acc {agg(la_s)['mean']*100:.1f}+/-{agg(la_s)['std']*100:.1f}%  "
              f"prec {agg(prec_s)['mean']*100:.1f}%  cov@100%prec {agg(cov_s)['mean']*100:.1f}%")

    # ---- active-sensing ablation (sigma=0.05, all seeds) -----------------
    abl_pre = agg(leak_pre); abl_post = agg(leak_post)

    out = {
        "provenance": {"network": "exa7", "seeds": SEEDS, "noises": NOISES,
                       "note": "real runs; leak twin-cache shared; no fabricated numbers"},
        "multiseed_sigma0.05": {
            "forced_retrieval_top1": agg(retr),
            "leak_partition_acc_no_active": abl_pre,
            "leak_partition_acc_with_active": abl_post,
            "decision_precision_at_maxcov": agg(prec),
            "coverage_at_100pct_precision": agg(cov100),
            "false_dispatch_at_maxcov": agg(fd),
        },
        "noise_sweep_seed42": {
            str(ns): {"leak_acc": leak_acc(noise_rows[ns]), **op_point(noise_rows[ns]),
                      "ece": ece(noise_rows[ns]),
                      "differential_confusion": M.differential_confusion(noise_rows[ns])}
            for ns in NOISES},
        "noise_sweep_multiseed": noise_multiseed,
        "calibration_ece_sigma0.05_seed42": ece(rows_main),
        "mcnemar_system_vs_retrieval_sigma0.05_seed42": mcnemar(rows_main),
        "risk_coverage_sigma0.05_seed42": M.risk_coverage_mahal(
            rows_main, [round(x, 3) for x in np.linspace(0.3, 8.0, 40)]),
        "risk_coverage_existence_only_sigma0.05_seed42": M.risk_coverage(
            rows_main, [round(x, 3) for x in np.linspace(0.0, 0.99, 34)], mode="existence_only"),
        "differential_confusion_sigma0.05_seed42": M.differential_confusion(rows_main),
    }
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    outp = os.path.join(ROOT, "artifacts", "results_full.json")
    json.dump(out, open(outp, "w", encoding="utf-8"), indent=2)

    # ---- report ----------------------------------------------------------
    ms = out["multiseed_sigma0.05"]
    print("\n--- HEADLINE (mean +/- std over", len(SEEDS), "seeds, sigma=0.05) ---")
    print(f"  forced retrieval Top-1     : {ms['forced_retrieval_top1']['mean']*100:.1f} "
          f"+/- {ms['forced_retrieval_top1']['std']*100:.1f}%")
    print(f"  leak-acc  (no active)      : {abl_pre['mean']*100:.1f} +/- {abl_pre['std']*100:.1f}%")
    print(f"  leak-acc  (with active)    : {abl_post['mean']*100:.1f} +/- {abl_post['std']*100:.1f}%  "
          f"(active sensing delta = +{(abl_post['mean']-abl_pre['mean'])*100:.1f}pp)")
    print(f"  decision-precision @maxcov : {ms['decision_precision_at_maxcov']['mean']*100:.1f} "
          f"+/- {ms['decision_precision_at_maxcov']['std']*100:.1f}%")
    print(f"  coverage @100% precision   : {ms['coverage_at_100pct_precision']['mean']*100:.1f} "
          f"+/- {ms['coverage_at_100pct_precision']['std']*100:.1f}%")
    print(f"  false-dispatch @maxcov     : {ms['false_dispatch_at_maxcov']['mean']*100:.1f} "
          f"+/- {ms['false_dispatch_at_maxcov']['std']*100:.1f}%")
    mc = out["mcnemar_system_vs_retrieval_sigma0.05_seed42"]
    print(f"  McNemar vs retrieval       : n10={mc['n10_system_only']} n01={mc['n01_retrieval_only']} "
          f"chi2={mc['chi2_cc']:.1f} sig={mc['significant_p<0.05']}")
    print(f"  calibration ECE            : {out['calibration_ece_sigma0.05_seed42']:.3f}")
    print(f"\nsaved -> {outp}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
