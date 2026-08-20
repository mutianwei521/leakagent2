"""
Demonstration selective-prediction evaluation on EXA7 (a real, reproducible run
— NO fabricated numbers). Generates a mixed 4-class sample, runs the
Executor<->Supervisor loop once per case, and sweeps the existence threshold to
produce risk-coverage curves for (a) the full physics-and-safety contract vs
(b) a plain existence-threshold selective baseline.

This is a METHODOLOGY DEMO at modest N, not the full multi-network paper eval.
Run:  python eval/run_eval.py            (via the wds_rag interpreter)
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context
from schemas.contract import GoalContract
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
SEED = 42
NOISE = 0.05          # moderate sensor noise -> a meaningful risk-coverage tradeoff
PERIOD = "day_normal"


def stat_row(ev, gr, scen):
    top1 = ev.top1
    best_alt = max((h.posterior for h in ev.hypotheses if h is not top1), default=0.0)
    rr = (top1.params.get("residual") if top1 else {}) or {}
    return {
        "case_id": scen.get("case_id"),
        "true_class": scen["true_class"],
        "true_region": int(scen.get("true_region", -1)),
        "top1_kind": top1.kind if top1 else "none",
        "pred_partition": (int(top1.partition) if top1 and top1.kind == "leak" else None),
        "retrieval_top1": int(gr.rank_partitions(ev.observation, top_k=1)[0][0]),
        "existence": float(ev.existence_mass),
        "margin": float(ev.margin),
        "top1_mahal": float(rr.get("mahalanobis_per_dof", 999.0)),
        "best_alt": float(best_alt),
        "region": int(len(ev.candidate_region)),
        "max_abs_dp": float(scen.get("max_abs_dp", 0.0)),
        "rate_Ls": scen.get("rate_Ls"),
    }


def build_sample(ctx, gr, rng):
    scens = []
    parts = ctx.partitions[:10]
    # LEAKS: KB representative node + one random junction, rates {5,20,50}
    for pid in parts:
        kb = gr.candidate_nodes(pid, max_nodes=1)
        kb_node = kb[0] if kb else None
        pnodes = ctx.partition_nodes_by_degree.get(pid, [])   # junctions only
        rand_node = str(rng.choice(pnodes)) if pnodes else None
        for node, tag in [(kb_node, "kb"), (rand_node, "rand")]:
            if node is None:
                continue
            for rate in (5.0, 20.0, 50.0):
                s = make_leak_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                       node, rate, PERIOD, noise_std=NOISE, rng=rng)
                if s:
                    s["case_id"] = f"leak_{tag}_P{pid}_N{node}_{int(rate)}"
                    scens.append(s)
    # DEMAND ANOMALIES: DMA-wide over-draw
    for pid in parts:
        for f in (1.6, 2.2):
            s = make_demand_anomaly_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                             pid, factor=f, period=PERIOD,
                                             noise_std=NOISE, rng=rng)
            if s:
                s["case_id"] = f"demand_P{pid}_f{f}"
                scens.append(s)
    # SENSOR FAULTS
    for i in range(15):
        k = int(rng.integers(1, 3))
        faulty = list(rng.choice(ctx.sensors, size=k, replace=False))
        mode = rng.choice(["bias", "stuck"])
        mag = float(rng.choice([0.5, 1.0, 1.5]))
        s = make_sensor_fault_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                       faulty, mode=mode, magnitude=mag,
                                       period=PERIOD, noise_std=NOISE, rng=rng)
        s["case_id"] = f"sensorfault_{i}"
        scens.append(s)
    # VALVE MIS-STATES: sample non-disconnecting pipe closures
    pipes = list(ctx.twin.wn.pipe_name_list)
    rng.shuffle(pipes)
    got = 0
    for pn in pipes:
        if got >= 15:
            break
        s = make_valve_misstate_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                         pn, period=PERIOD, noise_std=NOISE, rng=rng)
        if s and s["max_abs_dp"] < 200.0:
            s["case_id"] = f"valve_{pn}"
            scens.append(s)
            got += 1
    return scens


def main():
    print("=" * 74)
    print(f"SELECTIVE-PREDICTION DEMO — EXA7  (seed={SEED}, sensor_noise={NOISE} m)")
    print("=" * 74)
    ctx = build_context(CFG)
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt = DigitalTwinTool()
    rt = ResidualTool(sensor_std=NOISE, pass_threshold=3.0)
    be = BayesianEngine()
    # twin-fit over ALL partitions (retrieval only prioritizes), denser candidates
    ex = ExecutorAgent(ctx, gr, dt, rt, be, top_k_leak=len(ctx.partitions),
                       leak_cands=50, noise_std=NOISE)   # ~cover all nodes per partition
    asens = ActiveSensingTool()
    rng = np.random.default_rng(SEED)
    arng = np.random.default_rng(SEED + 1)

    scens = build_sample(ctx, gr, rng)
    by_class = {}
    for s in scens:
        by_class[s["true_class"]] = by_class.get(s["true_class"], 0) + 1
    print("sample:", by_class, " total:", len(scens))

    rows_pre, rows = [], []          # rows = post-active-sensing (the full system)
    for i, s in enumerate(scens):
        ev = EvidencePackage(observation=s["observation"], demand_period=PERIOD,
                             network=ctx.name, case_id=s["case_id"])
        ex.step(ev, 1)
        rows_pre.append(stat_row(ev, gr, s))                       # round-1 only (ablation)
        ex.active_sense_verify(ev, s.get("full_dp", {}), PERIOD, arng, asens,
                               n_extra=3, top_k=6)                 # P4 verification round
        rows.append(stat_row(ev, gr, s))
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(scens)} cases  (twin sims cached={len(ctx.twin._delta_cache)})")

    def _leak_acc(rs):
        lk = [r for r in rs if r["true_class"] == "leak"]
        return (sum(1 for r in lk if r["top1_kind"] == "leak"
                    and r["pred_partition"] == r["true_region"]) / len(lk)) if lk else 0.0
    sys_acc_pre, sys_acc_post = _leak_acc(rows_pre), _leak_acc(rows)

    ts = [round(x, 3) for x in np.linspace(0.0, 0.99, 34)]
    base = M.risk_coverage(rows, ts, mode="existence_only")
    # full contract: sweep the residual-fit gate to get a real precision-coverage curve
    mahal_grid = [round(x, 3) for x in np.linspace(0.3, 8.0, 40)]
    full = M.risk_coverage_mahal(rows, mahal_grid, min_margin=0.12, alt_max=0.20)
    forced = M.forced_top1_accuracy(rows)
    conf = M.differential_confusion(rows)

    summary = {
        "forced_localizer_top1_accuracy": forced,        # coverage=1.0 honest baseline
        "AURC_full_contract": M.aurc(full),
        "AURC_existence_only": M.aurc(base),
        "full@coverage>=0.3": M.precision_at_coverage(full, 0.30),
        "existence_only@coverage>=0.3": M.precision_at_coverage(base, 0.30),
        "full@coverage>=0.5": M.precision_at_coverage(full, 0.50),
    }

    out = {
        "provenance": {"network": "exa7", "seed": SEED, "sensor_noise_m": NOISE,
                       "n_cases": len(rows), "by_class": by_class,
                       "note": "real run, no fabricated numbers; methodology demo at modest N"},
        "summary": summary,
        "risk_coverage_full": full,
        "risk_coverage_existence_only": base,
        "differential_confusion": conf,
        "rows": rows,
    }
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    outpath = os.path.join(ROOT, "artifacts", "eval_demo_exa7.json")
    json.dump(out, open(outpath, "w", encoding="utf-8"), indent=2)

    # ---- report ----------------------------------------------------------
    # system leak-partition accuracy on ALL leak cases (coverage=1.0, full agent)
    leaks = [r for r in rows if r["true_class"] == "leak"]
    sys_acc = (sum(1 for r in leaks if r["top1_kind"] == "leak"
                   and r["pred_partition"] == r["true_region"]) / len(leaks)) if leaks else 0.0
    summary["system_leak_partition_accuracy"] = sys_acc

    print("\n--- HEADLINE (honest) ---")
    print(f"Forced RETRIEVAL Top-1 (coverage=1.0): "
          f"{forced['top1_accuracy']*100:.1f}%  (n_leaks={forced['n_leaks']})")
    print(f"System leak-partition acc (coverage=1.0): no-active-sensing {sys_acc_pre*100:.1f}% "
          f"-> +active-sensing {sys_acc_post*100:.1f}%")
    summary["leak_acc_no_active"] = sys_acc_pre
    summary["leak_acc_with_active"] = sys_acc_post
    print(f"AURC  full-contract = {summary['AURC_full_contract']:.4f}   "
          f"existence-only = {summary['AURC_existence_only']:.4f}   (lower=better)")
    for tag in ("full@coverage>=0.3", "existence_only@coverage>=0.3", "full@coverage>=0.5"):
        v = summary[tag]
        if v:
            print(f"{tag:32s}: precision={v['decision_precision']*100:.1f}%  "
                  f"coverage={v['coverage']*100:.1f}%  "
                  f"false_dispatch={v['false_dispatch_rate']*100:.1f}%")
    print("\n--- precision-coverage curve (full contract, sweep residual gate) ---")
    print("  coverage  precision  false_dispatch  n_acted")
    seen = set()
    for c in sorted(full, key=lambda c: -c["coverage"]):
        k = (round(c["coverage"], 2), round(c["decision_precision"], 2))
        if k in seen:
            continue
        seen.add(k)
        print(f"   {c['coverage']*100:5.1f}%    {c['decision_precision']*100:5.1f}%      "
              f"{c['false_dispatch_rate']*100:5.1f}%        {c['n_acted']}")

    print("\n--- differential confusion (rows=true, cols=pred top1) ---")
    cl = conf["classes"]
    print("           " + "  ".join(f"{c[:6]:>6}" for c in cl))
    for i, c in enumerate(cl):
        print(f"  {c[:9]:9s} " + "  ".join(f"{conf['matrix'][i][j]:6d}" for j in range(len(cl))))
    print(f"\nsaved -> {outpath}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
