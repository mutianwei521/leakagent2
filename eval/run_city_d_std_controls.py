# -*- coding: utf-8 -*-
"""
City D STANDARD-leg no-leak controls under three presets (sigma = 0.05 m).

The unified battery measured 2/50 pure-noise dispatches on the City D standard
leg at the transfer gate. This program reproduces those 50 controls on the
exact battery path (same executor construction, control seed 7, per-seed-42
operating gate) and evaluates the same three presets used for the register leg:

  A transfer (as run):  exist>=0.5, margin>=0.12, mahal<=gate42, alt<=0.20, region<=60
  B audit:              exist>=0.7, margin>=0.15, mahal<=gate42, alt<=0.20, region<=60
  C A + 3-sigma observation floor (max|dp_obs| >= 0.15 m)

Run:  <wds_rag python> eval/run_city_d_std_controls.py
Output: artifacts/results_city_d_std_controls.json
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
from schemas.evidence import EvidencePackage
from tools.graphrag_tool import GraphRAGTool
from tools.digital_twin import DigitalTwinTool
from tools.residual_analysis import ResidualTool
from tools.bayesian import BayesianEngine
from agents.executor import ExecutorAgent
from eval.metrics import gate_act
from eval.run_experiments import stat_row, PERIOD

NOISE, N, CTRL_SEED, FLOOR = 0.05, 50, 7, 0.15


def main():
    print("=" * 70)
    print("CITY_D STANDARD-LEG CONTROLS x THREE PRESETS (sigma=0.05, N=50, seed 7)")
    print("=" * 70)
    bat = json.load(open(os.path.join(ROOT, "artifacts", "results_unified_battery.json"),
                         encoding="utf-8"))
    gate42 = bat["city_d"]["per_seed_gates"]["42"]
    print(f"operating gate (seed 42): {gate42}")

    ctx = build_context(os.path.join(ROOT, "configs", "city_d.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    rt = ResidualTool(sensor_std=NOISE, pass_threshold=3.0)
    ex = ExecutorAgent(ctx, gr, DigitalTwinTool(), rt, BayesianEngine(),
                       top_k_leak=len(ctx.partitions), leak_cands=25, noise_std=NOISE)

    rng = np.random.RandomState(CTRL_SEED)
    counts = {"A_transfer": 0, "B_audit": 0, "C_floor": 0}
    fired = []
    for j in range(N):
        obs = {s: float(rng.normal(0.0, NOISE)) for s in ctx.sensors}
        maxdp = max(abs(v) for v in obs.values())
        ev = EvidencePackage(observation=obs, demand_period=PERIOD, network=ctx.name)
        ex.step(ev, 1)
        row = stat_row(ev, gr, {"true_class": "none", "true_region": -1})
        a = gate_act(row, 0.5, 0.12, gate42, 0.20, 60, mode="full")
        b = gate_act(row, 0.7, 0.15, gate42, 0.20, 60, mode="full")
        c = a and (maxdp >= FLOOR)
        counts["A_transfer"] += a
        counts["B_audit"] += b
        counts["C_floor"] += c
        if a or b:
            fired.append({"control": j, "max_abs_dp_obs_m": round(maxdp, 4),
                          "max_obs_in_sigma": round(maxdp / NOISE, 2),
                          "existence": round(row["existence"], 4),
                          "margin": round(row["margin"], 4),
                          "mahal": round(row["top1_mahal"], 3),
                          "best_alt": round(row["best_alt"], 4), "region": row["region"],
                          "passes": {"A": bool(a), "B": bool(b), "C": bool(c)}})
            print(f"  FIRED ctrl {j}: {fired[-1]}", flush=True)
        if (j + 1) % 10 == 0:
            print(f"  ...{j+1}/{N}", flush=True)

    out = {"provenance": {"network": "city_d standard leg (current placement)", "sigma_m": NOISE,
                          "n_controls": N, "seed": CTRL_SEED, "gate42": gate42,
                          "path": "identical to run_unified_battery controls (gate_act); presets "
                                  "B/C evaluated on the same rows"},
           "false_dispatches": counts,
           "false_alarm_rates": {k: v / N for k, v in counts.items()},
           "fired_controls": fired}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_city_d_std_controls.json"),
                        "w", encoding="utf-8"), indent=2)
    print(f"\ncounts: {counts}")
    print("saved -> artifacts/results_city_d_std_controls.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
