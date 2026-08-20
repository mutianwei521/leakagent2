# -*- coding: utf-8 -*-
"""
City D no-leak controls under three acceptance presets (authoritative path).

Replicates the register run's control loop EXACTLY (same construction order,
twin leak-cache loaded, seed 7, sigma = 0.15 m, diagnose() through the real
SupervisorAgent), so preset A reproduces the committed 1/50 bit-for-bit. The
stricter presets are then evaluated on the SAME final evidence packages via
evaluate_contract, so all three numbers describe one set of runs:

  A transfer preset (as run):  exist>=0.5, margin>=0.10, alt<=0.30, region<=80
  B audit-grade preset:        exist>=0.7, margin>=0.15, alt<=0.20, region<=80
  C preset A + the paper's ~3-sigma observation floor (max|dp_obs| >= 0.45 m)
    as a hard predicate.

Register events: preset A dispatches 5/194 (the committed headline); B and C
both yield zero register dispatches (every acted existence value is below 0.7,
and no register observation reaches the 0.45 m floor). The fired controls'
evidence is recorded, since they sit at the existence knife-edge (~0.51 vs the
0.50 threshold).

Run:  <wds_rag python> eval/run_city_d_controls.py
Output: artifacts/results_city_d_controls.json
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
from tools.graphrag_tool import GraphRAGTool
from tools.digital_twin import DigitalTwinTool
from tools.residual_analysis import ResidualTool
from tools.bayesian import BayesianEngine
from agents.executor import ExecutorAgent
from agents.supervisor import SupervisorAgent
from agents.goal_contract import evaluate_contract
from run_diagnosis import diagnose

SENSOR_STD, N, SEED, FLOOR, LEAK_CANDS = 0.15, 50, 7, 0.45, 25


def main():
    print("=" * 70)
    print("CITY_D NO-LEAK CONTROLS x THREE PRESETS (authoritative supervisor path)")
    print("=" * 70)
    ctx = build_context(os.path.join(ROOT, "configs", "city_d.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    from eval.run_city_d import RATE_GRID     # same register-leg hypothesis grid
    ex = ExecutorAgent(ctx, gr, DigitalTwinTool(),
                       ResidualTool(sensor_std=SENSOR_STD, pass_threshold=3.0), BayesianEngine(),
                       top_k_leak=len(ctx.partitions), leak_cands=LEAK_CANDS,
                       rate_grid=RATE_GRID, noise_std=SENSOR_STD)
    sv_a = SupervisorAgent(GoalContract(min_leak_probability=0.5, min_top1_margin=0.10,
                                        max_candidate_region_nodes=80, alt_falsify_prob=0.30,
                                        max_residual_mahalanobis=3.0))
    contract_b = GoalContract(min_leak_probability=0.7, min_top1_margin=0.15,
                              max_candidate_region_nodes=80, alt_falsify_prob=0.20,
                              max_residual_mahalanobis=3.0)
    nload = ctx.twin.load_cache(os.path.join(ROOT, "data", "city_d", "leak_cache.pkl"))
    print(f"loaded twin leak library: {nload} cached sims.", flush=True)

    rng = np.random.RandomState(SEED)
    counts = {"A_transfer": 0, "B_audit": 0, "C_floor": 0}
    fired = []
    for j in range(N):
        obs = {s: float(rng.normal(0.0, SENSOR_STD)) for s in ctx.sensors}
        maxdp = max(abs(v) for v in obs.values())
        res = diagnose(obs, ctx, ex, sv_a, "day_normal", case_id=f"noise{j}")
        ev = res.evidence
        a = bool(res.acted)
        b = (evaluate_contract(ev, contract_b).decision == "ACCEPT")
        c = a and (maxdp >= FLOOR)
        counts["A_transfer"] += a
        counts["B_audit"] += b
        counts["C_floor"] += c
        if a or b:
            t1 = ev.top1
            rr = (t1.params.get("residual") if t1 else {}) or {}
            fired.append({"control": j, "max_abs_dp_obs_m": round(maxdp, 4),
                          "max_obs_in_sigma": round(maxdp / SENSOR_STD, 2),
                          "existence": round(float(ev.existence_mass), 4),
                          "margin": round(float(ev.margin), 4),
                          "mahal": round(float(rr.get("mahalanobis_per_dof", 999.0)), 3),
                          "passes": {"A": a, "B": b, "C": c}})
            print(f"  FIRED ctrl {j}: {fired[-1]}", flush=True)
        if (j + 1) % 10 == 0:
            print(f"  ...{j+1}/{N}", flush=True)

    out = {"provenance": {"network": "city_d (optimized placement)", "sigma_m": SENSOR_STD,
                          "n_controls": N, "seed": SEED,
                          "path": "identical to eval/run_city_d.py controls (diagnose + "
                                  "SupervisorAgent, twin cache loaded); presets B/C evaluated "
                                  "on the same final evidence packages",
                          "presets": {"A_transfer": "exist 0.5 / margin 0.10 / alt 0.30 / region 80 / mahal 3.0",
                                      "B_audit": "exist 0.7 / margin 0.15 / alt 0.20 / region 80 / mahal 3.0",
                                      "C_floor": "preset A plus hard observation floor max|dp_obs| >= 0.45 m"},
                          "register_note": "preset A dispatches 5/194 on the register (the committed "
                                           "headline); B and C yield zero register dispatches (every "
                                           "acted existence value is below 0.7, and no register "
                                           "observation reaches the 0.45 m floor)"},
           "false_dispatches": counts,
           "false_alarm_rates": {k: v / N for k, v in counts.items()},
           "fired_controls": fired}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_city_d_controls.json"),
                        "w", encoding="utf-8"), indent=2)
    print(f"\ncounts: {counts}")
    print("saved -> artifacts/results_city_d_controls.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
