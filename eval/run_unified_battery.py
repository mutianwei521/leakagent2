"""
Unified-battery gap fill: the three cells missing from the cross-network
measurement matrix (main-text Table 1).

  1. EXA7 no-leak false-alarm control (50 pure-noise scenarios, same protocol
     as the City H/City D legs: N=50, noise-only observations, seed 7).
  2. KY4  no-leak false-alarm control (same).
  3. KY4  contract-vs-scalar-existence comparison at matched coverage,
     pooled over the five headline seeds.

Acceptance is the identical operating-point gate used by the headline runs:
gate_act(row, t_exist=0.5, min_margin=0.12, mahal<=gate, alt_max=0.20,
region<=60) with the per-seed maximum-coverage residual gate from the same
grid as run_experiments.op_point. Rows are regenerated with the same seeds,
so the acted sets are bit-identical to the committed headline artifacts.

Run:  <wds_rag python> eval/run_unified_battery.py
Output: artifacts/results_unified_battery.json
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
from tools.active_sensing import ActiveSensingTool
from agents.executor import ExecutorAgent
from eval import metrics as M
from eval.metrics import gate_act
from eval.run_experiments import generate, stat_row, PERIOD
from eval.run_experiments import run_once as run_once_exa7
from eval.run_ky4 import run_once as run_once_ky4
from eval.run_standard_net import run_once as run_once_std

NOISE, SEEDS, N_CONTROLS, CTRL_SEED = 0.05, [17, 42, 101, 2025, 31337], 50, 7
GRID = [round(x, 3) for x in np.linspace(0.3, 8.0, 40)]


def op_gate(rows):
    """The per-seed operating gate: the residual bound of the maximum-coverage
    point on the same grid as run_experiments.op_point."""
    curve = M.risk_coverage_mahal(rows, GRID, min_margin=0.12, alt_max=0.20)
    return max(curve, key=lambda c: c["coverage"])["max_mahal"]


def controls(ctx, gr, dt, asens, leak_cands, gate):
    """50 pure-noise observations through the identical executor + gate."""
    rt = ResidualTool(sensor_std=NOISE, pass_threshold=3.0)
    ex = ExecutorAgent(ctx, gr, dt, rt, BayesianEngine(), top_k_leak=len(ctx.partitions),
                       leak_cands=leak_cands, noise_std=NOISE)
    rng = np.random.RandomState(CTRL_SEED)
    dispatched = 0
    for j in range(N_CONTROLS):
        obs = {s: float(rng.normal(0.0, NOISE)) for s in ctx.sensors}
        ev = EvidencePackage(observation=obs, demand_period=PERIOD, network=ctx.name)
        ex.step(ev, 1)
        row = stat_row(ev, gr, {"true_class": "none", "true_region": -1})
        if gate_act(row, 0.5, 0.12, gate, 0.20, 60, mode="full"):
            dispatched += 1
    return dispatched


def main():
    print("=" * 70)
    print("UNIFIED BATTERY GAP FILL (EXA7 + KY4 controls; KY4 contract-vs-scalar)")
    print("=" * 70)
    out = {"provenance": {"noise_m": NOISE, "seeds": SEEDS, "n_controls": N_CONTROLS,
                          "control_seed": CTRL_SEED,
                          "gate": "per-seed max-coverage residual gate, t_exist=0.5, "
                                  "min_margin=0.12, alt_max=0.20, max_region=60 "
                                  "(identical to the headline operating point)"}}

    for net, cands, runner in (("exa7", 50, run_once_exa7), ("ky4", 25, run_once_ky4),
                           ("city_h", 25, run_once_std), ("city_d", 25, run_once_std)):
        print(f"\n--- {net} ---", flush=True)
        ctx = build_context(os.path.join(ROOT, "configs", f"{net}.yaml"))
        gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
        dt, asens = DigitalTwinTool(), ActiveSensingTool()

        pooled_acted, pooled_correct, pooled_scalar_correct, pooled_n = 0, 0, 0, 0
        gates = {}
        for sd in SEEDS:
            if runner is run_once_exa7:
                _, rows = runner(ctx, gr, dt, asens, NOISE, sd, active=True)
            else:
                _, rows = runner(ctx, gr, dt, asens, NOISE, sd)
            g = op_gate(rows)
            gates[sd] = g
            acted = [r for r in rows if gate_act(r, 0.5, 0.12, g, 0.20, 60, mode="full")]
            k = len(acted)
            correct = sum(1 for r in acted if M._is_correct(r))
            # scalar existence threshold at matched coverage: top-k by existence
            byex = sorted(rows, key=lambda r: r["existence"], reverse=True)[:k]
            sc = sum(1 for r in byex if M._is_correct(r))
            pooled_acted += k
            pooled_correct += correct
            pooled_scalar_correct += sc
            pooled_n += len(rows)
            print(f"  seed {sd}: gate {g:.2f}  acted {k}  contract {correct}/{k}  "
                  f"scalar {sc}/{k}", flush=True)

        gate42 = gates[42]
        fa = controls(ctx, gr, dt, asens, cands, gate42)
        print(f"  no-leak controls: {fa}/{N_CONTROLS} dispatched (gate {gate42:.2f})")

        out[net] = {
            "per_seed_gates": gates,
            "pooled": {"n_events": pooled_n, "n_acted": pooled_acted,
                       "coverage": pooled_acted / pooled_n,
                       "contract_precision": pooled_correct / pooled_acted if pooled_acted else None,
                       "scalar_existence_precision_matched": pooled_scalar_correct / pooled_acted if pooled_acted else None,
                       "contract_correct": pooled_correct,
                       "scalar_correct": pooled_scalar_correct},
            "no_leak_controls": {"n": N_CONTROLS, "dispatched": fa,
                                 "false_alarm_rate": fa / N_CONTROLS},
        }
        for f in ("temp.bin", "temp.inp", "temp.rpt"):
            try:
                os.remove(os.path.join(ROOT, f))
            except OSError:
                pass

    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_unified_battery.json"),
                        "w", encoding="utf-8"), indent=2)
    print("\nsaved -> artifacts/results_unified_battery.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
