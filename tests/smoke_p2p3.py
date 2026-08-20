"""
P2+P3 smoke test: the full Executor<->Supervisor loop (deterministic core).

Asserts the accountable-diagnosis behaviour:
  A. a leak near a KB node  -> ACTED, accepted partition == true region
  B. a sensor-fault         -> ABSTAINED (top hypothesis = sensor_fault, low leak mass)
  C. a DMA demand anomaly   -> NOT a false leak dispatch (abstain or top != leak)
  D. a micro leak (2 L/s)   -> reported (often abstain: signal ~ noise floor)

Run:  python tests/smoke_p2p3.py   (via the wds_rag interpreter)
"""
import os
import sys
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
from run_diagnosis import diagnose
from anomaly_sim.leak_anomaly import make_leak_scenario
from anomaly_sim.sensor_fault import make_sensor_fault_scenario
from anomaly_sim.demand_anomaly import make_demand_anomaly_scenario

CFG = os.path.join(ROOT, "configs", "exa7.yaml")
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def build(ctx, contract, noise_std):
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt = DigitalTwinTool()
    rt = ResidualTool(sensor_std=noise_std, pass_threshold=contract.max_residual_mahalanobis)
    be = BayesianEngine()
    ex = ExecutorAgent(ctx, gr, dt, rt, be, top_k_leak=3, noise_std=noise_std)
    sv = SupervisorAgent(contract)
    return gr, ex, sv


def main():
    print("=" * 72)
    print("P2+P3 SMOKE — Executor<->Supervisor accountable-diagnosis loop (EXA7)")
    print("=" * 72)
    noise = 0.02
    ctx = build_context(CFG)
    contract = GoalContract(min_leak_probability=0.55, min_top1_margin=0.12,
                            max_candidate_region_nodes=60, alt_falsify_prob=0.20,
                            max_residual_mahalanobis=3.0)
    gr, ex, sv = build(ctx, contract, noise)
    period = "day_normal"
    rng = np.random.default_rng(7)

    # ---- A: leak near a KB representative node ----------------------------
    pid = ctx.partitions[0]
    leak_node = gr.candidate_nodes(pid, max_nodes=1)[0]
    sA = make_leak_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                            leak_node, 20.0, period, noise_std=noise, rng=rng)
    rA = diagnose(sA["observation"], ctx, ex, sv, period, case_id="A")
    print(f"\nA leak  node={leak_node} true_region={sA['true_region']} "
          f"max|dp|={sA['max_abs_dp']:.2f}")
    print(f"   -> {rA.outcome} acc_part={rA.accepted_partition} "
          f"existence={rA.evidence.existence_mass:.3f} margin={rA.evidence.margin:.3f} "
          f"top1={rA.evidence.top1.kind}:P{rA.evidence.top1.partition}")
    check("A leak -> ACTED", rA.acted)
    check("A leak -> correct partition", rA.accepted_partition == sA["true_region"],
          f"(got P{rA.accepted_partition}, true P{sA['true_region']})")

    # ---- B: sensor fault --------------------------------------------------
    sB = make_sensor_fault_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                    faulty_sensors=ctx.sensors[:2], mode="bias",
                                    magnitude=1.0, noise_std=noise, rng=rng)
    rB = diagnose(sB["observation"], ctx, ex, sv, period, case_id="B")
    print(f"\nB sensor-fault faulty={sB['faulty_sensors']}")
    print(f"   -> {rB.outcome} existence={rB.evidence.existence_mass:.3f} "
          f"top1={rB.evidence.top1.kind}")
    check("B sensor-fault -> ABSTAINED (no false dispatch)", not rB.acted)
    check("B sensor-fault -> top hypothesis is sensor_fault",
          rB.evidence.top1.kind == "sensor_fault", f"(top1={rB.evidence.top1.kind})")

    # ---- C: DMA demand anomaly -------------------------------------------
    sC = make_demand_anomaly_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                      pid, factor=1.8, period=period,
                                      noise_std=noise, rng=rng)
    rC = diagnose(sC["observation"], ctx, ex, sv, period, case_id="C")
    print(f"\nC demand-anomaly partition={pid} max|dp|={sC['max_abs_dp']:.2f}")
    print(f"   -> {rC.outcome} existence={rC.evidence.existence_mass:.3f} "
          f"top1={rC.evidence.top1.kind}")
    check("C demand-anomaly -> no false leak dispatch",
          (not rC.acted) or rC.evidence.top1.kind != "leak",
          f"(outcome={rC.outcome}, top1={rC.evidence.top1.kind})")

    # ---- D: micro leak (2 L/s) -------------------------------------------
    sD = make_leak_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                            leak_node, 2.0, period, noise_std=noise, rng=rng)
    rD = diagnose(sD["observation"], ctx, ex, sv, period, case_id="D")
    print(f"\nD micro-leak (2 L/s) node={leak_node} max|dp|={sD['max_abs_dp']:.3f}")
    print(f"   -> {rD.outcome} existence={rD.evidence.existence_mass:.3f} "
          f"margin={rD.evidence.margin:.3f}  (informational)")

    print("\n" + "=" * 72)
    ok = all(results)
    print(f"RESULT: {'ALL PASS' if ok else 'SOME FAILED'}  "
          f"({sum(results)}/{len(results)})  twin_sims_cached={len(ctx.twin._delta_cache)}")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
