"""
P1 smoke test: validate the physics inner loop BEFORE any agent exists.

Asserts:
  (1) LEAK: predicting a leak at the TRUE node reproduces the observation
      (residual ~ noise), while a wrong-partition node does not (large residual).
  (2) SENSOR-FAULT: no leak hypothesis can reproduce a sensor-fault observation
      (the best-fit leak residual stays large) -> physically un-leak-like.
  (3) VALVE / DEMAND: the confounder simulators run and produce structured Δp.

Run:  conda run -n wds_rag python tests/smoke_p1.py
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:                                     # Windows consoles default to GBK
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context
from anomaly_sim.leak_anomaly import make_leak_scenario
from anomaly_sim.demand_anomaly import make_demand_anomaly_scenario
from anomaly_sim.sensor_fault import make_sensor_fault_scenario
from anomaly_sim.valve_misstate import make_valve_misstate_scenario
from tools.digital_twin import DigitalTwinTool
from tools.residual_analysis import ResidualTool

CFG = os.path.join(ROOT, "configs", "exa7.yaml")
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    results.append(cond)
    print(f"  [{tag}] {name}  {detail}")


def main():
    print("=" * 70)
    print("P1 SMOKE TEST — physics inner loop (EXA7)")
    print("=" * 70)
    ctx = build_context(CFG)
    print(f"network={ctx.name}  junctions={len(ctx.twin.junctions)}  "
          f"sensors={len(ctx.sensors)}  partitions={len(ctx.partitions)}")
    assert ctx.sensors, "no sensors loaded"
    rng = np.random.default_rng(42)
    dt = DigitalTwinTool()
    res = ResidualTool(sensor_std=0.02, pass_threshold=3.0)
    period = "day_normal"

    # representative node per partition (keeps the sim count small)
    rep = {}
    for n, p in ctx.node_to_partition.items():
        rep.setdefault(p, n)

    # pick a leak node with a usable signal
    leak_node, leak_pid, scen = None, None, None
    for p, n in rep.items():
        s = make_leak_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors, n,
                               20.0, period, noise_std=0.02, rng=rng)
        if s and s["max_abs_dp"] > 0.3:
            leak_node, leak_pid, scen = n, p, s
            break
    assert scen is not None, "could not find a leak node with signal"
    obs = scen["observation"]
    print(f"\nLeak scenario: node={leak_node} partition={leak_pid} "
          f"rate=20 L/s  max|Δp|={scen['max_abs_dp']:.3f} m")

    # (1a) true-node prediction -> small residual, passes
    pt = dt.predict_leak(ctx, leak_node, period, observation=obs)
    rt = res.analyze(obs, pt["predicted_dp"], ctx.sensors)
    print(f"  true-node : best_rate={pt['rate']} rmse={rt['rmse']:.4f} "
          f"mahal/dof={rt['mahalanobis_per_dof']:.2f} passed={rt['passed']}")

    # (1b) wrong-node (different partition) -> large residual, fails
    wrong = next(n for p, n in rep.items() if p != leak_pid)
    pw = dt.predict_leak(ctx, wrong, period, observation=obs)
    rw = res.analyze(obs, pw["predicted_dp"], ctx.sensors)
    print(f"  wrong-node: rmse={rw['rmse']:.4f} "
          f"mahal/dof={rw['mahalanobis_per_dof']:.2f} passed={rw['passed']}")

    check("leak: true-node residual < wrong-node residual",
          rt["rmse"] < rw["rmse"], f"({rt['rmse']:.4f} < {rw['rmse']:.4f})")
    check("leak: true-node passes residual gate", rt["passed"] is True)
    check("leak: wrong-node fails residual gate", rw["passed"] is False)

    # (2) sensor-fault: no leak hypothesis should reproduce it
    sf = make_sensor_fault_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                    faulty_sensors=ctx.sensors[:2], mode="bias",
                                    magnitude=1.0, noise_std=0.02, rng=rng)
    best_leak_rmse = min(
        res.analyze(sf["observation"],
                    dt.predict_leak(ctx, n, period, observation=sf["observation"])["predicted_dp"],
                    ctx.sensors)["rmse"]
        for n in rep.values()
    )
    print(f"\nSensor-fault scenario: faulty={sf['faulty_sensors']} mode=bias")
    print(f"  best-fit leak rmse over {len(rep)} candidates = {best_leak_rmse:.4f}")
    check("sensor-fault: no leak reproduces it (>> a true-leak fit)",
          best_leak_rmse > max(0.1, 5.0 * rt["rmse"]),
          f"(best leak rmse={best_leak_rmse:.4f} vs true-leak {rt['rmse']:.4f})")

    # (3) demand + valve confounders simulate and produce structured signal
    da = make_demand_anomaly_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                      leak_pid, factor=1.8, period=period,
                                      noise_std=0.02, rng=rng)
    check("demand-anomaly simulator runs", da is not None and da["max_abs_dp"] > 0.0,
          f"(max|dp|={da['max_abs_dp']:.3f} m)" if da else "")

    vm, vpipe = None, None
    pipes = ctx.twin.wn.pipe_name_list
    for pipe in pipes[::max(1, len(pipes) // 40)]:
        cand = make_valve_misstate_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                            pipe, period=period, noise_std=0.02, rng=rng)
        if cand is not None and cand["max_abs_dp"] < 200.0:
            vm, vpipe = cand, pipe
            break
    check("valve-misstate produces a sane (non-degenerate) scenario", vm is not None,
          (f"(pipe={vpipe}, max|dp|={vm['max_abs_dp']:.3f} m)" if vm
           else "(all sampled closures disconnect the network)"))

    print("\n" + "=" * 70)
    ok = all(results)
    print(f"RESULT: {'ALL PASS' if ok else 'SOME FAILED'}  "
          f"({sum(results)}/{len(results)})  twin_sim_failures={ctx.twin.sim_failures}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
