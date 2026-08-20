"""
Wall-clock runtime profile of a complete diagnostic episode (EXA7, sigma = 0.05).

Times the exact episode structure of the headline evaluation (run_experiments.
run_once): executor step (hypothesis instantiation + twin falsification +
Bayesian fusion) and, separately, the supervisor-directed active-sensing round.
One-time setup (context build, tool construction, response-library warm-up on
the first episode) is reported apart from the steady-state per-episode times, so
the numbers reflect what a deployed, cache-warm system would see. Hardware is
recorded alongside; all times are machine-dependent and are reported as such.

Run:  python eval/runtime_profile.py   (via the wds_rag interpreter)
Output: artifacts/results_runtime.json
"""
import os
import sys
import json
import time
import platform
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
from eval.run_experiments import generate, PERIOD

NOISE, SEED = 0.05, 42


def main():
    print("=" * 70)
    print("RUNTIME PROFILE (EXA7 headline episode structure, sigma=0.05)")
    print("=" * 70)
    t0 = time.perf_counter()
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt, asens = DigitalTwinTool(), ActiveSensingTool()
    rt = ResidualTool(sensor_std=NOISE, pass_threshold=3.0)
    ex = ExecutorAgent(ctx, gr, dt, rt, BayesianEngine(), top_k_leak=len(ctx.partitions),
                       leak_cands=50, noise_std=NOISE)
    setup_s = time.perf_counter() - t0

    rng = np.random.default_rng(SEED)
    arng = np.random.default_rng(SEED + 1)
    scens = generate(ctx, gr, rng, NOISE)

    step_t, active_t = [], []
    for s in scens:
        ev = EvidencePackage(observation=s["observation"], demand_period=PERIOD, network=ctx.name)
        t1 = time.perf_counter()
        ex.step(ev, 1)
        t2 = time.perf_counter()
        ex.active_sense_verify(ev, s.get("full_dp", {}), PERIOD, arng, asens, n_extra=3, top_k=6)
        t3 = time.perf_counter()
        step_t.append(t2 - t1)
        active_t.append(t3 - t2)

    step = np.array(step_t)
    act = np.array(active_t)
    # the first episode absorbs any remaining lazy cache warm-up: report it apart
    warm_step, warm_act = step[1:], act[1:]

    def q(a):
        return {"median_s": float(np.median(a)), "mean_s": float(a.mean()),
                "p90_s": float(np.quantile(a, 0.90)), "max_s": float(a.max())}

    hw = {"machine": platform.machine(), "processor": platform.processor(),
          "cpu_count": os.cpu_count(), "platform": platform.platform(),
          "python": platform.python_version()}

    out = {"provenance": {"network": "exa7", "noise_m": NOISE, "seed": SEED,
                          "n_episodes": len(scens),
                          "protocol": "run_experiments.run_once episode structure; "
                                      "steady-state stats exclude the first (cache-warming) episode"},
           "setup_s": setup_s,
           "first_episode_step_s": float(step[0]),
           "executor_step": q(warm_step),
           "active_sensing_round": q(warm_act),
           "full_episode": q(warm_step + warm_act),
           "hardware": hw}
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_runtime.json"), "w",
                        encoding="utf-8"), indent=2)

    print(f"setup (context + tools + executor): {setup_s:.1f} s")
    print(f"first episode (cache warm-up)     : {step[0]:.2f} s")
    print(f"executor step   median {np.median(warm_step):.3f} s  p90 {np.quantile(warm_step,0.9):.3f} s")
    print(f"active sensing  median {np.median(warm_act):.3f} s  p90 {np.quantile(warm_act,0.9):.3f} s")
    print(f"full episode    median {np.median(warm_step+warm_act):.3f} s")
    print(f"hardware: {hw['processor']} ({hw['cpu_count']} logical cores)")
    print("\nsaved -> artifacts/results_runtime.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
