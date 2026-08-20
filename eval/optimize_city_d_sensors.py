# -*- coding: utf-8 -*-
"""
Dedicated City D optimization program: sensitivity-maximizing sensor placement.

The original City D leg placed 23 sensors by a node-degree heuristic (a
topological observability proxy). Sensor placement is a simulation assumption
of this leg (the utility's real layout is unknown), so it is a legitimate
design degree of freedom. This program upgrades it to a greedy
detection-coverage placement computed ONLY from the pre-simulated leak-response
library (25 candidate nodes per zone x 6 rates = 2,250 cached responses):
no register event, register node or test observation enters the objective,
so there is no test leakage.

Objective: greedily pick the sensor that maximises the number of library
scenarios whose best-sensor |dp| clears theta = 0.05 m (tie-broken by summed
clipped |dp|). The same number of sensors (23) is kept for parity with the
original configuration.

Outputs:
  artifacts/results_city_d_placement.json  (old vs new placement metrics)
  data/city_d/sensor_fingerprints.json     (rebuilt for the new sensor set,
                                            same scenario list re-projected)

Run:  <wds_rag python> eval/optimize_city_d_sensors.py
"""
import os
import sys
import json
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_SENSORS = 23
THETA = 0.05          # detection threshold for the placement objective (m)
CLIP = 0.5            # tie-break signal clip (m)

DG = os.path.join(ROOT, "data", "city_d")


def metrics(A, idx):
    """Library detectability of a sensor set (columns idx of |dp| matrix A)."""
    best = A[:, idx].max(axis=1)
    return {"coverage_at_0.05m": float((best >= 0.05).mean()),
            "coverage_at_0.15m": float((best >= 0.15).mean()),
            "coverage_at_0.45m": float((best >= 0.45).mean()),
            "median_best_dp_m": float(np.median(best)),
            "mean_best_dp_m": float(best.mean())}


def main():
    cache = pickle.load(open(os.path.join(DG, "leak_cache.pkl"), "rb"))
    fp = json.load(open(os.path.join(DG, "sensor_fingerprints.json"), encoding="utf-8"))
    old_sensors = [str(s) for s in fp["sensor_nodes"]]

    keys = sorted(cache.keys())
    nodes = sorted(next(iter(cache.values())).keys(), key=lambda x: (len(x), x))
    node_ix = {n: j for j, n in enumerate(nodes)}
    A = np.zeros((len(keys), len(nodes)))
    for i, k in enumerate(keys):
        v = cache[k]
        for n, dp in v.items():
            A[i, node_ix[n]] = abs(float(dp))
    print(f"library matrix: {A.shape[0]} scenarios x {A.shape[1]} candidate nodes")

    # greedy detection-coverage placement
    chosen = []
    covered = np.zeros(len(keys), dtype=bool)
    for step in range(N_SENSORS):
        det = (A >= THETA)
        gain = (det & ~covered[:, None]).sum(axis=0).astype(float)
        # tie-break: summed clipped signal on not-yet-covered scenarios
        tie = (np.minimum(A, CLIP) * (~covered[:, None])).sum(axis=0)
        gain += 1e-6 * tie / (tie.max() + 1e-12)
        for j in [node_ix[c] for c in chosen]:
            gain[j] = -1
        j = int(gain.argmax())
        chosen.append(nodes[j])
        covered |= det[:, j]
        if (step + 1) % 5 == 0 or step == N_SENSORS - 1:
            print(f"  {step+1:2d} sensors: library coverage@{THETA} = {covered.mean()*100:.1f}%")

    old_idx = [node_ix[s] for s in old_sensors if s in node_ix]
    new_idx = [node_ix[s] for s in chosen]
    m_old, m_new = metrics(A, old_idx), metrics(A, new_idx)
    print("\nold (degree-heuristic):", {k: round(v, 4) for k, v in m_old.items()})
    print("new (greedy coverage) :", {k: round(v, 4) for k, v in m_new.items()})

    out = {"provenance": {"objective": f"greedy library detection coverage at theta={THETA} m, "
                                       f"tie-break summed clipped |dp| (clip {CLIP} m)",
                          "library": "leak_cache.pkl (25 cands/zone x 6 rates; NO register/test info)",
                          "n_sensors": N_SENSORS},
           "old_sensors": old_sensors, "new_sensors": chosen,
           "library_metrics_old": m_old, "library_metrics_new": m_new}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_city_d_placement.json"),
                        "w", encoding="utf-8"), indent=2)

    # rebuild fingerprints for the new sensor set (same scenario list, re-projected)
    new_fps = []
    for e in fp["fingerprints"]:
        k = f"leak|{e['leak_node']}|{e['leak_rate_Ls']}|day_normal"
        v = cache[k]
        vec = [float(v[s]) for s in chosen]
        new_fps.append({"sensor_fingerprint": vec,
                        "leak_partition": e["leak_partition"],
                        "leak_node": e["leak_node"],
                        "leak_rate_Ls": e["leak_rate_Ls"],
                        "max_pressure_drop": float(min(vec))})
    # NEVER overwrite sensor_fingerprints.json in place: doing so once silently
    # desynchronized the live sensor set from scenario_corpus.json and the
    # register leg evaluated all-zero observations. Write a versioned file and
    # switch configs/city_d.yaml explicitly, then REGENERATE the corpus.
    json.dump({"sensor_nodes": chosen, "num_sensors": len(chosen),
               "num_scenarios": len(new_fps), "fingerprints": new_fps},
              open(os.path.join(DG, "sensor_fingerprints_v1.json"), "w", encoding="utf-8"))
    print(f"\nwrote data/city_d/sensor_fingerprints_v1.json for {len(chosen)} optimized sensors")
    print("to adopt: point configs/city_d.yaml at the versioned file AND rebuild scenario_corpus.json")
    print("saved -> artifacts/results_city_d_placement.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
