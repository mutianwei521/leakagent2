# -*- coding: utf-8 -*-
"""
City D sensor placement v2: between-zone DISCRIMINABILITY objective.

Diagnosis: on the City D standard leg the twin-fit re-ranking (45.0%) adds
nothing over raw retrieval (44.7%), i.e. the failure is zone-signature
confusability, not detection. This optimizer therefore maximizes a direct
proxy of zone classification: greedy forward selection of 23 sensors that
maximizes nearest-(zone,rate)-centroid classification accuracy of the 2,250
cached library responses (375 nodes x 6 rates; signed dp signatures), averaged
over one clean and two sigma = 0.05 m noisy copies. Library-only objective:
no register or test information enters.

Outputs (side files; nothing overwritten until adoption is decided):
  data/city_d/sensor_fingerprints_v2.json
  configs/city_d_v2.yaml
  artifacts/results_city_d_placement_v2.json

Run:  <wds_rag python> eval/optimize_city_d_sensors_v2.py
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

N_SENSORS, NOISE, N_NOISY, SEED = 23, 0.05, 2, 7
DG = os.path.join(ROOT, "data", "city_d")


def detect_metrics(A_abs, idx):
    best = A_abs[:, idx].max(axis=1)
    return {"coverage_at_0.05m": float((best >= 0.05).mean()),
            "coverage_at_0.45m": float((best >= 0.45).mean()),
            "median_best_dp_m": float(np.median(best))}


def main():
    cache = pickle.load(open(os.path.join(DG, "leak_cache.pkl"), "rb"))
    parts = pickle.load(open(os.path.join(DG, "partitions.pkl"), "rb"))
    K = list(parts.keys())[0]
    n2z = {str(n): int(c) for n, c in parts[K]["node_to_community"].items()}
    fp_old = json.load(open(os.path.join(DG, "sensor_fingerprints.json"), encoding="utf-8"))
    old_sensors = [str(s) for s in fp_old["sensor_nodes"]]

    keys = sorted(cache.keys())
    nodes = sorted(next(iter(cache.values())).keys(), key=lambda x: (len(x), x))
    nix = {n: j for j, n in enumerate(nodes)}
    X = np.zeros((len(keys), len(nodes)))
    zone_of, rate_of = [], []
    for i, k in enumerate(keys):
        _, node, rate, _ = k.split("|")
        for n, dp in cache[k].items():
            X[i, nix[n]] = float(dp)          # signed
        zone_of.append(n2z[node])
        rate_of.append(float(rate))
    zone_of = np.array(zone_of)
    rate_of = np.array(rate_of)
    print(f"library: {X.shape[0]} responses x {X.shape[1]} candidate sensors")

    # class = (zone, rate); centroid matrix C [n_class, n_nodes]; class -> zone
    classes = sorted({(z, r) for z, r in zip(zone_of, rate_of)})
    cix = {c: i for i, c in enumerate(classes)}
    C = np.zeros((len(classes), len(nodes)))
    for c, i in cix.items():
        m = (zone_of == c[0]) & (rate_of == c[1])
        C[i] = X[m].mean(axis=0)
    class_zone = np.array([c[0] for c in classes])
    print(f"classes: {len(classes)} (zone,rate) centroids")

    rng = np.random.default_rng(SEED)
    Xd = [X] + [X + rng.normal(0, NOISE, X.shape) for _ in range(N_NOISY)]

    def acc_of(D):
        """D: list per draw of [entries, classes] partial sq-dists."""
        a = 0.0
        for d in D:
            pred = class_zone[np.argmin(d, axis=1)]
            a += float((pred == zone_of).mean())
        return a / len(D)

    chosen, D = [], [np.zeros((X.shape[0], len(classes))) for _ in Xd]
    for step in range(N_SENSORS):
        best_j, best_a, best_add = -1, -1.0, None
        for j in range(len(nodes)):
            if nodes[j] in chosen:
                continue
            adds = [(xd[:, j][:, None] - C[:, j][None, :]) ** 2 for xd in Xd]
            a = acc_of([d + ad for d, ad in zip(D, adds)])
            if a > best_a:
                best_j, best_a, best_add = j, a, adds
        chosen.append(nodes[best_j])
        D = [d + ad for d, ad in zip(D, best_add)]
        print(f"  {step+1:2d}/{N_SENSORS}: +{nodes[best_j]:>6s}  zone-classification proxy "
              f"{best_a*100:.1f}%", flush=True)

    # compare old vs new on the same objective
    def obj_of(sensor_list):
        idx = [nix[s] for s in sensor_list if s in nix]
        Do = [((xd[:, idx][:, :, None] - C[:, idx].T[None, :, :]) ** 2).sum(axis=1)
              for xd in Xd]
        return acc_of(Do)

    A_abs = np.abs(X)
    obj_old, obj_new = obj_of(old_sensors), obj_of(chosen)
    det_old = detect_metrics(A_abs, [nix[s] for s in old_sensors if s in nix])
    det_new = detect_metrics(A_abs, [nix[s] for s in chosen])
    print(f"\nzone-classification proxy: old {obj_old*100:.1f}%  ->  new {obj_new*100:.1f}%")
    print(f"detection coverage@0.45m : old {det_old['coverage_at_0.45m']*100:.1f}%  ->  "
          f"new {det_new['coverage_at_0.45m']*100:.1f}%")

    out = {"provenance": {"objective": "greedy nearest-(zone,rate)-centroid classification accuracy "
                                       f"over the cached library, averaged over 1 clean + {N_NOISY} "
                                       f"sigma={NOISE} noisy copies (seed {SEED}); library-only",
                          "n_sensors": N_SENSORS},
           "old_sensors": old_sensors, "new_sensors": chosen,
           "objective_old": obj_old, "objective_new": obj_new,
           "detection_old": det_old, "detection_new": det_new}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_city_d_placement_v2.json"),
                        "w", encoding="utf-8"), indent=2)

    # side fingerprints + side config (nothing overwritten)
    new_fps = []
    for e in fp_old["fingerprints"]:
        k = f"leak|{e['leak_node']}|{e['leak_rate_Ls']}|day_normal"
        vec = [float(cache[k][s]) for s in chosen]
        new_fps.append({"sensor_fingerprint": vec, "leak_partition": e["leak_partition"],
                        "leak_node": e["leak_node"], "leak_rate_Ls": e["leak_rate_Ls"],
                        "max_pressure_drop": float(min(vec))})
    json.dump({"sensor_nodes": chosen, "num_sensors": len(chosen),
               "num_scenarios": len(new_fps), "fingerprints": new_fps},
              open(os.path.join(DG, "sensor_fingerprints_v2.json"), "w", encoding="utf-8"))
    cfg = open(os.path.join(ROOT, "configs", "city_d.yaml"), encoding="utf-8").read()
    cfg = cfg.replace("sensor_fingerprints_file: data/city_d/sensor_fingerprints.json",
                      "sensor_fingerprints_file: data/city_d/sensor_fingerprints_v2.json")
    open(os.path.join(ROOT, "configs", "city_d_v2.yaml"), "w", encoding="utf-8").write(cfg)
    print("\nside files written: sensor_fingerprints_v2.json, configs/city_d_v2.yaml")
    print("saved -> artifacts/results_city_d_placement_v2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
