"""
Stage 1 for the City H municipal-WDN leg: build the diagnostic context from the
NOMINAL model (dataset/city_h.inp) -- Leiden zones, a degree-based (observability-proxy) sensor
placement (the utility's real sensor layout is confidential), a twin leak-fingerprint
library, topology + partition adjacency -- under data/city_h/ + configs/city_h.yaml.

Real SCADA pressure for City H is confidential and unavailable; the register-anchored
SEMI-SYNTHETIC observations are built separately (data/city_h_anchor_corpus.py) and
the leak-RESPONSE cache by data/city_h_build_library.py.

Run:  python data/city_h_setup.py    (via the wds_rag interpreter)
"""
import os
import sys
import json
import pickle
import shutil
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import wntr
from anomaly_sim.base_twin import HydraulicTwin
from vendor.partition_utils import similarity, hydraulic
from vendor.partition_utils.partitioning_leiden import (run_leiden_partitioning,
                                                        extract_partitions_with_merge)

SRC_INP = r"C:\mutianwei\paper\dualLLMWDS\dataset\city_h.inp"
OUT = os.path.join(ROOT, "data", "city_h")
INP_VENDOR = os.path.join(OUT, "city_h.inp")
TARGET_K = 25
TARGET_SENSORS = 38          # ~1 sensor per 24 nodes, matching the L-Town benchmark density
LIB_RATES = (2.0, 5.0, 10.0, 20.0, 35.0, 50.0)
NODES_PER_ZONE = 2           # fingerprint-library breadth for GraphRAG retrieval


def main():
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(SRC_INP, INP_VENDOR)          # vendor the nominal inp (self-contained)
    wn = wntr.network.WaterNetworkModel(INP_VENDOR)
    junc = set(str(x) for x in wn.junction_name_list)
    print(f"city_h: {len(junc)} junctions, {len(wn.pipe_name_list)} pipes", flush=True)

    # ---- partition the nominal network ----------------------------------
    print("baseline hydraulic sim + Leiden partition ...", flush=True)
    res = hydraulic.run_hydraulic_simulation(wn)
    avgp = hydraulic.calculate_average_pressure(res, wn)
    G, pos = similarity.create_network_graph(wn, avgp)
    parts = run_leiden_partitioning(G, resolution_range=(0.001, 5.0), num_iterations=300)
    merged = extract_partitions_with_merge(G, parts, merge_range=(2, 40))
    K = sorted(merged.keys(), key=lambda k: abs(k - TARGET_K))[0]
    n2c = {str(n): int(c) for n, c in merged[K]["node_to_community"].items()}
    pickle.dump({K: {"node_to_community": n2c, "resolution": merged[K].get("resolution")}},
                open(os.path.join(OUT, "partitions.pkl"), "wb"))
    zones = sorted(set(n2c.values()))
    print(f"K={K} zones", flush=True)

    # ---- per-zone candidate leak nodes (top degree) ---------------------
    deg = defaultdict(int)
    for ln in wn.link_name_list:
        l = wn.get_link(ln)
        deg[str(l.start_node_name)] += 1
        deg[str(l.end_node_name)] += 1
    zone_nodes = defaultdict(list)
    for n, z in n2c.items():
        if n in junc:
            zone_nodes[z].append((n, deg.get(n, 0)))
    zone_top = {z: [n for n, _ in sorted(v, key=lambda x: -x[1])] for z, v in zone_nodes.items()}

    # ---- degree-based sensor placement (topological observability proxy) -
    # the utility's real sensor layout is confidential and the EXA7 sensitivity-
    # maximizing rule is NOT re-run here; place the highest-degree junction in each
    # topological zone, then fill by global degree to TARGET_SENSORS.
    chosen = []
    for z in zones:
        if zone_top.get(z):
            chosen.append(zone_top[z][0])
    for n in sorted(junc, key=lambda x: -deg.get(x, 0)):
        if len(chosen) >= TARGET_SENSORS:
            break
        if n not in chosen:
            chosen.append(n)
    sensors = chosen[:TARGET_SENSORS]
    print(f"{len(sensors)} sensors placed (>=1 per zone, filled by degree)", flush=True)

    # ---- twin leak-fingerprint library (GraphRAG retrieval) -------------
    twin = HydraulicTwin(INP_VENDOR, period_multipliers={"day_normal": 1.0}, leak_pattern=None)
    print("building leak-fingerprint library ...", flush=True)
    fps = []
    for z in zones:
        for node in zone_top[z][:NODES_PER_ZONE]:
            for r in LIB_RATES:
                dp = twin.leak_delta(node, r, "day_normal")
                if dp is None:
                    continue
                vec = [float(dp.get(s, 0.0)) for s in sensors]
                fps.append({"sensor_fingerprint": vec, "leak_partition": int(z),
                            "leak_node": str(node), "leak_rate_Ls": float(r),
                            "max_pressure_drop": float(min(vec) if vec else 0.0)})
        if (z + 1) % 5 == 0:
            print(f"   ...zone {z+1}/{len(zones)} ({len(fps)} fps)", flush=True)
    json.dump({"sensor_nodes": sensors, "num_sensors": len(sensors),
               "num_scenarios": len(fps), "fingerprints": fps},
              open(os.path.join(OUT, "sensor_fingerprints.json"), "w"))
    print(f"library: {len(fps)} fingerprints", flush=True)

    # ---- topology index + partition adjacency ---------------------------
    json.dump({"partition_k": K, "node_to_partition": n2c, "sensor_nodes": sensors},
              open(os.path.join(OUT, "topology_index.json"), "w"))
    adj = defaultdict(set)
    for ln in wn.pipe_name_list:
        l = wn.get_link(ln)
        a, b = n2c.get(str(l.start_node_name)), n2c.get(str(l.end_node_name))
        if a is not None and b is not None and a != b:
            adj[a].add(b); adj[b].add(a)
    json.dump({"adjacency": {str(k): sorted(v) for k, v in adj.items()}},
              open(os.path.join(OUT, "partition_adjacency.json"), "w"))

    # ---- config ---------------------------------------------------------
    cfg = f"""# City H municipal WDN (diagnosed against the NOMINAL dataset/city_h.inp).
# Real SCADA pressure is confidential/unavailable; observations are register-anchored
# SEMI-SYNTHETIC (data/city_h_anchor_corpus.py + Methods). Sensor placement is
# a degree-based topological observability proxy (real sensor layout is confidential).
network: city_h
inp_file: data/city_h/city_h.inp
partition_file: data/city_h/partitions.pkl
partition_k: {K}
sensor_fingerprints_file: data/city_h/sensor_fingerprints.json
partition_adjacency_file: data/city_h/partition_adjacency.json
demand_periods:
  day_normal: 1.0
leak_pattern: null
noise:
  sensor_std: 0.15
"""
    open(os.path.join(ROOT, "configs", "city_h.yaml"), "w", encoding="utf-8").write(cfg)
    print(f"\nsaved context to data/city_h/ + configs/city_h.yaml "
          f"(K={K}, {len(sensors)} sensors, {len(fps)} fps)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
