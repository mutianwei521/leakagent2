"""
Stage 1 for the L-Town / BattLeDIM real-data leg: build the diagnostic context
from the NOMINAL model (L-TOWN.inp) — Leiden zones, the 33 real pressure sensors,
a twin leak-fingerprint library, topology and partition-adjacency — and write
them under data/ltown/ + a config. The real SCADA/labels are used later (Stage 2).

Run:  python data/ltown_setup.py     (via the wds_rag interpreter)
"""
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np

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

LT_DIR = os.environ.get("LTOWN_SCADA_DIR", r"data/ltown/scada")
INP = os.path.join(LT_DIR, "L-TOWN.inp")
OUT = os.path.join(ROOT, "data", "ltown")
TARGET_K = 25
LIB_RATES = (5.0, 10.0, 20.0, 30.0)
NODES_PER_ZONE = 3


def sensors_from_csv():
    header = open(os.path.join(LT_DIR, "2018_SCADA_Pressures.csv"), encoding="utf-8").readline().strip()
    cols = header.split(";")
    return [c for c in cols[1:]]      # drop 'Timestamp'


def main():
    os.makedirs(OUT, exist_ok=True)
    sensors = sensors_from_csv()
    print(f"33-sensor set: {len(sensors)} pressure sensors, sample {sensors[:5]}")

    wn = wntr.network.WaterNetworkModel(INP)
    junc = set(wn.junction_name_list)
    sensors = [s for s in sensors if s in junc]
    print(f"sensors that are junctions: {len(sensors)}")

    # ---- partition the nominal network ----------------------------------
    print("running baseline hydraulic sim + Leiden partition ...")
    res = hydraulic.run_hydraulic_simulation(wn)
    avgp = hydraulic.calculate_average_pressure(res, wn)
    G, pos = similarity.create_network_graph(wn, avgp)
    parts = run_leiden_partitioning(G, resolution_range=(0.001, 5.0), num_iterations=300)
    merged = extract_partitions_with_merge(G, parts, merge_range=(2, 40))   # -> {k: {...}}
    # pick the available K closest to TARGET_K
    ks = sorted(merged.keys(), key=lambda k: abs(k - TARGET_K))
    K = ks[0]
    n2c = {str(n): int(c) for n, c in merged[K]["node_to_community"].items()}
    parts_pkl = {K: {"node_to_community": n2c, "resolution": merged[K].get("resolution")}}
    pickle.dump(parts_pkl, open(os.path.join(OUT, "partitions.pkl"), "wb"))
    zones = sorted(set(n2c.values()))
    print(f"partitioned into K={K} zones")

    # ---- per-zone candidate leak nodes (top degree) ---------------------
    from collections import defaultdict
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

    # ---- twin leak-fingerprint library ----------------------------------
    twin = HydraulicTwin(INP, period_multipliers={"day_normal": 1.0}, leak_pattern=None)
    print("building leak-fingerprint library (constant-demand leaks) ...")
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
            print(f"   ...zone {z+1}/{len(zones)} ({len(fps)} fingerprints, cache={len(twin._delta_cache)})")
    json.dump({"sensor_nodes": sensors, "num_sensors": len(sensors), "num_scenarios": len(fps),
               "fingerprints": fps}, open(os.path.join(OUT, "sensor_fingerprints.json"), "w"))
    print(f"library: {len(fps)} fingerprints")

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
    cfg = f"""# L-Town / BattLeDIM real benchmark (diagnosed against the NOMINAL model).
network: ltown
inp_file: data/ltown/L-TOWN.inp
partition_file: data/ltown/partitions.pkl
partition_k: {K}
sensor_fingerprints_file: data/ltown/sensor_fingerprints.json
partition_adjacency_file: data/ltown/partition_adjacency.json
demand_periods:
  day_normal: 1.0
leak_pattern: null
noise:
  sensor_std: 0.05
"""
    open(os.path.join(ROOT, "configs", "ltown.yaml"), "w", encoding="utf-8").write(cfg)
    # vendor the nominal inp (small) so the project stays self-contained
    import shutil
    shutil.copy(INP, os.path.join(OUT, "L-TOWN.inp"))
    print(f"\nsaved context to data/ltown/ + configs/ltown.yaml  (K={K}, {len(sensors)} sensors, {len(fps)} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
