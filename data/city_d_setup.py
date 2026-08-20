# -*- coding: utf-8 -*-
"""
Stage 1 for the City D FIELD leg: build the diagnostic context from the utility's
own EPANET model (data/city_d/city_d.inp, vendored by data/city_d_extract_corpus.py)
-- Leiden zones, a degree-based (observability-proxy) sensor placement (the
utility's real sensor layout is not part of the deliverables), a twin
leak-fingerprint library, topology + partition adjacency -- under data/city_d/
+ configs/city_d.yaml. Mirrors data/city_h_setup.py: same placement rule, the
same ~1-sensor-per-24-nodes density and the same zone density (~37 nodes/zone),
so the field leg runs under the identical protocol.

Run:  python data/city_d_setup.py    (via the wds_rag interpreter)
"""
import os
import sys
import json
import pickle
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

OUT = os.path.join(ROOT, "data", "city_d")
INP_VENDOR = os.path.join(OUT, "city_d.inp")
TARGET_K = 15                # ~37 nodes/zone, the City H zone density (541/37)
TARGET_SENSORS = 23          # ~1 sensor per 24 nodes, the L-Town/City H density
LIB_RATES = (2.0, 5.0, 10.0, 20.0, 35.0, 50.0)
NODES_PER_ZONE = 2           # fingerprint-library breadth for GraphRAG retrieval


def main():
    wn = wntr.network.WaterNetworkModel(INP_VENDOR)
    junc = set(str(x) for x in wn.junction_name_list)
    print(f"city_d: {len(junc)} junctions, {len(wn.pipe_name_list)} pipes", flush=True)

    # ---- partition the utility network -----------------------------------
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

    # ---- per-zone candidate leak nodes (top degree) ----------------------
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

    # ---- degree-based sensor placement (topological observability proxy) --
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

    # ---- twin leak-fingerprint library (GraphRAG retrieval) --------------
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

    # ---- topology index + partition adjacency ----------------------------
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

    # ---- config -----------------------------------------------------------
    cfg = f"""# City D district WDN (FIELD leg: the utility's own EPANET model +
# the audited 2025 field leak register, data/city_d/leak_corpus.json).
# No SCADA pressures exist for these events: observations are twin-simulated
# with field noise. Sensor placement is a degree-based observability proxy.
network: city_d
inp_file: data/city_d/city_d.inp
partition_file: data/city_d/partitions.pkl
partition_k: {K}
sensor_fingerprints_file: data/city_d/sensor_fingerprints.json
partition_adjacency_file: data/city_d/partition_adjacency.json
demand_periods:
  day_normal: 1.0
leak_pattern: null
noise:
  sensor_std: 0.15
"""
    open(os.path.join(ROOT, "configs", "city_d.yaml"), "w", encoding="utf-8").write(cfg)
    print(f"\nsaved context to data/city_d/ + configs/city_d.yaml "
          f"(K={K}, {len(sensors)} sensors, {len(fps)} fps)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
