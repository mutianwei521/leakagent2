# -*- coding: utf-8 -*-
"""Rebuild ONLY the KY4 twin leak-fingerprint library (retrieval library) with the
current leak model, keeping the committed partitions, sensor placement, topology
index and adjacency untouched.

Why this exists: KY4's INP declares a default demand pattern, and before commit
2be6d25 (2026-07-18) the twin let a "constant" leak inherit that pattern, so the
KY4 fingerprints and leak cache built on 2026-07-05 encode leaks scaled by the
diurnal multiplier (about two thirds of the constant response at the simulated
period). The Leiden zones and the degree-based sensors do not depend on leak
simulation and are reused as committed. The leak-response cache is rebuilt
separately by data/ky4_build_library.py.

Run:  <wds_rag python> data/ky4_rebuild_fingerprints.py
"""
import os
import sys
import json
import pickle
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import wntr
from data.ky4_setup import INP_VENDOR, OUT, NODES_PER_ZONE, LIB_RATES
from anomaly_sim.base_twin import HydraulicTwin


def main():
    parts = pickle.load(open(os.path.join(OUT, "partitions.pkl"), "rb"))
    K = list(parts.keys())[0]
    n2c = {str(n): int(c) for n, c in parts[K]["node_to_community"].items()}
    old = json.load(open(os.path.join(OUT, "sensor_fingerprints.json"), encoding="utf-8"))
    sensors = [str(s) for s in old["sensor_nodes"]]
    print(f"K={K} zones, {len(sensors)} sensors (reused as committed); old library: {old['num_scenarios']} fps")

    wn = wntr.network.WaterNetworkModel(INP_VENDOR)
    junc = set(wn.junction_name_list)
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
    zones = sorted(set(n2c.values()))

    twin = HydraulicTwin(INP_VENDOR, period_multipliers={"day_normal": 1.0}, leak_pattern=None)
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
    print(f"library rebuilt: {len(fps)} fingerprints (was {old['num_scenarios']})")
    # seed the leak cache with what was just simulated (same keys as leak_delta)
    twin.save_cache(os.path.join(OUT, "leak_cache.pkl"))
    print("cache seeded ->", os.path.join(OUT, "leak_cache.pkl"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
