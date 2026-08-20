"""
NetworkContext: bundles everything a tool/agent needs about a network
(the hydraulic twin, the sensor set, the node->partition map, adjacency).

Loads from a YAML config so EXA7 / City H / L-Town differ only by config.
All paths are read-only references to the existing repo data (no copies).
"""
from __future__ import annotations
import glob
import json
import os
import pickle
from dataclasses import dataclass, field

import yaml

from anomaly_sim.base_twin import HydraulicTwin, DEFAULT_PERIODS


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH_KEYS = ("inp_file", "partition_file", "sensor_fingerprints_file", "sensor_csv_glob",
              "sensor_summary_glob", "partition_adjacency_file", "kb_dir")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # resolve repo-relative paths against the project root (portable across machines)
    for k in _PATH_KEYS:
        v = cfg.get(k)
        if isinstance(v, str) and not os.path.isabs(v):
            cfg[k] = os.path.join(PROJECT_ROOT, *v.split("/"))
    return cfg


def _numeric_key(x: str):
    s = str(x)
    return (0, int(s)) if s.lstrip("-").isdigit() else (1, s)


@dataclass
class NetworkContext:
    name: str
    inp_file: str
    twin: HydraulicTwin
    sensors: list                       # ordered sensor node ids (str)
    node_to_partition: dict             # {node(str): partition(int)}
    partitions: list                    # sorted partition ids
    partition_adjacency: dict           # {pid: [neighbour pids]} (may be empty)
    demand_periods: dict
    noise_sensor_std: float
    partition_nodes_by_degree: dict = field(default_factory=dict)  # {pid: [nodes hi->lo degree]}
    cfg: dict = field(default_factory=dict)

    def nodes_in(self, partition: int) -> list:
        return [n for n, p in self.node_to_partition.items() if p == partition]

    def top_degree_nodes(self, partition: int, k: int) -> list:
        return self.partition_nodes_by_degree.get(int(partition), [])[:k]


def _load_sensors(cfg: dict) -> list:
    sensors: list = []
    # 1) AUTHORITATIVE: sensor_fingerprints.json 'sensor_nodes' = exactly the Δp keys
    fp = cfg.get("sensor_fingerprints_file")
    if fp and os.path.exists(fp):
        try:
            d = json.load(open(fp, encoding="utf-8"))
            sensors = [str(x) for x in d.get("sensor_nodes", [])]
        except Exception:
            sensors = []
    # 2) sensor placement CSV — must use the 'Node_Name' (real junction) column,
    #    NOT 'Sensor_ID' (which holds labels like S001).
    if not sensors:
        g = cfg.get("sensor_csv_glob")
        if g:
            files = sorted(glob.glob(g))
            if files:
                try:
                    import pandas as pd
                    df = pd.read_csv(files[-1])
                    for cand in ("Node_Name", "Node", "node_name", "node"):
                        if cand in df.columns:
                            sensors = [str(x) for x in df[cand].tolist()]
                            break
                except Exception:
                    sensors = []
    # 3) fallback: sensor_summary partition_details
    if not sensors:
        g2 = cfg.get("sensor_summary_glob")
        if g2:
            files = sorted(glob.glob(g2))
            if files:
                d = json.load(open(files[-1], encoding="utf-8"))
                det = d.get("partition_details", {})
                vals = det.values() if isinstance(det, dict) else det
                for v in vals:
                    if isinstance(v, dict):
                        sensors += [str(x) for x in v.get("sensor_nodes", v.get("sensors", []))]
    seen, out = set(), []
    for s in sensors:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out, key=_numeric_key)


def _load_partition_adjacency(cfg: dict) -> dict:
    p = cfg.get("partition_adjacency_file")
    if p and os.path.exists(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            adj = d.get("adjacency", d)
            return {int(k): [int(x) for x in v] for k, v in adj.items()} if isinstance(adj, dict) else {}
        except Exception:
            return {}
    return {}


def build_context(config_path: str) -> NetworkContext:
    cfg = load_config(config_path)
    periods = cfg.get("demand_periods", dict(DEFAULT_PERIODS))
    twin = HydraulicTwin(cfg["inp_file"], period_multipliers=periods,
                         leak_pattern=cfg.get("leak_pattern"))

    with open(cfg["partition_file"], "rb") as f:
        parts = pickle.load(f)
    n2c_raw = parts[cfg["partition_k"]]["node_to_community"]
    n2c = {str(k): int(v) for k, v in n2c_raw.items()}
    partitions = sorted(set(n2c.values()))

    sensors = _load_sensors(cfg)
    adjacency = _load_partition_adjacency(cfg)

    # precompute node degree (one link pass) -> per-partition candidate nodes
    from collections import defaultdict
    deg = defaultdict(int)
    for ln in twin.wn.link_name_list:
        l = twin.wn.get_link(ln)
        deg[str(l.start_node_name)] += 1
        deg[str(l.end_node_name)] += 1
    junset = set(str(x) for x in twin.wn.junction_name_list)
    pnbd = defaultdict(list)
    for node, pid in n2c.items():
        if node in junset:                      # only junctions can host a leak
            pnbd[pid].append((node, deg.get(node, 0)))
    partition_nodes_by_degree = {
        pid: [n for n, _ in sorted(v, key=lambda x: -x[1])] for pid, v in pnbd.items()
    }

    return NetworkContext(
        name=cfg.get("network", "net"),
        inp_file=cfg["inp_file"],
        twin=twin,
        sensors=sensors,
        node_to_partition=n2c,
        partitions=partitions,
        partition_adjacency=adjacency,
        demand_periods=periods,
        noise_sensor_std=float(cfg.get("noise", {}).get("sensor_std", 0.02)),
        partition_nodes_by_degree=partition_nodes_by_degree,
        cfg=cfg,
    )
