"""
GraphRAGTool (T2): the TS-GraphRAG retrieval engine, demoted to ONE evidence
source. It proposes candidate leak partitions (and their representative leak
nodes) by matching the observation against the pre-simulated partition centroid
fingerprints. Its score is NOT trusted as a probability — the digital twin
falsifies and the Bayesian engine weighs it against non-leak hypotheses.

(This uses the numeric centroid/fingerprint channel of TS-GraphRAG, which is
self-contained. The full multi-channel LeakLocator is available in vendor/ for
later integration.)
"""
from __future__ import annotations
import json
from collections import defaultdict

import numpy as np

from .base import Tool


class GraphRAGTool(Tool):
    name = "graphrag_retrieve"
    cost = 0.0

    def __init__(self, fingerprints_file, sensors):
        d = json.load(open(fingerprints_file, encoding="utf-8"))
        self.order = [str(x) for x in d["sensor_nodes"]]
        by_p = defaultdict(list)
        nodes = defaultdict(lambda: defaultdict(float))
        for fp in d["fingerprints"]:
            p = int(fp["leak_partition"])
            by_p[p].append(np.asarray(fp["sensor_fingerprint"], dtype=float))
            # track candidate leak nodes by their typical drop magnitude
            nodes[p][str(fp["leak_node"])] += abs(float(fp.get("max_pressure_drop", 0.0)))
        self.centroids = {p: np.mean(np.stack(vs), axis=0) for p, vs in by_p.items()}
        self.cand_nodes = {
            p: [n for n, _ in sorted(v.items(), key=lambda kv: -kv[1])]
            for p, v in nodes.items()
        }

    def _vec(self, observation):
        return np.array([observation.get(s, 0.0) for s in self.order], dtype=float)

    def rank_partitions(self, observation, top_k=5):
        o = self._vec(observation)
        on = np.linalg.norm(o) + 1e-9
        scored = []
        for p, c in self.centroids.items():
            cn = np.linalg.norm(c) + 1e-9
            cos = float(np.dot(o, c) / (on * cn))
            euc = float(np.linalg.norm(o - c))
            euc_sim = 1.0 / (1.0 + euc / max(on, cn, 0.1))
            scored.append((int(p), 0.5 * max(cos, 0.0) + 0.5 * euc_sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def candidate_nodes(self, partition, max_nodes=2):
        return self.cand_nodes.get(int(partition), [])[:max_nodes]

    def run(self, ev, ctx, **kwargs):
        return {"ranked": self.rank_partitions(ev.observation, kwargs.get("top_k", 5))}
