"""
city_h_locator.py
===================
Lightweight TS-GraphRAG locator for City H evaluation.
Uses 6 channels (1-4, 6-7) directly from KB files; skips Channel 5
(semantic LanceDB) since vector store was not built.
Drop-in replacement for LeakLocator in eval_city_h_*.py.
"""
import os, json, math
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


class CityHLocator:
    """6-channel TS-GraphRAG locator (no LanceDB / no sensor_results dir)."""

    def __init__(self, kb_dir):
        self.kb_dir = kb_dir
        print(f"CityHLocator: loading KB from {kb_dir}...")

        self._load_topology()
        self._load_fingerprints()
        self._build_centroids()
        self._load_partition_adjacency()
        self._load_community_summaries()
        print("CityHLocator ready.\n")

    # ── Loaders ───────────────────────────────────────────────────────────────
    def _load_topology(self):
        p = os.path.join(self.kb_dir, 'topology_index.json')
        with open(p) as f:
            topo = json.load(f)
        self.node_to_partition = {}
        self.sensor_nodes = []
        self.partition_sensors = defaultdict(list)
        self.sensor_to_partition = {}
        for pid_str, pdata in topo['partitions'].items():
            pid = int(pid_str)
            for node in pdata.get('sensor_nodes', pdata.get('nodes', [])):
                if node not in self.sensor_nodes:
                    self.sensor_nodes.append(node)
                self.partition_sensors[pid].append(node)
                self.sensor_to_partition[node] = pid
            for node in pdata.get('nodes', []):
                self.node_to_partition[node] = pid
        self.sensor_nodes = sorted(set(self.sensor_nodes),
                                   key=lambda x: int(x) if str(x).isdigit() else x)
        print(f"  Topology: {len(self.node_to_partition)} nodes, "
              f"{len(self.partition_sensors)} partitions, "
              f"{len(self.sensor_nodes)} sensors")

    def _load_fingerprints(self):
        p = os.path.join(self.kb_dir, 'sensor_fingerprints.json')
        with open(p) as f:
            fp_data = json.load(f)
        self.fingerprints    = fp_data['fingerprints']
        self.fp_sensor_nodes = fp_data['sensor_nodes']
        self.fp_by_period    = defaultdict(list)
        for fp in self.fingerprints:
            self.fp_by_period[fp['demand_period']].append(fp)
        print(f"  Fingerprints: {len(self.fingerprints)} scenarios, "
              f"{fp_data['num_sensors']}-dim")
        # Rebuild sensor_nodes from fingerprints if topology missed them
        if not self.sensor_nodes:
            self.sensor_nodes = self.fp_sensor_nodes

    def _build_centroids(self):
        centroid_key = defaultdict(list)
        partition_vecs = defaultdict(list)
        for fp in self.fingerprints:
            key = (fp['leak_partition'], fp['demand_period'])
            v = np.array(fp['sensor_fingerprint'])
            centroid_key[key].append(v)
            partition_vecs[fp['leak_partition']].append(v)
        self.centroids = {k: np.mean(v, 0) for k, v in centroid_key.items()}
        self.partition_centroids = {k: np.mean(v, 0)
                                    for k, v in partition_vecs.items()}
        print(f"  Centroids: {len(self.centroids)} period-matched, "
              f"{len(self.partition_centroids)} partition-level")

    def _load_partition_adjacency(self):
        p = os.path.join(self.kb_dir, 'partition_adjacency.json')
        if not os.path.exists(p):
            print("  WARNING: No partition_adjacency.json — topological channel disabled")
            self.partition_adj  = {}
            self.adj_matrix_norm = None
            return
        with open(p) as f:
            pag = json.load(f)
        self.partition_adj = {int(k): v for k, v in pag['adjacency'].items()}
        am = np.array(pag['adjacency_matrix'], dtype=float)
        row_sums = am.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.adj_matrix_norm = am / row_sums
        print(f"  Adjacency: {len(self.partition_adj)} partitions, "
              f"{int(am.sum()/2)} edges")

    def _load_community_summaries(self):
        p = os.path.join(self.kb_dir, 'community_summaries.json')
        if not os.path.exists(p):
            print("  WARNING: No community_summaries.json — community channel disabled")
            self.community_summaries  = {}
            self.community_embeddings = {}
            return
        with open(p) as f:
            cs = json.load(f)
        self.community_summaries = {int(k): v
                                    for k, v in cs['level_1_communities'].items()}
        print("  Loading sentence embedder for community channel...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        texts = [self.community_summaries[pid]['summary']
                 for pid in sorted(self.community_summaries)]
        pids  = sorted(self.community_summaries.keys())
        embs  = self.embedder.encode(texts)
        self.community_embeddings = {pid: emb for pid, emb in zip(pids, embs)}
        print(f"  Community summaries: {len(self.community_summaries)} partitions")

    # ── Channel helpers ───────────────────────────────────────────────────────
    def build_observation_vector(self, obs):
        return np.array([obs.get(sn, obs.get(str(sn), 0.0))
                         for sn in self.fp_sensor_nodes], dtype=float)

    def ch1_sensor_prior(self, obs):
        part_total = defaultdict(float)
        part_max   = defaultdict(float)
        part_cnt   = defaultdict(int)
        for sn, dp in obs.items():
            pid = self.sensor_to_partition.get(sn,
                  self.sensor_to_partition.get(str(sn), -1))
            if pid < 0:
                continue
            a = abs(dp)
            part_total[pid] += a
            part_max[pid]    = max(part_max[pid], a)
            if a > 0.05:
                part_cnt[pid] += 1
        if not part_total:
            return {}
        mt = max(part_total.values())
        mm = max(part_max.values()) if part_max else 1.0
        mc = max(max(part_cnt.values()) if part_cnt else 0, 1)
        return {pid: 0.5*part_total[pid]/mt + 0.3*part_max[pid]/mm
                     + 0.2*part_cnt[pid]/mc
                for pid in part_total}

    def ch2_centroid(self, obs_vec, demand_period=None):
        obs_norm = np.linalg.norm(obs_vec)
        if obs_norm < 1e-10:
            return {}
        scores = {}
        for pid, c in self.partition_centroids.items():
            key = (pid, demand_period)
            c = self.centroids.get(key, c)
            cn = np.linalg.norm(c)
            if cn < 1e-10:
                continue
            cos = np.dot(obs_vec, c) / (obs_norm * cn)
            euc = np.linalg.norm(obs_vec - c)
            sim = 1.0/(1.0 + euc/max(obs_norm, cn, 0.1))
            scores[pid] = 0.5*max(cos, 0) + 0.5*sim
        return scores

    def ch3_rank(self, obs_vec, demand_period=None):
        pool = (self.fp_by_period.get(demand_period, self.fingerprints)
                if demand_period else self.fingerprints)
        obs_rank = np.argsort(obs_vec).argsort().astype(float)
        votes = defaultdict(float)
        scores = []
        for fp in pool:
            fv = np.array(fp['sensor_fingerprint'], dtype=float)
            fr = np.argsort(fv).argsort().astype(float)
            n  = len(obs_rank)
            d2 = np.sum((obs_rank - fr)**2)
            rho = (1.0 - 6*d2/(n*(n*n-1))) if n > 1 else 0
            scores.append((fp['leak_partition'], rho))
        scores.sort(key=lambda x: -x[1])
        for i, (pid, rho) in enumerate(scores[:20]):
            votes[pid] += max(rho, 0) * 1.0/(i+1)
        return dict(votes)

    def ch4_fingerprint(self, obs_vec, demand_period=None):
        obs_norm = np.linalg.norm(obs_vec)
        if obs_norm < 1e-10:
            return {}
        pool = (self.fp_by_period.get(demand_period, self.fingerprints)
                if demand_period else self.fingerprints)
        votes = defaultdict(float)
        scored = []
        for fp in pool:
            fv = np.array(fp['sensor_fingerprint'], dtype=float)
            fn = np.linalg.norm(fv)
            if fn < 1e-10:
                continue
            cos = np.dot(obs_vec, fv) / (obs_norm * fn)
            euc = np.linalg.norm(obs_vec - fv)
            s = 0.5*max(cos, 0) + 0.5/(1.0+euc/max(obs_norm,fn,0.1))
            scored.append((fp['leak_partition'], s))
        scored.sort(key=lambda x: -x[1])
        for i, (pid, s) in enumerate(scored[:20]):
            votes[pid] += s * 1.0/(i+1)
        return dict(votes)

    def ch6_community(self, obs, demand_period=None):
        if not self.community_embeddings:
            return {}
        # Build a simple query text
        sorted_obs = sorted(obs.items(), key=lambda x: x[1])
        max_node, max_dp = sorted_obs[0]
        text = (f"Pressure leak, max drop {abs(max_dp):.3f}m at node {max_node}, "
                f"period {demand_period}.")
        q = self.embedder.encode([text])[0]
        qn = np.linalg.norm(q)
        if qn < 1e-10:
            return {}
        scores = {}
        for pid, e in self.community_embeddings.items():
            en = np.linalg.norm(e)
            if en < 1e-10:
                continue
            scores[pid] = float(max(np.dot(q, e)/(qn*en), 0))
        return scores

    def ch7_topo_propagation(self, raw_scores, beta=0.15):
        if self.adj_matrix_norm is None:
            return raw_scores
        K = len(self.adj_matrix_norm)
        sv = np.zeros(K)
        for pid, s in raw_scores.items():
            if 0 <= pid < K:
                sv[pid] = s
        prop = (1-beta)*sv + beta*(self.adj_matrix_norm @ sv)
        return {pid: float(prop[pid]) for pid in range(K) if prop[pid] > 0}

    # ── Main localize ─────────────────────────────────────────────────────────
    def localize_leak(self, obs, demand_period=None, top_k=10):
        obs_vec = self.build_observation_vector(obs)
        max_obs = max(abs(v) for v in obs.values()) if obs else 0

        # Adaptive weights (channel 5 weight redistributed to 1 and 4)
        if max_obs >= 1.0:
            w = {'prior':0.28,'centroid':0.12,'rank':0.08,'fp':0.34,'comm':0.18}
        elif max_obs >= 0.1:
            w = {'prior':0.22,'centroid':0.20,'rank':0.15,'fp':0.22,'comm':0.21}
        else:
            w = {'prior':0.17,'centroid':0.27,'rank':0.20,'fp':0.14,'comm':0.22}

        def norm(d):
            if not d:
                return {}
            mv = max(d.values())
            return {k: v/mv for k, v in d.items()} if mv > 0 else d

        n1 = norm(self.ch1_sensor_prior(obs))
        n2 = norm(self.ch2_centroid(obs_vec, demand_period))
        n3 = norm(self.ch3_rank(obs_vec, demand_period))
        n4 = norm(self.ch4_fingerprint(obs_vec, demand_period))
        n6 = norm(self.ch6_community(obs, demand_period))

        all_pids = set()
        for d in [n1, n2, n3, n4, n6]:
            all_pids.update(d.keys())

        pre = {pid: (w['prior']*n1.get(pid,0) + w['centroid']*n2.get(pid,0)
                    + w['rank']*n3.get(pid,0) + w['fp']*n4.get(pid,0)
                    + w['comm']*n6.get(pid,0))
               for pid in all_pids}

        final = self.ch7_topo_propagation(pre)
        ranked = sorted(final.items(), key=lambda x: -x[1])

        predicted = ranked[0][0] if ranked else -1
        top3 = [p for p, _ in ranked[:3]]

        margin = (ranked[0][1]-ranked[1][1]) if len(ranked) >= 2 else 0
        confidence = 'high' if margin > 0.15 else ('medium' if margin > 0.05 else 'low')

        return {
            'predicted_partition': predicted,
            'confidence': confidence,
            'top3_partitions': top3,
            'top5_partitions': [p for p, _ in ranked[:5]],
            'final_scores': {str(p): round(s,4) for p, s in ranked[:top_k]},
        }
