"""
Phase 3: TS-GraphRAG Leak Localization Engine (v3 - High Accuracy)

Key accuracy improvements over v2:
  1. Period-matched retrieval: filter candidates to same demand period
  2. Sensor-partition prior: partition with max sensor drop gets bonus
  3. Partition centroid matching: compare against avg fingerprint per partition
  4. Rank-based matching: compare sensor drop RANKINGS (scale-invariant)
  5. Adaptive fusion: weight channels by signal strength
"""
import os
import json
import glob
import numpy as np
import lancedb
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# ==================== Configuration ====================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gpt-oss:120b-cloud"
LANCEDB_TABLE_NAME = "leak_scenarios"
TOP_K = 10


# ==================== Prompt Templates ====================
REASONING_SYSTEM = """You are a water distribution network (WDN) leak localization expert.
You will receive real-time pressure sensor readings, partition topology information,
community-level hydraulic profiles, and candidate leak partitions ranked by a
multi-channel retrieval engine.

Your task is to perform Chain-of-Thought reasoning to determine the most likely
leak partition. Follow these steps:

Step 1 - Signal Assessment: Identify the overall signal strength (severe/moderate/
small/micro) from the maximum pressure drop magnitude.

Step 2 - Spatial Analysis: Determine which partition's sensors show the largest
pressure drops (total and individual). The leak is most likely in the partition
with the highest aggregated sensor drop.

Step 3 - Pattern Comparison: For each top candidate partition, compare the
observed sensor drop PATTERN (relative ratios between sensors, not just absolutes)
against the community hydraulic profile. A good match means the drop distribution
across sensors is consistent with historical leaks in that partition.

Step 4 - Topological Consistency: Check if the candidate partition's topological
neighbours also show elevated sensor drops. Leaks in partition X affect adjacent
partitions more than distant ones.

Step 5 - Final Decision: Synthesize all evidence and choose the partition with
the strongest combined evidence.

Your final answer MUST be a JSON block:
```json
{
  "predicted_partition": <int>,
  "confidence": "<high|medium|low>",
  "reasoning_summary": "<2-3 sentence explanation citing specific evidence>"
}
```"""

REASONING_TEMPLATE = """## Real-Time Sensor Observations (Demand Period: {demand_period})

{sensor_table}

## Partition Sensor Drop Summary (aggregated by partition):
{partition_drop_summary}

## Partition Adjacency (physical pipe connections):
{adjacency_info}

## Community Hydraulic Profiles (historical leak characteristics):
{community_profiles}

## Top-{k} Candidates from Multi-Channel Retrieval Engine:
{candidates_text}

## Chain-of-Thought Analysis Required:
Perform the 5-step reasoning process described in your instructions.
Pay special attention to:
- Which partition has the HIGHEST total sensor drop?
- Does the drop PATTERN match the community hydraulic profile?
- Are adjacent partitions also showing drops (topological consistency)?
Output your final JSON answer."""

# Enhanced prompt for confidence-gated refinement mode
REFINEMENT_SYSTEM = """You are a water distribution network (WDN) leak localization expert.
The multi-channel retrieval engine has LOW CONFIDENCE in its current prediction.
You have been called to REFINE the prediction using structured physical reasoning.

You MUST perform these 3 consistency checks on the TOP candidates:

## Check 1 — Spatial Consistency
Does the top-1 candidate partition contain the sensor(s) with the largest pressure drop?
If NOT, which candidate partition does? Score: PASS / FAIL.

## Check 2 — Topological Consistency
Do the topological neighbours of the top-1 candidate also show elevated pressure drops?
A leak creates spatially correlated perturbations—adjacent partitions should show
secondary drops. Score: PASS / FAIL.

## Check 3 — Pattern Consistency
Does the observed sensor drop PATTERN (relative magnitudes and rank ordering)
match the community hydraulic profile of the top-1 candidate?
Compare the drop distribution shape, not absolute values. Score: PASS / FAIL.

## Decision Rule
- If top-1 passes ≥2/3 checks → KEEP top-1
- If top-1 passes ≤1/3 checks → choose the candidate in top-3 that passes the most checks
- You MUST choose from the provided top-3 candidates. Do NOT suggest a partition outside top-3.

Your final answer MUST be a JSON block:
```json
{
  "spatial_check": "PASS" or "FAIL",
  "topological_check": "PASS" or "FAIL",
  "pattern_check": "PASS" or "FAIL",
  "checks_passed": <int 0-3>,
  "predicted_partition": <int from top-3>,
  "confidence": "<high|medium|low>",
  "reasoning_summary": "<2-3 sentence explanation citing specific evidence>"
}
```"""

REFINEMENT_TEMPLATE = """## ATTENTION: Low-confidence scenario — your analysis is critical.

## The retrieval engine's top-3 candidates are:
{top3_text}

The score margin between #1 and #2 is only {margin:.4f} — very close.

## Real-Time Sensor Observations (Demand Period: {demand_period})
{sensor_table}

## Partition Sensor Drop Summary:
{partition_drop_summary}

## Partition Adjacency (physical pipe connections):
{adjacency_info}

## Community Hydraulic Profiles:
{community_profiles}

Perform the 3 consistency checks and make your decision.
REMEMBER: You MUST choose from the top-3 candidates only: {top3_ids}"""



class LeakLocator:
    """TS-GraphRAG Leak Localization Engine (v4 - True GraphRAG)."""

    def __init__(self, knowledge_base_dir='knowledge_base',
                 sensor_results_dir='sensor_results'):
        self.kb_dir = knowledge_base_dir

        # Load embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to LanceDB
        db_path = os.path.join(knowledge_base_dir, 'lancedb')
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(LANCEDB_TABLE_NAME)
        print(f"LanceDB connected: {self.table.count_rows()} scenarios")

        # Load topology
        topo_file = os.path.join(knowledge_base_dir, 'topology_index.json')
        with open(topo_file, 'r') as f:
            self.topology = json.load(f)

        self.node_to_partition = {}
        for pid_str, pdata in self.topology['partitions'].items():
            for node in pdata['nodes']:
                self.node_to_partition[node] = int(pid_str)

        # Load sensors
        self._load_sensors(sensor_results_dir)

        # Load fingerprints & build centroids
        self._load_fingerprints()
        self._build_centroids()

        # ---- GraphRAG components ----
        self._load_partition_adjacency()
        self._load_community_summaries()

        # Initialize LLM
        print(f"Initializing LLM: {OLLAMA_MODEL}")
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1,
                              num_predict=1024)

        print("LeakLocator v4 (True GraphRAG) ready.\n")

    def _load_partition_adjacency(self):
        """Load the Partition Adjacency Graph for topological propagation."""
        pag_file = os.path.join(self.kb_dir, 'partition_adjacency.json')
        if not os.path.exists(pag_file):
            print("WARNING: No partition adjacency graph. Run graph_rag.py.")
            self.partition_adj = {}
            self.adj_matrix = None
            return

        with open(pag_file, 'r') as f:
            pag = json.load(f)

        self.partition_adj = {
            int(k): v for k, v in pag['adjacency'].items()
        }
        self.adj_matrix = np.array(pag['adjacency_matrix'], dtype=float)

        # Normalize adjacency matrix for propagation
        row_sums = self.adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.adj_matrix_norm = self.adj_matrix / row_sums

        print(f"Partition adjacency: {len(self.partition_adj)} partitions, "
              f"{int(self.adj_matrix.sum()/2)} edges")

    def _load_community_summaries(self):
        """Load hierarchical community summaries for GraphRAG retrieval."""
        cs_file = os.path.join(self.kb_dir, 'community_summaries.json')
        if not os.path.exists(cs_file):
            print("WARNING: No community summaries. Run graph_rag.py.")
            self.community_summaries = {}
            self.community_embeddings = {}
            return

        with open(cs_file, 'r', encoding='utf-8') as f:
            cs = json.load(f)

        self.community_summaries = {
            int(k): v for k, v in cs['level_1_communities'].items()
        }

        # Pre-compute embeddings for community summaries
        texts = []
        pids = []
        for pid in sorted(self.community_summaries.keys()):
            texts.append(self.community_summaries[pid]['summary'])
            pids.append(pid)

        if texts:
            embeddings = self.embedder.encode(texts)
            self.community_embeddings = {
                pid: emb for pid, emb in zip(pids, embeddings)
            }
        else:
            self.community_embeddings = {}

        print(f"Community summaries: {len(self.community_summaries)} partitions, "
              f"embeddings computed")

    def _load_sensors(self, sensor_results_dir):
        """Load sensor placement results."""
        summaries = glob.glob(
            os.path.join(sensor_results_dir, 'sensor_summary_*.json'))
        if not summaries:
            raise FileNotFoundError(f"No sensor summaries in {sensor_results_dir}")
        with open(sorted(summaries)[-1], 'r') as f:
            summary = json.load(f)

        self.sensor_nodes = []
        self.partition_sensors = {}
        self.sensor_to_partition = {}

        for pid_str, details in summary['partition_details'].items():
            pid = int(pid_str)
            nodes = details['sensor_nodes']
            self.partition_sensors[pid] = nodes
            for n in nodes:
                self.sensor_to_partition[n] = pid
                self.sensor_nodes.append(n)

        self.sensor_nodes = sorted(set(self.sensor_nodes),
                                    key=lambda x: int(x) if x.isdigit() else x)
        print(f"Sensors: {len(self.sensor_nodes)} nodes, "
              f"{len(self.partition_sensors)} partitions")

    def _load_fingerprints(self):
        """Load pre-computed sensor fingerprints."""
        fp_file = os.path.join(self.kb_dir, 'sensor_fingerprints.json')
        if not os.path.exists(fp_file):
            self.fingerprints = []
            self.fp_sensor_nodes = []
            self.fp_by_period = {}
            return

        with open(fp_file, 'r') as f:
            fp_data = json.load(f)

        self.fingerprints = fp_data['fingerprints']
        self.fp_sensor_nodes = fp_data['sensor_nodes']

        # Index fingerprints by demand period for period-matched retrieval
        self.fp_by_period = defaultdict(list)
        for fp in self.fingerprints:
            self.fp_by_period[fp['demand_period']].append(fp)

        print(f"Fingerprints: {len(self.fingerprints)} scenarios, "
              f"{fp_data['num_sensors']}-dim")

    def _build_centroids(self):
        """
        Build average fingerprint per (partition, period) combination.
        This is a key innovation: instead of matching individual scenarios,
        match against the average fingerprint for each partition.
        """
        centroid_key = defaultdict(list)
        for fp in self.fingerprints:
            key = (fp['leak_partition'], fp['demand_period'])
            centroid_key[key].append(np.array(fp['sensor_fingerprint']))

        self.centroids = {}
        for key, vectors in centroid_key.items():
            self.centroids[key] = np.mean(vectors, axis=0)

        # Also build period-agnostic centroids
        partition_vecs = defaultdict(list)
        for fp in self.fingerprints:
            partition_vecs[fp['leak_partition']].append(
                np.array(fp['sensor_fingerprint']))
        self.partition_centroids = {
            pid: np.mean(vecs, axis=0) for pid, vecs in partition_vecs.items()
        }

        print(f"Centroids: {len(self.centroids)} period-matched, "
              f"{len(self.partition_centroids)} partition-level")

    # ==================== Core Methods ====================

    def compute_sensor_partition_prior(self, sensor_observations):
        """
        Key insight: The partition whose sensors show the largest total
        pressure drop is most likely the leak source.

        Returns dict {partition: prior_score}
        """
        partition_total_drop = defaultdict(float)
        partition_max_drop = defaultdict(float)
        partition_sensor_count_affected = defaultdict(int)

        for sn, dp in sensor_observations.items():
            pid = self.sensor_to_partition.get(sn, -1)
            if pid < 0:
                continue
            abs_drop = abs(dp)
            partition_total_drop[pid] += abs_drop
            partition_max_drop[pid] = max(partition_max_drop[pid], abs_drop)
            if abs_drop > 0.05:  # any notable drop
                partition_sensor_count_affected[pid] += 1

        if not partition_total_drop:
            return {}

        # Normalize scores
        max_total = max(partition_total_drop.values())
        if max_total < 1e-10:
            return {}

        prior = {}
        for pid in partition_total_drop:
            # Combined: total drop + max drop + affected count
            score = (
                0.5 * partition_total_drop[pid] / max_total +
                0.3 * partition_max_drop[pid] / max(partition_max_drop.values()) +
                0.2 * partition_sensor_count_affected[pid] /
                max(max(partition_sensor_count_affected.values()), 1)
            )
            prior[pid] = score

        return prior

    def centroid_match(self, obs_vector, demand_period=None):
        """
        Match observation against partition centroids.
        Returns sorted list of (partition_id, similarity_score).
        """
        obs = np.array(obs_vector, dtype=float)
        obs_norm = np.linalg.norm(obs)
        if obs_norm < 1e-10:
            return []

        scores = []

        # If demand period known, use period-matched centroids first
        if demand_period:
            for pid in self.partition_centroids:
                key = (pid, demand_period)
                centroid = self.centroids.get(key)
                if centroid is None:
                    centroid = self.partition_centroids[pid]

                c_norm = np.linalg.norm(centroid)
                if c_norm < 1e-10:
                    continue

                cosine = np.dot(obs, centroid) / (obs_norm * c_norm)
                euc = np.linalg.norm(obs - centroid)
                euc_sim = 1.0 / (1.0 + euc / max(obs_norm, c_norm, 0.1))

                score = 0.5 * max(cosine, 0) + 0.5 * euc_sim
                scores.append((pid, score))
        else:
            for pid, centroid in self.partition_centroids.items():
                c_norm = np.linalg.norm(centroid)
                if c_norm < 1e-10:
                    continue
                cosine = np.dot(obs, centroid) / (obs_norm * c_norm)
                euc = np.linalg.norm(obs - centroid)
                euc_sim = 1.0 / (1.0 + euc / max(obs_norm, c_norm, 0.1))
                score = 0.5 * max(cosine, 0) + 0.5 * euc_sim
                scores.append((pid, score))

        scores.sort(key=lambda x: -x[1])
        return scores

    def rank_match(self, obs_vector, demand_period=None):
        """
        Rank-based matching: compare the ORDERING of sensor drops rather
        than absolute values. This is scale-invariant and robust to noise.

        Returns sorted list of (scenario_id, rank_correlation, fp_dict).
        """
        obs = np.array(obs_vector, dtype=float)
        obs_rank = np.argsort(obs).argsort().astype(float)

        # Use period-matched fingerprints if available
        if demand_period and demand_period in self.fp_by_period:
            pool = self.fp_by_period[demand_period]
        else:
            pool = self.fingerprints

        scores = []
        for fp in pool:
            fp_vec = np.array(fp['sensor_fingerprint'], dtype=float)
            fp_rank = np.argsort(fp_vec).argsort().astype(float)

            # Spearman rank correlation
            n = len(obs_rank)
            d_sq = np.sum((obs_rank - fp_rank) ** 2)
            if n > 1:
                rho = 1.0 - (6 * d_sq) / (n * (n * n - 1))
            else:
                rho = 0

            scores.append((fp['scenario_id'], rho, fp))

        scores.sort(key=lambda x: -x[1])
        return scores

    def retrieve_fingerprint_matched(self, obs_vector, demand_period=None,
                                      top_k=20):
        """
        Period-matched fingerprint retrieval with cosine+euclidean scoring.
        """
        obs = np.array(obs_vector, dtype=float)
        obs_norm = np.linalg.norm(obs)
        if obs_norm < 1e-10:
            return {}

        # Use period-matched pool
        if demand_period and demand_period in self.fp_by_period:
            pool = self.fp_by_period[demand_period]
        else:
            pool = self.fingerprints

        candidates = {}
        for fp in pool:
            fp_vec = np.array(fp['sensor_fingerprint'], dtype=float)
            fp_norm = np.linalg.norm(fp_vec)
            if fp_norm < 1e-10:
                continue

            cosine = np.dot(obs, fp_vec) / (obs_norm * fp_norm)
            euc = np.linalg.norm(obs - fp_vec)
            euc_sim = 1.0 / (1.0 + euc / max(obs_norm, fp_norm, 0.1))

            score = 0.5 * max(cosine, 0) + 0.5 * euc_sim

            sid = fp['scenario_id']
            candidates[sid] = {
                'scenario_id': sid,
                'leak_partition': fp['leak_partition'],
                'leak_node': fp['leak_node'],
                'leak_rate_Ls': fp['leak_rate_Ls'],
                'leak_severity': fp.get('leak_severity', ''),
                'demand_period': fp['demand_period'],
                'demand_multiplier': fp['demand_multiplier'],
                'max_pressure_drop': fp['max_pressure_drop'],
                'num_affected_nodes': fp['num_affected_nodes'],
                'affected_partitions': fp.get('affected_partitions', []),
                'fingerprint_score': score,
            }

        sorted_c = sorted(candidates.values(),
                           key=lambda x: -x['fingerprint_score'])
        return {c['scenario_id']: c for c in sorted_c[:top_k]}

    def retrieve_semantic(self, query_text, top_k=20):
        """Retrieve via text semantic search."""
        query_vec = self.embedder.encode([query_text])[0].tolist()
        results = self.table.search(query_vec).limit(top_k).to_pandas()

        candidates = {}
        for _, row in results.iterrows():
            sid = row['scenario_id']
            candidates[sid] = {
                'scenario_id': sid,
                'summary_text': row['summary_text'],
                'leak_partition': int(row['leak_partition']),
                'leak_node': row['leak_node'],
                'leak_rate_Ls': float(row['leak_rate_Ls']),
                'leak_severity': row.get('leak_severity', ''),
                'demand_period': row['demand_period'],
                'demand_multiplier': float(row['demand_multiplier']),
                'max_pressure_drop': float(row['max_pressure_drop']),
                'num_affected_nodes': int(row['num_affected_nodes']),
                'affected_partitions': row['affected_partitions'],
                'semantic_score': 1.0 / (1.0 + float(row['_distance'])),
            }
        return candidates

    def build_query_text(self, sensor_observations, demand_period=None):
        """Build query text from sensor observations."""
        sorted_obs = sorted(sensor_observations.items(), key=lambda x: x[1])
        max_node, max_drop = sorted_obs[0]
        max_drop_abs = abs(max_drop)

        affected_parts = sorted(set(
            self.sensor_to_partition.get(n, -1)
            for n, dp in sorted_obs if dp < -0.1
        ))

        if max_drop_abs >= 5:
            severity = "severe"
        elif max_drop_abs >= 1:
            severity = "moderate"
        elif max_drop_abs >= 0.2:
            severity = "small"
        else:
            severity = "micro"

        sensor_text = ", ".join(
            f"Sensor {n} (P#{self.sensor_to_partition.get(n, '?')}): {dp:.3f}m"
            for n, dp in sorted_obs[:8]
        )

        return (
            f"A {severity} leak during {demand_period or 'unknown'}. "
            f"Max drop {max_drop_abs:.3f}m at node {max_node} "
            f"(P#{self.sensor_to_partition.get(max_node, '?')}). "
            f"Sensors: {sensor_text}. "
            f"Affected partitions: {affected_parts}."
        )

    def build_observation_vector(self, sensor_observations):
        """Build fingerprint vector from sensor observations."""
        return np.array([sensor_observations.get(sn, 0.0)
                         for sn in self.fp_sensor_nodes], dtype=float)

    def graph_topological_propagation(self, raw_scores, beta=0.15):
        """
        GraphRAG Channel: Topological propagation on the Partition Adjacency Graph.

        A partition's score is boosted if its topologically adjacent partitions
        also score highly. This implements one step of label propagation:
          score'(k) = (1-β) * score(k) + β * Σ_j A_kj * score(j)

        Args:
            raw_scores: dict {partition: score}
            beta: propagation strength (0=no propagation, 1=full propagation)
        Returns:
            dict {partition: propagated_score}
        """
        if self.adj_matrix is None:
            return raw_scores

        K = len(self.adj_matrix)
        score_vec = np.zeros(K)
        for pid, s in raw_scores.items():
            if 0 <= pid < K:
                score_vec[pid] = s

        # One-step propagation
        propagated = (1 - beta) * score_vec + beta * self.adj_matrix_norm @ score_vec

        return {pid: float(propagated[pid]) for pid in range(K) if propagated[pid] > 0}

    def hierarchical_community_retrieval(self, query_text, top_k=5):
        """
        GraphRAG Channel: Hierarchical retrieval via community summaries.

        Level 1: Compare query against community-level summaries
        This provides a coarse partition-level filter before scenario-level search.
        """
        if not self.community_embeddings:
            return {}

        query_emb = self.embedder.encode([query_text])[0]
        q_norm = np.linalg.norm(query_emb)
        if q_norm < 1e-10:
            return {}

        scores = {}
        for pid, emb in self.community_embeddings.items():
            e_norm = np.linalg.norm(emb)
            if e_norm < 1e-10:
                continue
            cosine = np.dot(query_emb, emb) / (q_norm * e_norm)
            scores[pid] = float(max(cosine, 0))

        return scores

    def localize_leak(self, sensor_observations, demand_period=None,
                       top_k=TOP_K, verbose=True, use_llm=False):
        """
        Main entry: localize leak with GraphRAG multi-signal fusion.

        Seven-channel scoring (True GraphRAG):
          1. Sensor-partition prior (direct signal)
          2. Centroid template matching
          3. Rank-based matching (Spearman)
          4. Period-matched fingerprint retrieval
          5. Semantic retrieval
          6. Hierarchical community retrieval (GraphRAG)
          7. Topological propagation (GraphRAG)

        Channel 7 is applied as a post-processing step on the fused scores
        from channels 1-6, propagating scores along the partition adjacency graph.
        """
        if verbose:
            print("=" * 60)
            print("TS-GraphRAG Leak Localization (v4 - True GraphRAG)")
            print("=" * 60)

        obs_vector = self.build_observation_vector(sensor_observations)

        # ---- Channel 1: Sensor-Partition Prior ----
        prior = self.compute_sensor_partition_prior(sensor_observations)

        # ---- Channel 2: Centroid Template Matching ----
        centroid_scores = self.centroid_match(obs_vector, demand_period)

        # ---- Channel 3: Rank-Based Matching ----
        rank_results = self.rank_match(obs_vector, demand_period)
        rank_partition_votes = defaultdict(float)
        for i, (sid, rho, fp) in enumerate(rank_results[:top_k]):
            weight = max(rho, 0) * (1.0 / (i + 1))
            rank_partition_votes[fp['leak_partition']] += weight

        # ---- Channel 4: Period-Matched Fingerprint ----
        fp_candidates = self.retrieve_fingerprint_matched(
            obs_vector, demand_period, top_k * 2)
        fp_partition_votes = defaultdict(float)
        for i, c in enumerate(sorted(fp_candidates.values(),
                                      key=lambda x: -x['fingerprint_score'])):
            weight = c['fingerprint_score'] * (1.0 / (i + 1))
            fp_partition_votes[c['leak_partition']] += weight

        # ---- Channel 5: Semantic Retrieval ----
        query_text = self.build_query_text(sensor_observations, demand_period)
        sem_candidates = self.retrieve_semantic(query_text, top_k * 2)
        sem_partition_votes = defaultdict(float)
        for i, c in enumerate(sorted(sem_candidates.values(),
                                      key=lambda x: -x['semantic_score'])):
            weight = c['semantic_score'] * (1.0 / (i + 1))
            sem_partition_votes[c['leak_partition']] += weight

        # ---- Channel 6: Hierarchical Community Retrieval (GraphRAG) ----
        community_scores = self.hierarchical_community_retrieval(query_text)

        # ---- Adaptive Fusion (Channels 1-6) ----
        max_obs = max(abs(dp) for dp in sensor_observations.values())

        if max_obs >= 1.0:
            w = {'prior': 0.25, 'centroid': 0.12, 'rank': 0.08,
                 'fingerprint': 0.30, 'semantic': 0.10, 'community': 0.15}
        elif max_obs >= 0.1:
            w = {'prior': 0.20, 'centroid': 0.20, 'rank': 0.15,
                 'fingerprint': 0.18, 'semantic': 0.10, 'community': 0.17}
        else:
            w = {'prior': 0.15, 'centroid': 0.25, 'rank': 0.20,
                 'fingerprint': 0.12, 'semantic': 0.10, 'community': 0.18}

        # Normalize each channel
        def normalize_votes(votes):
            if not votes:
                return {}
            max_v = max(votes.values())
            return {k: v / max_v if max_v > 0 else 0 for k, v in votes.items()}

        n_prior = normalize_votes(prior)
        n_centroid = {p: s for p, s in centroid_scores}
        if n_centroid:
            max_c = max(n_centroid.values())
            n_centroid = {p: s / max_c if max_c > 0 else 0
                         for p, s in n_centroid.items()}
        n_rank = normalize_votes(dict(rank_partition_votes))
        n_fp = normalize_votes(dict(fp_partition_votes))
        n_sem = normalize_votes(dict(sem_partition_votes))
        n_comm = normalize_votes(community_scores)

        # Combine channels 1-6
        all_partitions = set()
        for d in [n_prior, n_centroid, n_rank, n_fp, n_sem, n_comm]:
            all_partitions.update(d.keys())

        pre_prop_scores = {}
        for pid in all_partitions:
            score = (
                w['prior'] * n_prior.get(pid, 0) +
                w['centroid'] * n_centroid.get(pid, 0) +
                w['rank'] * n_rank.get(pid, 0) +
                w['fingerprint'] * n_fp.get(pid, 0) +
                w['semantic'] * n_sem.get(pid, 0) +
                w['community'] * n_comm.get(pid, 0)
            )
            pre_prop_scores[pid] = score

        # ---- Channel 7: Topological Propagation (GraphRAG) ----
        # Propagate scores along partition adjacency graph
        final_scores = self.graph_topological_propagation(
            pre_prop_scores, beta=0.15
        )

        # Sort by final score
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])
        predicted = ranked[0][0] if ranked else -1
        top3 = [p for p, _ in ranked[:3]]
        top5 = [p for p, _ in ranked[:5]]

        if verbose:
            print(f"\n[Fusion] 6-channel + topo propagation")
            print(f"[Fusion] Final ranking: "
                  f"{[(p, f'{s:.3f}') for p, s in ranked[:5]]}")
            print(f"\n  ★ Predicted: Partition #{predicted}")

        # Confidence
        if len(ranked) >= 2:
            margin = ranked[0][1] - ranked[1][1]
            if margin > 0.15:
                confidence = 'high'
            elif margin > 0.05:
                confidence = 'medium'
            else:
                confidence = 'low'
        else:
            confidence = 'low'

        result = {
            'predicted_partition': predicted,
            'confidence': confidence,
            'top3_partitions': top3,
            'top5_partitions': top5,
            'final_scores': {str(p): round(s, 4) for p, s in ranked[:10]},
            'channel_weights': w,
        }

        # ---- LLM-Enhanced Prediction (Confidence-Gated) ----
        # LLM role depends on engine confidence:
        #   low  → REFINE: LLM can override with structured consistency checks
        #   medium → SUGGEST: LLM provides suggestion, logged but low weight
        #   high → EXPLAIN: LLM explains only, no override
        if use_llm:
            result = self._llm_enhance(
                sensor_observations, result, demand_period, ranked,
                confidence, margin if len(ranked) >= 2 else 0.0, verbose)

        return result

    def _build_context(self, sensor_observations, ranked, demand_period):
        """Build shared context strings for LLM prompts."""
        sorted_obs = sorted(sensor_observations.items(), key=lambda x: x[1])

        # Sensor table
        lines = []
        for n, dp in sorted_obs:
            pid = self.sensor_to_partition.get(n, '?')
            lines.append(f"  Node {n} (Partition #{pid}): {dp:+.4f} m")
        sensor_table = "\n".join(lines)

        # Partition drop summary
        part_summary = []
        for pid in sorted(self.partition_sensors.keys()):
            sensors = self.partition_sensors[pid]
            drops = [sensor_observations.get(s, 0) for s in sensors]
            total = sum(abs(d) for d in drops)
            max_drop = max(abs(d) for d in drops) if drops else 0
            part_summary.append(
                f"  P#{pid}: total_drop={total:.4f}m, max_drop={max_drop:.4f}m "
                f"(sensors: {', '.join(f'{s}={sensor_observations.get(s,0):+.4f}' for s in sensors)})")
        partition_drop_summary = "\n".join(part_summary)

        # Adjacency info
        adj_lines = []
        for pid in sorted(self.partition_adj.keys()):
            neighbors = self.partition_adj[pid]
            adj_lines.append(f"  P#{pid} ↔ {['P#'+str(n) for n in neighbors]}")
        adjacency_info = "\n".join(adj_lines) if adj_lines else "  Not available"

        # Community hydraulic profiles
        profile_lines = []
        top_pids = [p for p, _ in ranked[:5]]
        for pid in top_pids:
            cs = self.community_summaries.get(pid)
            if cs:
                profile_lines.append(
                    f"  P#{pid}: {cs['node_count']} nodes, "
                    f"{cs['scenario_count']} scenarios, "
                    f"max_drop range [{cs['max_drop_range'][0]:.2f}, "
                    f"{cs['max_drop_range'][1]:.2f}]m, "
                    f"leak nodes: {', '.join(cs['leak_nodes'][:3])}")
            else:
                profile_lines.append(f"  P#{pid}: No community profile available")
        community_profiles = "\n".join(profile_lines)

        return {
            'sensor_table': sensor_table,
            'partition_drop_summary': partition_drop_summary,
            'adjacency_info': adjacency_info,
            'community_profiles': community_profiles,
        }

    def _llm_enhance(self, sensor_observations, result, demand_period,
                     ranked, confidence, margin, verbose=True):
        """
        Confidence-gated LLM enhancement.

        Three modes:
          - REFINE (low confidence): LLM performs structured consistency
            checks and can override prediction from top-3 candidates.
          - SUGGEST (medium confidence): LLM provides suggestion, logged
            but not applied.
          - EXPLAIN (high confidence): LLM explains only, no override.
        """
        ctx = self._build_context(sensor_observations, ranked, demand_period)
        top3 = result['top3_partitions']

        if confidence == 'low':
            mode = 'refine'
        elif confidence == 'medium':
            mode = 'suggest'
        else:
            mode = 'explain'

        if verbose:
            print(f"\n  [LLM] Mode: {mode.upper()} (confidence={confidence}, margin={margin:.4f})")

        result['llm_mode'] = mode
        result['llm_changed_prediction'] = False

        if mode == 'refine':
            # REFINE: structured consistency checks, can override
            result = self._llm_refine(ctx, result, demand_period, ranked,
                                      top3, margin, verbose)
        elif mode == 'suggest':
            # SUGGEST: standard reasoning, logged but no override
            result = self._llm_explain(ctx, result, demand_period, ranked,
                                        verbose, allow_override=False)
        else:
            # EXPLAIN: standard reasoning, explanation only
            result = self._llm_explain(ctx, result, demand_period, ranked,
                                        verbose, allow_override=False)

        return result

    def _llm_refine(self, ctx, result, demand_period, ranked,
                     top3, margin, verbose):
        """
        REFINE mode: LLM performs 3 structured consistency checks.
        Can override prediction ONLY if choosing from top-3.
        """
        # Build top-3 detail text
        top3_lines = []
        for i, (pid, score) in enumerate(ranked[:3], 1):
            adj = self.partition_adj.get(pid, [])
            top3_lines.append(
                f"  #{i}. Partition {pid} (score: {score:.4f}, "
                f"adjacent to: {adj})")
        top3_text = "\n".join(top3_lines)
        top3_ids = str(top3)

        prompt = REFINEMENT_TEMPLATE.format(
            demand_period=demand_period or 'unknown',
            top3_text=top3_text,
            margin=margin,
            top3_ids=top3_ids,
            **ctx,
        )

        messages = [SystemMessage(content=REFINEMENT_SYSTEM),
                    HumanMessage(content=prompt)]

        try:
            response = self.llm.invoke(messages)
            llm_result = self._parse_result(response.content.strip())

            if llm_result.get('predicted_partition', -1) >= 0:
                llm_pred = llm_result['predicted_partition']
                checks_passed = llm_result.get('checks_passed', 0)
                result['llm_prediction'] = llm_pred
                result['llm_confidence'] = llm_result.get('confidence', 'unknown')
                result['llm_reasoning'] = llm_result.get('reasoning_summary', '')
                result['llm_checks'] = {
                    'spatial': llm_result.get('spatial_check', 'N/A'),
                    'topological': llm_result.get('topological_check', 'N/A'),
                    'pattern': llm_result.get('pattern_check', 'N/A'),
                    'passed': checks_passed,
                }

                original_pred = result['predicted_partition']
                result['llm_agrees'] = (llm_pred == original_pred)

                # Override logic: LLM can change prediction ONLY if:
                # 1. LLM picks a partition within top-3
                # 2. LLM's pick differs from current prediction
                if llm_pred in top3 and llm_pred != original_pred:
                    result['predicted_partition'] = llm_pred
                    result['llm_changed_prediction'] = True
                    # Update top3/top5 if prediction changed
                    if verbose:
                        print(f"  [LLM-REFINE] Override: P#{original_pred} → P#{llm_pred} "
                              f"(checks: {checks_passed}/3, "
                              f"spatial={llm_result.get('spatial_check','?')}, "
                              f"topo={llm_result.get('topological_check','?')}, "
                              f"pattern={llm_result.get('pattern_check','?')})")
                elif llm_pred == original_pred:
                    if verbose:
                        print(f"  [LLM-REFINE] Confirmed P#{original_pred} "
                              f"(checks: {checks_passed}/3)")
                else:
                    if verbose:
                        print(f"  [LLM-REFINE] Suggested P#{llm_pred} (outside top-3), "
                              f"keeping P#{original_pred}")
        except Exception as e:
            if verbose:
                print(f"  [LLM-REFINE] Failed: {e}")

        return result

    def _llm_explain(self, ctx, result, demand_period, ranked,
                      verbose, allow_override=False):
        """
        EXPLAIN/SUGGEST mode: standard CoT reasoning.
        Logs LLM's independent prediction but does not override.
        """
        cand_lines = []
        for i, (pid, score) in enumerate(ranked[:5], 1):
            adj = self.partition_adj.get(pid, [])
            cand_lines.append(
                f"  {i}. Partition #{pid} (score: {score:.3f}, "
                f"adjacent to: {adj})")
        candidates_text = "\n".join(cand_lines)

        prompt = REASONING_TEMPLATE.format(
            demand_period=demand_period or 'unknown',
            k=5,
            candidates_text=candidates_text,
            **ctx,
        )

        messages = [SystemMessage(content=REASONING_SYSTEM),
                    HumanMessage(content=prompt)]

        try:
            response = self.llm.invoke(messages)
            llm_result = self._parse_result(response.content.strip())

            if llm_result.get('predicted_partition', -1) >= 0:
                llm_pred = llm_result['predicted_partition']
                result['llm_prediction'] = llm_pred
                result['llm_confidence'] = llm_result.get('confidence', 'unknown')
                result['llm_reasoning'] = llm_result.get('reasoning_summary', '')
                result['llm_agrees'] = (llm_pred == result['predicted_partition'])

                if verbose:
                    agree_str = '✓ agrees' if result['llm_agrees'] else f'✗ suggests P#{llm_pred}'
                    print(f"  [LLM-{result['llm_mode'].upper()}] {agree_str} "
                          f"({llm_result.get('confidence', 'unknown')})")
        except Exception as e:
            if verbose:
                print(f"  [LLM] Reasoning failed: {e}")

        return result

    def _parse_result(self, text):
        """Extract JSON from LLM response."""
        import re
        for pattern in [
            r'```json\s*(.*?)\s*```',
            r'\{[^{}]*"predicted_partition"\s*:\s*(\d+)[^{}]*\}'
        ]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1) if '```' in pattern else m.group(0))
                except (json.JSONDecodeError, IndexError):
                    if m.lastindex and m.lastindex >= 1:
                        try:
                            return {"predicted_partition": int(m.group(1))}
                        except ValueError:
                            pass
        return {"predicted_partition": -1}


def simulate_sensor_anomaly(knowledge_base_dir, scenario_id,
                             sensor_nodes, noise_std=0.02):
    """Simulate sensor-only observations from a known scenario."""
    scenario_file = os.path.join(
        knowledge_base_dir, 'simulation_results', f'{scenario_id}.json')
    with open(scenario_file, 'r') as f:
        scenario = json.load(f)

    observations = {}
    for sn in sensor_nodes:
        dp = scenario.get('delta_pressure', {}).get(sn, 0.0)
        observations[sn] = round(dp + np.random.normal(0, noise_std), 4)

    return observations, scenario


if __name__ == '__main__':
    locator = LeakLocator('knowledge_base', 'sensor_results')

    sim_dir = os.path.join('knowledge_base', 'simulation_results')
    files = sorted([f[:-5] for f in os.listdir(sim_dir) if f.endswith('.json')])
    test = [f for f in files if '20Ls' in f and 'day_normal' in f]
    test_scenario = test[len(test) // 2]

    print(f"Testing: {test_scenario}\n")
    obs, gt = simulate_sensor_anomaly(
        'knowledge_base', test_scenario, locator.sensor_nodes, 0.02)
    result = locator.localize_leak(obs, gt['demand_period'], verbose=True)

    print(f"\nGT=P#{gt['leak_partition']}, "
          f"Pred=P#{result['predicted_partition']} "
          f"{'✅' if result['predicted_partition'] == gt['leak_partition'] else '❌'}")
