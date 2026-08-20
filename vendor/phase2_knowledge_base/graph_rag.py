"""
TS-GraphRAG: True Graph-Structured Retrieval-Augmented Generation

This module implements the genuine GraphRAG architecture for WDN leak localization:

1. Scenario Knowledge Graph (SKG):
   - Nodes = leak scenarios (540)
   - Edges = topological adjacency (partitions sharing boundary pipes)
           + fingerprint similarity (cosine > threshold)
   - Node attributes = scenario metadata + fingerprint vector + embedding

2. Partition Adjacency Graph (PAG):
   - Nodes = partitions (15)
   - Edges = boundary pipes between partitions
   - Edge weights = number of cross-boundary pipes

3. Hierarchical Community Summarization:
   - Level 0 (global): entire network summary
   - Level 1 (community): per-partition summary
   - Level 2 (scenario): individual scenario summary

4. Graph-Based Retrieval:
   - Initial retrieval via multi-channel fusion
   - Graph expansion: traverse SKG neighbors
   - Topological propagation: adjacent-partition score boost
   - Hierarchical drill-down: community → scenario
"""
import os
import json
import numpy as np
import wntr
from collections import defaultdict
from itertools import combinations


def build_partition_adjacency(inp_file, topology_index_file, output_file):
    """
    Build the Partition Adjacency Graph (PAG) from the WDN model.
    Two partitions are adjacent if there exists at least one pipe
    connecting a node in one partition to a node in the other.

    Returns dict: {partition_id: [list of adjacent partition_ids]}
    """
    print("Building Partition Adjacency Graph...")

    # Load network
    wn = wntr.network.WaterNetworkModel(inp_file)

    # Load partition assignments
    with open(topology_index_file, 'r') as f:
        topo = json.load(f)

    node_to_partition = {}
    for pid_str, pdata in topo['partitions'].items():
        pid = int(pid_str)
        for node in pdata['nodes']:
            node_to_partition[node] = pid

    num_partitions = topo['num_partitions']

    # Build adjacency from pipes
    adjacency = defaultdict(set)
    boundary_pipes = defaultdict(list)

    for link_name, link in wn.links():
        start_node = str(link.start_node_name)
        end_node = str(link.end_node_name)

        p1 = node_to_partition.get(start_node, -1)
        p2 = node_to_partition.get(end_node, -1)

        if p1 >= 0 and p2 >= 0 and p1 != p2:
            adjacency[p1].add(p2)
            adjacency[p2].add(p1)
            boundary_pipes[(min(p1, p2), max(p1, p2))].append(link_name)

    # Build adjacency matrix (edge weights = number of boundary pipes)
    adj_matrix = np.zeros((num_partitions, num_partitions), dtype=int)
    for (p1, p2), pipes in boundary_pipes.items():
        adj_matrix[p1][p2] = len(pipes)
        adj_matrix[p2][p1] = len(pipes)

    pag = {
        'num_partitions': num_partitions,
        'adjacency': {str(k): sorted(list(v)) for k, v in adjacency.items()},
        'boundary_pipe_counts': {
            f"{p1}-{p2}": len(pipes) for (p1, p2), pipes in boundary_pipes.items()
        },
        'adjacency_matrix': adj_matrix.tolist(),
    }

    with open(output_file, 'w') as f:
        json.dump(pag, f, indent=2)

    print(f"  Partitions: {num_partitions}")
    print(f"  Adjacent pairs: {len(boundary_pipes)}")
    print(f"  Boundary pipes: {sum(len(p) for p in boundary_pipes.values())}")

    return pag


def build_scenario_knowledge_graph(fingerprint_file, pag_file,
                                    similarity_threshold=0.85,
                                    output_file=None):
    """
    Build the Scenario Knowledge Graph (SKG).

    Nodes: scenarios
    Edges:
      - Type 'topo': scenarios in adjacent partitions
      - Type 'sim': scenarios with fingerprint cosine similarity > threshold
    """
    print("Building Scenario Knowledge Graph...")

    with open(fingerprint_file, 'r') as f:
        fp_data = json.load(f)

    with open(pag_file, 'r') as f:
        pag = json.load(f)

    fingerprints = fp_data['fingerprints']
    adjacency = pag['adjacency']

    # Build scenario nodes
    nodes = {}
    by_partition = defaultdict(list)

    for fp in fingerprints:
        sid = fp['scenario_id']
        nodes[sid] = {
            'partition': fp['leak_partition'],
            'node': fp['leak_node'],
            'rate': fp['leak_rate_Ls'],
            'period': fp['demand_period'],
            'severity': fp.get('leak_severity', ''),
            'vector': np.array(fp['sensor_fingerprint'], dtype=float),
        }
        by_partition[fp['leak_partition']].append(sid)

    # Build edges
    topo_edges = []  # (sid1, sid2, weight)
    sim_edges = []

    # Type 1: Topological adjacency edges
    # Scenarios in adjacent partitions are connected
    for p1_str, neighbors in adjacency.items():
        p1 = int(p1_str)
        for p2 in neighbors:
            # Connect scenarios across adjacent partitions
            # Use boundary pipe count as edge weight
            bp_key = f"{min(p1,p2)}-{max(p1,p2)}"
            weight = pag['boundary_pipe_counts'].get(bp_key, 1)

            for s1 in by_partition.get(p1, []):
                for s2 in by_partition.get(p2, []):
                    # Only connect same-period scenarios for coherence
                    if nodes[s1]['period'] == nodes[s2]['period']:
                        topo_edges.append((s1, s2, weight))

    # Type 2: Fingerprint similarity edges
    scenario_ids = list(nodes.keys())
    n = len(scenario_ids)

    # Pre-compute norms
    norms = {}
    for sid in scenario_ids:
        norms[sid] = np.linalg.norm(nodes[sid]['vector'])

    # Compute pairwise cosine for same-partition scenarios (intra-community)
    for pid, sids in by_partition.items():
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                s1, s2 = sids[i], sids[j]
                n1, n2 = norms[s1], norms[s2]
                if n1 > 1e-10 and n2 > 1e-10:
                    cos = np.dot(nodes[s1]['vector'], nodes[s2]['vector']) / (n1 * n2)
                    if cos > similarity_threshold:
                        sim_edges.append((s1, s2, float(cos)))

    # Build graph as adjacency list
    graph = defaultdict(list)
    for s1, s2, w in topo_edges:
        graph[s1].append({'target': s2, 'type': 'topo', 'weight': w})
        graph[s2].append({'target': s1, 'type': 'topo', 'weight': w})

    for s1, s2, w in sim_edges:
        graph[s1].append({'target': s2, 'type': 'sim', 'weight': w})
        graph[s2].append({'target': s1, 'type': 'sim', 'weight': w})

    skg = {
        'num_nodes': len(nodes),
        'num_topo_edges': len(topo_edges),
        'num_sim_edges': len(sim_edges),
        'graph': {sid: edges for sid, edges in graph.items()},
        'partition_membership': {sid: n['partition'] for sid, n in nodes.items()},
    }

    if output_file:
        # Save without numpy arrays
        save_data = {k: v for k, v in skg.items()}
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

    print(f"  Scenario nodes: {len(nodes)}")
    print(f"  Topological edges: {len(topo_edges)}")
    print(f"  Similarity edges: {len(sim_edges)}")
    print(f"  Avg degree: {sum(len(v) for v in graph.values()) / max(len(graph), 1):.1f}")

    return skg


def build_community_summaries(summaries_file, topology_file, output_file):
    """
    Build hierarchical community summaries for GraphRAG.

    Level 0: Global network summary
    Level 1: Per-partition community summaries (aggregated from scenario summaries)
    Level 2: Individual scenario summaries (already exist)
    """
    print("Building hierarchical community summaries...")

    with open(summaries_file, 'r', encoding='utf-8') as f:
        all_summaries = json.load(f)

    with open(topology_file, 'r') as f:
        topo = json.load(f)

    # Group summaries by partition
    partition_scenarios = defaultdict(list)
    for item in all_summaries:
        pid = item.get('leak_partition', -1)
        partition_scenarios[pid].append(item)

    # Level 1: Community summaries
    community_summaries = {}
    for pid in range(topo['num_partitions']):
        scenarios = partition_scenarios.get(pid, [])
        if not scenarios:
            continue

        # Extract statistics
        rates = [s['leak_rate_Ls'] for s in scenarios]
        max_drops = [s.get('max_pressure_drop', 0) for s in scenarios]
        periods = set(s.get('demand_period', '') for s in scenarios)
        affected = [s.get('num_affected_nodes', 0) for s in scenarios]
        nodes_in_partition = topo['partitions'].get(str(pid), {}).get('node_count', 0)

        # Construct community-level summary
        summary = (
            f"Partition {pid} contains {nodes_in_partition} junction nodes. "
            f"This community has been characterized through {len(scenarios)} "
            f"leak simulation scenarios with leak rates ranging from "
            f"{min(rates):.0f} to {max(rates):.0f} L/s. "
            f"The hydraulic response shows maximum pressure drops between "
            f"{min(max_drops):.2f} and {max(max_drops):.2f} meters, "
            f"affecting {min(affected)} to {max(affected)} nodes. "
            f"Demand periods covered: {', '.join(sorted(periods))}. "
        )

        # Add leak node info
        leak_nodes = set(s.get('leak_node', '') for s in scenarios)
        summary += (
            f"Representative leak nodes in this community: {', '.join(sorted(leak_nodes))}. "
        )

        # Add severity distribution
        sev_counts = defaultdict(int)
        for s in scenarios:
            sev_counts[s.get('leak_severity', 'unknown')] += 1
        sev_str = ', '.join(f"{k}: {v}" for k, v in sorted(sev_counts.items()))
        summary += f"Severity distribution: {sev_str}."

        community_summaries[pid] = {
            'partition_id': pid,
            'node_count': nodes_in_partition,
            'scenario_count': len(scenarios),
            'summary': summary,
            'leak_rate_range': [min(rates), max(rates)],
            'max_drop_range': [min(max_drops), max(max_drops)],
            'affected_range': [min(affected), max(affected)],
            'periods': sorted(periods),
            'leak_nodes': sorted(leak_nodes),
        }

    # Level 0: Global summary
    total_scenarios = sum(c['scenario_count'] for c in community_summaries.values())
    total_nodes = sum(c['node_count'] for c in community_summaries.values())
    all_rates = set()
    for c in community_summaries.values():
        all_rates.update(c['leak_rate_range'])

    global_summary = (
        f"The water distribution network comprises {total_nodes} junction nodes "
        f"organized into {len(community_summaries)} hydraulically coherent partitions "
        f"via pressure-weighted Leiden community detection. "
        f"The knowledge base contains {total_scenarios} pre-simulated leak scenarios "
        f"spanning leak rates from {min(all_rates):.0f} to {max(all_rates):.0f} L/s "
        f"across 3 demand periods (night low, day normal, evening peak). "
        f"Each scenario is encoded as both a natural-language hydraulic summary "
        f"and a {30}-dimensional sensor-differential fingerprint vector."
    )

    output = {
        'level_0_global': {
            'summary': global_summary,
            'num_partitions': len(community_summaries),
            'total_scenarios': total_scenarios,
            'total_nodes': total_nodes,
        },
        'level_1_communities': community_summaries,
        'level_2_scenario_count': total_scenarios,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Global summary: {len(global_summary)} chars")
    print(f"  Community summaries: {len(community_summaries)}")
    print(f"  Total scenarios (L2): {total_scenarios}")

    return output


if __name__ == '__main__':
    import sys

    kb_dir = 'knowledge_base'
    inp_file = os.path.join('dataset', 'Exa7.inp')
    topo_file = os.path.join(kb_dir, 'topology_index.json')

    # Step 1: Partition adjacency graph
    pag_file = os.path.join(kb_dir, 'partition_adjacency.json')
    pag = build_partition_adjacency(inp_file, topo_file, pag_file)

    # Step 2: Scenario knowledge graph
    fp_file = os.path.join(kb_dir, 'sensor_fingerprints.json')
    skg_file = os.path.join(kb_dir, 'scenario_knowledge_graph.json')
    skg = build_scenario_knowledge_graph(fp_file, pag_file,
                                          similarity_threshold=0.85,
                                          output_file=skg_file)

    # Step 3: Hierarchical community summaries
    summaries_file = os.path.join(kb_dir, 'scenario_summaries.json')
    hier_file = os.path.join(kb_dir, 'community_summaries.json')
    build_community_summaries(summaries_file, topo_file, hier_file)

    print("\n✅ GraphRAG knowledge graph construction complete.")
