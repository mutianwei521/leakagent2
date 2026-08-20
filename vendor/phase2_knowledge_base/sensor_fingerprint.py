"""
Sensor Differential Fingerprint Builder

For each simulation scenario, extracts pressure drops ONLY at sensor nodes
and builds a compact numerical fingerprint vector for fast matching.
This is the key innovation for improving small/moderate leak detection:
instead of relying on text-based semantic search, we match on precise
numerical sensor readings.
"""
import os
import json
import glob
import numpy as np
from sentence_transformers import SentenceTransformer


def load_sensor_nodes(sensor_results_dir='sensor_results'):
    """
    Load sensor node list from sensor placement results.

    Returns:
        sorted list of sensor node names
        dict mapping partition_id -> [sensor_nodes]
    """
    # Find the most recent summary file
    summaries = glob.glob(
        os.path.join(sensor_results_dir, 'sensor_summary_*.json')
    )
    if not summaries:
        raise FileNotFoundError(
            f"No sensor summary files found in {sensor_results_dir}"
        )

    summary_file = sorted(summaries)[-1]  # most recent
    print(f"  Loading sensor summary: {summary_file}")

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    # Extract all sensor nodes
    all_sensor_nodes = []
    partition_sensors = {}
    for pid_str, details in summary['partition_details'].items():
        nodes = details['sensor_nodes']
        partition_sensors[int(pid_str)] = nodes
        all_sensor_nodes.extend(nodes)

    # Sort for consistent ordering
    all_sensor_nodes = sorted(set(all_sensor_nodes), key=lambda x: int(x) if x.isdigit() else x)

    print(f"  Total sensors: {len(all_sensor_nodes)}")
    print(f"  Partitions with sensors: {len(partition_sensors)}")

    return all_sensor_nodes, partition_sensors


def build_fingerprints(knowledge_base_dir='knowledge_base',
                       sensor_results_dir='sensor_results'):
    """
    For every simulation scenario, extract the pressure drop at each
    sensor node to form a fixed-length numerical fingerprint vector.

    Returns:
        fingerprints: list of dicts, each with scenario_id, vector, metadata
        sensor_nodes: ordered list of sensor node names (defines vector dims)
    """
    print("\n" + "=" * 60)
    print("Building Sensor Differential Fingerprints")
    print("=" * 60)

    # Load sensor nodes
    sensor_nodes, partition_sensors = load_sensor_nodes(sensor_results_dir)
    n_sensors = len(sensor_nodes)

    # Load all simulation results
    sim_dir = os.path.join(knowledge_base_dir, 'simulation_results')
    scenario_files = sorted(glob.glob(os.path.join(sim_dir, 'leak_*.json')))
    print(f"  Scenario files: {len(scenario_files)}")

    fingerprints = []
    for i, sf in enumerate(scenario_files):
        with open(sf, 'r') as f:
            scenario = json.load(f)

        # Extract pressure drop at each sensor node
        delta_pressure = scenario.get('delta_pressure', {})
        vector = []
        for sn in sensor_nodes:
            dp = delta_pressure.get(sn, 0.0)
            vector.append(dp)

        fingerprints.append({
            'scenario_id': scenario['scenario_id'],
            'leak_partition': scenario['leak_partition'],
            'leak_node': scenario['leak_node'],
            'leak_rate_Ls': scenario['leak_rate_Ls'],
            'leak_severity': scenario.get('leak_severity', ''),
            'demand_period': scenario['demand_period'],
            'demand_multiplier': scenario['demand_multiplier'],
            'max_pressure_drop': scenario['max_pressure_drop'],
            'num_affected_nodes': scenario['num_affected_nodes'],
            'affected_partitions': scenario.get('affected_partitions', []),
            'sensor_fingerprint': vector,
        })

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(scenario_files)}] processed")

    print(f"  Built {len(fingerprints)} fingerprints "
          f"({n_sensors}-dim vectors)")

    # Save fingerprints
    output_file = os.path.join(knowledge_base_dir, 'sensor_fingerprints.json')
    output = {
        'sensor_nodes': sensor_nodes,
        'partition_sensors': {str(k): v for k, v in partition_sensors.items()},
        'num_sensors': n_sensors,
        'num_scenarios': len(fingerprints),
        'fingerprints': fingerprints,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {output_file} "
          f"({os.path.getsize(output_file) / 1024:.0f} KB)")

    return fingerprints, sensor_nodes, partition_sensors


def match_fingerprint(observation_vector, fingerprints, top_k=10):
    """
    Match an observation vector against all stored fingerprints
    using cosine similarity and Euclidean distance.

    Args:
        observation_vector: numpy array of pressure drops at sensor nodes
        fingerprints: list of fingerprint dicts (from build_fingerprints)
        top_k: number of best matches to return

    Returns:
        list of (scenario_id, similarity_score, fingerprint_dict)
    """
    obs = np.array(observation_vector, dtype=float)
    obs_norm = np.linalg.norm(obs)

    if obs_norm < 1e-10:
        # All zeros — no meaningful signal
        return []

    scores = []
    for fp in fingerprints:
        fp_vec = np.array(fp['sensor_fingerprint'], dtype=float)
        fp_norm = np.linalg.norm(fp_vec)

        if fp_norm < 1e-10:
            scores.append((fp['scenario_id'], 0.0, fp))
            continue

        # Cosine similarity
        cosine_sim = np.dot(obs, fp_vec) / (obs_norm * fp_norm)

        # Euclidean distance (normalized)
        euclidean_dist = np.linalg.norm(obs - fp_vec)
        max_range = max(obs_norm, fp_norm, 1.0)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist / max_range)

        # Combined score (cosine is better for direction, euclidean for magnitude)
        combined = 0.5 * cosine_sim + 0.5 * euclidean_sim

        scores.append((fp['scenario_id'], combined, fp))

    # Sort by score descending
    scores.sort(key=lambda x: -x[1])

    return scores[:top_k]


if __name__ == '__main__':
    build_fingerprints()
