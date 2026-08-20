"""
Step 2.2: Batch Leak Simulation

Systematically injects leaks across all partitions under different demand
periods and leak rates, runs WNTR hydraulic simulations, and extracts
multi-dimensional hydraulic response features (pressure drops, flow changes).
"""
import os
import json
import copy
import numpy as np
import wntr
import pickle
import warnings

warnings.filterwarnings('ignore')


# ==================== Demand period configurations ====================
DEMAND_PERIODS = {
    'night_low': {
        'multiplier': 0.7,
        'label': 'Night low-demand period (0.7x)'
    },
    'day_normal': {
        'multiplier': 1.0,
        'label': 'Daytime normal demand (1.0x)'
    },
    'evening_peak': {
        'multiplier': 1.2,
        'label': 'Evening peak demand (1.2x)'
    },
}

# Leak rates in L/s
LEAK_RATES = [2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
LEAK_RATE_LABELS = {
    2.0: 'micro', 5.0: 'small', 10.0: 'small-moderate',
    20.0: 'moderate', 35.0: 'moderate-severe', 50.0: 'severe'
}


def select_representative_nodes(wn, node_to_community, partition_id,
                                 max_nodes=2):
    """
    Select representative leak nodes within a partition based on
    degree centrality (most connected nodes).

    Args:
        wn: WNTR network model
        node_to_community: dict mapping node name -> community id
        partition_id: target partition id
        max_nodes: max number of nodes to select

    Returns:
        list of selected junction node names
    """
    # Get all junctions in this partition
    partition_junctions = [
        n for n in wn.junction_name_list
        if node_to_community.get(n) == partition_id
    ]

    if not partition_junctions:
        return []

    # Calculate degree (number of connected links) for each junction
    node_degrees = {}
    for junc_name in partition_junctions:
        degree = 0
        for link_name in wn.link_name_list:
            link = wn.get_link(link_name)
            if (link.start_node_name == junc_name or
                    link.end_node_name == junc_name):
                degree += 1
        node_degrees[junc_name] = degree

    # Sort by degree descending, pick top-N
    sorted_nodes = sorted(node_degrees.items(), key=lambda x: -x[1])
    selected = [n for n, d in sorted_nodes[:max_nodes]]

    return selected


def run_baseline_simulation(wn, demand_multiplier=1.0):
    """
    Run baseline (no-leak) simulation with a given demand multiplier.

    Returns:
        baseline_pressure: dict {node_name: avg_pressure}
        baseline_flow: dict {link_name: avg_flow}
    """
    wn_copy = copy.deepcopy(wn)

    # Apply demand multiplier to all junctions
    for junc_name in wn_copy.junction_name_list:
        junc = wn_copy.get_node(junc_name)
        for demand in junc.demand_timeseries_list:
            demand.base_value *= demand_multiplier

    # Run simulation
    sim = wntr.sim.EpanetSimulator(wn_copy)
    results = sim.run_sim()

    # Extract average pressures
    pressure_df = results.node['pressure']
    baseline_pressure = pressure_df.mean(axis=0).to_dict()

    # Extract average flows (absolute values for comparison)
    flow_df = results.link['flowrate']
    baseline_flow = flow_df.mean(axis=0).to_dict()

    return baseline_pressure, baseline_flow


def run_leak_simulation(wn, leak_node, leak_rate_Ls, demand_multiplier=1.0):
    """
    Run a leak simulation by adding an emitter at the specified node.

    Args:
        wn: WNTR network model
        leak_node: node name where the leak occurs
        leak_rate_Ls: leak rate in L/s
        demand_multiplier: multiplier for all demands

    Returns:
        leak_pressure: dict {node_name: avg_pressure}
        leak_flow: dict {link_name: avg_flow}
    """
    wn_copy = copy.deepcopy(wn)

    # Apply demand multiplier
    for junc_name in wn_copy.junction_name_list:
        junc = wn_copy.get_node(junc_name)
        for demand in junc.demand_timeseries_list:
            demand.base_value *= demand_multiplier

    # Add leak as additional demand at the target node
    # (using emitter coefficient approach or simple demand addition)
    leak_node_obj = wn_copy.get_node(leak_node)
    leak_node_obj.add_demand(leak_rate_Ls / 1000.0, '881')  # Convert L/s to m3/s

    # Run simulation
    sim = wntr.sim.EpanetSimulator(wn_copy)
    results = sim.run_sim()

    # Extract average pressures
    pressure_df = results.node['pressure']
    leak_pressure = pressure_df.mean(axis=0).to_dict()

    # Extract average flows
    flow_df = results.link['flowrate']
    leak_flow = flow_df.mean(axis=0).to_dict()

    return leak_pressure, leak_flow


def compute_response_features(baseline_pressure, baseline_flow,
                               leak_pressure, leak_flow,
                               node_to_community, pressure_threshold=0.5):
    """
    Compute hydraulic response features by comparing leak vs baseline.

    Args:
        baseline_pressure: dict of baseline node pressures
        baseline_flow: dict of baseline link flows
        leak_pressure: dict of leak-scenario node pressures
        leak_flow: dict of leak-scenario link flows
        node_to_community: partition assignment
        pressure_threshold: threshold (m) for "affected" classification

    Returns:
        features dict
    """
    # Pressure differences
    delta_pressure = {}
    for node in baseline_pressure:
        if node in leak_pressure:
            dp = leak_pressure[node] - baseline_pressure[node]
            if abs(dp) > 0.01:  # filter negligible changes
                delta_pressure[node] = round(dp, 4)

    # Flow differences
    delta_flow = {}
    for link in baseline_flow:
        if link in leak_flow:
            df = leak_flow[link] - baseline_flow[link]
            if abs(df) > 0.0001:
                delta_flow[link] = round(df, 6)

    # Affected nodes (pressure drop exceeds threshold)
    affected_nodes = [
        n for n, dp in delta_pressure.items()
        if dp < -pressure_threshold
    ]

    # Affected partitions
    affected_partitions = sorted(set(
        node_to_community.get(n, -1) for n in affected_nodes
    ))

    # Max pressure drop
    if delta_pressure:
        max_drop_node = min(delta_pressure, key=delta_pressure.get)
        max_pressure_drop = abs(delta_pressure[max_drop_node])
    else:
        max_drop_node = "N/A"
        max_pressure_drop = 0.0

    # Top-5 pressure drops
    sorted_drops = sorted(delta_pressure.items(), key=lambda x: x[1])[:5]
    top5_pressure_drops = {n: round(abs(dp), 4) for n, dp in sorted_drops}

    # Top-5 flow changes (by absolute change)
    sorted_flows = sorted(delta_flow.items(), key=lambda x: abs(x[1]),
                          reverse=True)[:5]
    top5_flow_changes = {l: round(df, 6) for l, df in sorted_flows}

    return {
        'delta_pressure': delta_pressure,
        'delta_flow': delta_flow,
        'affected_nodes': affected_nodes,
        'affected_partitions': affected_partitions,
        'max_drop_node': max_drop_node,
        'max_pressure_drop': round(max_pressure_drop, 4),
        'num_affected_nodes': len(affected_nodes),
        'top5_pressure_drops': top5_pressure_drops,
        'top5_flow_changes': top5_flow_changes,
    }


def run_batch_simulation(inp_file, partition_file, partition_k, output_dir,
                          max_scenarios=None, nodes_per_partition=2):
    """
    Main function: run batch leak simulations across all partitions.

    Args:
        inp_file: path to EPANET INP file
        partition_file: path to partitions.pkl
        partition_k: number of communities
        output_dir: directory to save results
        max_scenarios: limit total scenarios (for testing)
        nodes_per_partition: how many representative nodes per partition
    """
    sim_dir = os.path.join(output_dir, 'simulation_results')
    os.makedirs(sim_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Step 2.2: Batch Leak Simulation")
    print("=" * 60)

    # Load network
    wn = wntr.network.WaterNetworkModel(inp_file)
    print(f"  Network: {len(wn.junction_name_list)} junctions")

    # Load partition
    with open(partition_file, 'rb') as f:
        all_partitions = pickle.load(f)
    data = all_partitions[partition_k]
    node_to_community = data['node_to_community']
    partitions = sorted(set(node_to_community.values()))
    print(f"  Partitions: {len(partitions)} (k={partition_k})")

    # Select representative nodes for each partition
    print("  Selecting representative leak nodes per partition...")
    partition_nodes = {}
    for pid in partitions:
        selected = select_representative_nodes(
            wn, node_to_community, pid, max_nodes=nodes_per_partition
        )
        partition_nodes[pid] = selected
        print(f"    Partition #{pid}: {selected}")

    # Count total scenarios
    total_leak_nodes = sum(len(v) for v in partition_nodes.values())
    total_scenarios = total_leak_nodes * len(DEMAND_PERIODS) * len(LEAK_RATES)
    print(f"\n  Total scenarios to simulate: {total_scenarios}")
    print(f"    = {total_leak_nodes} nodes × {len(DEMAND_PERIODS)} periods "
          f"× {len(LEAK_RATES)} leak rates")

    if max_scenarios:
        print(f"  (Limited to {max_scenarios} scenarios for testing)")

    # Run baseline simulations for each demand period
    print("\n  Running baseline simulations...")
    baselines = {}
    for period_name, period_cfg in DEMAND_PERIODS.items():
        mult = period_cfg['multiplier']
        bp, bf = run_baseline_simulation(wn, mult)
        baselines[period_name] = (bp, bf)
        print(f"    Baseline [{period_name}] (x{mult}): done")

    # Run leak scenarios
    print("\n  Running leak scenarios...")
    scenario_count = 0
    all_scenario_ids = []

    for pid in partitions:
        for leak_node in partition_nodes[pid]:
            for period_name, period_cfg in DEMAND_PERIODS.items():
                for leak_rate in LEAK_RATES:
                    if max_scenarios and scenario_count >= max_scenarios:
                        break

                    scenario_id = (
                        f"leak_P{pid}_N{leak_node}_"
                        f"{int(leak_rate)}Ls_{period_name}"
                    )
                    scenario_file = os.path.join(
                        sim_dir, f"{scenario_id}.json"
                    )

                    # Skip if already exists (resume support)
                    if os.path.exists(scenario_file):
                        all_scenario_ids.append(scenario_id)
                        scenario_count += 1
                        continue

                    try:
                        # Run leak simulation
                        lp, lf = run_leak_simulation(
                            wn, leak_node, leak_rate,
                            period_cfg['multiplier']
                        )

                        # Compute features
                        bp, bf = baselines[period_name]
                        features = compute_response_features(
                            bp, bf, lp, lf, node_to_community
                        )

                        # Build scenario record
                        scenario = {
                            'scenario_id': scenario_id,
                            'leak_node': leak_node,
                            'leak_partition': pid,
                            'leak_rate_Ls': leak_rate,
                            'leak_severity': LEAK_RATE_LABELS[leak_rate],
                            'demand_period': period_name,
                            'demand_multiplier': period_cfg['multiplier'],
                            'demand_label': period_cfg['label'],
                            'max_pressure_drop': features['max_pressure_drop'],
                            'max_drop_node': features['max_drop_node'],
                            'num_affected_nodes': features['num_affected_nodes'],
                            'affected_nodes': features['affected_nodes'],
                            'affected_partitions': features['affected_partitions'],
                            'top5_pressure_drops': features['top5_pressure_drops'],
                            'top5_flow_changes': features['top5_flow_changes'],
                            'delta_pressure': features['delta_pressure'],
                            'delta_flow': features['delta_flow'],
                        }

                        # Save
                        with open(scenario_file, 'w') as f:
                            json.dump(scenario, f, indent=2)

                        scenario_count += 1
                        all_scenario_ids.append(scenario_id)

                        # Print progress
                        if scenario_count % 10 == 0 or scenario_count <= 3:
                            print(
                                f"    [{scenario_count}/{total_scenarios}] "
                                f"{scenario_id}: "
                                f"max_drop={features['max_pressure_drop']:.2f}m, "
                                f"affected={features['num_affected_nodes']} nodes"
                            )

                    except Exception as e:
                        print(f"    WARNING: {scenario_id} failed: {e}")
                        continue

                if max_scenarios and scenario_count >= max_scenarios:
                    break
            if max_scenarios and scenario_count >= max_scenarios:
                break
        if max_scenarios and scenario_count >= max_scenarios:
            break

    print(f"\n  Completed: {scenario_count} scenarios simulated")
    print(f"  Results saved in: {sim_dir}/")

    # Save scenario index
    index_file = os.path.join(output_dir, 'scenario_index.json')
    with open(index_file, 'w') as f:
        json.dump(all_scenario_ids, f, indent=2)
    print(f"  Index saved: {index_file}")

    return all_scenario_ids


if __name__ == '__main__':
    run_batch_simulation(
        inp_file='dataset/Exa7.inp',
        partition_file='partition_results_leiden/partitions.pkl',
        partition_k=15,
        output_dir='knowledge_base',
        max_scenarios=None,
        nodes_per_partition=2,
    )
