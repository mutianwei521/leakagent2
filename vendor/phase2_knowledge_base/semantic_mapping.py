"""
Step 2.1: Static Semantic Mapping

Converts physical parameters of nodes, pipes, pumps, tanks, and reservoirs
into descriptive English text. Annotates each element with its Leiden partition ID.
"""
import os
import json
import wntr
import pickle
import numpy as np


def load_partition(partition_file, partition_k):
    """Load a specific partition result (k communities) from the pickled file."""
    with open(partition_file, 'rb') as f:
        all_partitions = pickle.load(f)

    if partition_k not in all_partitions:
        available = sorted(all_partitions.keys())
        raise ValueError(
            f"Partition k={partition_k} not found. Available: {available}"
        )

    data = all_partitions[partition_k]
    return data['node_to_community'], data['resolution']


def get_node_connections(wn, node_name):
    """Get all pipes/links connected to a given node."""
    connections = []
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        if link.start_node_name == node_name or link.end_node_name == node_name:
            connections.append(link_name)
    return connections


def map_junctions(wn, node_to_community):
    """Generate semantic descriptions for all junctions."""
    semantic_nodes = {}

    for junc_name in wn.junction_name_list:
        junc = wn.get_node(junc_name)
        partition_id = node_to_community.get(junc_name, -1)
        connections = get_node_connections(wn, junc_name)

        # Get base demand
        base_demand = 0.0
        demand_pattern = "None"
        if junc.demand_timeseries_list:
            for demand in junc.demand_timeseries_list:
                base_demand += demand.base_value
                if demand.pattern_name:
                    demand_pattern = demand.pattern_name

        # Build connection descriptions
        conn_descs = []
        for link_name in connections:
            link = wn.get_link(link_name)
            link_type = type(link).__name__
            if hasattr(link, 'diameter'):
                diameter_in = link.diameter * 39.3701  # m to inches
                conn_descs.append(
                    f"{link_name}({link_type}, {diameter_in:.0f}in)"
                )
            else:
                conn_descs.append(f"{link_name}({link_type})")

        conn_str = ", ".join(conn_descs) if conn_descs else "none"

        # Classify demand level
        if base_demand > 30:
            demand_level = "major consumer"
        elif base_demand > 10:
            demand_level = "moderate consumer"
        elif base_demand > 0:
            demand_level = "minor consumer"
        else:
            demand_level = "zero-demand node"

        desc = (
            f"Junction {junc_name} is located in Partition #{partition_id}, "
            f"at elevation {junc.elevation:.1f}m. "
            f"Base demand: {base_demand:.2f} L/s ({demand_level}), "
            f"pattern: {demand_pattern}. "
            f"Connected links: [{conn_str}]. "
            f"Coordinates: ({junc.coordinates[0]:.1f}, {junc.coordinates[1]:.1f})."
        )

        semantic_nodes[junc_name] = {
            'type': 'junction',
            'partition_id': partition_id,
            'elevation': junc.elevation,
            'base_demand': base_demand,
            'demand_pattern': demand_pattern,
            'demand_level': demand_level,
            'connections': connections,
            'coordinates': list(junc.coordinates),
            'description': desc,
        }

    return semantic_nodes


def map_reservoirs(wn, node_to_community):
    """Generate semantic descriptions for reservoirs."""
    semantic = {}
    for res_name in wn.reservoir_name_list:
        res = wn.get_node(res_name)
        partition_id = node_to_community.get(res_name, -1)
        connections = get_node_connections(wn, res_name)
        conn_str = ", ".join(connections) if connections else "none"

        desc = (
            f"Reservoir {res_name} in Partition #{partition_id}, "
            f"head = {res.base_head:.1f}m. "
            f"This is a water source node supplying the network. "
            f"Connected links: [{conn_str}]."
        )

        semantic[res_name] = {
            'type': 'reservoir',
            'partition_id': partition_id,
            'base_head': res.base_head,
            'connections': connections,
            'coordinates': list(res.coordinates),
            'description': desc,
        }

    return semantic


def map_tanks(wn, node_to_community):
    """Generate semantic descriptions for tanks."""
    semantic = {}
    for tank_name in wn.tank_name_list:
        tank = wn.get_node(tank_name)
        partition_id = node_to_community.get(tank_name, -1)
        connections = get_node_connections(wn, tank_name)
        conn_str = ", ".join(connections) if connections else "none"

        desc = (
            f"Tank {tank_name} in Partition #{partition_id}, "
            f"elevation {tank.elevation:.1f}m, "
            f"init level {tank.init_level:.1f}m, "
            f"min level {tank.min_level:.1f}m, max level {tank.max_level:.1f}m, "
            f"diameter {tank.diameter:.1f}m. "
            f"Connected links: [{conn_str}]."
        )

        semantic[tank_name] = {
            'type': 'tank',
            'partition_id': partition_id,
            'elevation': tank.elevation,
            'init_level': tank.init_level,
            'min_level': tank.min_level,
            'max_level': tank.max_level,
            'diameter': tank.diameter,
            'connections': connections,
            'coordinates': list(tank.coordinates),
            'description': desc,
        }

    return semantic


def map_pipes(wn, node_to_community):
    """Generate semantic descriptions for all pipes."""
    semantic_pipes = {}

    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        start = pipe.start_node_name
        end = pipe.end_node_name
        start_part = node_to_community.get(start, -1)
        end_part = node_to_community.get(end, -1)

        diameter_in = pipe.diameter * 39.3701  # m to inches
        length_m = pipe.length

        # Classify pipe
        if diameter_in >= 16:
            pipe_class = "trunk main"
        elif diameter_in >= 8:
            pipe_class = "distribution main"
        else:
            pipe_class = "service line"

        # Cross-partition?
        if start_part != end_part:
            boundary_note = (
                f" This pipe crosses partition boundaries "
                f"(P#{start_part} <-> P#{end_part})."
            )
        else:
            boundary_note = ""

        desc = (
            f"Pipe {pipe_name}: {start} -> {end}, "
            f"length {length_m:.0f}m, diameter {diameter_in:.0f}in "
            f"({pipe_class}), roughness {pipe.roughness:.0f}.{boundary_note}"
        )

        semantic_pipes[pipe_name] = {
            'type': 'pipe',
            'start_node': start,
            'end_node': end,
            'start_partition': start_part,
            'end_partition': end_part,
            'length_m': length_m,
            'diameter_in': round(diameter_in, 1),
            'roughness': pipe.roughness,
            'pipe_class': pipe_class,
            'is_boundary_pipe': start_part != end_part,
            'description': desc,
        }

    return semantic_pipes


def map_pumps(wn, node_to_community):
    """Generate semantic descriptions for all pumps."""
    semantic_pumps = {}

    for pump_name in wn.pump_name_list:
        pump = wn.get_link(pump_name)
        start = pump.start_node_name
        end = pump.end_node_name

        desc = (
            f"Pump {pump_name}: {start} -> {end}, "
            f"speed {pump.speed_timeseries.base_value:.1f}. "
            f"This is an active hydraulic element boosting pressure."
        )

        semantic_pumps[pump_name] = {
            'type': 'pump',
            'start_node': start,
            'end_node': end,
            'speed': pump.speed_timeseries.base_value,
            'description': desc,
        }

    return semantic_pumps


def run_semantic_mapping(inp_file, partition_file, partition_k, output_dir):
    """
    Main function: run full semantic mapping.

    Args:
        inp_file: path to EPANET INP file
        partition_file: path to partitions.pkl
        partition_k: number of communities to use
        output_dir: directory to save JSON outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Step 2.1: Static Semantic Mapping")
    print("=" * 60)

    # Load network
    wn = wntr.network.WaterNetworkModel(inp_file)
    print(f"  Network loaded: {len(wn.junction_name_list)} junctions, "
          f"{len(wn.pipe_name_list)} pipes, "
          f"{len(wn.pump_name_list)} pumps, "
          f"{len(wn.tank_name_list)} tanks, "
          f"{len(wn.reservoir_name_list)} reservoirs")

    # Load partition
    node_to_community, resolution = load_partition(partition_file, partition_k)
    print(f"  Partition loaded: k={partition_k}, resolution={resolution}")

    # Map junctions
    print("  Mapping junctions...")
    semantic_junctions = map_junctions(wn, node_to_community)

    # Map reservoirs
    print("  Mapping reservoirs...")
    semantic_reservoirs = map_reservoirs(wn, node_to_community)

    # Map tanks
    print("  Mapping tanks...")
    semantic_tanks = map_tanks(wn, node_to_community)

    # Combine all nodes
    semantic_nodes = {}
    semantic_nodes.update(semantic_junctions)
    semantic_nodes.update(semantic_reservoirs)
    semantic_nodes.update(semantic_tanks)
    print(f"  Total node descriptions: {len(semantic_nodes)}")

    # Map pipes
    print("  Mapping pipes...")
    semantic_pipes = map_pipes(wn, node_to_community)

    # Map pumps
    print("  Mapping pumps...")
    semantic_pumps = map_pumps(wn, node_to_community)

    # Combine all links
    semantic_links = {}
    semantic_links.update(semantic_pipes)
    semantic_links.update(semantic_pumps)
    print(f"  Total link descriptions: {len(semantic_links)}")

    # Count boundary pipes
    boundary_count = sum(
        1 for v in semantic_pipes.values() if v.get('is_boundary_pipe')
    )
    print(f"  Boundary pipes (cross-partition): {boundary_count}")

    # Save outputs
    nodes_file = os.path.join(output_dir, 'semantic_nodes.json')
    with open(nodes_file, 'w', encoding='utf-8') as f:
        json.dump(semantic_nodes, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {nodes_file}")

    links_file = os.path.join(output_dir, 'semantic_links.json')
    with open(links_file, 'w', encoding='utf-8') as f:
        json.dump(semantic_links, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {links_file}")

    return semantic_nodes, semantic_links


if __name__ == '__main__':
    run_semantic_mapping(
        inp_file='dataset/Exa7.inp',
        partition_file='partition_results_leiden/partitions.pkl',
        partition_k=15,
        output_dir='knowledge_base',
    )
