"""
Step 2.4: Vector Store

Embeds scenario summaries using sentence-transformers (all-MiniLM-L6-v2)
and stores them in LanceDB for efficient similarity search.
Also builds a topology index for partition-aware retrieval.
"""
import os
import json
import numpy as np
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LANCEDB_TABLE_NAME = "leak_scenarios"


def build_topology_index(knowledge_base_dir, partition_file, partition_k):
    """
    Build a topology index recording partition structure and inter-partition
    pipe connections.
    """
    import pickle

    with open(partition_file, 'rb') as f:
        all_partitions = pickle.load(f)

    data = all_partitions[partition_k]
    node_to_community = data['node_to_community']

    # Group nodes by partition
    partition_nodes = {}
    for node, pid in node_to_community.items():
        if pid not in partition_nodes:
            partition_nodes[pid] = []
        partition_nodes[pid].append(node)

    # Load semantic links to find boundary pipes
    links_file = os.path.join(knowledge_base_dir, 'semantic_links.json')
    boundary_connections = {}
    if os.path.exists(links_file):
        with open(links_file, 'r', encoding='utf-8') as f:
            semantic_links = json.load(f)

        for link_name, link_data in semantic_links.items():
            if link_data.get('is_boundary_pipe'):
                sp = link_data['start_partition']
                ep = link_data['end_partition']
                pair = f"{min(sp, ep)}-{max(sp, ep)}"
                if pair not in boundary_connections:
                    boundary_connections[pair] = []
                boundary_connections[pair].append(link_name)

    topology_index = {
        'partition_k': partition_k,
        'num_partitions': len(partition_nodes),
        'partitions': {
            str(pid): {
                'node_count': len(nodes),
                'nodes': sorted(nodes),
            }
            for pid, nodes in sorted(partition_nodes.items())
        },
        'boundary_pipes': boundary_connections,
    }

    index_file = os.path.join(knowledge_base_dir, 'topology_index.json')
    with open(index_file, 'w') as f:
        json.dump(topology_index, f, indent=2)
    print(f"  Topology index saved: {index_file}")

    return topology_index


def run_vector_store(knowledge_base_dir, partition_file, partition_k):
    """
    Main function: embed scenario summaries and store in LanceDB.

    Args:
        knowledge_base_dir: directory containing scenario_summaries.json
        partition_file: path to partitions.pkl
        partition_k: number of partitions for topology index
    """
    print("\n" + "=" * 60)
    print("Step 2.4: Vector Embedding & Storage")
    print("=" * 60)

    # Load scenario summaries
    summaries_file = os.path.join(knowledge_base_dir, 'scenario_summaries.json')
    print(f"  Loading summaries: {summaries_file}")
    with open(summaries_file, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    print(f"  Loaded {len(summaries)} scenario summaries")

    if not summaries:
        print("  ERROR: No summaries to process!")
        return

    # Initialize sentence transformer
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {embedding_dim}")

    # Generate embeddings
    print("  Generating embeddings...")
    texts = [s['summary_text'] for s in summaries]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"  Generated {len(embeddings)} embeddings")

    # Prepare data for LanceDB
    records = []
    for i, summary in enumerate(summaries):
        records.append({
            'scenario_id': summary['scenario_id'],
            'summary_text': summary['summary_text'],
            'vector': embeddings[i].tolist(),
            'leak_partition': summary['leak_partition'],
            'leak_node': summary['leak_node'],
            'leak_rate_Ls': summary['leak_rate_Ls'],
            'leak_severity': summary.get('leak_severity', ''),
            'demand_period': summary['demand_period'],
            'demand_multiplier': summary['demand_multiplier'],
            'max_pressure_drop': summary['max_pressure_drop'],
            'num_affected_nodes': summary['num_affected_nodes'],
            'affected_partitions': json.dumps(
                summary['affected_partitions']
            ),
        })

    # Create LanceDB database
    db_path = os.path.join(knowledge_base_dir, 'lancedb')
    print(f"  Creating LanceDB at: {db_path}")
    db = lancedb.connect(db_path)

    # Drop existing table if present
    if LANCEDB_TABLE_NAME in db.table_names():
        db.drop_table(LANCEDB_TABLE_NAME)
        print(f"  Dropped existing table: {LANCEDB_TABLE_NAME}")

    # Create table
    table = db.create_table(LANCEDB_TABLE_NAME, data=records)
    print(f"  Created table '{LANCEDB_TABLE_NAME}' with {len(records)} rows")

    # Build topology index
    print("\n  Building topology index...")
    build_topology_index(knowledge_base_dir, partition_file, partition_k)

    # Verification: run a test query
    print("\n  Verification: running test similarity search...")
    test_query = "moderate leak causing pressure drop in partition 5"
    test_embedding = model.encode([test_query])[0].tolist()
    results = table.search(test_embedding).limit(3).to_pandas()

    print(f"  Query: \"{test_query}\"")
    print(f"  Top-3 results:")
    for _, row in results.iterrows():
        print(
            f"    - {row['scenario_id']}: "
            f"partition={row['leak_partition']}, "
            f"drop={row['max_pressure_drop']:.2f}m"
        )

    print(f"\n  Vector store complete! "
          f"Database at: {db_path}")

    return db, table


if __name__ == '__main__':
    run_vector_store(
        knowledge_base_dir='knowledge_base',
        partition_file='partition_results_leiden/partitions.pkl',
        partition_k=15,
    )
