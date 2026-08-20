"""
Step 2.3: Scenario Summary Generator

Uses LangChain + Ollama (gpt-oss:120b-cloud) to generate standardized
English scenario description texts from simulation results + semantic data.
Supports checkpoint/resume for batch processing.
"""
import os
import json
import glob
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# LLM Configuration
OLLAMA_MODEL = "gpt-oss:120b-cloud"

SYSTEM_PROMPT = """You are a water distribution network leak analysis expert. 
Given hydraulic simulation data, generate a concise English scenario description 
(150-200 words). Focus on:
1. Topological characteristics of the leak location
2. Hydraulic response propagation pattern  
3. Extent of affected areas
4. Severity assessment

Be precise with numbers. Use engineering terminology. Do NOT use bullet points, 
write in flowing paragraph form."""

SCENARIO_TEMPLATE = """## Leak Information
- Leak Node: {leak_node} (Partition #{leak_partition})
- Leak Rate: {leak_rate_Ls} L/s ({leak_severity})
- Demand Period: {demand_label} (multiplier: {demand_multiplier})

## Leak Node Properties
{node_description}

## Hydraulic Response
- Maximum pressure drop: {max_pressure_drop:.2f} m at Node {max_drop_node}
- Number of affected nodes (>0.5m drop): {num_affected_nodes}
- Affected partitions: {affected_partitions}

## Top-5 Pressure Drops (node: drop in meters):
{top5_pressure_drops_text}

## Top-5 Flow Changes (link: change in m³/s):
{top5_flow_changes_text}

Generate a 150-200 word scenario description based on the above data."""


def format_top5(data_dict):
    """Format a top-5 dict into readable lines."""
    lines = []
    for key, value in data_dict.items():
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines) if lines else "  - None"


def build_prompt(scenario, semantic_nodes):
    """Build a filled prompt for one scenario."""
    leak_node = scenario['leak_node']
    node_desc = "No description available."
    if leak_node in semantic_nodes:
        node_desc = semantic_nodes[leak_node].get('description', node_desc)

    prompt_text = SCENARIO_TEMPLATE.format(
        leak_node=leak_node,
        leak_partition=scenario['leak_partition'],
        leak_rate_Ls=scenario['leak_rate_Ls'],
        leak_severity=scenario.get('leak_severity', 'unknown'),
        demand_label=scenario.get('demand_label', scenario['demand_period']),
        demand_multiplier=scenario['demand_multiplier'],
        node_description=node_desc,
        max_pressure_drop=scenario['max_pressure_drop'],
        max_drop_node=scenario['max_drop_node'],
        num_affected_nodes=scenario['num_affected_nodes'],
        affected_partitions=scenario['affected_partitions'],
        top5_pressure_drops_text=format_top5(scenario['top5_pressure_drops']),
        top5_flow_changes_text=format_top5(scenario['top5_flow_changes']),
    )

    return prompt_text


def generate_summary(llm, scenario, semantic_nodes):
    """Generate one summary using the LLM."""
    prompt = build_prompt(scenario, semantic_nodes)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def run_scenario_summary_generation(knowledge_base_dir, test_single=False):
    """
    Main function: generate English summaries for all simulation scenarios.

    Args:
        knowledge_base_dir: directory containing semantic_nodes.json & simulation_results/
        test_single: if True, only process one scenario for testing
    """
    print("\n" + "=" * 60)
    print("Step 2.3: Scenario Summary Generation (LLM)")
    print("=" * 60)

    # Load semantic node data
    semantic_file = os.path.join(knowledge_base_dir, 'semantic_nodes.json')
    print(f"  Loading semantic data: {semantic_file}")
    with open(semantic_file, 'r', encoding='utf-8') as f:
        semantic_nodes = json.load(f)
    print(f"  Loaded {len(semantic_nodes)} node descriptions")

    # Load existing summaries (checkpoint support)
    summaries_file = os.path.join(knowledge_base_dir, 'scenario_summaries.json')
    existing_summaries = {}
    if os.path.exists(summaries_file):
        with open(summaries_file, 'r', encoding='utf-8') as f:
            existing_list = json.load(f)
        existing_summaries = {s['scenario_id']: s for s in existing_list}
        print(f"  Found {len(existing_summaries)} existing summaries (checkpoint)")

    # Discover simulation result files
    sim_dir = os.path.join(knowledge_base_dir, 'simulation_results')
    scenario_files = sorted(glob.glob(os.path.join(sim_dir, 'leak_*.json')))
    print(f"  Found {len(scenario_files)} simulation scenarios")

    if test_single:
        scenario_files = scenario_files[:1]
        print("  [TEST MODE] Processing only 1 scenario")

    # Initialize LLM
    print(f"  Initializing LLM: ollama/{OLLAMA_MODEL}")
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.3,
        num_predict=512,
    )

    # Process scenarios
    all_summaries = list(existing_summaries.values())
    new_count = 0
    skip_count = 0
    error_count = 0

    for i, sf in enumerate(scenario_files):
        # Load scenario data
        with open(sf, 'r') as f:
            scenario = json.load(f)

        sid = scenario['scenario_id']

        # Skip if already processed
        if sid in existing_summaries:
            skip_count += 1
            continue

        try:
            # Generate summary
            summary_text = generate_summary(llm, scenario, semantic_nodes)

            summary_record = {
                'scenario_id': sid,
                'summary_text': summary_text,
                'leak_partition': scenario['leak_partition'],
                'leak_node': scenario['leak_node'],
                'leak_rate_Ls': scenario['leak_rate_Ls'],
                'leak_severity': scenario.get('leak_severity', ''),
                'demand_period': scenario['demand_period'],
                'demand_multiplier': scenario['demand_multiplier'],
                'max_pressure_drop': scenario['max_pressure_drop'],
                'num_affected_nodes': scenario['num_affected_nodes'],
                'affected_partitions': scenario['affected_partitions'],
            }

            all_summaries.append(summary_record)
            new_count += 1

            # Progress
            total = len(scenario_files)
            print(
                f"    [{i+1}/{total}] {sid}: "
                f"generated ({len(summary_text)} chars)"
            )

            # Checkpoint: save every 10 new summaries
            if new_count % 10 == 0:
                with open(summaries_file, 'w', encoding='utf-8') as f:
                    json.dump(all_summaries, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"    [{i+1}/{len(scenario_files)}] ERROR {sid}: {e}")
            error_count += 1
            continue

    # Final save
    with open(summaries_file, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print(f"\n  Summary generation complete:")
    print(f"    New: {new_count}")
    print(f"    Skipped (checkpoint): {skip_count}")
    print(f"    Errors: {error_count}")
    print(f"    Total: {len(all_summaries)}")
    print(f"  Saved: {summaries_file}")

    return all_summaries


if __name__ == '__main__':
    import sys
    test_mode = '--test-single' in sys.argv
    run_scenario_summary_generation(
        knowledge_base_dir='knowledge_base',
        test_single=test_mode,
    )
