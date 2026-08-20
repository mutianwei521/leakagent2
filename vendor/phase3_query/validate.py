"""
Phase 3 Validation: Sensor-Enhanced Leak Localization

Tests the improved TS-GraphRAG pipeline with sensor-only observations,
dual-channel retrieval, and LLM reasoning across multiple scenarios.
Reports accuracy by leak severity.
"""
import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase3_query.leak_locator import LeakLocator, simulate_sensor_anomaly


def run_validation(kb_dir='knowledge_base', sensor_dir='sensor_results',
                   num_tests=10, seed=42):
    """Run validation across diverse scenarios grouped by severity."""

    print("=" * 60)
    print("TS-GraphRAG Phase 3 Validation (Sensor-Enhanced)")
    print("=" * 60)

    locator = LeakLocator(kb_dir, sensor_dir)

    # Discover all scenarios
    sim_dir = os.path.join(kb_dir, 'simulation_results')
    all_files = sorted([f[:-5] for f in os.listdir(sim_dir) if f.endswith('.json')])

    # Group by severity
    severity_groups = {
        'micro': [], 'small': [], 'small-moderate': [],
        'moderate': [], 'moderate-severe': [], 'severe': []
    }
    for f in all_files:
        for sev in severity_groups:
            tag = sev.replace('-', '')  # e.g., smallmoderate
            # Parse from filename
            parts = f.split('_')
            rate_str = parts[3]  # e.g., '2Ls', '10Ls'
            rate = int(rate_str.replace('Ls', ''))
            rate_to_sev = {2: 'micro', 5: 'small', 10: 'small-moderate',
                           20: 'moderate', 35: 'moderate-severe', 50: 'severe'}
            actual_sev = rate_to_sev.get(rate, 'unknown')
            if actual_sev not in severity_groups:
                severity_groups[actual_sev] = []
            severity_groups[actual_sev].append(f)
            break

    # Select test scenarios: pick from each severity
    np.random.seed(seed)
    per_severity = max(1, num_tests // len([g for g in severity_groups.values() if g]))

    test_scenarios = []
    for sev in ['micro', 'small', 'small-moderate', 'moderate',
                'moderate-severe', 'severe']:
        pool = severity_groups.get(sev, [])
        if pool:
            np.random.shuffle(pool)
            test_scenarios.extend(pool[:per_severity])

    test_scenarios = test_scenarios[:num_tests]

    print(f"\nSelected {len(test_scenarios)} test scenarios:\n")
    for ts in test_scenarios:
        print(f"  - {ts}")

    # Run
    results = []
    correct_total = 0
    severity_stats = {}

    for i, scenario_id in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_scenarios)}: {scenario_id}")
        print("=" * 60)

        obs, gt = simulate_sensor_anomaly(
            kb_dir, scenario_id, locator.sensor_nodes, noise_std=0.02
        )
        gt_partition = gt['leak_partition']
        sev = gt.get('leak_severity', 'unknown')

        # Run localization (quiet mode)
        try:
            result = locator.localize_leak(
                obs, demand_period=gt['demand_period'], top_k=10, verbose=False
            )
            pred = result.get('predicted_partition', -1)
        except Exception as e:
            print(f"  ERROR: {e}")
            pred = -1
            result = {'predicted_partition': -1, 'confidence': 'error'}

        is_correct = pred == gt_partition
        if is_correct:
            correct_total += 1

        print(f"  GT: P#{gt_partition} | Pred: P#{pred} | "
              f"{'✅' if is_correct else '❌'} | "
              f"Severity: {sev} | "
              f"Confidence: {result.get('confidence', 'N/A')}")

        if sev not in severity_stats:
            severity_stats[sev] = {'total': 0, 'correct': 0}
        severity_stats[sev]['total'] += 1
        if is_correct:
            severity_stats[sev]['correct'] += 1

        results.append({
            'scenario_id': scenario_id,
            'ground_truth_partition': gt_partition,
            'predicted_partition': pred,
            'correct': is_correct,
            'severity': sev,
            'confidence': result.get('confidence', 'N/A'),
        })

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Overall: {correct_total}/{len(results)} "
          f"({100*correct_total/len(results):.0f}%)")

    print(f"\n  By Severity:")
    for sev in ['micro', 'small', 'small-moderate',
                'moderate', 'moderate-severe', 'severe']:
        stats = severity_stats.get(sev)
        if stats:
            pct = 100 * stats['correct'] / stats['total']
            print(f"    {sev:20s}: {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    # Save
    out_file = os.path.join(kb_dir, 'validation_results_v2.json')
    with open(out_file, 'w') as f:
        json.dump({'results': results, 'severity_stats': severity_stats,
                   'overall_accuracy': correct_total / len(results)}, f, indent=2)
    print(f"\n  Saved: {out_file}")

    return results


if __name__ == '__main__':
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run_validation(num_tests=num)
